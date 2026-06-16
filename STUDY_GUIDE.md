# AES-GCM Engine — Technical Study & Oral Examination Guide
### Principal Cryptographic Engineering Review

This document is a defense-ready technical dossier for the AES-GCM implementation: a NIST SP 800-38D–conformant CPU baseline (Phase 1) co-designed with, and now fully realized by, a CUDA acceleration pipeline (Phases 2–4). It is structured for panel-style interrogation — each section anticipates the "why," not just the "what."

---

## 1. Theoretical Foundations & Mathematics

### 1.1 The Field: GF(2¹²⁸)

GHASH operates over the finite field **GF(2¹²⁸)**, constructed as the quotient ring:
`GF(2^128) = GF(2)[x] / (x^128 + x^7 + x^2 + x + 1)`

Every element is a polynomial of degree ≤ 127 with binary coefficients — equivalently, a 128-bit string where **bit `i` is the coefficient of `x^i`**. Addition in this field is defined as polynomial addition with coefficients in GF(2), which collapses to **bitwise XOR** — there is no carry, because `1 + 1 = 0` in GF(2). This is why `gf128_xor` ([gf128.h:30](include/gf128.h#L30)) is a trivial two-instruction XOR of the `hi`/`lo` limbs: field addition and bitwise XOR are *the same operation*, not an analogy.

Multiplication is polynomial multiplication followed by **reduction modulo the irreducible polynomial** `x^128 + x^7 + x^2 + x + 1`. This specific polynomial was chosen by NIST because it is a primitive, irreducible trinomial-adjacent polynomial in GF(2)[x] of minimal Hamming weight above degree 128 (only 5 nonzero terms: `x^128, x^7, x^2, x, 1`), which keeps the reduction step cheap — reducing a degree-254 product back into the field only requires correcting for 4 low-order terms (`x^7, x^2, x^1, x^0`) every time the multiplication overflows past bit 128.

### 1.2 Reflected Bit-Ordering and the 0xE1 Reduction Constant

NIST SP 800-38D specifies GHASH using a **reflected (bit-reversed) representation**, a historical artifact of the original Galois/Counter Mode hardware design that processes bits LSB-first. This implementation faithfully follows that convention rather than a "natural" big-endian polynomial encoding — see the struct documentation at [gf128.h:13-22](include/gf128.h#L13-L22):

```cpp
struct GF128 {
    uint64_t hi;   // bits 127..64  (MSB = coeff of x^0)
    uint64_t lo;   // bits  63..0
};
```
The critical, easily-misunderstood detail: `hi`'s most-significant bit is the coefficient of `x^0`, not `x^127`. The byte string `0x80 0x00 ... 0x00` — which looks numerically large in a naive big-endian reading — represents the multiplicative identity, element 1 (`gf128_one()`, `gf128.h:28`). This single inversion is the root of nearly every subtle GHASH bug in re-implementations; it is also precisely the convention that produced the device/host endianness defect found and fixed in `mapreduce.cu` during GPU bring-up (§5.3 below).

Given this reflection, the reduction polynomial `x^128 + x^7 + x^2 + x + 1` does not get represented as its "natural" bit pattern. In the reflected domain, the reduction constant is:

```cpp
const uint64_t R = 0xE100000000000000ULL;
```
Decompose `0xE1` in binary: `1110 0001`. Reading this as reflected coefficients of `x^7, x^6, x^5, ..., x^0` recovers exactly the non-`x^128` terms of the reduction polynomial — `x^7 + x^2 + x + 1` — placed in the high byte of the 64-bit word because reduction acts on the bit that overflows out of the low end of the reflected representation. `R` is not an arbitrary magic constant; it is the field's defining polynomial, re-expressed in the coordinate system the rest of the codebase uses.

### 1.3 Right-to-Left Multiplication and Conditional Reduction
`gf128_mul` (`gf128.h:70-90`) implements the standard shift-and-add (peasant/Russian) multiplication adapted to the reflected field:

```cpp
GF128_FUNC GF128 gf128_mul(GF128 X, GF128 Y) {
    const uint64_t R = 0xE100000000000000ULL;
    uint64_t Zhi = 0, Zlo = 0;
    uint64_t Vhi = X.hi, Vlo = X.lo;

    for (int i = 0; i < 128; i++) {
        uint64_t ybit = (i < 64) ? ((Y.hi >> (63 - i)) & 1)
                                  : ((Y.lo >> (127 - i)) & 1);
        uint64_t mask = -(uint64_t)ybit;
        Zhi ^= Vhi & mask;
        Zlo ^= Vlo & mask;

        uint64_t lsb = Vlo & 1;
        Vlo = (Vlo >> 1) | (Vhi << 63);
        Vhi >>= 1;
        Vhi ^= R & -(uint64_t)lsb;
    }
    return { Zhi, Zlo };
}
```
The algorithm walks `Y`'s coefficients from `x^0` toward `x^127` (note `i` indexing `Y.hi` MSB-first, then `Y.lo`), maintaining `V` as a running copy of `X` shifted by the current power. At each of the 128 iterations:

1.  **Conditional accumulation:** if the current bit of `Y` is 1, XOR the current `V` (`= X · x^i`) into the accumulator `Z`. This is the "add" half of shift-and-add — `Z` accumulates `Σ X·x^i` for every set bit `i` in `Y`, which by distributivity equals `X · Y`.
2.  **Shift with carry propagation:** `V` is shifted right by one bit across the 128-bit pair (`Vlo` absorbs the bit falling out of `Vhi`'s bottom via `Vhi << 63`), representing multiplication of the running term by `x`.
3.  **Conditional reduction:** the bit that fell off the bottom of `V` (`Vlo & 1`, captured as `lsb` before the shift) indicates the term would have produced an `x^128` coefficient — i.e., the polynomial degree just exceeded the field's modulus. Because `x^128 ≡ x^7 + x^2 + x + 1 (mod f(x))`, this overflow is corrected immediately by XORing `R` into `Vhi`, exactly substituting the high-degree term for its low-degree equivalent before the next iteration uses `V`.

This is degree-reduction performed incrementally, every step, rather than once at the end — a property essential for the **constant-time guarantee**: every iteration does the identical sequence of operations (shift, conditional mask, XOR) regardless of the operand bits, with both conditionals implemented via arithmetic masking (`-(uint64_t)bit`) instead of branches. There is no `if` statement whose branch depends on secret data (`X` is the running hash state — secret-dependent; `Y` is the public hash subkey power) — eliminating a timing side-channel that a naive branch-on-bit implementation would introduce.

---

## 2. Codebase Anatomy & Execution Flow
### 2.1 The AesGcmCtx Structure

```cpp
struct AesGcmCtx {
    uint8_t  key[AES_256_KEY_LEN];   // up to 256-bit key
    uint8_t  iv[GCM_IV_LEN];         // 96-bit IV
    uint8_t  H[AES_BLOCK_SIZE];      // hash subkey: H = AES(K, 0^128)
    uint32_t round_keys[60];         // 15 round keys × 4 words (covers AES-256)
    int      key_len;                // 16 (AES-128) or 32 (AES-256)
    int      num_rounds;             // 10 or 14
};
```

This struct is the entire mutable cryptographic state for a session, and its layout is deliberately key-length-polymorphic:
*   **`key[32]`** is sized for the AES-256 maximum, but only the first `key_len` bytes are meaningful — `key_len` is the runtime discriminator (16 vs. 32) that lets a single struct represent either cipher variant without a tagged union or virtual dispatch.
*   **`H[16]`** is not the AES key — it is the GHASH hash subkey, derived once at initialization as `H = AES_K(0^128)`. Deriving `H` from the same key under all-zero plaintext means an attacker who doesn't know `K` cannot predict `H`, binding the authentication mechanism cryptographically to the encryption key without requiring a second secret.
*   **`round_keys[60]`** holds the unrolled key schedule — 4 words × up to 15 round-key sets (`AES_256_ROUNDS = 14` rounds → 15 round keys including the initial whitening key). This precomputation at `aes_gcm_init` time means every subsequent block encryption is a pure lookup-and-XOR loop with zero per-block key-derivation cost — critical for both CPU throughput and, later, for placing the schedule in CUDA `__constant__` memory where it is broadcast-cached identically to every thread in a warp.
*   **`num_rounds`** (10 or 14) is the loop bound for every AES block operation and the template parameter selecting which compiled GPU kernel variant (`aes_ctr_kernel<10>` vs. `<14>`) is dispatched at the CUDA layer — the CPU struct field and the GPU compile-time specialization are kept in lockstep by this one integer.

### 2.2 Encryption Data Flow — Step by Step

**Step 1 — Counter block configuration (the J0/counter-2 isolation).**
`aes_ctr_crypt(plaintext, ciphertext, pt_len, ctx->iv, ctx->round_keys, ctx->num_rounds, 2)`
The CTR keystream generator is invoked with `start_ctr = 2`. This is not arbitrary — NIST SP 800-38D §7.1 defines the initial counter block `J0` as `IV || 0^31 || 1` (32-bit counter field = 1), and mandates that plaintext encryption begin at `inc32(J0)`, i.e., counter value 2. Counter value 1 is reserved exclusively for masking the authentication tag in Step 5. If plaintext encryption were ever allowed to consume counter 1, the keystream block `E(K, J0)` used for the tag mask would coincide with a keystream block already XORed into (and therefore recoverable from) the ciphertext — collapsing the tag's secrecy and enabling forgery. 

**Step 2 — Ciphertext block streaming via AES-CTR.**
Inside `aes_ctr_crypt`, each 16-byte block is produced by forming a counter block `IV (96 bits) || counter (32 bits, big-endian)`, AES-encrypting it once under the expanded `round_keys`, and XORing the result with the corresponding plaintext block. Because CTR mode's keystream blocks are mutually independent — block `i`'s keystream depends only on `IV` and `i`, never on block `i-1`'s output — this loop has no data dependency between iterations. This independence is the load-bearing property that makes Phase 2's "one CUDA thread per block" mapping valid.

**Step 3 — GHASH input formatting.**
NIST §7.1 defines the authenticated data fed to GHASH as: `pad(A) || pad(C) || len(A)_64 || len(C)_64`. The implementation zero-pads AAD (`A`) and ciphertext (`C`) independently to 16-byte boundaries, then appends a final 16-byte length block encoding `len(A)` and `len(C)` in bits. This length block is what gives GCM its protection against truncation and length-extension forgeries.

**Step 4 — Horner's method evaluation.**
```cpp
GF128 y = gf128_zero();
for (size_t i = 0; i < full_blocks; i++) {
    GF128 xi = gf128_load(data + i * AES_BLOCK_SIZE);
    y = gf128_mul(gf128_xor(y, xi), h);
}
```
This is a direct, sequential implementation of Horner's rule applied to the GHASH polynomial: `Y_i = (Y_{i-1} ⊕ X_i) · H`. This sequential dependency is precisely the bottleneck Phase 3's tree-reduction is engineered to eliminate by algebraically re-expanding the recurrence.

**Step 5 — Tag computation.**
`T = S ⊕ E(K, J0)`, where `S` is the final Horner accumulator and `J0` is AES-encrypted directly.

---

## 3. Command-Line Playbook

| Mode | Command Signature | Architectural Description |
|---|---|---|
| Validation | `./run_tests` | Loads the compiled NIST SP 800-38D test vector suite and runs both AES-128 and AES-256 variants. Exit criterion: 7/7 vectors must match bit-for-bit. |
| Encrypt | `./aes_gcm_demo encrypt --key ... --iv ... --in <file> --out <file>` | Parses hex keys, dispatches on length. Calls `aes_gcm_init`. Internally triggers the full Step 1→5 flow. Prints the resulting 128-bit tag in hex. |
| Decrypt | `./aes_gcm_demo decrypt --key ... --iv ... --tag ... --in <file> --out <file>` | Recomputes the tag identically to encryption, then performs `ct_memcmp` against the caller-supplied tag. Decryption itself does not execute unless the comparison succeeds (`aes_gcm_decrypt` returns -1 before the AES-CTR pass is ever invoked on mismatch). |
| Tamper | `./aes_gcm_demo tamper --key ... --iv ... --in <file>` | Encrypts the input file, deliberately flips a single bit of the resulting ciphertext, then immediately attempts `aes_gcm_decrypt`. `ct_memcmp` reports inequality, decryption is refused, and the program prints `Authentication FAILED — plaintext withheld`. |

---

## 4. Critical Security & Engineering Defenses

### 4.1 The Timing Side-Channel on Tag Comparison
A textbook `memcmp(computed_tag, received_tag, 16)` terminates as soon as the first differing byte is found. In an authentication context, it becomes an oracle: the time the comparison takes is a side-channel that leaks how many leading bytes of the attacker's guessed tag were correct. `ct_memcmp` eliminates this by construction. The loop always executes exactly `len` iterations regardless of where (or whether) bytes differ.

### 4.2 Payload Isolation — Scrubbing Plaintext on Authentication Failure
GCM's entire security model rests on the principle that plaintext must never be released to a caller until its integrity has been verified. In `aes_gcm_decrypt`, GHASH and tag comparison happen entirely before the AES-CTR decryption pass is ever invoked. On failure, `memset(plaintext, 0, ct_len)` proactively zeroes the caller's output buffer before returning.

---

## 5. Strategic Blueprint for CUDA Parallelization

### 5.1 Sequential Phase 1 vs. Parallel Phases 2–4
Phase 1's CPU implementation is built around two `O(N)` sequential loops: AES-CTR and GHASH. The CUDA transition's central insight is that these two loops have fundamentally different parallelization profiles — CTR is "embarrassingly parallel", but GHASH required algebraic re-derivation.

### 5.2 Routine-by-Routine Translation Table

*   **AES-CTR keystream generation:** Translated to one CUDA thread per 16-byte block. S-box cooperatively staged once per block from `__device__` global memory into `__shared__`. Round keys placed in `__constant__` memory. AES state held in 4 `uint32_t` registers. 128-bit coalesced memory access via `uint4`, with `__byte_perm` correcting the GPU's native little-endian word layout.
*   **GHASH accumulator:** Algebraic re-expansion from the Horner recurrence into its expanded summation form: `GHASH = X₁·Hᴺ ⊕ X₂·Hᴺ⁻¹ ⊕ ... ⊕ Xₙ·H¹`. Realized as a two-stage MAP→REDUCE kernel. The power table itself (`H_powers[i] = H^{N-i}`) is built via binary exponentiation against precomputed doublings.

### 5.3 A Note on the Translation's Hardest-Won Lesson
The reflected bit-ordering convention — where struct fields are stored in a machine-native (little-endian) layout, but `gf128_load`/`gf128_store` assume a big-endian byte-stream input/output — created a defect during GPU bring-up. The fix (`mapreduce.cu`) replaced the byte-buffer round-trip with a direct struct-to-struct `cudaMemcpy`, preserving the limb values exactly as the device computed them. This is precisely the kind of distinction an oral panel is likely to probe.
