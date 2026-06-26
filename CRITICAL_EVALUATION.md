# Critical Technical Evaluation — AES-GCM GPU Acceleration Project

*Written from the perspective of a skeptical systems professor, CUDA engineer, and cryptography researcher.*
*This is not a supportive document. It is a brutally honest technical audit.*

---

## Preamble: The Core Risk of This Project

This project sits at the intersection of three fields that are each individually capable of hiding shallow understanding behind impressive-sounding terminology: **GPU systems programming**, **applied cryptography**, and **high-performance computing**. The combination makes it easy to produce something that *reads* like a serious systems contribution while containing fundamental errors that any domain expert would catch immediately.

The review below is structured to expose exactly those gaps.

---

## Part I — Assessment of the Three Proposed Additions

### Addition 1: CPU vs GPU Profiling Analysis

**Verdict: Correct instinct. Weak in practice unless executed properly.**

This is not an optional nice-to-have — it is the difference between a systems paper and a blog post. The problem is that students routinely *claim* to do profiling analysis while demonstrating no real understanding of the numbers.

**What would genuinely impress an evaluator:**
- Actual `ncu` (Nsight Compute) output attached as appendix, showing: achieved occupancy, SM throughput, memory bandwidth utilization percentage, L1/L2/global load/store hit rates, stall reasons per warp (memory dependency, long scoreboard, execution dependency)
- An explanation of *why* the achieved occupancy is what it is — not just reporting a number. If occupancy is 62%, what is the limiting resource? Register file pressure? Shared memory? Block size?
- A roofline model: where does the GHASH kernel sit relative to the compute-bound and memory-bound ceilings on your specific GPU?

**What would immediately expose weak understanding:**
- Reporting "occupancy: 75%" without specifying theoretical vs achieved, and without explaining the bottleneck
- Claiming high memory bandwidth utilization while using `uint8_t` loads (these are 1-byte loads — catastrophically inefficient; coalescing requires 128-bit aligned `uint4` or `float4` accesses)
- Confusing latency and throughput occupancy metrics
- Any claim about optimization that isn't backed by before/after ncu comparison

**The key profiling breakdown a serious evaluator wants to see:**

```
Kernel: ghash_parallel_kernel
─────────────────────────────────────────────────────────────
Achieved occupancy:          ??% (theoretical: ??)
Limiting factor:             [ ] registers  [ ] shared mem  [ ] block size
Memory throughput:           ?? GB/s of ?? GB/s peak (??%)
L1 hit rate:                 ??%
L2 hit rate:                 ??%
Compute throughput:          ?? TFLOPS of ?? peak (??%)
Primary stall reason:        ??
Warp execution efficiency:   ??%
─────────────────────────────────────────────────────────────
```

Without filling this table with real numbers from a real GPU run, the profiling section is decoration.

---

### Addition 2: Sequential vs Parallel GHASH Comparison

**Verdict: Absolutely essential — but contains a subtle, project-destroying logical flaw in how it's typically framed.**

The comparison table you proposed:

```
GHASH Type                    Throughput
Sequential GPU GHASH          8 GB/s
Parallel Tree Reduction        52 GB/s
```

This *looks* like the right experiment, but it conceals a critical conceptual problem that a CUDA engineer will catch immediately:

**Sequential GHASH on a GPU is not a meaningful baseline.**

A GPU core runs at ~2 GHz with terrible single-thread IPC compared to a CPU core at 5 GHz with out-of-order execution, branch prediction, and deep pipeline. If you run Horner's sequential GHASH on a GPU, you are wasting the GPU almost entirely — a single thread drives the chain while thousands of cores idle. The "8 GB/s" for sequential GPU GHASH is actually *worse* than a CPU with AES-NI, which makes the GPU look bad without context.

**The comparison that actually demonstrates the project's contribution:**

```
Implementation                         Throughput     Notes
─────────────────────────────────────────────────────────────────────
CPU, software AES (no AES-NI)          ~0.4 GB/s      Your reference baseline
CPU, OpenSSL with AES-NI + PCLMULQDQ   ~4–8 GB/s      The real-world opponent
GPU, AES-CTR only (no auth)            ~60 GB/s       Theoretical max without GHASH
GPU, sequential GHASH (Horner)         ~0.2 GB/s      GPU misused — WORSE than CPU
GPU, parallel GHASH (tree reduction)   ~40–60 GB/s    The paper's contribution
```

The sequential-GPU row being *worse* than CPU is the point — it motivates the entire parallelization. Showing it being "8 GB/s" implies sequential GPU GHASH is still competitive, which is wrong and confusing.

**The second flaw**: Without including PCIe transfer time in all GPU numbers, every GPU throughput figure is a lie. See Part II for the full dissection of this.

---

### Addition 3: CUDA Streams Pipeline

**Verdict: Good instinct. Your proposed pipeline diagram has a dependency violation that would be caught immediately.**

Your proposed pipeline:
```
Chunk 1 → encrypt
Chunk 2 → transfer
Chunk 3 → GHASH
```

This diagram implies `encrypt(chunk 1)` and `GHASH(chunk 3)` can run simultaneously. **They cannot overlap on the same chunk** — GHASH over the ciphertext requires the ciphertext to exist, which requires AES-CTR to complete first. The dependency within a chunk is strict:

```
Transfer(chunk N) → AES-CTR(chunk N) → GHASH(chunk N) → [partial tag]
```

The actual valid pipeline overlaps *across* chunks:

```
Stream 1: [Transfer C1] → [AES-CTR C1] → [GHASH C1]
Stream 2:                 [Transfer C2] → [AES-CTR C2] → [GHASH C2]
Stream 3:                               → [Transfer C3] → [AES-CTR C3] → [GHASH C3]
```

Where `[Transfer C2]` overlaps with `[AES-CTR C1]` on the copy engine vs SM hardware.

**What makes this genuinely complex to implement correctly:**

1. **Pinned memory is mandatory** — `cudaMemcpyAsync` only achieves true DMA overlap when the source is pinned host memory (`cudaMallocHost`). With pageable memory, the async copy blocks until the page is pinned, defeating the purpose.

2. **Stream synchronization for partial tag combination** — the final GHASH reduction cannot start until all per-chunk GHASH kernels complete. You need `cudaStreamWaitEvent` across streams, not just `cudaDeviceSynchronize`.

3. **The benefit may not materialize** — if the compute time per chunk >> transfer time, the pipeline stalls at the compute stage anyway. Only worth the complexity if the bottleneck is transfer, which means PCIe is your limiting factor, which means your 60 GB/s compute claim is already misleading.

If you implement streams, you must measure and show the actual overlap using Nsight Systems timeline view. Claiming pipeline without showing the timeline is not evidence.

---

## Part II — Fatal Architectural Flaws in the Current Plan

### Flaw 1: The H Power Table Memory Catastrophe

The current plan casually mentions "precompute a look-up table of hash key powers." Consider the actual memory cost:

For a **1 GB** input:
- Number of 16-byte AES blocks: `2^30 / 16 = 2^26 ≈ 67 million blocks`
- H power table size: `67 million × 16 bytes = 1,073,741,824 bytes = 1 GB`

So to encrypt 1 GB of data, the plan requires **1 GB of GPU memory just for the hash key powers**, plus 1 GB for plaintext, plus 1 GB for ciphertext = **3 GB minimum VRAM**, before accounting for the round key table, intermediate partial tags, and CUDA runtime overhead.

For a 4 GB file (entirely plausible in an astronomical catalog application, which the paper explicitly targets): the power table alone is 4 GB.

This is not a minor inefficiency. **This is a fundamental architectural flaw that renders the MapReduce design infeasible for its stated purpose at scale.**

The paper's actual solution almost certainly uses a block-level power table — powers up to the block size, not the total message size — and handles the per-chunk H-power scaling algebraically. The plan as written skips over this entirely.

**The question a professor will ask:** *"What is the memory footprint of your H power table for a 256 MB input on an 8 GB GPU? How much VRAM is left for actual data?"*

---

### Flaw 2: GF(2¹²⁸) Multiplication — The Real Performance Bottleneck

The plan says "GF(2¹²⁸) multiply" as if it's a well-understood primitive. It is not. This is the **hardest function to get right and fast in this entire project**, and the plan spends two lines on it.

**On x86 CPU with PCLMULQDQ**: A 128-bit carryless multiply is a single instruction (~3 cycles). OpenSSL uses this. Your software CPU baseline does not. This alone makes every CPU comparison invalid.

**On NVIDIA GPU**: There is **no hardware carryless multiply instruction** in CUDA PTX. You must emulate it with:
- Schoolbook method: 128 × 128 bit operations ≈ 256 XORs + 128 ANDs minimum, plus reduction
- Karatsuba: reduces to ~3 × 64-bit sub-multiplications, still emulated
- Russian-peasant: 128 iterations of shift + conditional XOR

The Russian-peasant method the plan recommends for "clarity" runs in O(128) iterations — that's 128 loop iterations per GF multiply, per thread, per block, in a kernel that calls this function once per data block. At 67 million blocks for 1 GB of data, this is **8.6 billion loop iterations** before any other work.

**The uncomfortable question**: Is parallelized, emulated GF(2¹²⁸) multiply on a GPU actually faster than PCLMULQDQ-accelerated sequential GHASH on a CPU? The answer is not obvious, and the plan assumes the answer without measuring it.

---

### Flaw 3: The Authenticated Decryption API is Missing

The plan defines only encryption:

```cpp
void aes_gcm_encrypt(..., uint8_t tag[16]);
```

**AES-GCM without decryption is not a usable cryptographic primitive.** More critically, authenticated decryption has a security requirement that the plan violates by omission:

> **The plaintext must not be released until the authentication tag is verified.**

A correct AES-GCM decryption API must:
1. Decrypt the ciphertext
2. Recompute the authentication tag over the ciphertext
3. Compare tags in **constant time** (preventing timing oracle attacks)
4. Return plaintext **only** if tags match; return an error and zero the plaintext buffer otherwise

If a project claims to implement AES-GCM but only implements encryption, it has implemented AES-CTR with a MAC that is never checked. Any cryptographer on an evaluation panel will flag this immediately.

---

### Flaw 4: The Benchmarking Numbers Exclude PCIe Transfer

The plan claims GPU throughputs of "30–80 GB/s." Let's examine whether this survives contact with reality.

**PCIe 4.0 x16 peak bandwidth**: 32 GB/s bidirectional, ~16 GB/s in each direction under ideal conditions, ~12–14 GB/s realistic sustained.

**PCIe 3.0 x16 (still common)**: ~12 GB/s peak per direction, ~8–10 GB/s realistic.

For a 1 GB encryption job:
- Transfer plaintext to GPU: 1 GB / 12 GB/s ≈ **83 ms**
- GPU computation (AES-CTR + GHASH at 50 GB/s): 1 GB / 50 GB/s ≈ **20 ms**
- Transfer ciphertext back: 1 GB / 12 GB/s ≈ **83 ms**

**Total wall-clock time: ~186 ms** for what the plan claims is a "50 GB/s" operation.

**Actual end-to-end throughput: 1 GB / 0.186 s ≈ 5.4 GB/s**

This is *better* than an unoptimized CPU but competitive with OpenSSL, not a dramatic win. The only scenario where the GPU wins decisively is if the data already lives on the GPU (e.g., processing data already in GPU memory for a compute pipeline, which is exactly the astronomical catalog use case). The plan never specifies whether benchmarks include or exclude transfer time. This omission fundamentally invalidates the performance claims.

---

### Flaw 5: The CPU Baseline is Unfairly Slow

The plan uses a "software AES (no AES-NI)" CPU baseline. Every x86-64 processor since 2010 has AES-NI. Benchmarking against a software AES implementation is equivalent to benchmarking a GPU against a CPU deliberately crippled by running in interpreted mode.

A fair CPU comparison is against **OpenSSL EVP_aead_aes_128_gcm**, which uses:
- AES-NI for encryption (1 instruction per round)
- PCLMULQDQ for GHASH (1 instruction per GF multiply)
- Pipeline interleaving of 4–8 blocks at once

This achieves 4–8 GB/s on modern hardware. Your GPU implementation must beat this *including PCIe transfer time* to be a meaningful contribution.

---

## Part III — CUDA-Specific Technical Errors in the Current Plan

### Error 1: `__byte_perm` Applied to a 64-bit Value

The plan contains:
```cuda
uint32_t ctr_val = __byte_perm(block_idx + 1, 0, 0x0123);
```

`__byte_perm` is a 32-bit intrinsic — it operates on two 32-bit inputs and a selector. Using it on `block_idx` (a `uint64_t`) silently truncates to the low 32 bits. For AES-GCM, the counter is only 32 bits big-endian, so this accidentally works for inputs up to `2^32 × 16 = 64 GB`. But the code is wrong for the stated reason: it doesn't byte-swap a 64-bit value, it truncates and byte-swaps 32 bits. A reviewer who understands CUDA intrinsics will notice the type mismatch.

### Error 2: Warp Divergence in Tree Reduction

The plan's reduction loop:
```cuda
for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
        s_partial[tid].lo ^= s_partial[tid + stride].lo;
        ...
    }
    __syncthreads();
}
```

For `stride < 32`, the `if (tid < stride)` condition causes warp divergence — within a single warp of 32 threads, some satisfy the condition and some do not. The hardware serializes these paths, halving throughput for the last 5 reduction rounds. The standard fix is to unroll the last 32 iterations using warp shuffle intrinsics:

```cuda
if (blockDim.x >= 64 && tid < 32) {
    // warp-level reduction using __shfl_down_sync
    GF128 val = s_partial[tid];
    val.lo ^= __shfl_down_sync(0xffffffff, val.lo, 16);
    val.hi ^= __shfl_down_sync(0xffffffff, val.hi, 16);
    // ... down to stride 1
}
```

A CUDA engineer reviewing the plan will flag the naive loop immediately.

### Error 3: The Double-atomicXor Race Condition

The global reduce kernel uses:
```cuda
atomicXor((unsigned long long*)&final_tag->lo, weighted.lo);
atomicXor((unsigned long long*)&final_tag->hi, weighted.hi);
```

These are two separate atomic operations. Between the first and second atomic, another thread can interleave its own pair of atomics, producing a partially-written intermediate state. While both individual atomics are atomic, the *pair* is not. For XOR reduction this is actually fine mathematically (XOR is commutative and associative, so any interleaving produces the correct final result), but many students cannot explain *why* it's correct and will claim it's safe for the wrong reasons.

More importantly: `atomicXor` for `unsigned long long` requires `sm_12` minimum and CUDA 3.2+. At `sm_75` and above it's fine, but the plan should state this explicitly.

### Error 4: Shared Memory S-Box Load Coverage

```cuda
if (threadIdx.x < 256)
    sbox[threadIdx.x] = d_sbox[threadIdx.x];
__syncthreads();
```

The plan uses 256 threads per block, so all 256 threads participate and all 256 bytes are loaded. But if this kernel is ever launched with fewer than 256 threads per block (e.g., 128 for an occupancy experiment), only half the S-Box is loaded. The code has a silent correctness dependency on block size. Robust code uses a loop: `for (int i = threadIdx.x; i < 256; i += blockDim.x)`.

### Error 5: The GHASH Bit-Ordering Trap

The plan mentions this in the pitfalls table but understates the severity. GHASH uses a **non-standard bit ordering**: within each byte, bit 7 is the most significant bit of the polynomial representation, but the byte order is big-endian while the *bit* order within the GF element is reflected compared to the natural representation.

Concretely: the byte string `0x80 0x00 ... 0x00` represents the polynomial element `x^0 = 1`, *not* `x^127`. Every GF(2¹²⁸) operation must account for this reflection consistently. This is the number one source of "implementation produces a tag that's a fixed transformation away from correct" bugs. It is trivially hidden by only testing a single test vector where the tag happens to match by coincidence with an incorrect bit ordering.

**The test that exposes it**: Run 10 NIST vectors with varying key/IV/plaintext combinations. A reflection bug will fail on some but pass on others, depending on the specific bit patterns.

---

## Part IV — Cryptographic Correctness Risks

### Risk 1: IV Reuse Catastrophe

AES-GCM's security guarantee collapses completely if the same (Key, IV) pair is ever reused. The consequences:
- An attacker XORs two ciphertexts encrypted under the same keystream to get the XOR of plaintexts
- The H key is revealed from the authentication tags
- Both the confidentiality *and* integrity of all past messages under that key are compromised

The plan has no IV management strategy. For the astronomical catalog use case, how are IVs assigned across concurrent encryption requests? This is not a minor detail — it is the most critical operational security requirement of AES-GCM.

### Risk 2: Tag Length Not Specified

NIST allows AES-GCM tags of 32 to 128 bits. The plan implicitly assumes 128-bit tags. Shorter tags are commonly used in practice (e.g., 96-bit in TLS 1.3). The plan has no support for this, and the NIST test suite includes vectors with shorter tags.

### Risk 3: No Timing-Safe Tag Comparison

The CPU's `aes_gcm_decrypt` (which the plan doesn't implement) must compare the computed tag against the received tag using `CRYPTO_memcmp` or equivalent — a byte-by-byte comparison that always runs to completion regardless of where the first mismatch occurs. Using `memcmp` creates a timing oracle that leaks information about the expected tag.

### Risk 4: The NIST Test Vector Coverage Claim

Saying "pass NIST GCM test vectors" sounds rigorous. The actual NIST CAVS GCM test suite (from `gcmEncryptExtIV128.rsp`) contains roughly 1,000 vectors per tag length. The plan mentions "all 15 vectors" — this is a misunderstanding of the NIST test suite structure. The full suite has hundreds of vectors covering varying key lengths, tag lengths, data lengths, and AAD lengths. Running 15 vectors is a smoke test, not validation.

---

## Part V — "Looks Impressive" vs "Is Technically Impressive"

### Things That LOOK Impressive

| Claim | Reality |
|-------|---------|
| "O(log N) tree reduction" | Standard parallel pattern. Undergraduate algorithms. |
| "Shared memory S-Box optimization" | 256 bytes. Trivially small. Benefit is marginal on sm_75+ with 32KB L1 cache per SM. |
| "MapReduce architecture" | Using the word MapReduce for a chunked reduction is terminology inflation. |
| "GF(2¹²⁸) arithmetic" | Sounds advanced. Is a 128-line function. The *correct* version is hard; calling it impressive without profiling is not. |
| "CUDA occupancy optimization" | Meaningless without measured occupancy before and after. |
| "Pipeline with CUDA streams" | Impressive only if the Nsight Systems timeline shows actual overlap. |

### Things That Are ACTUALLY Technically Impressive (if evidence exists)

| Achievement | Why It's Real |
|-------------|---------------|
| Passing all NIST CAVS vectors including non-aligned lengths, empty PT, empty AAD | Requires correctly implementing edge cases that break most implementations |
| Showing Nsight Compute output with a specific occupancy bottleneck identified and resolved | Demonstrates real profiling workflow |
| GF(2¹²⁸) multiply that beats the naive method with measured speedup | Requires understanding Karatsuba or CLMUL emulation |
| Honest benchmark including PCIe transfer time with analysis of when GPU wins | Requires understanding the full system, not just the kernel |
| Correct authenticated decryption with constant-time tag comparison | Shows cryptographic understanding beyond encryption-only |
| Roofline analysis placing each kernel on the compute/memory-bound spectrum | Demonstrates HPC maturity |

---

## Part VI — Final Verdict by Evaluator Type

### Average College Faculty (CS, not systems-specialized)
**Perception: Impressive.**
The vocabulary is correct, the phases are logical, and the code snippets look real. A non-specialist will not check whether the benchmarks include PCIe time or whether the H power table scales to the stated use case. Grade: A or A+.

### Strong Systems Professor / HPC Researcher
**Perception: Promising scaffold, suspicious depth.**
The architecture is described correctly at a high level. The profiling section is thin. The benchmarking methodology is flawed (PCIe exclusion, unfair CPU baseline). The GF multiply implementation is hand-waved. The MapReduce memory scaling flaw is absent. Would probe hard in a viva. Grade depends entirely on whether you can answer questions live. Could be A- or C depending on depth of actual understanding.

### Senior CUDA Engineer (e.g., from NVIDIA, Meta, Google HPC)
**Perception: Surface-level. Several red flags.**
Would immediately ask about the power table memory cost, warp divergence in the reduction loop, the shared memory S-Box benefit measurement, and the PCIe bandwidth calculation. The absence of Nsight Compute output in a CUDA optimization project is a major red flag. This person has seen many undergraduate CUDA projects that claim optimization without profiling evidence. **Would not be hired based on this project alone without a live coding interview.**

### Cryptography Researcher
**Perception: Serious concern about correctness.**
No authenticated decryption. No IV management. Unclear NIST coverage. The bit-ordering issue in GF arithmetic is flagged. Would want to see constant-time comparisons and a discussion of IV uniqueness guarantees. Would consider this "AES-CTR with a tag generator" rather than "AES-GCM."

### Technical Recruiter (Big Tech, systems role)
**Perception: Strong candidate signal.**
The project demonstrates initiative, breadth, and familiarity with advanced topics. Most recruiters cannot evaluate technical depth independently. This project signals CUDA experience, cryptography knowledge, and systems thinking — all valuable. Would pass to technical screen confidently.

---

## Part VII — Top 5 Additions That Most Increase Credibility

**1. Attach Real Nsight Compute Output**
A single screenshot of `ncu` output for the GHASH kernel — showing achieved occupancy, memory throughput percentage, primary stall reason, and the L1/L2 cache hit rates — is worth more than five pages of architectural description. It proves you ran the code on real hardware and understand how to read profiler output.

**2. Implement Correct Authenticated Decryption with Constant-Time Tag Check**
`aes_gcm_decrypt` that refuses to return plaintext on tag mismatch, uses `CRYPTO_memcmp`, and zeroes the output buffer on failure. This is the security-critical half of AES-GCM and its absence is the first thing a cryptographer notices.

**3. Honest End-to-End Benchmark Including PCIe Transfer**
A benchmark table with two rows per payload size: "compute-only (data pre-loaded on GPU)" and "end-to-end (includes H2D + D2H transfer)." Then explain when each scenario applies. The astronomical catalog use case may have data pre-loaded; most general-purpose use cases do not. This level of nuance signals genuine systems understanding.

**4. Solve the H Power Table Memory Problem**
Describe and implement a block-local power table — compute powers only up to `CHUNK_SIZE`, not `N`. The per-chunk weight factor `H^(chunk_start)` is computed separately and applied during global reduction. Quantify: "For a 1 GB input, the full power table requires 1 GB VRAM; our block-local approach requires only `CHUNK_SIZE × 16` bytes = 64 KB." This shows you found and fixed a critical flaw.

**5. Full NIST CAVS Test Suite Pass (with evidence)**
Run the complete `gcmEncryptExtIV128.rsp` and `gcmEncryptExtIV256.rsp` test suites (hundreds of vectors each). Print the pass count in your README. The difference between "pass 15 sample vectors" and "pass 1,000 CAVS vectors including 32-bit and 96-bit tag lengths" is the difference between smoke-tested and validated.

---

## Part VIII — Top 5 Red Flags That Most Damage Credibility

**1. No Nsight Compute or Nsight Systems Output Anywhere**
Claiming CUDA optimization without profiler evidence is the single most reliable marker of a project that was designed but not implemented. If you have profiler output, show it everywhere.

**2. Benchmarks That Do Not Specify Whether PCIe Transfer Is Included**
Every GPU throughput number must state: "compute-only (data resident on device)" or "end-to-end (includes H2D and D2H transfer)." Unmarked numbers are assumed to be hiding the transfer cost.

**3. CPU Baseline Using Software AES Without AES-NI**
This is the easiest way to make a GPU look good. An evaluator who knows the field will immediately ask "did you compare against OpenSSL?" and if the answer is no, they know why.

**4. GF(2¹²⁸) Multiplication Treated as a Black Box**
If you cannot derive the reduction modulo `x¹²⁸ + x⁷ + x² + x + 1` from scratch, explain why the bit-reflected representation is used, and show a unit test of your GF multiply against known vectors, you do not understand your own core algorithm.

**5. No Authenticated Decryption**
Implementing encrypt-only and calling it AES-GCM signals either that you do not understand AEAD semantics, or that you know decryption is harder and avoided it. Either interpretation damages credibility with cryptography evaluators.

---

## Part IX — Top 10 Interview Questions

These are the questions that separate genuine implementation from surface-level description. The correct answer to each requires knowledge that cannot be acquired from reading a high-level description.

**1.** "Your H power table for a 1 GB input would require how much GPU memory? Walk me through the calculation." *(Tests whether you understand the memory scaling flaw.)*

**2.** "What is the bit ordering used in GHASH, and how does it differ from the natural binary representation of a 128-bit integer? Show me the code that handles this." *(Tests cryptographic correctness of the core algorithm.)*

**3.** "Your profiler shows 62% achieved occupancy on the GHASH kernel. What is the limiting resource, and what would you do to increase it?" *(Tests real profiling experience.)*

**4.** "Your benchmark reports 45 GB/s for AES-GCM. Does this include PCIe transfer time? If not, what is the actual end-to-end throughput on a PCIe 3.0 system?" *(Tests benchmark validity.)*

**5.** "In your tree reduction, why is `__syncthreads()` required at every stride, and what exactly happens at the hardware level if you remove it?" *(Tests understanding of GPU memory consistency, not just "it's required for synchronization.")*

**6.** "Compare your GF(2¹²⁸) multiply performance against a CPU using PCLMULQDQ. How many clock cycles does each take, and does your parallelism compensate?" *(Tests whether the core claim — GPU GHASH is faster — is actually true.)*

**7.** "What happens to your authentication tag if you encrypt two different messages with the same key and IV? Why is this catastrophic for AES-GCM specifically?" *(Tests cryptographic understanding beyond the implementation.)*

**8.** "Your code uses `atomicXor` twice — once for `lo` and once for `hi` of the 128-bit tag. Is this atomic? Is the result correct? Explain the difference between the two questions." *(Tests precision about atomicity vs. correctness.)*

**9.** "CUDA streams require pinned memory for asynchronous transfers to actually overlap with computation. Is your host allocation pinned? How did you verify the overlap occurred?" *(Tests whether the streams optimization was actually implemented or just described.)*

**10.** "The NIST GCM test suite includes vectors with 32-bit authentication tags. Does your implementation support them? What changes when the tag length is not 128 bits?" *(Tests completeness of NIST validation and understanding of tag truncation.)*

---

## Closing Assessment

This project, as currently planned, has a **strong architectural skeleton and serious implementation risks**. The vocabulary is correct, the algorithmic choices are well-motivated, and the phased structure is sound. These are genuine strengths.

The vulnerabilities — H power table scaling, unfair benchmarking, missing authenticated decryption, GF arithmetic correctness, PCIe transfer exclusion — are not minor polish items. They are the substance of the project. An evaluator who knows any one of these fields deeply will find the gap within five minutes of questioning.

The difference between a project that survives expert scrutiny and one that does not is straightforward: **run the code, measure the numbers, show the output.** An attached Nsight Compute screenshot, a full NIST CAVS pass log, a benchmark table that honestly includes transfer time, and a GF multiply that you can derive at a whiteboard — these turn a description into evidence.

Build the thing. Measure everything. Show your work.
