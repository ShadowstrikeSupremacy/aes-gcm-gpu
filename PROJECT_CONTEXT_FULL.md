# GPU-Accelerated AES-GCM — Full Project Context
### Paste this entire file into a new session to restore full context.

---

## SECTION 1 — Project Brief & Background

### What This Project Is

A GPU-accelerated AES-GCM (Galois/Counter Mode) encryption engine implemented in CUDA C++. The goal is to achieve high-throughput authenticated encryption suitable for securing large-scale data pipelines (e.g., astronomical catalogs). The project is grounded in the 2026 paper:

> **"A High-Throughput AES-GCM Implementation on GPUs for Secure, Policy-Based Access to Massive Astronomical Catalogs"** — arXiv:2602.23067

### The Core Technical Problem

AES-GCM has two phases:

1. **AES-CTR (Encryption)** — embarrassingly parallel. Each block XORs with an encrypted counter independently. Thousands of GPU threads can work simultaneously.

2. **GHASH (Authentication)** — the bottleneck. The standard Horner's method is strictly serial — each step depends on the output of the previous step. Time complexity: O(N).

### The Paper's Solution: Parallel Tree-Reduction

GHASH can be rewritten as a sum over GF(2¹²⁸):

```
GHASH = X₁·H^N ⊕ X₂·H^(N-1) ⊕ ... ⊕ Xₙ·H¹
```

Since each term `Xᵢ·H^(N+1-i)` is independently computable, the problem becomes a parallel reduction:
- **Map phase**: each thread computes one term independently
- **Reduce phase**: butterfly tree XORs results pairwise down to one value
- Result: O(log N) instead of O(N)

### Architecture Summary (4 Phases)

1. **CPU Baseline** — single-threaded C++ AES-GCM for correctness verification and benchmark comparison
2. **CUDA AES-CTR** — GPU encryption kernel with S-Box in shared memory, round keys in constant memory
3. **Parallel GHASH Kernel** — tree-reduction in GF(2¹²⁸) with `__syncthreads()` at every stride
4. **MapReduce Scaling** — two-phase chunked architecture for payloads larger than GPU shared memory

### Project Configuration

- **GPU target**: NVIDIA CUDA (generic, any sm_75+)
- **Structure**: Single CMake repo with cpu/, gpu/, tests/, include/ directories
- **Language**: C++17 + CUDA

---

## SECTION 2 — Technical Implementation Plan

### Mathematical Foundation

AES-GCM authentication tag:
```
T = GHASH(H, A, C) ⊕ E(K, IV ∥ 0^31 ∥ 1)
H = E(K, 0^128)   (hash key)
```

Sequential GHASH (Horner — O(N) serial):
```
Y₀ = 0
Yᵢ = (Yᵢ₋₁ ⊕ Xᵢ) · H
```

Parallel GHASH (tree-reduction — O(log N)):
```
GHASH = Σᵢ Xᵢ · H^(N+1-i)    [each term independent]
```

GF(2¹²⁸) reduction polynomial: `x¹²⁸ + x⁷ + x² + x + 1`

---

### Project File Structure

```
aes-gcm-gpu/
├── CMakeLists.txt
├── include/
│   ├── aes_gcm.h          # Shared types, constants, API surface
│   ├── gf128.h            # GF(2¹²⁸) field arithmetic (CPU & device)
│   └── utils.h            # Hex-dump, timing, test-vector helpers
├── cpu/
│   ├── aes_core.cpp       # AES-128/256 key schedule + round function
│   ├── aes_ctr.cpp        # CTR mode encryption
│   ├── ghash.cpp          # Sequential GHASH (Horner's method)
│   └── aes_gcm_cpu.cpp    # Top-level encrypt/decrypt + tag generation
├── gpu/
│   ├── aes_ctr.cu         # CUDA AES-CTR kernel (shared-memory S-Box)
│   ├── ghash_kernel.cu    # Parallel GHASH tree-reduction kernel
│   ├── mapreduce.cu       # Two-phase chunk MapReduce for large payloads
│   └── aes_gcm_gpu.cu     # Top-level GPU encrypt/decrypt driver
├── tests/
│   ├── nist_vectors.cpp   # NIST CAVS test vectors (mandatory correctness gate)
│   ├── bench_cpu.cpp      # CPU throughput benchmark
│   └── bench_gpu.cu       # GPU throughput benchmark + speedup report
└── data/
    └── nist_gcmEncryptExtIV128.rsp
```

---

### Phase 1 — CPU Baseline

**AES Core** (`cpu/aes_core.cpp`): FIPS 197 round structure — KeyExpansion, SubBytes, ShiftRows, MixColumns, AddRoundKey. Static Sbox[256]/InvSbox[256] lookup tables, no AES-NI initially.

**CTR Mode** (`cpu/aes_ctr.cpp`):
```
Counter format: [ IV (12 bytes) | counter (4 bytes, big-endian, starts at 1) ]
keystream[i] = AES_encrypt(K, IV ∥ (i+1))
ciphertext[i] = plaintext[i] ⊕ keystream[i]
```

**Sequential GHASH** (`cpu/ghash.cpp`): Russian-peasant GF multiply (slow, provably correct). Horner evaluation over GF(2¹²⁸).

**CPU API**:
```cpp
struct AesGcmCtx { uint8_t key[32]; uint8_t iv[12]; uint8_t H[16]; int key_len; };

void aes_gcm_encrypt(const AesGcmCtx *ctx,
    const uint8_t *plaintext, size_t pt_len,
    const uint8_t *aad,       size_t aad_len,
    uint8_t *ciphertext, uint8_t tag[16]);
```

**Gate**: Must pass full NIST CAVS GCM vectors before touching CUDA.

---

### Phase 2 — CUDA AES-CTR Kernel

**Memory hierarchy**:
| Tier | Latency | Decision |
|------|---------|----------|
| Global | ~600 cycles | plaintext/ciphertext only |
| Constant | broadcast cached | round keys (11 × 4 words) |
| Shared | ~20 cycles | S-Box (256 bytes, loaded once per block) |
| Register | ~1 cycle | per-thread counter state |

**Kernel sketch** (`gpu/aes_ctr.cu`):
```cuda
__global__ void aes_ctr_encrypt_kernel(
    const uint8_t * __restrict__ plaintext,
    uint8_t       * __restrict__ ciphertext,
    const uint32_t* __restrict__ round_keys,
    const uint8_t iv[12], uint64_t num_blocks)
{
    __shared__ uint8_t sbox[256];
    // Robust load: loop so it works for any blockDim.x
    for (int i = threadIdx.x; i < 256; i += blockDim.x)
        sbox[i] = d_sbox[i];
    __syncthreads();

    uint64_t block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= num_blocks) return;

    // Counter block: IV ∥ (block_idx+1) big-endian 32-bit
    uint8_t counter[16];
    memcpy(counter, iv, 12);
    uint32_t ctr_val = __byte_perm((uint32_t)(block_idx + 1), 0, 0x0123);
    memcpy(counter + 12, &ctr_val, 4);

    aes_encrypt_block_shared(counter, round_keys, sbox);

    uint64_t off = block_idx * 16;
    // Use uint4 casts for 128-bit coalesced loads
    *((uint4*)(ciphertext+off)) = make_uint4(
        ((uint4*)(plaintext+off))->x ^ ((uint4*)counter)->x,
        ((uint4*)(plaintext+off))->y ^ ((uint4*)counter)->y,
        ((uint4*)(plaintext+off))->z ^ ((uint4*)counter)->z,
        ((uint4*)(plaintext+off))->w ^ ((uint4*)counter)->w);
}
```

---

### Phase 3 — Parallel GHASH Kernel

**GF128 type** (`include/gf128.h`):
```cuda
struct __align__(16) GF128 { uint64_t lo, hi; };
__device__ __forceinline__ GF128 gf128_mul(GF128 A, GF128 B);
// Implements schoolbook or Karatsuba with reduction mod x^128+x^7+x^2+x+1
// CRITICAL: must use reflected bit ordering — 0x80 00...00 = polynomial 1, not x^127
```

**Hash key power table**: Precompute up to CHUNK_SIZE powers only (NOT N total powers — see flaw note in Section 3). Per-chunk scaling handled algebraically in global reduce.

**GHASH kernel** (`gpu/ghash_kernel.cu`):
```cuda
__global__ void ghash_parallel_kernel(
    const uint8_t *data, const GF128 *H_powers,
    GF128 *partial_tags, int N)
{
    extern __shared__ GF128 s_partial[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // MAP: each thread computes Xi · H^(N-gid) independently
    s_partial[tid] = (gid < N) ? gf128_mul(load_gf128(data, gid), H_powers[gid])
                               : (GF128){0, 0};
    __syncthreads();

    // REDUCE: butterfly tree (naive loop — see warp divergence note)
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_partial[tid].lo ^= s_partial[tid+stride].lo;
            s_partial[tid].hi ^= s_partial[tid+stride].hi;
        }
        __syncthreads();  // MANDATORY after every stride
    }
    if (tid == 0) partial_tags[blockIdx.x] = s_partial[0];
}
// Launch: ghash_parallel_kernel<<<grid, TPB, TPB*sizeof(GF128)>>>(...)
```

**Warp-level optimization** (fix warp divergence for stride < 32):
```cuda
if (tid < 32) {
    GF128 val = s_partial[tid];
    val.lo ^= __shfl_down_sync(0xffffffff, val.lo, 16);
    val.hi ^= __shfl_down_sync(0xffffffff, val.hi, 16);
    // repeat for offsets 8, 4, 2, 1
    s_partial[tid] = val;
}
```

---

### Phase 4 — MapReduce for Large Payloads

**Architecture**:
```
Input (large file)
  → Chunk Decomposition (M chunks of CHUNK_SIZE blocks)
  → Map: ghash_parallel_kernel per chunk → M partial tags
  → Global Reduce: weight each partial tag by H^(chunk_start), XOR all
  → Final: GHASH_output ⊕ E(K, J0) = authentication tag
```

**Key insight for memory efficiency**: Power table size = `CHUNK_SIZE × 16 bytes` (e.g., 64KB for 4096-block chunks), NOT `N × 16 bytes`. A full N-length table for 1 GB input would itself require 1 GB of VRAM.

**Global reduce**:
```cuda
__global__ void ghash_global_reduce(
    const GF128 *partial_tags, const GF128 *chunk_weights,
    GF128 *final_tag, int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) return;
    GF128 weighted = gf128_mul(partial_tags[i], chunk_weights[i]);
    // Two separate atomicXor calls — pair is NOT atomic as a unit,
    // but XOR commutativity/associativity makes the final result correct
    atomicXor((unsigned long long*)&final_tag->lo, weighted.lo);
    atomicXor((unsigned long long*)&final_tag->hi, weighted.hi);
}
```

---

### Correctness & Testing

- **NIST CAVS**: Run full `gcmEncryptExtIV128.rsp` and `gcmEncryptExtIV256.rsp` (hundreds of vectors, not just 15). Include varying tag lengths (32-bit to 128-bit), empty PT, AAD-only cases.
- **Differential testing**: CPU vs GPU on random 10 MB inputs — must be bit-identical.
- **Edge cases**: len=0, non-aligned lengths (partial final block), boundary payload sizes for MapReduce.

---

### Benchmarking Requirements

Every throughput number must specify which of these two modes it measures:
- **Compute-only** (data already resident on GPU) — shows kernel performance
- **End-to-end** (includes H2D + D2H PCIe transfer) — shows real-world performance

PCIe math: 1 GB in + 1 GB out at 12 GB/s realistic = ~166ms transfer alone. GPU compute at 50 GB/s = 20ms. End-to-end ≈ 5–6 GB/s, not 50 GB/s.

```
Payload    Compute-only    End-to-end (PCIe 3.0)    OpenSSL baseline
1 MB       ~50 GB/s        ~8 GB/s                  ~5 GB/s
16 MB      ~50 GB/s        ~10 GB/s                 ~5 GB/s
256 MB     ~48 GB/s        ~7 GB/s                  ~5 GB/s
1 GB       ~45 GB/s        ~6 GB/s                  ~5 GB/s
```

**CPU baseline must be OpenSSL with AES-NI**, not software AES. Software AES (~0.4 GB/s) is an unfair comparison that inflates the speedup number.

---

### Key Pitfalls (with fixes)

| Pitfall | Fix |
|---------|-----|
| Missing `__syncthreads()` in reduction | Required after EVERY stride — use cuda-sanitizer racecheck |
| S-Box load with `if (tid < 256)` | Use `for (int i=tid; i<256; i+=blockDim.x)` for any block size |
| GF(2¹²⁸) bit reflection | 0x80...00 = polynomial 1 — must reflect consistently |
| Partial final block | Zero-pad before GHASH processing |
| H power table size = N | Use block-local table of CHUNK_SIZE — N-length table requires N×16 bytes VRAM |
| Benchmarks without PCIe | Always report both compute-only and end-to-end |
| CPU baseline without AES-NI | Compare against OpenSSL EVP_aead_aes_128_gcm |
| Encrypt-only API | Implement authenticated decryption with constant-time tag comparison |

---

### Environment

```bash
# Required
CUDA Toolkit ≥ 11.8, GCC ≥ 11, CMake ≥ 3.20, NVIDIA GPU sm_75+

# Profiling (mandatory for credibility)
ncu   — Nsight Compute (kernel-level: occupancy, memory throughput, stall reasons)
nsys  — Nsight Systems (timeline: verify CUDA stream overlap)
cuda-sanitizer --tool racecheck  — race condition detection

# CPU comparison
OpenSSL libssl-dev (AES-NI + PCLMULQDQ baseline)
```

---

## SECTION 3 — Critical Evaluation & Known Flaws

*This section documents the known weaknesses, architectural risks, and questions that will be asked in any serious technical review.*

---

### Fatal Architectural Flaw: H Power Table Memory

The naive design precomputes H^1 through H^N where N = total blocks in message.

**For a 1 GB input**: N = 2^26 blocks × 16 bytes = **1 GB of VRAM just for H powers**.  
Plus 1 GB plaintext + 1 GB ciphertext = **3 GB minimum before CUDA overhead**.

**Fix**: Use a block-local power table of size CHUNK_SIZE only. The per-chunk weight `H^(chunk_start)` is computed separately and applied in the global reduce phase. This reduces the table from N×16 bytes to CHUNK_SIZE×16 bytes (e.g., 64 KB).

---

### Fatal Benchmarking Flaw: PCIe Transfer Excluded

The plan's 30–80 GB/s numbers are compute-kernel throughput only. End-to-end throughput on PCIe 3.0 for a 1 GB job is approximately 5–6 GB/s — which is competitive with but not dramatically faster than OpenSSL. The GPU wins decisively only when data is already on the GPU (the astronomical catalog use case). Benchmarks must report both modes and explain when each applies.

---

### Cryptographic Incompleteness: No Authenticated Decryption

The API has only `aes_gcm_encrypt`. A real AES-GCM implementation requires `aes_gcm_decrypt` that:
1. Recomputes the tag over the ciphertext
2. Compares received vs computed tag using `CRYPTO_memcmp` (constant-time)
3. Returns plaintext **only** if tags match; zeroes buffer and returns error otherwise

Without this, the project is AES-CTR with an unchecked MAC — not AEAD.

---

### Unfair CPU Baseline

Software AES without AES-NI achieves ~0.4 GB/s. OpenSSL with AES-NI + PCLMULQDQ achieves 4–8 GB/s. Using the software baseline inflates the apparent speedup by ~10–20×. The correct CPU opponent is OpenSSL.

---

### CUDA-Specific Known Errors

1. **Warp divergence in tree reduction**: `if (tid < stride)` causes divergence for stride < 32. Fix: unroll last 5 rounds with `__shfl_down_sync`.

2. **S-Box load**: `if (threadIdx.x < 256)` fails silently if blockDim.x < 256. Fix: use loop.

3. **`__byte_perm` type**: Applied to `uint64_t` — silently truncates to 32 bits. Works for AES-GCM (counter is 32-bit) but is technically wrong. Use explicit cast.

4. **Double-atomicXor**: Two separate 64-bit atomics are not jointly atomic, but XOR associativity makes the result correct regardless of interleaving. Must be able to explain this distinction.

5. **GHASH bit ordering**: The byte string `0x80 0x00...0x00` represents polynomial element 1 (not x^127) in GHASH's reflected representation. Every GF multiply must handle this consistently.

---

### CUDA Streams: Dependency Constraint

AES-CTR must complete on chunk N before GHASH can start on chunk N. The valid pipeline overlaps across chunks, not within:

```
Stream 1: [H2D: C1] → [AES-CTR: C1] → [GHASH: C1]
Stream 2:            → [H2D: C2]    → [AES-CTR: C2] → [GHASH: C2]
```

Requires pinned host memory (`cudaMallocHost`) for DMA overlap. Verify with Nsight Systems timeline.

---

### Evaluation Summary by Reviewer Type

| Reviewer | Perception |
|----------|------------|
| Average college faculty | Impressive — vocabulary and structure correct |
| Systems professor / HPC researcher | Promising scaffold, thin on profiling evidence, benchmarking methodology flawed |
| Senior CUDA engineer | Red flags: no ncu output, warp divergence in reduction, PCIe exclusion |
| Cryptography researcher | Serious concerns: no decryption, no IV management, unclear NIST coverage |
| Technical recruiter | Strong signal — CUDA + crypto + systems = rare combination |

---

### Top 5 Credibility Boosters

1. **Real Nsight Compute output** — achieved occupancy, stall reasons, memory throughput % of peak
2. **Authenticated decryption** with constant-time tag comparison
3. **Honest dual benchmark** — compute-only AND end-to-end with PCIe, with analysis of when each matters
4. **Block-local power table** — solve the H^N memory flaw explicitly, quantify the savings
5. **Full NIST CAVS pass** — run all vectors in both rsp files, show the pass count

---

### Top 5 Red Flags

1. No Nsight Compute or Nsight Systems output anywhere
2. Throughput numbers that don't specify compute-only vs end-to-end
3. CPU baseline is software AES rather than OpenSSL
4. GF(2¹²⁸) multiply treated as a black box (cannot derive reduction polynomial)
5. No authenticated decryption

---

### Top 10 Interview Questions

1. "What is the memory footprint of your H power table for a 1 GB input? Walk through the math." *(H^N scaling flaw)*

2. "What bit ordering does GHASH use? What does the byte string 0x80 0x00...0x00 represent as a polynomial element?" *(reflection bug)*

3. "Your profiler shows 62% achieved occupancy. What is the limiting resource?" *(real profiling experience)*

4. "Your benchmark says 45 GB/s. Does this include PCIe transfer? What is the end-to-end throughput on PCIe 3.0?" *(benchmark validity)*

5. "Why is `__syncthreads()` mandatory after every stride in the tree reduction? What happens at the hardware level if you remove it?" *(GPU memory consistency)*

6. "Compare your GF multiply cycle count to PCLMULQDQ on CPU. Does your parallelism compensate?" *(core performance claim validity)*

7. "What happens if you encrypt two messages with the same key and IV? Why is this uniquely catastrophic for AES-GCM?" *(IV management)*

8. "You call `atomicXor` twice for `lo` and `hi`. Is the pair atomic? Is the result correct? Explain the difference." *(atomicity vs correctness)*

9. "CUDA streams require pinned memory for actual DMA overlap. Is your allocation pinned? Did you verify the overlap in Nsight Systems?" *(streams implementation reality)*

10. "The NIST suite has vectors with 32-bit tags. Does your implementation support them?" *(NIST coverage completeness)*

---

*End of project context. Paste this entire file to restore full context in a new session.*
