# High-Throughput AES-GCM on GPUs — Technical Implementation Plan

*Based on: "A High-Throughput AES-GCM Implementation on GPUs for Secure, Policy-Based Access to Massive Astronomical Catalogs" (arXiv 2602.23067)*

---

## 0. Background & Mathematical Foundation

### AES-GCM Primer

AES-GCM (Galois/Counter Mode) combines two orthogonal operations:

| Component | Role | Native Parallelism |
|-----------|------|--------------------|
| AES-CTR | Confidentiality — encrypts a running counter, XORs result with plaintext | **Embarrassingly parallel** — blocks are independent |
| GHASH | Integrity — polynomial MAC over GF(2¹²⁸) | **Sequential by definition** — each block depends on the previous |

The authentication tag `T` is computed as:

```
T = GHASH(H, A, C) ⊕ E(K, IV ∥ 0^31 ∥ 1)
```

where `H = E(K, 0^128)` is the hash key and GHASH is:

```
GHASH(H, X) = X₁·H^N ⊕ X₂·H^(N-1) ⊕ ... ⊕ Xₙ·H¹   (all arithmetic in GF(2¹²⁸))
```

### The Bottleneck: Standard Sequential GHASH

The naïve evaluation uses Horner's method:

```
Y₀ = 0
Yᵢ = (Yᵢ₋₁ ⊕ Xᵢ) · H
```

Each step reads `Yᵢ₋₁`, making the chain **strictly serial: O(N) latency**.  
On a GPU with thousands of idle cores this wastes >99% of available parallelism.

### The Paper's Breakthrough: Parallel Tree-Reduction

Because GF(2¹²⁸) multiplication is associative and the full sum is:

```
GHASH = Σᵢ Xᵢ · H^(N+1-i)
```

each term `Xᵢ · H^(N+1-i)` is **independently computable** — the terms share no data dependency between each other. The algorithm is therefore:

1. **Map phase**: Each thread computes its own partial term `Xᵢ · H^(N+1-i)` using a precomputed power table `{H¹, H², ..., Hⁿ}`.
2. **Reduce phase**: A butterfly tree-reduction XORs partial results pairwise, halving the active thread count each round until one final result remains.

This transforms GHASH from O(N) serial to O(log N) parallel.

---

## 1. Project Structure

```
aes-gcm-gpu/
├── CMakeLists.txt
├── README.md
│
├── include/
│   ├── aes_gcm.h          # Shared types, constants, API surface
│   ├── gf128.h            # GF(2¹²⁸) field arithmetic (CPU & device)
│   └── utils.h            # Hex-dump, timing, test-vector helpers
│
├── cpu/
│   ├── aes_core.cpp       # AES-128/256 key schedule + round function
│   ├── aes_ctr.cpp        # CTR mode encryption
│   ├── ghash.cpp          # Sequential GHASH (Horner's method)
│   └── aes_gcm_cpu.cpp    # Top-level encrypt/decrypt + tag generation
│
├── gpu/
│   ├── aes_ctr.cu         # CUDA AES-CTR kernel (shared-memory S-Box)
│   ├── ghash_kernel.cu    # Parallel GHASH tree-reduction kernel
│   ├── mapreduce.cu       # Two-phase chunk MapReduce for large payloads
│   └── aes_gcm_gpu.cu     # Top-level GPU encrypt/decrypt driver
│
├── tests/
│   ├── nist_vectors.cpp   # NIST CAVS test vectors (mandatory correctness gate)
│   ├── bench_cpu.cpp      # CPU throughput benchmark
│   └── bench_gpu.cu       # GPU throughput benchmark + speedup report
│
└── data/
    └── nist_gcmEncryptExtIV128.rsp   # NIST test vector file
```

### Build System (CMakeLists.txt sketch)

```cmake
cmake_minimum_required(VERSION 3.20)
project(aes_gcm_gpu LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 75 86 89)  # Turing / Ampere / Ada

# CPU library
add_library(aes_gcm_cpu STATIC
    cpu/aes_core.cpp cpu/aes_ctr.cpp cpu/ghash.cpp cpu/aes_gcm_cpu.cpp)
target_include_directories(aes_gcm_cpu PUBLIC include)
target_compile_options(aes_gcm_cpu PRIVATE -O3 -march=native)

# GPU library
add_library(aes_gcm_gpu STATIC
    gpu/aes_ctr.cu gpu/ghash_kernel.cu gpu/mapreduce.cu gpu/aes_gcm_gpu.cu)
target_include_directories(aes_gcm_gpu PUBLIC include)
target_compile_options(aes_gcm_gpu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -lineinfo>)

# Test & benchmark executables
add_executable(run_tests tests/nist_vectors.cpp)
target_link_libraries(run_tests aes_gcm_cpu aes_gcm_gpu)

add_executable(bench tests/bench_gpu.cu)
target_link_libraries(bench aes_gcm_cpu aes_gcm_gpu)
```

---

## 2. Phase 1 — CPU Baseline

### 2.1 AES Core (`cpu/aes_core.cpp`)

Implement AES-128 and AES-256 using the standard FIPS 197 round structure:
- Key expansion via `KeyExpansion()` → produces 11 (AES-128) or 15 (AES-256) round keys
- Per-round operations: `SubBytes`, `ShiftRows`, `MixColumns`, `AddRoundKey`
- Use static lookup tables (`Sbox[256]`, `InvSbox[256]`, `xtime[]`) — no AES-NI initially

**Critical invariant**: The CPU AES block cipher output must pass NIST CAVS AES ECB test vectors before any further work proceeds.

### 2.2 CTR Mode (`cpu/aes_ctr.cpp`)

```
Counter block format (96-bit IV + 32-bit counter, big-endian):
  [ IV (12 bytes) | counter (4 bytes, starts at 1) ]

For each 16-byte block i:
  keystream[i] = AES_encrypt(K, IV ∥ (i+1))
  ciphertext[i] = plaintext[i] ⊕ keystream[i]
```

The final block handles padding by XORing only the significant bytes.

### 2.3 Sequential GHASH (`cpu/ghash.cpp`)

```cpp
// GF(2^128) multiply: CLMUL-free, bit-by-bit for reference clarity
void gf128_mul(const uint8_t X[16], const uint8_t Y[16], uint8_t Z[16]);

// Horner evaluation
void ghash(const uint8_t H[16],
           const uint8_t *data, size_t len,
           uint8_t tag[16]);
```

The reduction polynomial for GF(2¹²⁸) is: `x¹²⁸ + x⁷ + x² + x + 1`

Implement `gf128_mul` using the Russian-peasant (shift-and-XOR) method first — it's slow but provably correct and matches the spec bit-for-bit.

### 2.4 CPU API (`cpu/aes_gcm_cpu.cpp`)

```cpp
struct AesGcmCtx {
    uint8_t key[32];
    uint8_t iv[12];
    uint8_t H[16];       // H = AES(K, 0^128)
    int key_len;         // 128 or 256
};

void aes_gcm_encrypt(
    const AesGcmCtx *ctx,
    const uint8_t *plaintext, size_t pt_len,
    const uint8_t *aad,       size_t aad_len,
    uint8_t *ciphertext,      // out, same length as plaintext
    uint8_t tag[16]           // out
);
```

**Correctness gate**: Must pass all 15 NIST GCM test vectors (from `gcmEncryptExtIV128.rsp`) before proceeding.

---

## 3. Phase 2 — CUDA AES-CTR Kernel

### 3.1 Memory Architecture Decision

The most critical architectural choice on the GPU is where to place the AES substitution tables:

| Memory tier | Latency | Size | Visibility |
|-------------|---------|------|------------|
| Global | ~600 cycles | GBs | All threads |
| L2 cache | ~200 cycles | MBs | All threads |
| Shared memory | ~20 cycles | 48–164 KB/SM | Block-local |
| Registers | ~1 cycle | 255 regs/thread | Thread-local |

**Decision**: Load the AES S-Box (256 bytes) into shared memory once per block during kernel launch. Each thread then reads from `__shared__` rather than global memory.

### 3.2 AES-CTR CUDA Kernel Design (`gpu/aes_ctr.cu`)

```cuda
__global__ void aes_ctr_encrypt_kernel(
    const uint8_t * __restrict__ plaintext,
    uint8_t       * __restrict__ ciphertext,
    const uint32_t* __restrict__ round_keys,  // 11 × 4 words, in constant memory
    const uint8_t               iv[12],
    uint64_t                    num_blocks
) {
    // 1. Load S-Box into shared memory — one coalesced 256-byte load
    __shared__ uint8_t sbox[256];
    if (threadIdx.x < 256)
        sbox[threadIdx.x] = d_sbox[threadIdx.x];
    __syncthreads();

    // 2. Each thread owns one 16-byte counter block
    uint64_t block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= num_blocks) return;

    // 3. Form counter block: IV ∥ (block_idx + 1) big-endian
    uint8_t counter[16];
    memcpy(counter, iv, 12);
    uint32_t ctr_val = __byte_perm(block_idx + 1, 0, 0x0123);  // bswap
    memcpy(counter + 12, &ctr_val, 4);

    // 4. AES encrypt the counter block using shared sbox
    aes_encrypt_block_shared(counter, round_keys, sbox);

    // 5. XOR with plaintext (coalesced global load)
    uint64_t byte_offset = block_idx * 16;
    #pragma unroll
    for (int i = 0; i < 16; i++)
        ciphertext[byte_offset + i] = plaintext[byte_offset + i] ^ counter[i];
}
```

**Key optimizations**:
- Round keys go in `__constant__` memory (broadcast-cached, read-only)
- Plaintext/ciphertext accesses must be 128-bit aligned for coalesced loads (`float4` or `uint4` casting)
- Use `__launch_bounds__(256, 4)` to give the compiler register spill hints
- Occupancy target: ≥50% — profile with `ncu --metrics sm__warps_active`

### 3.3 Launch Configuration

```cpp
constexpr int THREADS_PER_BLOCK = 256;
int num_blocks_grid = (num_aes_blocks + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
aes_ctr_encrypt_kernel<<<num_blocks_grid, THREADS_PER_BLOCK>>>(...)
```

---

## 4. Phase 3 — Parallel GHASH Kernel

This is the mathematically complex core of the paper.

### 4.1 GF(2¹²⁸) Multiplication on the GPU (`include/gf128.h`)

All GHASH arithmetic lives in GF(2¹²⁸). On CUDA, implement it using 128-bit values as two `uint64_t` words (lo, hi):

```cuda
struct __align__(16) GF128 {
    uint64_t lo, hi;  // lo = bits 0–63, hi = bits 64–127
};

__device__ __forceinline__
GF128 gf128_mul(GF128 A, GF128 B) {
    // Karatsuba or schoolbook CLMUL emulation
    // Reduction modulo x^128 + x^7 + x^2 + x + 1
    // This is the hottest function in the entire codebase
}
```

**If targeting sm_70+**: Use `__umul64hi()` and manual carry propagation — no native CLMUL in CUDA PTX, but bit-parallel implementations can still be fast with unrolled inner loops.

### 4.2 Hash Key Power Table (precomputed on GPU)

Before the reduction kernel runs, precompute `H^1, H^2, ..., H^N` on the device:

```cuda
__global__ void build_power_table(
    GF128 *d_H_powers,   // out: [H^1, H^2, ..., H^N]
    GF128  H,
    int    N
) {
    // Thread 0 seeds H^1 = H
    // Each thread i computes H^(2i) = H^i * H^i (squaring chain)
    // Fill remaining powers by sequential multiply from the squared values
    // This only runs once per message — amortized cost is negligible
}
```

### 4.3 Parallel GHASH Tree-Reduction Kernel (`gpu/ghash_kernel.cu`)

```cuda
__global__ void ghash_parallel_kernel(
    const uint8_t *data,          // ciphertext + AAD blocks
    const GF128   *H_powers,      // precomputed [H^N, H^(N-1), ..., H^1]
    GF128         *partial_tags,  // out: one partial tag per block
    int            N              // total number of 128-bit blocks
) {
    extern __shared__ GF128 s_partial[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // ── MAP PHASE ─────────────────────────────────────────────────────────
    // Each thread computes its independent term: X[gid] · H^(N - gid)
    GF128 Xi = load_block_as_gf128(data, gid);
    GF128 Hi = H_powers[gid];               // H^(N - gid), precomputed
    s_partial[tid] = gf128_mul(Xi, Hi);
    __syncthreads();

    // ── REDUCE PHASE (butterfly tree) ────────────────────────────────────
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_partial[tid].lo ^= s_partial[tid + stride].lo;
            s_partial[tid].hi ^= s_partial[tid + stride].hi;
        }
        __syncthreads();   // MANDATORY: prevents race on shared memory
    }

    // Thread 0 writes this block's partial result to global memory
    if (tid == 0)
        partial_tags[blockIdx.x] = s_partial[0];
}
```

**Why `__syncthreads()` is mandatory here**: Without it, thread `tid` might read `s_partial[tid + stride]` before another thread has written it, producing a non-deterministic data race. The synchronization barrier guarantees all writes in the current stride are visible before reads begin in the next.

### 4.4 Dynamic Shared Memory Sizing

```cpp
size_t shared_mem = threads_per_block * sizeof(GF128);  // 16 bytes × 256 = 4KB
ghash_parallel_kernel<<<grid, threads, shared_mem>>>(...)
```

Verify `shared_mem` ≤ `cudaDeviceProp.sharedMemPerBlock` (typically 48KB, extendable to 96–164KB with `cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, ...)`).

---

## 5. Phase 4 — MapReduce for Large Payloads

A single CUDA block can hold at most ~164KB of shared memory. For a 1 GB payload (~67M blocks), we need a two-level reduction.

### 5.1 Architecture Overview

```
Input (1 GB ciphertext, ~67M × 16-byte blocks)
         │
         ▼
┌─────────────────────────────────────────────────┐
│  CHUNK DECOMPOSITION                            │
│  Split into M chunks, each of C blocks          │
│  (C chosen so one GPU grid handles one chunk)   │
└─────────────────────────────────────────────────┘
         │
         ▼  [kernel launch per chunk OR single large launch]
┌─────────────────────────────────────────────────┐
│  MAP PHASE (ghash_parallel_kernel)              │
│  Each thread block produces one partial_tag[]   │
│  Output: M intermediate GF128 values            │
└─────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│  GLOBAL REDUCE (single small kernel)            │
│  XOR all M partial tags with correct H-power    │
│  weights to produce final GHASH output          │
└─────────────────────────────────────────────────┘
         │
         ▼
    Final 128-bit authentication tag
```

### 5.2 Chunk Size Selection

```cpp
// Empirical target: each kernel launch occupies all SMs
int sm_count;
cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
int warps_per_sm = 32;        // for high occupancy
int threads_per_block = 256;
int blocks_per_launch = sm_count * warps_per_sm * 32;  // ~wave of work
int blocks_per_aes_chunk = blocks_per_launch;          // same granularity
```

### 5.3 Global Reduce Kernel

The M intermediate partial tags must be combined accounting for the positional weight of each chunk:

```cuda
__global__ void ghash_global_reduce(
    const GF128 *partial_tags,   // one per chunk, M values
    const GF128 *chunk_weights,  // H^(chunk_i_start) for each chunk
    GF128       *final_tag,
    int          M
) {
    // Each thread handles one partial tag
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) return;

    GF128 weighted = gf128_mul(partial_tags[i], chunk_weights[i]);

    // Atomic XOR reduction into final_tag
    atomicXor((unsigned long long*)&final_tag->lo, weighted.lo);
    atomicXor((unsigned long long*)&final_tag->hi, weighted.hi);
}
```

### 5.4 Host Orchestration (`gpu/mapreduce.cu`)

```cpp
void aes_gcm_gpu_encrypt(
    const GPUCtx *ctx,
    const uint8_t *d_plaintext, size_t len,
    uint8_t *d_ciphertext,
    uint8_t tag_out[16]
) {
    // 1. Launch AES-CTR kernel (all blocks in one shot — always parallelizable)
    launch_aes_ctr(ctx, d_plaintext, d_ciphertext, len);

    // 2. Build H power table up to N = ceil(len / 16)
    int N = (len + 15) / 16;
    build_power_table<<<...>>>(d_H_powers, ctx->H, N);

    // 3. Map: chunked GHASH partial reduction
    int num_chunks = (N + CHUNK_SIZE - 1) / CHUNK_SIZE;
    GF128 *d_partial_tags;
    cudaMalloc(&d_partial_tags, num_chunks * sizeof(GF128));
    for (int c = 0; c < num_chunks; c++) {
        int offset = c * CHUNK_SIZE;
        int count  = min(CHUNK_SIZE, N - offset);
        ghash_parallel_kernel<<<count/TPB, TPB, TPB*sizeof(GF128)>>>(
            d_ciphertext + offset * 16,
            d_H_powers + offset,
            d_partial_tags + c,
            count
        );
    }

    // 4. Reduce: combine partial tags globally
    ghash_global_reduce<<<...>>>(d_partial_tags, d_chunk_weights,
                                 d_final_tag, num_chunks);

    // 5. Final XOR with E(K, IV ∥ 0^31 ∥ 1)
    // Copy d_final_tag to host, XOR with precomputed J0 encryption
    cudaMemcpy(tag_out, d_final_tag, 16, cudaMemcpyDeviceToHost);
}
```

---

## 6. Correctness & Testing Strategy

### 6.1 NIST Test Vectors (mandatory first gate)

The NIST Cryptographic Algorithm Validation System (CAVS) provides official AES-GCM test vectors. File format: `gcmEncryptExtIV128.rsp`.

Each vector specifies: `Key`, `IV`, `PT` (plaintext), `AAD`, expected `CT` (ciphertext), and expected `Tag`. The test harness must:

1. Run the CPU implementation against all vectors — **zero failures required**
2. Run the GPU implementation against the same vectors — **bit-identical to CPU**

Any divergence between CPU and GPU output is a correctness bug, not a performance issue.

### 6.2 Differential Testing

For large random inputs, run CPU and GPU in parallel and diff the outputs:

```cpp
// Generate 10MB of random plaintext
// Encrypt with CPU → cpu_ct, cpu_tag
// Encrypt with GPU → gpu_ct, gpu_tag
// Assert: cpu_ct == gpu_ct && cpu_tag == gpu_tag
```

### 6.3 Edge Cases

- Empty plaintext (len = 0): GHASH is computed over AAD only
- AAD only, no ciphertext
- Non-16-byte-aligned input lengths (partial final block)
- Payload exactly 1, 2, 4, ... GB (stress MapReduce boundary conditions)

---

## 7. Benchmarking & Expected Performance

### 7.1 Metrics to Capture

```
metric                  tool
─────────────────────────────────────────────────────────
Throughput (GB/s)       custom timer + payload size
Latency (ms)            cudaEvent_t before/after
SM occupancy (%)        ncu --metrics sm__warps_active
L2 hit rate (%)         ncu --metrics l2_global_hit_rate
Shared mem util (%)     ncu --metrics shared_load_transactions
DRAM bandwidth util     ncu --metrics dram__bytes_read
Speedup vs CPU          ratio of throughputs
```

### 7.2 Expected Throughput Targets (based on paper)

| Configuration | Expected throughput |
|---------------|---------------------|
| CPU (single thread, no AES-NI) | ~0.3–0.5 GB/s |
| CPU (with OpenSSL AES-NI) | ~1–5 GB/s |
| GPU AES-CTR only (RTX 3080) | ~50–100 GB/s |
| GPU AES-GCM full (parallel GHASH) | ~30–80 GB/s |
| GPU AES-GCM naïve (sequential GHASH) | ~5–15 GB/s (GHASH bottleneck visible) |

The parallel GHASH should yield **5–10× throughput improvement** over sequential GHASH on the GPU. This is the key result to reproduce from the paper.

### 7.3 Benchmark Driver

```cpp
// bench_gpu.cu
for (size_t sz : {1<<20, 1<<24, 1<<28, 1<<30}) {  // 1MB, 16MB, 256MB, 1GB
    // warmup 3 runs, measure 10 runs, report mean ± stddev
    double throughput = measure_throughput(sz);
    printf("%-10zu MB: %.2f GB/s\n", sz >> 20, throughput);
}
```

---

## 8. Phase-by-Phase Execution Order

```
Week 1:  CPU AES core + CTR → pass NIST ECB vectors
Week 1:  CPU GHASH (sequential) → pass NIST GCM vectors
Week 2:  CUDA AES-CTR kernel → match CPU output, profile occupancy
Week 2:  GF(2¹²⁸) multiply device function → unit test against CPU
Week 3:  Parallel GHASH kernel (single block) → match CPU GHASH output
Week 3:  MapReduce two-phase driver → correct tags on 1GB inputs
Week 4:  Benchmarking, optimization (occupancy tuning, memory layout)
Week 4:  Final report with speedup graphs
```

---

## 9. Key Pitfalls & How to Avoid Them

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| Missing `__syncthreads()` in tree reduction | Non-deterministic wrong tags | Add sync after every stride; use `cuda-memcheck --tool racecheck` |
| S-Box in global memory | Low occupancy, high latency | Load into `__shared__` at kernel start |
| Non-coalesced memory access | Low memory throughput | Ensure consecutive threads access consecutive addresses |
| Counter endianness mismatch | Wrong ciphertext vs CPU | AES-GCM specifies big-endian counters — use `__byte_perm()` to swap |
| GF(2¹²⁸) bit ordering | Wrong authentication tags | GHASH uses a reflected bit order (LSB-first per byte) — match the spec exactly |
| Partial block not zeroed | Authentication tag wrong | Zero-pad the final partial block before GHASH processing |
| Power table off-by-one | Tag wrong for specific lengths | Term i uses H^(N+1-i), not H^(N-i) — verify against single-block test vector |

---

## 10. Dependencies & Environment

```bash
# Required
- CUDA Toolkit ≥ 11.8 (for sm_86 support)
- GCC ≥ 11 or Clang ≥ 14
- CMake ≥ 3.20
- NVIDIA GPU (sm_75 minimum, sm_86+ recommended)

# Optional but strongly recommended
- Nsight Compute (ncu) — kernel profiling
- Nsight Systems (nsys) — timeline profiling
- cuda-sanitizer — race condition detection
- OpenSSL (libssl-dev) — for AES-NI CPU baseline comparison
```

---

*Next step: implement Phase 1 (CPU baseline) and run the NIST test vectors before touching any CUDA code.*
