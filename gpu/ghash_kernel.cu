// gpu/ghash_kernel.cu  —  Parallel GHASH tree-reduction (Phase 3).
//
// Mathematical foundation (NIST SP 800-38D §6.4):
//
//   Sequential Horner form:  Y_i = (Y_{i-1} XOR X_i) * H
//   Expanded:  GHASH = X_1*H^N XOR X_2*H^(N-1) XOR ... XOR X_N*H^1
//
// All N terms are independently computable.
//
// Kernel design:
//   ghash_map_reduce_kernel:
//     MAP:    thread gid computes X_{gid+1} * H_powers[gid]
//             where H_powers[gid] = H^{N-gid} (precomputed, see below).
//     REDUCE: butterfly tree-XOR within the thread block (via shared memory).
//     OUTPUT: one GF128 partial sum per thread block → partial[blockIdx.x].
//
//   ghash_final_reduce_kernel:
//     Combines all partial sums from the first kernel into a single GF128.
//     Uses a grid-stride loop so a single block handles arbitrarily many partials.
//
// Power table H_powers[i] = H^{N-i} for i in [0, N):
//   Built on the host (cpu_build_ghash_power_table) and uploaded to the device.
//   For large N a GPU parallel builder is available (gpu_build_ghash_power_table).
//
// __syncthreads() is mandatory between every stride of the butterfly reduction:
//   without it, a thread may read smem[tid + stride] before another thread has
//   written it, producing a non-deterministic data race.

#include "aes_gcm.h"
#include "aes_gcm_gpu.h"
#include "gf128.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>

// ── H^{2^k} table for GPU parallel power table construction ──────────────────
// Covers N up to 2^27 ≈ 134M blocks ≈ 2 GB of ciphertext.
__constant__ GF128 d_H_squares[27];   // H^1, H^2, H^4, ..., H^{2^26}

// ── Map + intra-block reduce ──────────────────────────────────────────────────
// data:      device pointer to GHASH input bytes (16 bytes per block).
// H_powers:  H_powers[i] = H^{N-i}.
// partial:   output — one GF128 per launched thread block.
// N:         total number of 128-bit blocks.
__global__
void ghash_map_reduce_kernel(
    const uint8_t  * __restrict__ data,
    const GF128    * __restrict__ H_powers,
          GF128    * __restrict__ partial,
    int N
) {
    extern __shared__ GF128 smem[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // MAP: each thread multiplies its block by the corresponding H power.
    // Out-of-range threads contribute zero (identity for XOR reduction).
    GF128 acc = gf128_zero();
    if (gid < N) {
        GF128 Xi = gf128_load(data + (size_t)gid * AES_BLOCK_SIZE);
        acc = gf128_mul(Xi, H_powers[gid]);
    }
    smem[tid] = acc;
    __syncthreads();

    // REDUCE: butterfly tree-XOR within the thread block.
    // Each stride halves the active thread count; __syncthreads() ensures
    // all writes at stride s are visible before reads at stride s/2.
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride)
            smem[tid] = gf128_xor(smem[tid], smem[tid + stride]);
        __syncthreads();
    }

    if (tid == 0)
        partial[blockIdx.x] = smem[0];
}

// ── Final reduce (single launch block) ───────────────────────────────────────
// Combines partial[] into a single result via grid-stride accumulation then
// in-block butterfly reduction.  Launch with exactly one thread block.
__global__
void ghash_final_reduce_kernel(
    const GF128 * __restrict__ partial,
          GF128 * __restrict__ result,
    int num_partial
) {
    extern __shared__ GF128 smem2[];

    int tid = threadIdx.x;

    // Grid-stride accumulation: each thread handles ceil(num_partial / blockDim.x) values.
    GF128 acc = gf128_zero();
    for (int i = tid; i < num_partial; i += blockDim.x)
        acc = gf128_xor(acc, partial[i]);
    smem2[tid] = acc;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride)
            smem2[tid] = gf128_xor(smem2[tid], smem2[tid + stride]);
        __syncthreads();
    }

    if (tid == 0)
        *result = smem2[0];
}

// ── GPU parallel power table builder ─────────────────────────────────────────
// Each thread independently reconstructs H^exp by binary exponentiation
// over the precomputed doublings d_H_squares[k] = H^{2^k}.
// Call cpu_upload_h_squares() once before using this kernel.
__global__
void build_power_table_kernel(
    GF128   *H_powers,  // output: H_powers[i] = H^{N-i}
    int      N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int exp = N - i;    // this thread computes H^exp (exp in [1, N])
    GF128 acc = gf128_one();

    // Decompose exp in binary, multiplying by H^{2^k} for each set bit.
    #pragma unroll 27
    for (int k = 0; k < 27; k++) {
        if ((exp >> k) & 1)
            acc = gf128_mul(acc, d_H_squares[k]);
    }

    H_powers[i] = acc;
}

// ── Host-side helpers ─────────────────────────────────────────────────────────

// Upload the H^{2^k} doublings to constant memory.
// H: the hash subkey (16 bytes, host pointer).
// max_N: upper bound on the number of blocks; determines how many doublings to compute.
static void upload_h_squares(const uint8_t H_bytes[AES_BLOCK_SIZE], int max_N) {
    GF128 squares[27];
    squares[0] = gf128_load(H_bytes);
    int k;
    for (k = 1; k < 27 && (1 << (k-1)) < max_N; k++)
        squares[k] = gf128_mul(squares[k-1], squares[k-1]);
    cudaMemcpyToSymbol(d_H_squares, squares, (size_t)k * sizeof(GF128));
}

// Build the full H_powers table on the GPU.
// Requires upload_h_squares() to have been called first.
// d_H_powers: device buffer of N GF128 elements (must be pre-allocated).
static void gpu_build_power_table(GF128 *d_H_powers, int N) {
    constexpr int TPB = 256;
    int grid = (N + TPB - 1) / TPB;
    build_power_table_kernel<<<grid, TPB>>>(d_H_powers, N);
}

// ── Main GHASH entry point ────────────────────────────────────────────────────
// d_data:     device pointer to GHASH input bytes (N × 16 bytes).
// d_H_powers: device pointer to H_powers[i] = H^{N-i}, N elements.
// N:          total number of 128-bit blocks.
// d_result:   device pointer to output GF128 (single element, pre-allocated).
void gpu_ghash(const uint8_t *d_data, const GF128 *d_H_powers,
               int N, GF128 *d_result)
{
    constexpr int TPB = 256;
    int num_blocks = (N + TPB - 1) / TPB;
    size_t smem    = (size_t)TPB * sizeof(GF128);   // 256 × 16 = 4 KB

    // Allocate temporary partial sums.
    GF128 *d_partial;
    cudaMalloc(&d_partial, (size_t)num_blocks * sizeof(GF128));

    // Phase 1: map + intra-block reduce → partial[]
    ghash_map_reduce_kernel<<<num_blocks, TPB, smem>>>(
        d_data, d_H_powers, d_partial, N);

    // Phase 2: combine partials into final result.
    // A single block with grid-stride loop handles any number of partials.
    ghash_final_reduce_kernel<<<1, TPB, smem>>>(d_partial, d_result, num_blocks);

    cudaFree(d_partial);
}

// ── CPU power table builder (used for moderate-size messages) ─────────────────
// Builds H_powers[i] = H^{N-i} on the host, uploads to device.
// For N > ~1M, prefer gpu_build_power_table to avoid O(N) CPU GF128 muls.
// Returns device pointer (caller must cudaFree).
GF128 *cpu_build_and_upload_power_table(const uint8_t H_bytes[AES_BLOCK_SIZE],
                                         int N)
{
    GF128 *h_powers = (GF128*)malloc((size_t)N * sizeof(GF128));
    if (!h_powers) return nullptr;

    // Build ascending: h_powers_asc[i] = H^{i+1}, then reverse.
    GF128 H = gf128_load(H_bytes);
    GF128 cur = H;
    for (int i = N - 1; i >= 0; i--) {
        // h_powers[i] = H^{N-i}
        // h_powers[N-1] = H^1, h_powers[N-2] = H^2, ...
        // Build from index N-1 downward.
        h_powers[i] = cur;
        if (i > 0) cur = gf128_mul(cur, H);
    }

    GF128 *d_powers;
    cudaMalloc(&d_powers, (size_t)N * sizeof(GF128));
    cudaMemcpy(d_powers, h_powers, (size_t)N * sizeof(GF128), cudaMemcpyHostToDevice);
    free(h_powers);
    return d_powers;
}

// Threshold above which we use the GPU power table builder instead of the CPU.
// Roughly: N × 128 GF128-muls GPU < N CPU muls + PCIe upload time.
static constexpr int GPU_POWER_TABLE_THRESHOLD = 1 << 18;   // ~4M blocks = 64 MB

// Build power table via either CPU or GPU path depending on N.
// Returns device pointer; caller must cudaFree.
GF128 *build_power_table(const uint8_t H_bytes[AES_BLOCK_SIZE], int N) {
    if (N < GPU_POWER_TABLE_THRESHOLD)
        return cpu_build_and_upload_power_table(H_bytes, N);

    // GPU path: upload doublings, launch parallel builder.
    upload_h_squares(H_bytes, N);
    GF128 *d_powers;
    cudaMalloc(&d_powers, (size_t)N * sizeof(GF128));
    gpu_build_power_table(d_powers, N);
    return d_powers;
}
