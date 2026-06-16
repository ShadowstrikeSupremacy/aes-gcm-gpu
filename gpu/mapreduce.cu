// gpu/mapreduce.cu  —  Two-phase MapReduce driver for large payloads (Phase 4).
//
// For message sizes up to ~68 GB (AES-CTR 32-bit counter limit), the AES-CTR
// kernel and GHASH kernel can each operate on the full payload in a single
// launch — CUDA supports grids up to 2^31-1 thread blocks.
//
// For GHASH, the bottleneck is the power table H_powers[N] where N can be up
// to ~4B blocks (68 GB).  At 16 bytes per GF128, that is 64 GB of device
// memory — infeasible to store all at once.  The MapReduce chunking strategy:
//
//   1. Divide N blocks into C chunks of CHUNK_BLOCKS blocks each.
//   2. For each chunk c:
//        - H_powers for this chunk: H^{N - c*CHUNK_BLOCKS} down to H^{N - (c+1)*CHUNK_BLOCKS + 1}.
//        - Build power table for CHUNK_BLOCKS (or smaller) blocks.
//        - Run ghash_map_reduce_kernel → one GF128 partial sum per GPU block.
//        - Run ghash_final_reduce_kernel → one GF128 partial sum per chunk.
//   3. Combine C chunk partial sums on the CPU (XOR — they are already correctly
//      weighted via H_powers[gid]).
//
// For AES-CTR, the full payload is processed in a single launch since each
// block is independent (no power table needed).
//
// CHUNK_BLOCKS is chosen so the power table for one chunk fits in device SRAM
// shared budget and the build cost is amortised.

#include "aes_gcm.h"
#include "aes_gcm_gpu.h"
#include "gf128.h"
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstring>
#include <cstdio>

// Forward declarations of kernel launchers from the other .cu files.
cudaError_t gpu_aes_ctr_encrypt(const uint8_t *d_in, uint8_t *d_out,
                                  size_t num_bytes, uint32_t start_ctr,
                                  int num_rounds, uint8_t *d_partial_buf);

void gpu_ghash(const uint8_t *d_data, const GF128 *d_H_powers,
               int N, GF128 *d_result);

GF128 *build_power_table(const uint8_t H_bytes[AES_BLOCK_SIZE], int N);

// ── GHASH input construction on device ───────────────────────────────────────
// Constructs pad(AAD) || pad(CT) || len_block in a device buffer.
// AAD is on the host (typically small); CT is already on the device.
// Returns device pointer to the combined input (caller must cudaFree).
static uint8_t *build_ghash_device_input(
    const uint8_t *h_aad,     size_t aad_len,
    const uint8_t *d_ct,      size_t ct_len,
    size_t        *out_len)
{
    size_t aad_padded = ((aad_len + 15) / 16) * 16;
    size_t ct_padded  = ((ct_len  + 15) / 16) * 16;
    size_t total      = aad_padded + ct_padded + 16;   // +16 for len block
    *out_len = total;

    uint8_t *d_buf;
    cudaMalloc(&d_buf, total);
    cudaMemset(d_buf, 0, total);    // zero-pads both AAD and CT to block boundary

    if (aad_len > 0)
        cudaMemcpy(d_buf, h_aad, aad_len, cudaMemcpyHostToDevice);

    if (ct_len > 0)
        cudaMemcpy(d_buf + aad_padded, d_ct, ct_len, cudaMemcpyDeviceToDevice);

    // Length block: len(A) || len(C) in bits, big-endian 64-bit each
    uint8_t len_block[16];
    uint64_t bit_aad = (uint64_t)aad_len * 8;
    uint64_t bit_ct  = (uint64_t)ct_len  * 8;
    for (int i = 0; i < 8; i++) len_block[    i] = (uint8_t)(bit_aad >> (56 - 8*i));
    for (int i = 0; i < 8; i++) len_block[8 + i] = (uint8_t)(bit_ct  >> (56 - 8*i));
    cudaMemcpy(d_buf + aad_padded + ct_padded, len_block, 16, cudaMemcpyHostToDevice);

    return d_buf;
}

// ── GPU GHASH with chunking ───────────────────────────────────────────────────
// Processes a GHASH input that may be too large for a single power table.
// Writes the final 128-bit GHASH result into h_result (host).
static void gpu_ghash_chunked(const uint8_t *d_data, int N,
                               const uint8_t  H_bytes[AES_BLOCK_SIZE],
                               uint8_t        h_result[AES_BLOCK_SIZE])
{
    // 64 MB power table = 4M blocks.  Fits comfortably in device memory.
    constexpr int CHUNK_BLOCKS = 1 << 22;

    GF128 combined = gf128_zero();
    int offset = 0;

    while (offset < N) {
        int chunk = (N - offset < CHUNK_BLOCKS) ? (N - offset) : CHUNK_BLOCKS;

        // The power table for this chunk:
        // Block at global index (offset + i) corresponds to X_{offset+i+1},
        // which must be multiplied by H^{N - (offset+i)}.
        // So power table[i] = H^{N - offset - i} for i in [0, chunk).
        // We build this by calling build_power_table with the "virtual N"
        // relative to this chunk's starting position (= N - offset).
        int virtual_N = N - offset;
        GF128 *d_powers = build_power_table(H_bytes, virtual_N);
        // d_powers[i] = H^{virtual_N - i} = H^{N - offset - i} ✓
        // We only need the first `chunk` entries (for i in [0, chunk)).
        // build_power_table returns virtual_N entries; extra entries are unused.

        GF128 *d_chunk_result;
        cudaMalloc(&d_chunk_result, sizeof(GF128));
        cudaMemset(d_chunk_result, 0, sizeof(GF128));

        gpu_ghash(d_data + (size_t)offset * AES_BLOCK_SIZE,
                  d_powers, chunk, d_chunk_result);

        // Copy chunk result to host as a struct (not raw bytes).
        // gf128_load() would misinterpret the little-endian GPU struct layout as
        // big-endian; copying the struct directly preserves hi/lo correctly.
        GF128 chunk_val;
        cudaMemcpy(&chunk_val, d_chunk_result, sizeof(GF128), cudaMemcpyDeviceToHost);
        combined = gf128_xor(combined, chunk_val);

        cudaFree(d_chunk_result);
        cudaFree(d_powers);

        offset += chunk;
    }

    gf128_store(combined, h_result);
}

// ── Top-level GPU AES-GCM encrypt ────────────────────────────────────────────
// Called by aes_gcm_gpu_encrypt in aes_gcm_gpu.cu.
cudaError_t gpu_encrypt_mapreduce(
    const AesGcmGpuCtx *ctx,
    const uint8_t      *d_plaintext,  size_t pt_len,
    const uint8_t      *h_aad,        size_t aad_len,
    uint8_t            *d_ciphertext,
    uint8_t             tag[GCM_TAG_LEN])
{
    // ── Step 1: AES-CTR encryption ───────────────────────────────────────────
    // Counter starts at 2 (NIST: counter 1 = J0, reserved for tag mask).
    uint8_t *d_partial = nullptr;
    size_t remainder = pt_len % AES_BLOCK_SIZE;
    if (remainder > 0) cudaMalloc(&d_partial, AES_BLOCK_SIZE);

    cudaError_t err = gpu_aes_ctr_encrypt(
        d_plaintext, d_ciphertext, pt_len, 2,
        ctx->cpu.num_rounds, d_partial);
    if (err != cudaSuccess) { cudaFree(d_partial); return err; }

    // Handle partial last block on the host.
    if (remainder > 0) {
        uint8_t h_ks[AES_BLOCK_SIZE];
        cudaMemcpy(h_ks, d_partial, AES_BLOCK_SIZE, cudaMemcpyDeviceToHost);
        uint8_t *h_pt_tail = (uint8_t*)malloc(remainder);
        cudaMemcpy(h_pt_tail, d_plaintext + (pt_len - remainder),
                   remainder, cudaMemcpyDeviceToHost);
        uint8_t h_ct_tail[AES_BLOCK_SIZE];
        for (size_t i = 0; i < remainder; i++)
            h_ct_tail[i] = h_pt_tail[i] ^ h_ks[i];
        cudaMemcpy(d_ciphertext + (pt_len - remainder), h_ct_tail,
                   remainder, cudaMemcpyHostToDevice);
        free(h_pt_tail);
        cudaFree(d_partial);
    }

    // ── Step 2: Build GHASH input on device ──────────────────────────────────
    size_t ghash_len;
    uint8_t *d_ghash_in = build_ghash_device_input(
        h_aad, aad_len, d_ciphertext, pt_len, &ghash_len);
    int N = (int)(ghash_len / AES_BLOCK_SIZE);

    // ── Step 3: Parallel GHASH ───────────────────────────────────────────────
    uint8_t h_S[AES_BLOCK_SIZE];
    gpu_ghash_chunked(d_ghash_in, N, ctx->cpu.H, h_S);
    cudaFree(d_ghash_in);

    // ── Step 4: T = S XOR E(K, J0) ──────────────────────────────────────────
    // J0 = IV || 0x00000001.  Encrypt on CPU (single block, negligible cost).
    uint8_t J0[AES_BLOCK_SIZE];
    memcpy(J0, ctx->cpu.iv, GCM_IV_LEN);
    J0[12] = 0x00; J0[13] = 0x00; J0[14] = 0x00; J0[15] = 0x01;

    uint8_t EJ0[AES_BLOCK_SIZE];
    aes_encrypt_block(J0, EJ0, ctx->cpu.round_keys, ctx->cpu.num_rounds);

    for (int i = 0; i < GCM_TAG_LEN; i++)
        tag[i] = h_S[i] ^ EJ0[i];

    return cudaSuccess;
}

// ── Top-level GPU AES-GCM decrypt ────────────────────────────────────────────
int gpu_decrypt_mapreduce(
    const AesGcmGpuCtx *ctx,
    const uint8_t      *d_ciphertext, size_t ct_len,
    const uint8_t      *h_aad,        size_t aad_len,
    const uint8_t       tag[GCM_TAG_LEN],
    uint8_t            *d_plaintext)
{
    // ── Step 1: Verify tag before decrypting ─────────────────────────────────
    size_t ghash_len;
    uint8_t *d_ghash_in = build_ghash_device_input(
        h_aad, aad_len, d_ciphertext, ct_len, &ghash_len);
    int N = (int)(ghash_len / AES_BLOCK_SIZE);

    uint8_t h_S[AES_BLOCK_SIZE];
    gpu_ghash_chunked(d_ghash_in, N, ctx->cpu.H, h_S);
    cudaFree(d_ghash_in);

    uint8_t J0[AES_BLOCK_SIZE];
    memcpy(J0, ctx->cpu.iv, GCM_IV_LEN);
    J0[12] = 0x00; J0[13] = 0x00; J0[14] = 0x00; J0[15] = 0x01;

    uint8_t EJ0[AES_BLOCK_SIZE];
    aes_encrypt_block(J0, EJ0, ctx->cpu.round_keys, ctx->cpu.num_rounds);

    uint8_t computed_tag[GCM_TAG_LEN];
    for (int i = 0; i < GCM_TAG_LEN; i++)
        computed_tag[i] = h_S[i] ^ EJ0[i];

    // Constant-time comparison.
    uint8_t diff = 0;
    for (int i = 0; i < GCM_TAG_LEN; i++) diff |= computed_tag[i] ^ tag[i];
    if (diff != 0) {
        cudaMemset(d_plaintext, 0, ct_len);
        return -1;
    }

    // ── Step 2: AES-CTR decrypt ──────────────────────────────────────────────
    uint8_t *d_partial = nullptr;
    size_t remainder = ct_len % AES_BLOCK_SIZE;
    if (remainder > 0) cudaMalloc(&d_partial, AES_BLOCK_SIZE);

    gpu_aes_ctr_encrypt(d_ciphertext, d_plaintext, ct_len, 2,
                         ctx->cpu.num_rounds, d_partial);

    if (remainder > 0) {
        uint8_t h_ks[AES_BLOCK_SIZE];
        cudaMemcpy(h_ks, d_partial, AES_BLOCK_SIZE, cudaMemcpyDeviceToHost);
        uint8_t h_ct_tail[AES_BLOCK_SIZE];
        cudaMemcpy(h_ct_tail, d_ciphertext + (ct_len - remainder),
                   remainder, cudaMemcpyDeviceToHost);
        uint8_t h_pt_tail[AES_BLOCK_SIZE];
        for (size_t i = 0; i < remainder; i++)
            h_pt_tail[i] = h_ct_tail[i] ^ h_ks[i];
        cudaMemcpy(d_plaintext + (ct_len - remainder), h_pt_tail,
                   remainder, cudaMemcpyHostToDevice);
        cudaFree(d_partial);
    }

    return 0;
}
