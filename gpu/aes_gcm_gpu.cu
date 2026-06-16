// gpu/aes_gcm_gpu.cu  —  Top-level GPU AES-GCM API.
//
// Thin wrapper that:
//   1. Initialises the context (CPU key schedule + uploads to constant memory).
//   2. Delegates to the MapReduce driver (mapreduce.cu) for encrypt/decrypt.

#include "aes_gcm_gpu.h"
#include <cstring>

// From mapreduce.cu
cudaError_t gpu_encrypt_mapreduce(
    const AesGcmGpuCtx *ctx,
    const uint8_t      *d_plaintext,  size_t pt_len,
    const uint8_t      *h_aad,        size_t aad_len,
    uint8_t            *d_ciphertext,
    uint8_t             tag[GCM_TAG_LEN]);

int gpu_decrypt_mapreduce(
    const AesGcmGpuCtx *ctx,
    const uint8_t      *d_ciphertext, size_t ct_len,
    const uint8_t      *h_aad,        size_t aad_len,
    const uint8_t       tag[GCM_TAG_LEN],
    uint8_t            *d_plaintext);

// From aes_ctr.cu
void gpu_aes_ctr_upload_key(const uint32_t *rk, int num_rounds,
                              const uint8_t  iv[GCM_IV_LEN]);

// ── Initialisation ────────────────────────────────────────────────────────────
void aes_gcm_gpu_init(AesGcmGpuCtx *gctx,
                      const uint8_t *key, int key_len,
                      const uint8_t  iv[GCM_IV_LEN])
{
    aes_gcm_init(&gctx->cpu, key, key_len, iv);
    gpu_aes_ctr_upload_key(gctx->cpu.round_keys, gctx->cpu.num_rounds,
                            gctx->cpu.iv);
}

// ── Encrypt ───────────────────────────────────────────────────────────────────
cudaError_t aes_gcm_gpu_encrypt(
    const AesGcmGpuCtx *ctx,
    const uint8_t *d_plaintext,  size_t pt_len,
    const uint8_t *aad,          size_t aad_len,
    uint8_t       *d_ciphertext,
    uint8_t        tag[GCM_TAG_LEN])
{
    return gpu_encrypt_mapreduce(ctx, d_plaintext, pt_len,
                                  aad, aad_len, d_ciphertext, tag);
}

// ── Decrypt ───────────────────────────────────────────────────────────────────
int aes_gcm_gpu_decrypt(
    const AesGcmGpuCtx *ctx,
    const uint8_t *d_ciphertext, size_t ct_len,
    const uint8_t *aad,          size_t aad_len,
    const uint8_t  tag[GCM_TAG_LEN],
    uint8_t       *d_plaintext)
{
    return gpu_decrypt_mapreduce(ctx, d_ciphertext, ct_len,
                                  aad, aad_len, tag, d_plaintext);
}
