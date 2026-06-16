#pragma once
#include <cuda_runtime.h>
#include "aes_gcm.h"
#include "gf128.h"

// ── GPU context ───────────────────────────────────────────────────────────────
// Wraps a CPU AesGcmCtx and holds device-side precomputed data.
// Call aes_gcm_gpu_init() once per key/IV pair before any encrypt/decrypt.
struct AesGcmGpuCtx {
    AesGcmCtx cpu;    // CPU context (round keys, H, IV, etc.)
};

// ── Initialisation ────────────────────────────────────────────────────────────
// Initialises the CPU sub-context and uploads round keys + IV to constant memory.
void aes_gcm_gpu_init(AesGcmGpuCtx *gctx,
                      const uint8_t *key, int key_len,
                      const uint8_t  iv[GCM_IV_LEN]);

// ── Authenticated encryption (GPU) ───────────────────────────────────────────
// d_plaintext / d_ciphertext: device pointers, must be 16-byte aligned and
//   backed by at least ((pt_len + 15) & ~15) bytes (padding to block boundary).
// aad: host pointer (AAD is typically small; GPU handles large ct).
// tag: host output, 16 bytes.
//
// Returns cudaSuccess on success, or a CUDA error code.
// Returns cudaErrorInvalidValue if the payload exceeds the AES-CTR 32-bit
// counter limit (~68 GB).
cudaError_t aes_gcm_gpu_encrypt(
    const AesGcmGpuCtx *ctx,
    const uint8_t *d_plaintext,  size_t pt_len,
    const uint8_t *aad,          size_t aad_len,
    uint8_t       *d_ciphertext,
    uint8_t        tag[GCM_TAG_LEN]);

// ── Authenticated decryption (GPU) ───────────────────────────────────────────
// Returns 0 on success (tag verified), -1 on authentication failure.
// On tag mismatch d_plaintext is zeroed — plaintext is never exposed.
int aes_gcm_gpu_decrypt(
    const AesGcmGpuCtx *ctx,
    const uint8_t *d_ciphertext, size_t ct_len,
    const uint8_t *aad,          size_t aad_len,
    const uint8_t  tag[GCM_TAG_LEN],
    uint8_t       *d_plaintext);

// ── Low-level kernel launchers (used by bench_gpu.cu) ───────────────────────

// Upload round keys and IV to __constant__ memory.
// Must be called before gpu_aes_ctr_encrypt on the same device.
void gpu_aes_ctr_upload_key(const uint32_t *rk, int num_rounds,
                              const uint8_t  iv[GCM_IV_LEN]);

// Encrypt/decrypt num_bytes bytes of AES-CTR keystream.
// d_in / d_out must be 16-byte aligned; the last partial block (< 16 bytes)
// is left in d_partial_out (a 16-byte host buffer) for the caller to apply.
// start_ctr: counter value for the first block (2 for AES-GCM plaintext).
cudaError_t gpu_aes_ctr_encrypt(const uint8_t *d_in, uint8_t *d_out,
                                  size_t         num_bytes,
                                  uint32_t       start_ctr,
                                  int            num_rounds,
                                  uint8_t       *d_partial_buf);

// Parallel GHASH over a pre-formatted input block (pad(A)||pad(C)||len).
// H_powers[i] = H^{N-i} for i in [0, N), must be on the device.
// Result is written to *d_result (device pointer).
void gpu_ghash(const uint8_t *d_data, const GF128 *d_H_powers,
               int N, GF128 *d_result);
