#include "aes_gcm.h"
#include "gf128.h"
#include <cstring>

// Declare so ghash.cpp can call it without exposing it in the header
int aes_ctr_crypt(const uint8_t *input, uint8_t *output, size_t len,
                  const uint8_t iv[GCM_IV_LEN],
                  const uint32_t *rk, int num_rounds,
                  uint32_t start_ctr);

// ── GHASH (NIST SP 800-38D §6.4) ─────────────────────────────────────────────
//
// GHASH_H(X) = X_1·H^m ⊕ X_2·H^(m-1) ⊕ ... ⊕ X_m·H
//
// Evaluated via Horner's method (sequential, O(N)):
//   Y_0 = 0
//   Y_i = (Y_{i-1} ⊕ X_i) · H
//
// Input data is processed as a sequence of 128-bit blocks.
// The last block is zero-padded if not 16-byte aligned.
//
// This implements the full GHASH over a pre-formatted input that already
// includes the length block (len(A) || len(C)).  The caller (aes_gcm_cpu.cpp)
// is responsible for building the formatted input.

void ghash(const uint8_t H[AES_BLOCK_SIZE],
           const uint8_t *data, size_t data_len,
           uint8_t tag[AES_BLOCK_SIZE])
{
    GF128 h = gf128_load(H);
    GF128 y = gf128_zero();

    size_t full_blocks = data_len / AES_BLOCK_SIZE;
    size_t remainder   = data_len % AES_BLOCK_SIZE;

    for (size_t i = 0; i < full_blocks; i++) {
        GF128 xi = gf128_load(data + i * AES_BLOCK_SIZE);
        y = gf128_mul(gf128_xor(y, xi), h);
    }

    if (remainder > 0) {
        uint8_t padded[AES_BLOCK_SIZE] = {0};
        memcpy(padded, data + full_blocks * AES_BLOCK_SIZE, remainder);
        GF128 xi = gf128_load(padded);
        y = gf128_mul(gf128_xor(y, xi), h);
    }

    gf128_store(y, tag);
}
