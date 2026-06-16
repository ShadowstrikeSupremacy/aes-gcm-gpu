#pragma once
#include <cstdint>
#include <cstddef>

// ── Constants ─────────────────────────────────────────────────────────────────
static constexpr int AES_BLOCK_SIZE  = 16;
static constexpr int AES_128_KEY_LEN = 16;
static constexpr int AES_256_KEY_LEN = 32;
static constexpr int GCM_IV_LEN      = 12;
static constexpr int GCM_TAG_LEN     = 16;

// AES-128: 10 rounds → 11 round keys (each 16 bytes = 4 words)
// AES-256: 14 rounds → 15 round keys
static constexpr int AES_128_ROUNDS  = 10;
static constexpr int AES_256_ROUNDS  = 14;

// ── Context ───────────────────────────────────────────────────────────────────
struct AesGcmCtx {
    uint8_t  key[AES_256_KEY_LEN];   // up to 256-bit key
    uint8_t  iv[GCM_IV_LEN];         // 96-bit IV
    uint8_t  H[AES_BLOCK_SIZE];      // hash subkey: H = AES(K, 0^128)
    uint32_t round_keys[60];         // 15 round keys × 4 words (covers AES-256)
    int      key_len;                // 16 (AES-128) or 32 (AES-256)
    int      num_rounds;             // 10 or 14
};

// ── CPU API ───────────────────────────────────────────────────────────────────

// Initialise context: expands key, derives H = AES(K, 0^128)
void aes_gcm_init(AesGcmCtx *ctx, const uint8_t *key, int key_len,
                  const uint8_t iv[GCM_IV_LEN]);

// Authenticated encryption.
// ciphertext must be at least pt_len bytes.
// tag is always GCM_TAG_LEN (16) bytes.
// Returns 0 on success, -1 if payload exceeds the 32-bit counter limit (~68 GB).
int  aes_gcm_encrypt(const AesGcmCtx *ctx,
                     const uint8_t *plaintext,  size_t pt_len,
                     const uint8_t *aad,         size_t aad_len,
                     uint8_t       *ciphertext,
                     uint8_t        tag[GCM_TAG_LEN]);

// Authenticated decryption.
// Returns 0 on success (tag verified), -1 on authentication failure.
// On failure the output buffer is zeroed — plaintext is never exposed.
int  aes_gcm_decrypt(const AesGcmCtx *ctx,
                     const uint8_t *ciphertext, size_t ct_len,
                     const uint8_t *aad,         size_t aad_len,
                     const uint8_t  tag[GCM_TAG_LEN],
                     uint8_t       *plaintext);

// ── Low-level AES primitives (used by GPU code too) ──────────────────────────
void aes_key_expansion(const uint8_t *key, int key_len,
                       uint32_t *round_keys, int *num_rounds);

void aes_encrypt_block(const uint8_t in[AES_BLOCK_SIZE],
                       uint8_t       out[AES_BLOCK_SIZE],
                       const uint32_t *round_keys, int num_rounds);
