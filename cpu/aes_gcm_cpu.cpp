#include "aes_gcm.h"
#include "utils.h"
#include <cstring>
#include <cstdlib>
#include <cstdio>

// Forward declarations of internal functions
int aes_ctr_crypt(const uint8_t *input, uint8_t *output, size_t len,
                  const uint8_t iv[GCM_IV_LEN],
                  const uint32_t *rk, int num_rounds,
                  uint32_t start_ctr);

void ghash(const uint8_t H[AES_BLOCK_SIZE],
           const uint8_t *data, size_t data_len,
           uint8_t tag[AES_BLOCK_SIZE]);

// ── Context Initialisation ────────────────────────────────────────────────────
void aes_gcm_init(AesGcmCtx *ctx, const uint8_t *key, int key_len,
                  const uint8_t iv[GCM_IV_LEN])
{
    memcpy(ctx->key, key, key_len);
    memcpy(ctx->iv,  iv,  GCM_IV_LEN);
    ctx->key_len = key_len;

    aes_key_expansion(key, key_len, ctx->round_keys, &ctx->num_rounds);

    // H = AES_K(0^128)
    uint8_t zero_block[AES_BLOCK_SIZE] = {0};
    aes_encrypt_block(zero_block, ctx->H, ctx->round_keys, ctx->num_rounds);
}

// ── GHASH input construction ──────────────────────────────────────────────────
// NIST SP 800-38D §7.1 defines the GHASH input as:
//   pad(A) || pad(C) || len(A)_64 || len(C)_64
// where pad(X) zero-pads X to a multiple of 128 bits,
// and len(X)_64 is the bit-length of X as a 64-bit big-endian integer.
static void build_ghash_input(const uint8_t *aad, size_t aad_len,
                               const uint8_t *ct,  size_t ct_len,
                               uint8_t **out, size_t *out_len)
{
    // Guard against size_t overflow in padding arithmetic.
    // NIST GCM data limit is 2^39 - 256 bits; inputs near SIZE_MAX are invalid.
    if (aad_len > (SIZE_MAX / 2) || ct_len > (SIZE_MAX / 2)) {
        fprintf(stderr, "aes_gcm: input too large\n");
        abort();
    }

    size_t aad_padded = ((aad_len + 15) / 16) * 16;
    size_t ct_padded  = ((ct_len  + 15) / 16) * 16;
    *out_len = aad_padded + ct_padded + 16;  // +16 for length block
    *out = (uint8_t *)calloc(*out_len, 1);
    if (!*out) {
        fprintf(stderr, "aes_gcm: allocation failed\n");
        abort();
    }

    if (aad_len > 0) memcpy(*out, aad, aad_len);
    if (ct_len  > 0) memcpy(*out + aad_padded, ct, ct_len);

    // Length block: len(A) || len(C) in bits, big-endian 64-bit each
    uint64_t bit_aad = (uint64_t)aad_len * 8;
    uint64_t bit_ct  = (uint64_t)ct_len  * 8;
    uint8_t *len_block = *out + aad_padded + ct_padded;
    len_block[0] = (uint8_t)(bit_aad >> 56); len_block[1] = (uint8_t)(bit_aad >> 48);
    len_block[2] = (uint8_t)(bit_aad >> 40); len_block[3] = (uint8_t)(bit_aad >> 32);
    len_block[4] = (uint8_t)(bit_aad >> 24); len_block[5] = (uint8_t)(bit_aad >> 16);
    len_block[6] = (uint8_t)(bit_aad >>  8); len_block[7] = (uint8_t)(bit_aad      );
    len_block[8]  = (uint8_t)(bit_ct >> 56); len_block[9]  = (uint8_t)(bit_ct >> 48);
    len_block[10] = (uint8_t)(bit_ct >> 40); len_block[11] = (uint8_t)(bit_ct >> 32);
    len_block[12] = (uint8_t)(bit_ct >> 24); len_block[13] = (uint8_t)(bit_ct >> 16);
    len_block[14] = (uint8_t)(bit_ct >>  8); len_block[15] = (uint8_t)(bit_ct      );
}

// ── Encrypt ───────────────────────────────────────────────────────────────────
// Returns 0 on success, -1 if payload exceeds the 32-bit counter limit (~68 GB).
// NOTE: in-place encryption (plaintext == ciphertext) is NOT supported.
int aes_gcm_encrypt(const AesGcmCtx *ctx,
                    const uint8_t *plaintext,  size_t pt_len,
                    const uint8_t *aad,         size_t aad_len,
                    uint8_t       *ciphertext,
                    uint8_t        tag[GCM_TAG_LEN])
{
    // Step 1: Encrypt plaintext with AES-CTR.
    // NIST SP 800-38D §7.1: plaintext uses ICB = inc32(J0), i.e. counter=2.
    // Counter=1 (J0) is used only for the tag XOR mask below.
    if (aes_ctr_crypt(plaintext, ciphertext, pt_len,
                      ctx->iv, ctx->round_keys, ctx->num_rounds, 2) != 0)
        return -1;

    // Step 2: Build GHASH input and compute S = GHASH_H(pad(A)||pad(C)||len)
    uint8_t *ghash_in;
    size_t   ghash_len;
    build_ghash_input(aad, aad_len, ciphertext, pt_len, &ghash_in, &ghash_len);

    uint8_t S[AES_BLOCK_SIZE];
    ghash(ctx->H, ghash_in, ghash_len, S);
    free(ghash_in);

    // Step 3: T = S XOR E(K, IV || 0^31 || 1)
    // The J0 block is IV || 0x00000001 (counter = 1 in AES-CTR notation = 0 for J0)
    // NIST: J0 = IV || 0^31 || 1 means the 32-bit counter field = 1.
    // We encrypt J0 directly (not via CTR mode which starts counter at 1).
    uint8_t J0[AES_BLOCK_SIZE];
    memcpy(J0, ctx->iv, GCM_IV_LEN);
    J0[12] = 0x00; J0[13] = 0x00; J0[14] = 0x00; J0[15] = 0x01;

    uint8_t EJ0[AES_BLOCK_SIZE];
    aes_encrypt_block(J0, EJ0, ctx->round_keys, ctx->num_rounds);

    for (int i = 0; i < GCM_TAG_LEN; i++)
        tag[i] = S[i] ^ EJ0[i];
    return 0;
}

// ── Decrypt ───────────────────────────────────────────────────────────────────
// NOTE: in-place decryption (ciphertext == plaintext) is NOT supported.
// On auth failure the plaintext buffer is zeroed; if it aliased the ciphertext
// the caller's only copy of the ciphertext would be destroyed.
int aes_gcm_decrypt(const AesGcmCtx *ctx,
                    const uint8_t *ciphertext, size_t ct_len,
                    const uint8_t *aad,         size_t aad_len,
                    const uint8_t  tag[GCM_TAG_LEN],
                    uint8_t       *plaintext)
{
    // Step 1: Recompute authentication tag over the received ciphertext
    uint8_t *ghash_in;
    size_t   ghash_len;
    build_ghash_input(aad, aad_len, ciphertext, ct_len, &ghash_in, &ghash_len);

    uint8_t S[AES_BLOCK_SIZE];
    ghash(ctx->H, ghash_in, ghash_len, S);
    free(ghash_in);

    uint8_t J0[AES_BLOCK_SIZE];
    memcpy(J0, ctx->iv, GCM_IV_LEN);
    J0[12] = 0x00; J0[13] = 0x00; J0[14] = 0x00; J0[15] = 0x01;

    uint8_t EJ0[AES_BLOCK_SIZE];
    aes_encrypt_block(J0, EJ0, ctx->round_keys, ctx->num_rounds);

    uint8_t computed_tag[GCM_TAG_LEN];
    for (int i = 0; i < GCM_TAG_LEN; i++)
        computed_tag[i] = S[i] ^ EJ0[i];

    // Step 2: Constant-time tag comparison — never short-circuits
    if (ct_memcmp(computed_tag, tag, GCM_TAG_LEN) != 0) {
        // Zero the output buffer — plaintext must never be exposed on failure
        memset(plaintext, 0, ct_len);
        return -1;
    }

    // Step 3: Decrypt (AES-CTR is its own inverse — same counter offset)
    // Counter overflow here would mean the ciphertext itself violated the limit,
    // which is impossible if it was produced by aes_gcm_encrypt.
    return aes_ctr_crypt(ciphertext, plaintext, ct_len,
                         ctx->iv, ctx->round_keys, ctx->num_rounds, 2);
}
