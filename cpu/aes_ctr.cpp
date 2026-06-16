#include "aes_gcm.h"
#include <cstring>

// AES-CTR encryption/decryption (symmetric — same operation).
//
// Counter block format (NIST SP 800-38D §7.1):
//   [ IV (12 bytes) | counter (4 bytes, big-endian) ]
//
// start_ctr: the counter value for the FIRST block.
// AES-GCM plaintext uses start_ctr=2 because counter=1 (J0) is reserved
// for the tag computation: T = GHASH_result XOR E(K, J0).
// Returns 0 on success, -1 if the payload would wrap the 32-bit counter.
// NIST SP 800-38D §5.2.1.1: counter field is 32 bits; wraparound reuses keystream.
int aes_ctr_crypt(const uint8_t *input, uint8_t *output, size_t len,
                  const uint8_t iv[GCM_IV_LEN],
                  const uint32_t *rk, int num_rounds,
                  uint32_t start_ctr)
{
    uint8_t counter_block[AES_BLOCK_SIZE];
    uint8_t keystream[AES_BLOCK_SIZE];

    memcpy(counter_block, iv, GCM_IV_LEN);

    size_t full_blocks = len / AES_BLOCK_SIZE;
    size_t remainder   = len % AES_BLOCK_SIZE;

    // Guard: last counter used is start_ctr + full_blocks (partial block case).
    // If that would exceed 0xFFFFFFFF the keystream would wrap and repeat.
    if ((uint64_t)full_blocks > (uint64_t)0xFFFFFFFFU - start_ctr)
        return -1;

    for (size_t i = 0; i < full_blocks; i++) {
        uint32_t ctr = start_ctr + (uint32_t)i;
        counter_block[12] = (uint8_t)(ctr >> 24);
        counter_block[13] = (uint8_t)(ctr >> 16);
        counter_block[14] = (uint8_t)(ctr >>  8);
        counter_block[15] = (uint8_t)(ctr      );

        aes_encrypt_block(counter_block, keystream, rk, num_rounds);

        const uint8_t *in  = input  + i * AES_BLOCK_SIZE;
        uint8_t       *out = output + i * AES_BLOCK_SIZE;
        for (int j = 0; j < AES_BLOCK_SIZE; j++)
            out[j] = in[j] ^ keystream[j];
    }

    // Partial final block
    if (remainder > 0) {
        uint32_t ctr = start_ctr + (uint32_t)full_blocks;
        counter_block[12] = (uint8_t)(ctr >> 24);
        counter_block[13] = (uint8_t)(ctr >> 16);
        counter_block[14] = (uint8_t)(ctr >>  8);
        counter_block[15] = (uint8_t)(ctr      );

        aes_encrypt_block(counter_block, keystream, rk, num_rounds);

        const uint8_t *in  = input  + full_blocks * AES_BLOCK_SIZE;
        uint8_t       *out = output + full_blocks * AES_BLOCK_SIZE;
        for (size_t j = 0; j < remainder; j++)
            out[j] = in[j] ^ keystream[j];
    }
    return 0;
}
