#include "aes_gcm.h"
#include <cstring>

// ── FIPS 197 S-Box and inverse S-Box ─────────────────────────────────────────
static const uint8_t SBOX[256] = {
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16
};

// xtime: multiply by x in GF(2^8) mod x^8+x^4+x^3+x+1
// Branchless: mask is 0xFF when MSB is set, 0x00 otherwise
static inline uint8_t xtime(uint8_t a) {
    return (uint8_t)((a << 1) ^ (0x1b & (uint8_t)(-(int)(a >> 7))));
}

// MixColumns single-column multiply
static inline uint32_t mix_col(uint32_t w) {
    uint8_t s0 = (w >> 24) & 0xff;
    uint8_t s1 = (w >> 16) & 0xff;
    uint8_t s2 = (w >>  8) & 0xff;
    uint8_t s3 = (w      ) & 0xff;
    uint8_t r0 = xtime(s0) ^ (xtime(s1) ^ s1) ^ s2 ^ s3;
    uint8_t r1 = s0 ^ xtime(s1) ^ (xtime(s2) ^ s2) ^ s3;
    uint8_t r2 = s0 ^ s1 ^ xtime(s2) ^ (xtime(s3) ^ s3);
    uint8_t r3 = (xtime(s0) ^ s0) ^ s1 ^ s2 ^ xtime(s3);
    return ((uint32_t)r0 << 24) | ((uint32_t)r1 << 16) | ((uint32_t)r2 << 8) | r3;
}

// SubWord: apply S-Box to each byte of a 32-bit word
static inline uint32_t sub_word(uint32_t w) {
    return ((uint32_t)SBOX[(w >> 24) & 0xff] << 24)
         | ((uint32_t)SBOX[(w >> 16) & 0xff] << 16)
         | ((uint32_t)SBOX[(w >>  8) & 0xff] <<  8)
         | ((uint32_t)SBOX[(w      ) & 0xff]);
}

// RotWord: rotate word left by 8 bits
static inline uint32_t rot_word(uint32_t w) {
    return (w << 8) | (w >> 24);
}

// Round constants (FIPS 197 Table 5)
static const uint32_t RCON[11] = {
    0x00000000,
    0x01000000, 0x02000000, 0x04000000, 0x08000000,
    0x10000000, 0x20000000, 0x40000000, 0x80000000,
    0x1b000000, 0x36000000
};

// ── Key Expansion (FIPS 197 §5.2) ────────────────────────────────────────────
void aes_key_expansion(const uint8_t *key, int key_len,
                       uint32_t *rk, int *num_rounds) {
    int Nk = key_len / 4;   // words in key: 4 (AES-128) or 8 (AES-256)
    *num_rounds = Nk + 6;   // 10 (AES-128) or 14 (AES-256)
    int total_words = (*num_rounds + 1) * 4;

    // Load key as big-endian words
    for (int i = 0; i < Nk; i++) {
        rk[i] = ((uint32_t)key[4*i+0] << 24) | ((uint32_t)key[4*i+1] << 16)
              | ((uint32_t)key[4*i+2] <<  8) | ((uint32_t)key[4*i+3]);
    }

    for (int i = Nk; i < total_words; i++) {
        uint32_t temp = rk[i - 1];
        if (i % Nk == 0)
            temp = sub_word(rot_word(temp)) ^ RCON[i / Nk];
        else if (Nk > 6 && i % Nk == 4)
            temp = sub_word(temp);
        rk[i] = rk[i - Nk] ^ temp;
    }
}

// ── AES Block Encryption (FIPS 197 §5.1) ─────────────────────────────────────
// State is stored column-major as four 32-bit words.
void aes_encrypt_block(const uint8_t in[16], uint8_t out[16],
                       const uint32_t *rk, int num_rounds) {
    // Load state
    uint32_t s0 = ((uint32_t)in[0]<<24)|((uint32_t)in[1]<<16)|((uint32_t)in[2]<<8)|in[3];
    uint32_t s1 = ((uint32_t)in[4]<<24)|((uint32_t)in[5]<<16)|((uint32_t)in[6]<<8)|in[7];
    uint32_t s2 = ((uint32_t)in[8]<<24)|((uint32_t)in[9]<<16)|((uint32_t)in[10]<<8)|in[11];
    uint32_t s3 = ((uint32_t)in[12]<<24)|((uint32_t)in[13]<<16)|((uint32_t)in[14]<<8)|in[15];

    // Initial round key addition
    s0 ^= rk[0]; s1 ^= rk[1]; s2 ^= rk[2]; s3 ^= rk[3];

    // Main rounds (SubBytes + ShiftRows + MixColumns + AddRoundKey)
    // ShiftRows reorders bytes across columns; we implement it inline via
    // the T-table approach using only SubBytes + manual byte extraction.
    for (int r = 1; r < num_rounds; r++) {
        uint32_t t0, t1, t2, t3;

        // SubBytes + ShiftRows in one pass (row i is shifted left by i)
        // Row 0: bytes at positions 0,4,8,12  → no shift
        // Row 1: bytes at positions 1,5,9,13  → shift 1
        // Row 2: bytes at positions 2,6,10,14 → shift 2
        // Row 3: bytes at positions 3,7,11,15 → shift 3
        uint8_t b[16];
        b[0]  = SBOX[(s0>>24)&0xff]; b[4]  = SBOX[(s1>>24)&0xff];
        b[8]  = SBOX[(s2>>24)&0xff]; b[12] = SBOX[(s3>>24)&0xff];
        b[1]  = SBOX[(s1>>16)&0xff]; b[5]  = SBOX[(s2>>16)&0xff];
        b[9]  = SBOX[(s3>>16)&0xff]; b[13] = SBOX[(s0>>16)&0xff];
        b[2]  = SBOX[(s2>> 8)&0xff]; b[6]  = SBOX[(s3>> 8)&0xff];
        b[10] = SBOX[(s0>> 8)&0xff]; b[14] = SBOX[(s1>> 8)&0xff];
        b[3]  = SBOX[(s3    )&0xff]; b[7]  = SBOX[(s0    )&0xff];
        b[11] = SBOX[(s1    )&0xff]; b[15] = SBOX[(s2    )&0xff];

        t0 = ((uint32_t)b[0]<<24)|((uint32_t)b[1]<<16)|((uint32_t)b[2]<<8)|b[3];
        t1 = ((uint32_t)b[4]<<24)|((uint32_t)b[5]<<16)|((uint32_t)b[6]<<8)|b[7];
        t2 = ((uint32_t)b[8]<<24)|((uint32_t)b[9]<<16)|((uint32_t)b[10]<<8)|b[11];
        t3 = ((uint32_t)b[12]<<24)|((uint32_t)b[13]<<16)|((uint32_t)b[14]<<8)|b[15];

        // MixColumns
        s0 = mix_col(t0) ^ rk[4*r+0];
        s1 = mix_col(t1) ^ rk[4*r+1];
        s2 = mix_col(t2) ^ rk[4*r+2];
        s3 = mix_col(t3) ^ rk[4*r+3];
    }

    // Final round: SubBytes + ShiftRows + AddRoundKey (no MixColumns)
    uint8_t b[16];
    b[0]  = SBOX[(s0>>24)&0xff]; b[4]  = SBOX[(s1>>24)&0xff];
    b[8]  = SBOX[(s2>>24)&0xff]; b[12] = SBOX[(s3>>24)&0xff];
    b[1]  = SBOX[(s1>>16)&0xff]; b[5]  = SBOX[(s2>>16)&0xff];
    b[9]  = SBOX[(s3>>16)&0xff]; b[13] = SBOX[(s0>>16)&0xff];
    b[2]  = SBOX[(s2>> 8)&0xff]; b[6]  = SBOX[(s3>> 8)&0xff];
    b[10] = SBOX[(s0>> 8)&0xff]; b[14] = SBOX[(s1>> 8)&0xff];
    b[3]  = SBOX[(s3    )&0xff]; b[7]  = SBOX[(s0    )&0xff];
    b[11] = SBOX[(s1    )&0xff]; b[15] = SBOX[(s2    )&0xff];

    uint32_t t0 = ((uint32_t)b[0]<<24)|((uint32_t)b[1]<<16)|((uint32_t)b[2]<<8)|b[3];
    uint32_t t1 = ((uint32_t)b[4]<<24)|((uint32_t)b[5]<<16)|((uint32_t)b[6]<<8)|b[7];
    uint32_t t2 = ((uint32_t)b[8]<<24)|((uint32_t)b[9]<<16)|((uint32_t)b[10]<<8)|b[11];
    uint32_t t3 = ((uint32_t)b[12]<<24)|((uint32_t)b[13]<<16)|((uint32_t)b[14]<<8)|b[15];

    t0 ^= rk[4*num_rounds+0];
    t1 ^= rk[4*num_rounds+1];
    t2 ^= rk[4*num_rounds+2];
    t3 ^= rk[4*num_rounds+3];

    out[0]=(t0>>24)&0xff; out[1]=(t0>>16)&0xff; out[2]=(t0>>8)&0xff; out[3]=t0&0xff;
    out[4]=(t1>>24)&0xff; out[5]=(t1>>16)&0xff; out[6]=(t1>>8)&0xff; out[7]=t1&0xff;
    out[8]=(t2>>24)&0xff; out[9]=(t2>>16)&0xff; out[10]=(t2>>8)&0xff; out[11]=t2&0xff;
    out[12]=(t3>>24)&0xff; out[13]=(t3>>16)&0xff; out[14]=(t3>>8)&0xff; out[15]=t3&0xff;
}
