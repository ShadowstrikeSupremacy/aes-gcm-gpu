#pragma once
#include <cstdint>
#include <cstring>

// When compiled by nvcc, mark all functions callable from both host and device.
// When compiled by a plain C++ compiler, they are just inline.
#ifdef __CUDACC__
#  define GF128_FUNC __device__ __host__ __forceinline__
#else
#  define GF128_FUNC inline
#endif

// GF(2^128) element — two 64-bit words.
//
// Bit ordering follows GHASH (NIST SP 800-38D):
//   hi holds bits 127..64, lo holds bits 63..0.
//   The MSB of hi is the coefficient of x^0 (the constant term).
//   Byte string 0x80 0x00 ... 0x00 represents element 1.
struct GF128 {
    uint64_t hi;   // bits 127..64  (MSB = coeff of x^0)
    uint64_t lo;   // bits  63..0
};

GF128_FUNC GF128 gf128_zero() { return {0, 0}; }

// Multiplicative identity: element 1 in the GHASH reflected convention.
// hi = 0x8000000000000000 (x^0 coefficient = 1), lo = 0.
GF128_FUNC GF128 gf128_one() { return {0x8000000000000000ULL, 0}; }

GF128_FUNC GF128 gf128_xor(GF128 a, GF128 b) {
    return { a.hi ^ b.hi, a.lo ^ b.lo };
}

// Load 16 bytes (big-endian) → GF128
GF128_FUNC GF128 gf128_load(const uint8_t src[16]) {
    GF128 r;
    r.hi  = (uint64_t)src[0]  << 56 | (uint64_t)src[1]  << 48
          | (uint64_t)src[2]  << 40 | (uint64_t)src[3]  << 32
          | (uint64_t)src[4]  << 24 | (uint64_t)src[5]  << 16
          | (uint64_t)src[6]  <<  8 | (uint64_t)src[7];
    r.lo  = (uint64_t)src[8]  << 56 | (uint64_t)src[9]  << 48
          | (uint64_t)src[10] << 40 | (uint64_t)src[11] << 32
          | (uint64_t)src[12] << 24 | (uint64_t)src[13] << 16
          | (uint64_t)src[14] <<  8 | (uint64_t)src[15];
    return r;
}

// GF128 → 16 bytes big-endian
GF128_FUNC void gf128_store(const GF128 &a, uint8_t dst[16]) {
    dst[0]  = (uint8_t)(a.hi >> 56); dst[1]  = (uint8_t)(a.hi >> 48);
    dst[2]  = (uint8_t)(a.hi >> 40); dst[3]  = (uint8_t)(a.hi >> 32);
    dst[4]  = (uint8_t)(a.hi >> 24); dst[5]  = (uint8_t)(a.hi >> 16);
    dst[6]  = (uint8_t)(a.hi >>  8); dst[7]  = (uint8_t)(a.hi);
    dst[8]  = (uint8_t)(a.lo >> 56); dst[9]  = (uint8_t)(a.lo >> 48);
    dst[10] = (uint8_t)(a.lo >> 40); dst[11] = (uint8_t)(a.lo >> 32);
    dst[12] = (uint8_t)(a.lo >> 24); dst[13] = (uint8_t)(a.lo >> 16);
    dst[14] = (uint8_t)(a.lo >>  8); dst[15] = (uint8_t)(a.lo);
}

// ── GF(2^128) multiplication ──────────────────────────────────────────────────
// Reduction polynomial: x^128 + x^7 + x^2 + x + 1
// In the reflected representation the reduction constant is 0xE1 << 56.
//
// Algorithm: right-to-left binary method on Y's bits (MSB first).
//   Each round: accumulate V into Z if current Y bit is 1,
//               then shift V right by 1 and reduce if the bit that fell off was 1.
//
// Constant-time: both conditionals use branchless masking to avoid
// leaking X or Y bits through timing.
GF128_FUNC GF128 gf128_mul(GF128 X, GF128 Y) {
    const uint64_t R = 0xE100000000000000ULL;

    uint64_t Zhi = 0, Zlo = 0;
    uint64_t Vhi = X.hi, Vlo = X.lo;

    for (int i = 0; i < 128; i++) {
        uint64_t ybit = (i < 64) ? ((Y.hi >> (63 - i)) & 1)
                                 : ((Y.lo >> (127 - i)) & 1);
        uint64_t mask = -(uint64_t)ybit;
        Zhi ^= Vhi & mask;
        Zlo ^= Vlo & mask;

        uint64_t lsb = Vlo & 1;
        Vlo = (Vlo >> 1) | (Vhi << 63);
        Vhi >>= 1;
        Vhi ^= R & -(uint64_t)lsb;
    }

    return { Zhi, Zlo };
}
