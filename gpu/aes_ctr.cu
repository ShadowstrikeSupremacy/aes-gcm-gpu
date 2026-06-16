// gpu/aes_ctr.cu  —  CUDA AES-CTR encryption kernel.
//
// Design:
//   - One thread per 16-byte AES block.
//   - S-Box loaded from global device memory into __shared__ once per block.
//     (20-cycle shared vs 600-cycle global latency; 256 bytes = 4 cache lines.)
//   - Round keys stored in __constant__ memory (broadcast-cached, read-only).
//   - AES state kept in 4 uint32_t registers — no uint8_t b[16] local array
//     that would spill to device DRAM.
//   - 128-bit coalesced loads/stores via uint4.
//   - Endianness: GPU is little-endian; AES state is big-endian word format.
//     After AES, keystream words are byte-swapped (__byte_perm) before XOR
//     with the plaintext loaded as little-endian uint4.
//   - Templated on NUM_ROUNDS so the inner loop is fully unrolled by nvcc.
//   - __launch_bounds__(256, 4) hints the register allocator to cap usage so
//     at least 4 warps per SM can reside concurrently.

#include "aes_gcm.h"
#include <cuda_runtime.h>
#include <cstdint>

// ── Constant memory ───────────────────────────────────────────────────────────
__constant__ uint32_t d_round_keys[60];   // AES-256 max: 15 rk × 4 words
__constant__ uint8_t  d_iv[GCM_IV_LEN];  // 12-byte IV, set per message

// AES S-Box in global device memory — cooperatively loaded into __shared__.
// __device__ (not __constant__) because threads access different indices
// (non-broadcast), making constant memory serialise; shared memory does not.
static __device__ const uint8_t d_sbox[256] = {
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

// ── Device helpers ────────────────────────────────────────────────────────────

// xtime in GF(2^8), mod x^8+x^4+x^3+x+1 — branchless, register-friendly.
__device__ __forceinline__
static uint32_t xtime_d(uint32_t a) {
    return ((a << 1) ^ (0x1bu & (uint32_t)(-(int)((a >> 7) & 1)))) & 0xffu;
}

// MixColumns for one 32-bit word (big-endian column).
__device__ __forceinline__
static uint32_t mix_col_d(uint32_t w) {
    uint32_t s0 = (w >> 24) & 0xff;
    uint32_t s1 = (w >> 16) & 0xff;
    uint32_t s2 = (w >>  8) & 0xff;
    uint32_t s3 =  w        & 0xff;
    uint32_t x0 = xtime_d(s0), x1 = xtime_d(s1);
    uint32_t x2 = xtime_d(s2), x3 = xtime_d(s3);
    return ((x0 ^ x1 ^ s1 ^ s2 ^ s3) << 24)
         | ((s0 ^ x1 ^ x2 ^ s2 ^ s3) << 16)
         | ((s0 ^ s1 ^ x2 ^ x3 ^ s3) <<  8)
         | ( x0 ^ s0 ^ s1 ^ s2 ^ x3);
}

// SubBytes + ShiftRows combined, reading from shared-memory sbox.
// Produces t0..t3 (new columns) from s0..s3 (current state).
// ShiftRows: row i is shifted left by i positions in the column-major layout.
//   t_col_j byte_i = SBOX[ s_col_{(i+j) mod 4} byte_i ]
#define SUB_SHIFT(s0,s1,s2,s3, t0,t1,t2,t3, sbox)                          \
    t0 = ((uint32_t)(sbox)[(s0>>24)&0xff] << 24)                            \
       | ((uint32_t)(sbox)[(s1>>16)&0xff] << 16)                            \
       | ((uint32_t)(sbox)[(s2>> 8)&0xff] <<  8)                            \
       | ((uint32_t)(sbox)[ s3     &0xff]);                                  \
    t1 = ((uint32_t)(sbox)[(s1>>24)&0xff] << 24)                            \
       | ((uint32_t)(sbox)[(s2>>16)&0xff] << 16)                            \
       | ((uint32_t)(sbox)[(s3>> 8)&0xff] <<  8)                            \
       | ((uint32_t)(sbox)[ s0     &0xff]);                                  \
    t2 = ((uint32_t)(sbox)[(s2>>24)&0xff] << 24)                            \
       | ((uint32_t)(sbox)[(s3>>16)&0xff] << 16)                            \
       | ((uint32_t)(sbox)[(s0>> 8)&0xff] <<  8)                            \
       | ((uint32_t)(sbox)[ s1     &0xff]);                                  \
    t3 = ((uint32_t)(sbox)[(s3>>24)&0xff] << 24)                            \
       | ((uint32_t)(sbox)[(s0>>16)&0xff] << 16)                            \
       | ((uint32_t)(sbox)[(s1>> 8)&0xff] <<  8)                            \
       | ((uint32_t)(sbox)[ s2     &0xff])

// ── AES-CTR kernel ────────────────────────────────────────────────────────────
// Template on NUM_ROUNDS so nvcc can unroll the round loop at compile time.
// AES-128 → NUM_ROUNDS=10, AES-256 → NUM_ROUNDS=14.
template<int NUM_ROUNDS>
__global__ __launch_bounds__(256, 4)
void aes_ctr_kernel(
    const uint4 * __restrict__ in,
          uint4 * __restrict__ out,
    uint64_t  num_blocks,
    uint32_t  start_ctr
) {
    // ── Cooperative S-Box load: 256 threads × 1 byte each ───────────────────
    __shared__ uint8_t sbox[256];
    if (threadIdx.x < 256)
        sbox[threadIdx.x] = d_sbox[threadIdx.x];
    __syncthreads();

    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_blocks) return;

    // ── Form counter block as 4 big-endian uint32_t words ───────────────────
    // Words 0-2: IV bytes 0-11.  Word 3: counter value (NIST big-endian).
    uint32_t s0 = ((uint32_t)d_iv[0]  << 24) | ((uint32_t)d_iv[1]  << 16)
                | ((uint32_t)d_iv[2]  <<  8) |  (uint32_t)d_iv[3];
    uint32_t s1 = ((uint32_t)d_iv[4]  << 24) | ((uint32_t)d_iv[5]  << 16)
                | ((uint32_t)d_iv[6]  <<  8) |  (uint32_t)d_iv[7];
    uint32_t s2 = ((uint32_t)d_iv[8]  << 24) | ((uint32_t)d_iv[9]  << 16)
                | ((uint32_t)d_iv[10] <<  8) |  (uint32_t)d_iv[11];
    uint32_t s3 = start_ctr + (uint32_t)idx;   // counter value in big-endian

    // ── Initial AddRoundKey ──────────────────────────────────────────────────
    s0 ^= d_round_keys[0];
    s1 ^= d_round_keys[1];
    s2 ^= d_round_keys[2];
    s3 ^= d_round_keys[3];

    // ── Main rounds: SubBytes + ShiftRows + MixColumns + AddRoundKey ─────────
    // Unrolled by the compiler because NUM_ROUNDS is a template constant.
    uint32_t t0, t1, t2, t3;
    #pragma unroll
    for (int r = 1; r < NUM_ROUNDS; r++) {
        SUB_SHIFT(s0, s1, s2, s3, t0, t1, t2, t3, sbox);
        s0 = mix_col_d(t0) ^ d_round_keys[4*r + 0];
        s1 = mix_col_d(t1) ^ d_round_keys[4*r + 1];
        s2 = mix_col_d(t2) ^ d_round_keys[4*r + 2];
        s3 = mix_col_d(t3) ^ d_round_keys[4*r + 3];
    }

    // ── Final round: SubBytes + ShiftRows + AddRoundKey (no MixColumns) ─────
    SUB_SHIFT(s0, s1, s2, s3, t0, t1, t2, t3, sbox);
    s0 = t0 ^ d_round_keys[4*NUM_ROUNDS + 0];
    s1 = t1 ^ d_round_keys[4*NUM_ROUNDS + 1];
    s2 = t2 ^ d_round_keys[4*NUM_ROUNDS + 2];
    s3 = t3 ^ d_round_keys[4*NUM_ROUNDS + 3];

    // ── XOR keystream with plaintext (128-bit coalesced) ────────────────────
    // s0..s3 are big-endian keystream words; the GPU loads uint4 little-endian.
    // __byte_perm(x, 0, 0x0123) byte-swaps x so that keystream byte[0] (MSB of s0)
    // aligns with plaintext byte[0] (LSB of pt.x after little-endian load).
    uint4 pt = in[idx];
    uint4 ct;
    ct.x = pt.x ^ __byte_perm(s0, 0, 0x0123u);
    ct.y = pt.y ^ __byte_perm(s1, 0, 0x0123u);
    ct.z = pt.z ^ __byte_perm(s2, 0, 0x0123u);
    ct.w = pt.w ^ __byte_perm(s3, 0, 0x0123u);
    out[idx] = ct;
}

// ── Host-side API ─────────────────────────────────────────────────────────────

void gpu_aes_ctr_upload_key(const uint32_t *rk, int num_rounds,
                              const uint8_t  iv[GCM_IV_LEN])
{
    cudaMemcpyToSymbol(d_round_keys, rk,
                       (size_t)(num_rounds + 1) * 4 * sizeof(uint32_t));
    cudaMemcpyToSymbol(d_iv, iv, GCM_IV_LEN);
}

// Encrypt/decrypt in-place (AES-CTR is its own inverse).
// num_bytes may be non-multiple of 16; the partial final block is handled
// by a CPU fallback (too small to justify a device kernel).
//
// d_partial_buf: a 16-byte device buffer; the partial-block keystream is
//   written here so the caller can XOR the last bytes on the host.
//   Pass NULL if num_bytes is a multiple of 16.
cudaError_t gpu_aes_ctr_encrypt(const uint8_t *d_in,  uint8_t *d_out,
                                  size_t         num_bytes,
                                  uint32_t       start_ctr,
                                  int            num_rounds,
                                  uint8_t       *d_partial_buf)
{
    size_t full_blocks = num_bytes / AES_BLOCK_SIZE;
    size_t remainder   = num_bytes % AES_BLOCK_SIZE;

    if (full_blocks > 0) {
        constexpr int TPB = 256;
        int grid = (int)((full_blocks + TPB - 1) / TPB);

        if (num_rounds == 10)
            aes_ctr_kernel<10><<<grid, TPB>>>(
                (const uint4*)d_in, (uint4*)d_out, (uint64_t)full_blocks, start_ctr);
        else
            aes_ctr_kernel<14><<<grid, TPB>>>(
                (const uint4*)d_in, (uint4*)d_out, (uint64_t)full_blocks, start_ctr);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) return err;
    }

    // Partial final block: encrypt a zero block to get the keystream, then
    // the caller XORs the remaining bytes on the host.
    if (remainder > 0 && d_partial_buf != nullptr) {
        // Reuse the kernel with a single-element zero input block.
        // The partial keystream word is the encrypted counter for
        // (start_ctr + full_blocks).
        static uint8_t h_zero[AES_BLOCK_SIZE] = {};
        uint8_t *d_zero;
        cudaMalloc(&d_zero, AES_BLOCK_SIZE);
        cudaMemcpy(d_zero, h_zero, AES_BLOCK_SIZE, cudaMemcpyHostToDevice);

        if (num_rounds == 10)
            aes_ctr_kernel<10><<<1, 256>>>(
                (const uint4*)d_zero, (uint4*)d_partial_buf,
                1, start_ctr + (uint32_t)full_blocks);
        else
            aes_ctr_kernel<14><<<1, 256>>>(
                (const uint4*)d_zero, (uint4*)d_partial_buf,
                1, start_ctr + (uint32_t)full_blocks);

        cudaFree(d_zero);
    }

    return cudaGetLastError();
}
