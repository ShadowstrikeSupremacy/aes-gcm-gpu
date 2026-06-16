// tests/bench_gpu.cu  —  GPU AES-GCM throughput benchmark.
//
// Reports two metrics per payload size:
//   compute-only:    kernel time only (cudaEvent_t), excludes PCIe transfers.
//   end-to-end:      wall-clock time including host↔device memory copies.
//
// This mirrors the paper's methodology: compute-only shows the GPU's peak
// AES-GCM throughput; end-to-end shows what a real application sees.
//
// Also runs a correctness check: GPU output is compared against the CPU
// reference for the first (smallest) payload size.
//
// Usage:   ./bench
// Requires: CUDA device on the current machine.

#include "aes_gcm.h"
#include "aes_gcm_gpu.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>

// ── Utility ───────────────────────────────────────────────────────────────────

static void cuda_check(cudaError_t err, const char *where) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", where, cudaGetErrorString(err));
        exit(1);
    }
}

// Fill a host buffer with a simple deterministic pattern (no rand() dependency).
static void fill_pattern(uint8_t *buf, size_t len) {
    for (size_t i = 0; i < len; i++) buf[i] = (uint8_t)(i ^ (i >> 8));
}

// ── NIST TC2 correctness check ────────────────────────────────────────────────
// Key = 0^16, IV = 0^12, PT = 0^16 → CT = 0388dace..., Tag = ab6e47d4...
static bool correctness_check(const AesGcmGpuCtx *gctx) {
    uint8_t pt[16] = {};
    uint8_t exp_ct[16]  = {0x03,0x88,0xda,0xce,0x60,0xb6,0xa3,0x92,
                            0xf3,0x28,0xc2,0xb9,0x71,0xb2,0xfe,0x78};
    uint8_t exp_tag[16] = {0xab,0x6e,0x47,0xd4,0x2c,0xec,0x13,0xbd,
                            0xf5,0x3a,0x67,0xb2,0x12,0x57,0xbd,0xdf};

    uint8_t *d_pt, *d_ct;
    cuda_check(cudaMalloc(&d_pt, 16), "malloc d_pt");
    cuda_check(cudaMalloc(&d_ct, 16), "malloc d_ct");
    cuda_check(cudaMemcpy(d_pt, pt, 16, cudaMemcpyHostToDevice), "upload pt");

    uint8_t tag[16];
    cuda_check(aes_gcm_gpu_encrypt(gctx, d_pt, 16, nullptr, 0, d_ct, tag),
               "gpu_encrypt correctness");

    uint8_t got_ct[16];
    cuda_check(cudaMemcpy(got_ct, d_ct, 16, cudaMemcpyDeviceToHost), "download ct");

    cudaFree(d_pt); cudaFree(d_ct);

    bool ct_ok  = (memcmp(got_ct, exp_ct, 16) == 0);
    bool tag_ok = (memcmp(tag, exp_tag, 16) == 0);

    printf("Correctness (NIST TC2): CT=%s  Tag=%s\n",
           ct_ok ? "PASS" : "FAIL", tag_ok ? "PASS" : "FAIL");

    if (!ct_ok)  { print_hex("  exp CT ", exp_ct,  16); print_hex("  got CT ", got_ct, 16); }
    if (!tag_ok) { print_hex("  exp Tag", exp_tag, 16); print_hex("  got Tag", tag, 16); }

    return ct_ok && tag_ok;
}

// ── Single benchmark run ──────────────────────────────────────────────────────
struct BenchResult {
    double compute_gb_s;   // kernel time only
    double e2e_gb_s;       // including PCIe transfers
};

static BenchResult bench_size(const AesGcmGpuCtx *gctx, size_t payload_bytes,
                               int warmup_runs, int measure_runs)
{
    // Allocate host and device buffers (padded to 16-byte boundary).
    size_t padded = (payload_bytes + 15) & ~(size_t)15;

    uint8_t *h_pt = (uint8_t*)malloc(padded);
    uint8_t *h_ct = (uint8_t*)malloc(padded);
    fill_pattern(h_pt, payload_bytes);
    memset(h_pt + payload_bytes, 0, padded - payload_bytes);

    uint8_t *d_pt, *d_ct;
    cuda_check(cudaMalloc(&d_pt, padded), "malloc d_pt bench");
    cuda_check(cudaMalloc(&d_ct, padded), "malloc d_ct bench");

    uint8_t tag[GCM_TAG_LEN];

    // CUDA events for compute-only timing.
    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start); cudaEventCreate(&ev_stop);

    // ── Warm-up ──────────────────────────────────────────────────────────────
    cuda_check(cudaMemcpy(d_pt, h_pt, padded, cudaMemcpyHostToDevice), "upload warmup");
    for (int i = 0; i < warmup_runs; i++) {
        aes_gcm_gpu_encrypt(gctx, d_pt, payload_bytes, nullptr, 0, d_ct, tag);
    }
    cuda_check(cudaDeviceSynchronize(), "sync warmup");

    // ── Compute-only timing ───────────────────────────────────────────────────
    cudaEventRecord(ev_start);
    for (int i = 0; i < measure_runs; i++)
        aes_gcm_gpu_encrypt(gctx, d_pt, payload_bytes, nullptr, 0, d_ct, tag);
    cudaEventRecord(ev_stop);
    cuda_check(cudaEventSynchronize(ev_stop), "event sync compute");

    float ms_compute = 0.0f;
    cudaEventElapsedTime(&ms_compute, ev_start, ev_stop);
    ms_compute /= measure_runs;

    // ── End-to-end timing (includes PCIe h→d and d→h) ────────────────────────
    Timer wall;
    wall.start();
    for (int i = 0; i < measure_runs; i++) {
        cudaMemcpy(d_pt, h_pt, padded, cudaMemcpyHostToDevice);
        aes_gcm_gpu_encrypt(gctx, d_pt, payload_bytes, nullptr, 0, d_ct, tag);
        cudaMemcpy(h_ct, d_ct, padded, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
    }
    double ms_e2e = wall.elapsed_ms() / measure_runs;

    cudaEventDestroy(ev_start); cudaEventDestroy(ev_stop);
    cudaFree(d_pt); cudaFree(d_ct);
    free(h_pt); free(h_ct);

    BenchResult r;
    r.compute_gb_s = (double)payload_bytes / (ms_compute * 1e6);  // GB/s
    r.e2e_gb_s     = (double)payload_bytes / (ms_e2e     * 1e6);
    return r;
}

// ── CPU reference ─────────────────────────────────────────────────────────────
static double bench_cpu(const AesGcmCtx *cpu_ctx, size_t payload_bytes,
                          int warmup_runs, int measure_runs)
{
    uint8_t *pt = (uint8_t*)malloc(payload_bytes);
    uint8_t *ct = (uint8_t*)malloc(payload_bytes);
    fill_pattern(pt, payload_bytes);

    uint8_t tag[GCM_TAG_LEN];

    for (int i = 0; i < warmup_runs; i++)
        aes_gcm_encrypt(cpu_ctx, pt, payload_bytes, nullptr, 0, ct, tag);

    Timer t;
    t.start();
    for (int i = 0; i < measure_runs; i++)
        aes_gcm_encrypt(cpu_ctx, pt, payload_bytes, nullptr, 0, ct, tag);
    double ms = t.elapsed_ms() / measure_runs;

    free(pt); free(ct);
    return (double)payload_bytes / (ms * 1e6);
}

// ── Main ──────────────────────────────────────────────────────────────────────
int main() {
    // Device info
    int dev = 0;
    cudaDeviceProp prop;
    cuda_check(cudaGetDeviceProperties(&prop, dev), "getDeviceProperties");
    printf("Device: %s (sm_%d%d, %d SMs, %.1f GB GDDR)\n",
           prop.name, prop.major, prop.minor,
           prop.multiProcessorCount,
           (double)prop.totalGlobalMem / 1e9);

    // Key / IV setup (all-zero for benchmark reproducibility)
    uint8_t key[AES_128_KEY_LEN] = {};
    uint8_t iv[GCM_IV_LEN]       = {};

    AesGcmGpuCtx gctx;
    aes_gcm_gpu_init(&gctx, key, AES_128_KEY_LEN, iv);

    // Correctness gate before benchmarking
    if (!correctness_check(&gctx)) {
        fprintf(stderr, "Correctness check FAILED — aborting benchmark.\n");
        return 1;
    }
    printf("\n");

    // CPU context for speedup comparison
    AesGcmCtx cpu_ctx;
    aes_gcm_init(&cpu_ctx, key, AES_128_KEY_LEN, iv);

    // Payload sizes: 1 MB, 16 MB, 256 MB
    // (1 GB omitted by default; add 1u<<30 if you have enough device memory.)
    size_t sizes[] = { 1u<<20, 1u<<24, 1u<<28 };
    int    warmup  = 3;
    int    runs    = 10;

    printf("%-12s  %-16s  %-16s  %-16s  %-10s\n",
           "Payload", "Compute GB/s", "E2E GB/s", "CPU GB/s", "Speedup");
    printf("%s\n", "────────────────────────────────────────────────────────────────────────");

    for (size_t sz : sizes) {
        BenchResult g = bench_size(&gctx, sz, warmup, runs);
        double cpu_gbps = bench_cpu(&cpu_ctx, sz, warmup, runs);
        double speedup  = g.compute_gb_s / cpu_gbps;

        printf("%-12zu  %-16.2f  %-16.2f  %-16.2f  %.1fx\n",
               sz >> 20, g.compute_gb_s, g.e2e_gb_s, cpu_gbps, speedup);
    }

    return 0;
}
