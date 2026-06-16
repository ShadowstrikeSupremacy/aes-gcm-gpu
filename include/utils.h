#pragma once
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <chrono>

// ── Hex utilities ─────────────────────────────────────────────────────────────

inline void print_hex(const char *label, const uint8_t *buf, size_t len) {
    printf("%s: ", label);
    for (size_t i = 0; i < len; i++) printf("%02x", buf[i]);
    printf("\n");
}

// Parse a hex string (no spaces) into bytes. Returns false on bad input.
inline bool hex_to_bytes(const char *hex, uint8_t *out, size_t out_len) {
    size_t hex_len = strlen(hex);
    if (hex_len != out_len * 2) return false;
    for (size_t i = 0; i < out_len; i++) {
        unsigned int byte;
        if (sscanf(hex + 2 * i, "%02x", &byte) != 1) return false;
        out[i] = (uint8_t)byte;
    }
    return true;
}

// Constant-time comparison — never short-circuits.
inline int ct_memcmp(const uint8_t *a, const uint8_t *b, size_t len) {
    uint8_t diff = 0;
    for (size_t i = 0; i < len; i++) diff |= a[i] ^ b[i];
    return diff;   // 0 → equal, non-zero → different
}

// ── Timing ────────────────────────────────────────────────────────────────────

struct Timer {
    std::chrono::high_resolution_clock::time_point t0;

    void start() { t0 = std::chrono::high_resolution_clock::now(); }

    // Returns elapsed milliseconds
    double elapsed_ms() const {
        auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    double throughput_gb_s(size_t bytes) const {
        double sec = elapsed_ms() / 1000.0;
        return (double)bytes / (sec * 1e9);
    }
};
