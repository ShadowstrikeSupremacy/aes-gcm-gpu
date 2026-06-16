#include "aes_gcm.h"
#include "utils.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <string>

static void usage(const char *prog) {
    fprintf(stderr,
        "Usage:\n"
        "  %s encrypt --key <hex32> --iv <hex24> --in <file> --out <file>\n"
        "  %s decrypt --key <hex32> --iv <hex24> --tag <hex32> --in <file> --out <file>\n"
        "  %s tamper  --key <hex32> --iv <hex24> --in <file>\n",
        prog, prog, prog);
}

static std::vector<uint8_t> read_file(const char *path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    return { std::istreambuf_iterator<char>(f), {} };
}

static void write_file(const char *path, const uint8_t *data, size_t len) {
    std::ofstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "Cannot write %s\n", path); exit(1); }
    f.write((const char *)data, len);
}

int main(int argc, char **argv) {
    if (argc < 2) { usage(argv[0]); return 1; }
    std::string mode = argv[1];

    // Parse flags
    const char *key_hex = nullptr, *iv_hex = nullptr;
    const char *tag_hex = nullptr;
    const char *in_path = nullptr, *out_path = nullptr;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--key") == 0 && i+1 < argc) key_hex = argv[++i];
        else if (strcmp(argv[i], "--iv")  == 0 && i+1 < argc) iv_hex  = argv[++i];
        else if (strcmp(argv[i], "--tag") == 0 && i+1 < argc) tag_hex = argv[++i];
        else if (strcmp(argv[i], "--in")  == 0 && i+1 < argc) in_path  = argv[++i];
        else if (strcmp(argv[i], "--out") == 0 && i+1 < argc) out_path = argv[++i];
    }

    if (!key_hex || !iv_hex || !in_path) {
        fprintf(stderr, "Missing required arguments.\n");
        usage(argv[0]); return 1;
    }

    uint8_t key[32] = {0};
    uint8_t iv[GCM_IV_LEN] = {0};
    int key_len = (int)(strlen(key_hex) / 2);
    if (key_len != 16 && key_len != 32) {
        fprintf(stderr, "Key must be 32 or 64 hex chars (128 or 256-bit).\n");
        return 1;
    }
    if (!hex_to_bytes(key_hex, key, key_len)) { fprintf(stderr, "Bad key hex\n"); return 1; }
    if (!hex_to_bytes(iv_hex, iv, GCM_IV_LEN)) { fprintf(stderr, "Bad IV hex\n"); return 1; }

    AesGcmCtx ctx;
    aes_gcm_init(&ctx, key, key_len, iv);

    // ── Encrypt ───────────────────────────────────────────────────────────────
    if (mode == "encrypt") {
        if (!out_path) { fprintf(stderr, "--out required\n"); return 1; }

        auto pt = read_file(in_path);
        std::vector<uint8_t> ct(pt.size());
        uint8_t tag[GCM_TAG_LEN];

        Timer t; t.start();
        aes_gcm_encrypt(&ctx, pt.data(), pt.size(), nullptr, 0, ct.data(), tag);
        double ms = t.elapsed_ms();

        write_file(out_path, ct.data(), ct.size());

        printf("[AES-GCM Encrypt]\n");
        printf("  Input:      %s (%zu bytes)\n", in_path, pt.size());
        printf("  Output:     %s\n", out_path);
        print_hex("  Auth Tag ", tag, GCM_TAG_LEN);
        printf("  Time:       %.2f ms\n", ms);
        if (ms > 0)
            printf("  Throughput: %.2f MB/s\n", (double)pt.size() / (ms / 1000.0) / 1e6);

    // ── Decrypt ───────────────────────────────────────────────────────────────
    } else if (mode == "decrypt") {
        if (!out_path || !tag_hex) {
            fprintf(stderr, "--out and --tag required for decrypt\n"); return 1;
        }

        uint8_t tag[GCM_TAG_LEN];
        if (!hex_to_bytes(tag_hex, tag, GCM_TAG_LEN)) {
            fprintf(stderr, "Bad tag hex\n"); return 1;
        }

        auto ct = read_file(in_path);
        std::vector<uint8_t> pt(ct.size());

        int rc = aes_gcm_decrypt(&ctx, ct.data(), ct.size(), nullptr, 0, tag, pt.data());

        if (rc != 0) {
            printf("[AES-GCM Decrypt]\n");
            printf("  ERROR: authentication tag mismatch — plaintext withheld.\n");
            printf("  Error code: AEAD_DECRYPT_FAILED\n");
            return 1;
        }

        write_file(out_path, pt.data(), pt.size());
        printf("[AES-GCM Decrypt]\n");
        printf("  Input:  %s (%zu bytes)\n", in_path, ct.size());
        printf("  Output: %s\n", out_path);
        printf("  Tag verified: OK\n");
        printf("  Plaintext written successfully.\n");

    // ── Tamper Demo ───────────────────────────────────────────────────────────
    } else if (mode == "tamper") {
        // Encrypt the file, flip one bit in the ciphertext, try to decrypt
        auto pt = read_file(in_path);
        std::vector<uint8_t> ct(pt.size());
        uint8_t tag[GCM_TAG_LEN];
        aes_gcm_encrypt(&ctx, pt.data(), pt.size(), nullptr, 0, ct.data(), tag);

        printf("[Tamper Demo]\n");
        printf("  Original file: %s (%zu bytes)\n", in_path, pt.size());
        print_hex("  Auth Tag    ", tag, GCM_TAG_LEN);

        // Flip one bit at byte 100 (or byte 0 for small files)
        size_t flip_pos = (ct.size() > 100) ? 100 : 0;
        ct[flip_pos] ^= 0x01;
        printf("\n  >> Flipping bit in ciphertext at byte %zu ...\n\n", flip_pos);

        std::vector<uint8_t> recovered(ct.size());
        int rc = aes_gcm_decrypt(&ctx, ct.data(), ct.size(), nullptr, 0, tag, recovered.data());

        if (rc != 0) {
            printf("  [RESULT] Authentication FAILED — plaintext withheld.\n");
            printf("  One flipped bit in %zu bytes was detected.\n", pt.size());
            printf("  This demonstrates AES-GCM integrity protection.\n");
        } else {
            printf("  [BUG] Decrypt succeeded on tampered data — this should never happen!\n");
            return 1;
        }

    } else {
        fprintf(stderr, "Unknown mode: %s\n", mode.c_str());
        usage(argv[0]); return 1;
    }

    return 0;
}
