# AES-GCM GPU Engine

AES-128/256-GCM implemented from scratch in C++ and CUDA — no OpenSSL, no libsodium.
The CPU baseline passes all 7 NIST SP 800-38D test vectors. The GPU pipeline pushes
**18.7× speedup** over the CPU at 16 MB payloads on an RTX 3050 Laptop.

Built following the architecture in
*A High-Throughput AES-GCM Implementation on GPUs for Secure, Policy-Based Access to
Massive Astronomical Catalogs* — the target workload is bulk encryption of multi-gigabyte
scientific catalogs where AES-NI alone bottlenecks on I/O throughput.

---

## How it's structured

The implementation is split into four phases, each one independently testable:

| Phase | What it does |
|---|---|
| **CPU baseline** | Reference AES-GCM in plain C++. Correct GHASH via sequential Horner's method, constant-time tag comparison, strict J0/counter-2 isolation. Verified against NIST. |
| **CUDA CTR kernel** | One thread per 16-byte block. S-Box in `__shared__`, round keys in `__constant__`, AES state in registers (no DRAM spill), 128-bit coalesced loads via `uint4`. Templated on round count so the loop unrolls at compile time. |
| **Parallel GHASH** | Re-expresses the Horner recurrence as `X₁·Hᴺ ⊕ X₂·Hᴺ⁻¹ ⊕ ... ⊕ Xₙ·H¹` — all terms independent. Each thread multiplies its block by a precomputed H-power, then butterfly tree-reduction in shared memory collapses everything to one GF(2¹²⁸) element. |
| **MapReduce driver** | Chunks payloads that would require a power table too large for device memory (CHUNK_BLOCKS = 4M blocks / 64 MB). Partial GHASH sums are XOR'd on the host. Handles the full ~68 GB counter limit. |

## Benchmarks

Measured on RTX 3050 Laptop GPU (sm_86, 16 SMs, 4.3 GB GDDR6), AES-128:

| Payload | Compute GB/s | End-to-end GB/s | CPU GB/s | Speedup |
|---|---|---|---|---|
| 1 MB | 0.07 | 0.07 | 0.04 | 1.7× |
| **16 MB** | **0.76** | 0.66 | 0.04 | **18.7×** |
| 256 MB | 0.34 | 0.30 | 0.04 | 9.0× |

The 1 MB case is kernel-launch-overhead-bound. The 256 MB drop comes from rebuilding the
H-power table four times (once per chunk) — an intentional tradeoff to keep device memory
use bounded.

---

## Build

**Requirements:** CMake ≥ 3.20, a C++17 compiler. CUDA Toolkit 12+ is optional — if
`nvcc` isn't found, only the CPU targets are built.

```bash
git clone https://github.com/ShadowstrikeSupremacy/aes-gcm-gpu.git
cd aes-gcm-gpu
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

On Windows with CUDA, run from a Visual Studio Developer Command Prompt or call
`vcvars64.bat` first so `nvcc` can find `cl.exe`.

---

## Usage

**Run the NIST test suite**
```bash
./build/Release/run_tests
```

**Encrypt a file**
```bash
./build/Release/aes_gcm_demo encrypt \
  --key <32 or 64 hex chars>  \
  --iv  <24 hex chars>         \
  --in  plain.txt              \
  --out cipher.bin
```
Prints the auth tag to stdout — you need it to decrypt.

**Decrypt and verify**
```bash
./build/Release/aes_gcm_demo decrypt \
  --key <hex>  --iv <hex>  --tag <32 hex chars> \
  --in  cipher.bin  --out recovered.txt
```
Returns exit code 1 and writes nothing on tag mismatch.

**Tamper demo** — encrypts a file, flips one ciphertext bit, tries to decrypt:
```bash
./build/Release/aes_gcm_demo tamper --key <hex> --iv <hex> --in plain.txt
```

**GPU benchmark**
```bash
./build/Release/bench
```
Runs NIST TC2 correctness check first, then reports compute-only and end-to-end
throughput for 1 / 16 / 256 MB payloads against the CPU baseline.

---

## Web UI

A browser interface for encrypting, decrypting, and testing tamper detection — useful
for demos without touching the CLI.

```bash
pip install -r requirements.txt
python server.py
# open http://localhost:8080
```

On Windows: double-click `run_ui.bat`.

The server uses the compiled `aes_gcm_demo` binary when available. If you deploy it
somewhere without the binary (e.g. a Python-only cloud host), it falls back to Python's
`cryptography` library — same algorithm, NIST-identical output.

---

## Security notes

A few things that are easy to get wrong in AES-GCM:

**Counter isolation** — NIST mandates that plaintext encryption starts at counter 2.
Counter 1 (J0) is reserved exclusively for masking the authentication tag
(`T = GHASH(...) ⊕ E(K, J0)`). Using J0 as a data counter would let an attacker recover
the tag mask from the ciphertext.

**Constant-time tag comparison** — `ct_memcmp` in `include/utils.h` always runs all 16
iterations regardless of where bytes differ. A standard `memcmp` exits early on the first
mismatch, leaking timing information that can be used to forge tags one byte at a time.

**Plaintext isolation** — `aes_gcm_decrypt` verifies the tag *before* running AES-CTR.
On mismatch, the output buffer is zeroed and the function returns -1. Decryption never
runs on unauthenticated ciphertext.

---

## Repo layout

```
├── include/
│   ├── aes_gcm.h          core API and AesGcmCtx struct
│   ├── aes_gcm_gpu.h      GPU context and kernel launcher declarations
│   ├── gf128.h            GF(2¹²⁸) arithmetic (host + device)
│   └── utils.h            ct_memcmp, hex helpers, Timer
├── cpu/
│   ├── aes_core.cpp       key expansion, single-block encrypt
│   ├── aes_ctr.cpp        CTR keystream
│   ├── ghash.cpp          sequential GHASH
│   └── aes_gcm_cpu.cpp    encrypt/decrypt orchestration
├── gpu/
│   ├── aes_ctr.cu         Phase 2: AES-CTR kernel
│   ├── ghash_kernel.cu    Phase 3: parallel GHASH + tree-reduction
│   ├── mapreduce.cu       Phase 4: chunked driver
│   └── aes_gcm_gpu.cu     host-side wrapper
├── tests/
│   ├── nist_vectors.cpp   NIST SP 800-38D test suite
│   ├── demo.cpp           CLI tool (encrypt / decrypt / tamper)
│   └── bench_gpu.cu       GPU benchmark
├── server.py              web UI server
└── CMakeLists.txt
```

---

## License

MIT
