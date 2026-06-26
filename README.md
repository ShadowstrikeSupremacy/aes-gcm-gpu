# 🔐 AES-GCM: A GPU-Accelerated Authenticated Encryption Engine

[![Status](https://img.shields.io/badge/CPU%20Baseline-Verified-success)]()
[![Status](https://img.shields.io/badge/GPU%20Pipeline-4%2F4%20Phases%20Complete-success)]()
[![Tests](https://img.shields.io/badge/NIST%20Vectors-7%2F7%20PASS-success)]()
[![Web UI](https://img.shields.io/badge/Web%20UI-Live-blue)]()
[![Deploy](https://img.shields.io/badge/Deploy-Render%20%7C%20Docker-blueviolet)]()
[![License](https://img.shields.io/badge/license-MIT-blue)]()

A from-scratch, dependency-free implementation of **AES-128/256-GCM** (Galois/Counter Mode authenticated encryption), built around a **security-hardened, NIST-verified CPU baseline** and engineered explicitly as the substrate for a **4-phase CUDA acceleration pipeline**.

This is not a toy AES implementation wrapped in CUDA boilerplate. Every primitive — the CTR keystream generation, the GHASH polynomial evaluation, the tag comparison — was designed with its eventual **massively-parallel GPU execution** in mind: register-resident state, GF(2¹²⁸) tree-reduction, MapReduce-style chunking for arbitrarily large payloads. The result is a single codebase that runs correctly on a CPU today and saturates a GPU's SMs without architectural rewrites.

> 📡 **Why this exists:** Built against the methodology in *"A High-Throughput AES-GCM Implementation on GPUs for Secure, Policy-Based Access to Massive Astronomical Catalogs"* — the target workload is bulk, policy-gated decryption of multi-gigabyte scientific datasets, where AES-NI alone can't keep up with modern storage I/O.

---

## 🌐 Web UI

An interactive browser-based interface for encrypting, decrypting, and demonstrating tamper detection — no command line required.

**Encrypt panel** — type any message, optionally supply your own key/IV (or click Generate for a cryptographically random one), hit Encrypt. The ciphertext, auth tag, key, and IV are all displayed with one-click copy buttons.

**Decrypt & Verify panel** — paste ciphertext + tag + key + IV, or click **⇒ Fill Decrypt Panel** after encrypting to auto-populate everything. The result shows either a green **✅ AUTHENTICATED** badge with the recovered plaintext, or a red **❌ AUTHENTICATION FAILED** badge (plaintext is withheld and the output buffer is zeroed).

**Tamper Simulation** — one button flips a single bit in the ciphertext before decrypting, proving that even a 1-bit change anywhere in the message is reliably detected. The badge shakes on failure.

A status badge in the header shows whether the engine is running in **C++ Engine** mode (local build) or **Python / OpenSSL** mode (cloud deployment) — both produce NIST-identical results.

### Run locally

```bash
# First build the project (see Build & Execution Guide below)
python server.py
# Open http://localhost:8080
```

On Windows, double-click `run_ui.bat` — it starts the server and opens the browser automatically.

---

## 🚀 Deploy Online

### Option 1 — Instant public link with ngrok

The fastest way to share a live demo from your own machine. The C++ engine (including CUDA, if available) runs locally; ngrok creates a secure tunnel to it.

```bash
# 1. Download ngrok from https://ngrok.com/download  (free account required)
# 2. Authenticate once
ngrok config add-authtoken <YOUR_TOKEN>

# 3. Start the server
python server.py

# 4. In a second terminal, open the tunnel
ngrok http 8080
```

ngrok prints a public HTTPS URL like `https://abc123.ngrok-free.app` — share it with anyone. The tunnel stays live as long as both processes are running. No port forwarding or firewall changes needed.

### Option 2 — Permanent cloud hosting with Render

`server.py` automatically falls back to Python's `cryptography` library (OpenSSL-backed, NIST-identical) when the C++ binary isn't present, making it deployable on any Linux cloud with zero native dependencies.

1. Push this repository to GitHub.
2. Go to [render.com](https://render.com) → **New Web Service** → connect your repo.
3. Set the following in the Render dashboard:

   | Setting | Value |
   |---|---|
   | **Runtime** | Python 3 |
   | **Build command** | `pip install -r requirements.txt` |
   | **Start command** | `python server.py` |

4. Click **Deploy**. Render sets the `PORT` environment variable automatically.

The live URL is available immediately after the first deploy completes.

### Option 3 — Docker

```bash
docker build -t aes-gcm-ui .
docker run -p 8080:8080 aes-gcm-ui
# Open http://localhost:8080
```

Push to any container registry (Docker Hub, GHCR, ECR) and deploy to Cloud Run, Fly.io, or any container platform.

---

## 📐 Architectural Roadmap

The engine is built in four strictly-ordered phases. Each phase is independently testable and produces a working artifact before the next begins.

| Phase | Name | What It Does | Status |
|---|---|---|---|
| **1** | 🧱 **CPU Baseline** | Reference-correct AES-128/256-GCM in portable C++. Constant-time tag verification, NIST-compliant counter management, zero external crypto dependencies. The "ground truth" every GPU kernel is checked against. | ✅ **Complete** — 7/7 NIST test vectors |
| **2** | ⚡ **CUDA CTR Kernel** | One CUDA thread per 16-byte AES block. S-Box cooperatively staged into `__shared__` memory, round keys in `__constant__` memory, AES state held in 4 registers (zero spill to DRAM), 128-bit coalesced `uint4` loads/stores. | ✅ **Complete** |
| **3** | 🌳 **Parallel GHASH (Tree-Reduction)** | Re-expresses the sequential GHASH recurrence as an embarrassingly-parallel sum: `GHASH = X₁·Hᴺ ⊕ X₂·Hᴺ⁻¹ ⊕ ... ⊕ Xₙ·H¹`. Each term computed independently in GF(2¹²⁸), then combined via a `__syncthreads()`-gated butterfly tree reduction in shared memory. | ✅ **Complete** |
| **4** | 🗺️ **MapReduce Driver** | Chunks payloads exceeding the in-memory power-table budget (64 MB / 4M-block chunks) so GHASH scales to the full 32-bit-counter payload ceiling (~68 GB) without exhausting device memory. Orchestrates the full encrypt/decrypt pipeline: CTR → GHASH → tag. | ✅ **Complete** |

### Measured Throughput (RTX 3050 Laptop GPU — sm_86, 16 SMs)

| Payload | Compute-Only | End-to-End (incl. PCIe) | CPU Baseline | Speedup |
|---|---|---|---|---|
| 1 MB | 0.07 GB/s | 0.07 GB/s | 0.04 GB/s | 1.7× |
| **16 MB** | **0.76 GB/s** | 0.66 GB/s | 0.04 GB/s | **18.7×** |
| 256 MB | 0.34 GB/s | 0.30 GB/s | 0.04 GB/s | 9.0× |

> Small payloads are launch-overhead-bound; 16 MB is the kernel's saturation sweet spot. The 256 MB dip reflects MapReduce chunk-boundary power-table rebuilds — the deliberate memory/throughput tradeoff that keeps device memory usage bounded regardless of payload size.

---

## 🚀 Build & Execution Guide

### Prerequisites

- **CMake ≥ 3.20**
- A C++17 compiler (MSVC, GCC, or Clang)
- *(Optional, for GPU targets)* **CUDA Toolkit 12+** with `nvcc` and a CUDA-capable GPU (sm_75+)
  - On Windows, `nvcc` requires the MSVC host compiler — run from a **Developer Command Prompt** or call `vcvars64.bat` first.

CMake **dynamically detects CUDA** via `check_language(CUDA)`. If no toolkit is found, only the CPU targets are built — no manual flags required.

### Clone & Build

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/aes-gcm-gpu.git
cd aes-gcm-gpu

# 2. Create an out-of-source build directory
mkdir build && cd build

# 3. Configure — CUDA is auto-detected; falls back to CPU-only if absent
cmake .. -DCMAKE_BUILD_TYPE=Release

# 4. Compile in Release mode (enables -O3 / /O2 and --use_fast_math for CUDA)
cmake --build . --config Release
```

*Windows + CUDA note:* if `nvcc` can't find `cl.exe`, configure from inside a Visual Studio Developer Prompt, or point CMake at the compiler explicitly:
```bash
cmake .. -G "Visual Studio 17 2022" -A x64 \
  -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.x/bin/nvcc.exe"
```

Build artifacts land in `build/Release/` (MSVC) or `build/` (Makefiles/Ninja):
`run_tests`, `aes_gcm_demo`, and — if CUDA was found — `bench`.

### 🖥️ Execution Options

**1. Run the Validation Test Suite**
Verifies the CPU engine against official NIST SP 800-38D test vectors (AES-128 and AES-256, multiple payload/AAD sizes).
```bash
./run_tests
```
> Expected output: `7/7 NIST vectors PASSED.`

**2. Encrypt a File**
```bash
./aes_gcm_demo encrypt \
  --key <hex32-or-hex64> \
  --iv  <hex24> \
  --in  plaintext.bin \
  --out ciphertext.bin
```
*   `--key` — 32 hex chars (128-bit) or 64 hex chars (256-bit)
*   `--iv` — 24 hex chars (96-bit, NIST-mandated IV length)

Prints the authentication tag (keep it — you'll need it to decrypt) and measured throughput.

**3. Decrypt & Authenticate**
```bash
./aes_gcm_demo decrypt \
  --key <hex32-or-hex64> \
  --iv  <hex24> \
  --tag <hex32> \
  --in  ciphertext.bin \
  --out recovered.bin
```
If the tag matches, `recovered.bin` is written and `Tag verified: OK` is printed. If it doesn't match, no plaintext is written at all.

**4. Tamper Simulation 🧪**
Demonstrates AES-GCM's integrity guarantee live: encrypts a file, flips a single ciphertext bit, then attempts to decrypt it.
```bash
./aes_gcm_demo tamper --key <hex32-or-hex64> --iv <hex24> --in plaintext.bin
```

**5. GPU Benchmark (CUDA build only)**
```bash
./bench
```
Runs a correctness gate against NIST test vector TC2 before reporting compute-only and end-to-end throughput across 1 MB / 16 MB / 256 MB payloads, with a CPU-baseline speedup comparison.

---

## 📁 Repository Structure

```text
aes-gcm-gpu/
├── include/                  # Public headers
│   ├── aes_gcm.h              # Core API
│   ├── aes_gcm_gpu.h          # GPU context + kernel launcher declarations
│   ├── gf128.h                # GF(2¹²⁸) field arithmetic
│   └── utils.h                # Hex conversion, ct_memcmp, Timer
│
├── cpu/                      # Phase 1 — verified reference implementation
│   ├── aes_core.cpp           # Key expansion, single-block AES encryption
│   ├── aes_ctr.cpp            # AES-CTR keystream generation
│   ├── ghash.cpp              # Sequential GHASH
│   └── aes_gcm_cpu.cpp        # Top-level encrypt/decrypt orchestration
│
├── gpu/                      # Phases 2-4 — CUDA acceleration pipeline
│   ├── aes_ctr.cu              # Phase 2: templated AES-CTR kernel
│   ├── ghash_kernel.cu         # Phase 3: parallel GHASH + butterfly tree-reduction
│   ├── mapreduce.cu            # Phase 4: chunked GHASH driver
│   └── aes_gcm_gpu.cu          # Thin host-side wrapper
│
├── tests/
│   ├── nist_vectors.cpp        # NIST SP 800-38D conformance suite
│   ├── demo.cpp                # CLI: encrypt / decrypt / tamper
│   └── bench_gpu.cu            # GPU correctness gate + throughput benchmark
│
├── server.py                 # Web UI server — C++ binary or Python/OpenSSL fallback
├── run_ui.bat                # Windows one-click launcher
├── requirements.txt          # Python deps for cloud deployment (cryptography)
├── Dockerfile                # Container image for cloud platforms
└── CMakeLists.txt            # Dynamic CUDA detection
```

---

## 🛡️ Key Security Implementations

This implementation treats AES-GCM's well-documented footguns as first-class engineering constraints:

*   **🔒 Constant-Time Tag Comparison (`ct_memcmp`)**: Tag verification never short-circuits on the first mismatched byte, preventing timing-oracle attacks.
*   **🔢 Strict NIST Counter Isolation**: The keystream counter for data blocks always starts at 2. Counter value 1 ($J_0$) is permanently reserved for masking the authentication tag, preventing tag-mask recovery attacks.
*   **🧹 Immediate Zeroing on Authentication Failure**: Decryption verifies the tag *before* any plaintext is exposed. On mismatch, the output buffer is explicitly zeroed and a hard failure is returned.
*   **🌐 Endianness-Correct GF(2¹²⁸) Arithmetic**: The GPU port carefully tracks the distinction between the GPU's native little-endian layout and AES/GHASH's big-endian wire format using raw PTX instructions (`__byte_perm`).

## 📄 License

MIT — see LICENSE for details.
