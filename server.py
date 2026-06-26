#!/usr/bin/env python3
"""
AES-GCM Web UI server.
Wraps the aes_gcm_demo binary with a browser-based interface.

Usage:
    python server.py
    Then open http://localhost:8080
"""

import http.server
import json
import os
import re
import secrets
import subprocess
import sys
import tempfile
from pathlib import Path

PORT = int(os.environ.get("PORT", 8080))
SCRIPT_DIR = Path(__file__).parent.resolve()

# Prefer the MSVC Release build; fall back to a Makefile/Ninja build.
_candidates = [
    SCRIPT_DIR / "build" / "Release" / "aes_gcm_demo.exe",
    SCRIPT_DIR / "build" / "aes_gcm_demo.exe",
    SCRIPT_DIR / "build" / "aes_gcm_demo",
]
BINARY = next((p for p in _candidates if p.exists()), _candidates[0])

# ---------------------------------------------------------------------------
# Crypto mode: prefer the compiled C++ binary; fall back to the Python
# 'cryptography' library (OpenSSL-backed) for cloud deployments where
# the binary is not present.  Results are NIST-identical in both modes.
# ---------------------------------------------------------------------------
USE_BINARY = BINARY.exists()

_PY_CRYPTO_OK = False
if not USE_BINARY:
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM as _AESGCM
        from cryptography.exceptions import InvalidTag as _InvalidTag
        _PY_CRYPTO_OK = True
    except ImportError:
        pass

ENGINE_LABEL = (
    f"C++ Engine ({BINARY.name})" if USE_BINARY
    else ("Python / OpenSSL (cloud)" if _PY_CRYPTO_OK else "NO CRYPTO BACKEND")
)

# ---------------------------------------------------------------------------
# HTML  (raw string so JS regex escapes like \s, \n are left untouched)
# ---------------------------------------------------------------------------
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AES-GCM Engine</title>
<style>
:root {
  --bg:          #070c14;
  --surface:     #0d1521;
  --surface-2:   #131f30;
  --border:      #1e2d40;
  --accent:      #00d97e;
  --accent-dim:  #003d22;
  --blue:        #3b82f6;
  --blue-dim:    #1e3a5f;
  --red:         #f87171;
  --red-dim:     #450a0a;
  --text:        #e2e8f0;
  --muted:       #64748b;
  --mono: 'Cascadia Code', 'Fira Code', 'JetBrains Mono', Consolas, monospace;
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html { scroll-behavior: smooth; }
body {
  background: var(--bg);
  color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
  font-size: 14px;
  line-height: 1.6;
  min-height: 100vh;
}

/* ── Scrollbars ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 99px; }

/* ── Header ── */
header {
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  padding: 0.875rem 1.75rem;
  display: flex;
  align-items: center;
  gap: 1.25rem;
  position: sticky;
  top: 0;
  z-index: 10;
  backdrop-filter: blur(8px);
}
.logo {
  font-size: 1.05rem;
  font-weight: 700;
  letter-spacing: -0.01em;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}
.logo-icon {
  width: 28px; height: 28px;
  background: linear-gradient(135deg, var(--accent-dim), #005c35);
  border: 1px solid var(--accent);
  border-radius: 7px;
  display: flex; align-items: center; justify-content: center;
  font-size: 0.85rem;
}
.badges { display: flex; gap: 0.4rem; flex-wrap: wrap; }
.badge {
  font-size: 0.6rem; font-weight: 700;
  padding: 0.18rem 0.55rem;
  border-radius: 99px;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}
.badge-g { background: var(--accent-dim); color: var(--accent);  border: 1px solid #00804a; }
.badge-b { background: var(--blue-dim);   color: #93c5fd;         border: 1px solid #2563eb; }

/* ── Main grid ── */
main {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1.25rem;
  padding: 1.25rem 1.75rem;
  max-width: 1300px;
  margin: 0 auto;
}
@media (max-width: 860px) {
  main { grid-template-columns: 1fr; }
}

/* ── Panel ── */
.panel {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}
.panel-header {
  background: var(--surface-2);
  border-bottom: 1px solid var(--border);
  padding: 0.75rem 1.1rem;
  display: flex;
  align-items: center;
  gap: 0.75rem;
}
.ph-icon {
  width: 30px; height: 30px;
  border-radius: 8px;
  display: flex; align-items: center; justify-content: center;
  font-size: 0.9rem;
  flex-shrink: 0;
}
.ph-icon.enc { background: var(--accent-dim); border: 1px solid #00804a; }
.ph-icon.dec { background: var(--blue-dim);   border: 1px solid #2563eb; }
.ph-title { font-weight: 700; font-size: 0.85rem; }
.ph-sub   { font-size: 0.7rem; color: var(--muted); margin-top: 0.05rem; }
.panel-body { padding: 1.1rem; flex: 1; }

/* ── Fields ── */
.field { margin-bottom: 0.9rem; }
.field-label {
  font-size: 0.68rem; font-weight: 700;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.07em;
  margin-bottom: 0.35rem;
  display: flex; align-items: center; justify-content: space-between;
}
.field-hint { font-weight: 400; text-transform: none; letter-spacing: 0; font-size: 0.65rem; }
textarea, input[type="text"] {
  width: 100%;
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 7px;
  color: var(--text);
  padding: 0.55rem 0.75rem;
  font-size: 0.82rem;
  transition: border-color 0.15s;
  outline: none;
  appearance: none;
}
textarea:focus, input[type="text"]:focus { border-color: var(--accent); }
textarea { resize: vertical; font-family: inherit; }
.mono-input { font-family: var(--mono); font-size: 0.72rem; }

/* ── Key row ── */
.key-row { display: flex; gap: 0.45rem; }
.key-row input { flex: 1; min-width: 0; }
.gen-btn {
  background: var(--surface-2);
  border: 1px solid var(--border);
  border-radius: 6px;
  color: var(--muted);
  cursor: pointer;
  font-size: 0.62rem;
  font-weight: 700;
  padding: 0 0.6rem;
  white-space: nowrap;
  transition: all 0.15s;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  height: 33px;
}
.gen-btn:hover { border-color: var(--accent); color: var(--accent); }

/* ── Buttons ── */
.btn {
  width: 100%;
  padding: 0.65rem 1rem;
  border-radius: 7px;
  font-size: 0.82rem;
  font-weight: 600;
  cursor: pointer;
  border: none;
  transition: all 0.15s;
  display: flex; align-items: center; justify-content: center; gap: 0.45rem;
  letter-spacing: 0.01em;
}
.btn:disabled { cursor: not-allowed; opacity: 0.6; }
.btn-enc { background: var(--accent); color: #fff; }
.btn-enc:hover:not(:disabled) { background: #00b86a; }
.btn-dec { background: var(--blue); color: #fff; }
.btn-dec:hover:not(:disabled) { background: #2563eb; }
.btn-fill {
  background: transparent; color: var(--accent);
  border: 1px solid var(--accent);
  font-size: 0.78rem;
  margin-top: 0.75rem;
}
.btn-fill:hover:not(:disabled) { background: var(--accent-dim); }
.btn-tamper {
  background: transparent; color: var(--red);
  border: 1px solid var(--red);
  font-size: 0.75rem;
  margin-top: 0.45rem;
}
.btn-tamper:hover:not(:disabled) { background: var(--red-dim); }

/* ── Output ── */
.output-area { margin-top: 1rem; border-top: 1px solid var(--border); padding-top: 1rem; }
.out-field {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 7px;
  padding: 0.55rem 0.75rem;
  margin-bottom: 0.65rem;
}
.out-label {
  font-size: 0.62rem; font-weight: 700;
  color: var(--muted);
  text-transform: uppercase; letter-spacing: 0.07em;
  margin-bottom: 0.25rem;
  display: flex; justify-content: space-between; align-items: center;
}
.out-value {
  font-family: var(--mono);
  font-size: 0.69rem;
  color: #6ee7b7;
  word-break: break-all;
  line-height: 1.65;
  max-height: 90px;
  overflow-y: auto;
}
.out-value.plain {
  font-family: inherit; font-size: 0.82rem;
  color: var(--text); max-height: 130px;
}
.copy-btn {
  background: none; border: none;
  color: var(--muted); cursor: pointer;
  font-size: 0.62rem; padding: 0.1rem 0.35rem;
  border-radius: 4px; transition: all 0.12s;
}
.copy-btn:hover { color: var(--text); background: var(--border); }
.copy-btn.ok { color: var(--accent); }

/* ── Integrity badge ── */
.badge-wrap { margin-bottom: 0.8rem; }
.int-badge {
  border-radius: 9px;
  padding: 0.85rem 1rem;
  display: flex; align-items: center; gap: 0.85rem;
}
.int-badge.pass {
  background: linear-gradient(135deg, #022c1c, var(--accent-dim));
  border: 1px solid #009955;
}
.int-badge.fail {
  background: linear-gradient(135deg, #2d0a0a, var(--red-dim));
  border: 1px solid #b91c1c;
  animation: shake 0.35s ease-in-out;
}
@keyframes shake {
  0%,100% { transform: translateX(0); }
  20%      { transform: translateX(-5px); }
  40%      { transform: translateX(5px); }
  60%      { transform: translateX(-3px); }
  80%      { transform: translateX(3px); }
}
.int-icon { font-size: 1.6rem; flex-shrink: 0; }
.int-title { font-weight: 700; font-size: 0.85rem; }
.int-badge.pass .int-title { color: var(--accent); }
.int-badge.fail .int-title { color: var(--red); }
.int-desc { font-size: 0.7rem; color: var(--muted); margin-top: 0.1rem; }

/* ── Error ── */
.err-box {
  background: var(--red-dim);
  border: 1px solid #7f1d1d;
  border-radius: 7px;
  padding: 0.65rem 0.85rem;
  font-size: 0.78rem; color: #fca5a5;
  margin-top: 0.85rem; line-height: 1.5;
}

/* ── Spinner ── */
.spin {
  display: inline-block;
  width: 13px; height: 13px;
  border: 2px solid transparent;
  border-top-color: currentColor;
  border-radius: 50%;
  animation: rot 0.55s linear infinite;
}
@keyframes rot { to { transform: rotate(360deg); } }

.hidden { display: none !important; }

/* ── Footer ── */
footer {
  text-align: center;
  padding: 1.1rem;
  color: var(--muted);
  font-size: 0.68rem;
  border-top: 1px solid var(--border);
  margin-top: 0.5rem;
  letter-spacing: 0.03em;
}
footer span { color: var(--accent); }
</style>
</head>
<body>

<header>
  <div class="logo">
    <div class="logo-icon">🔐</div>
    AES-GCM Engine
  </div>
  <div class="badges">
    <span class="badge badge-g">NIST SP 800-38D</span>
    <span class="badge badge-b">AES-128 / 256</span>
    <span class="badge badge-g">CUDA Accelerated</span>
    <span class="badge badge-g">Constant-Time Tag Verify</span>
    <span class="badge" id="eng-badge" style="opacity:0">...</span>
  </div>
</header>

<main>

  <!-- ── Encrypt ─────────────────────────────────────────────────────────── -->
  <div class="panel">
    <div class="panel-header">
      <div class="ph-icon enc">🔒</div>
      <div>
        <div class="ph-title">Encrypt</div>
        <div class="ph-sub">AES-CTR keystream · GHASH authentication tag</div>
      </div>
    </div>
    <div class="panel-body">

      <div class="field">
        <div class="field-label">Plaintext</div>
        <textarea id="enc-pt" rows="4" placeholder="Type the message to encrypt…"></textarea>
      </div>

      <div class="field">
        <div class="field-label">
          AES Key (hex)
          <span class="field-hint">blank → auto-generate 128-bit</span>
        </div>
        <div class="key-row">
          <input type="text" id="enc-key" class="mono-input"
                 placeholder="32 hex chars (AES-128)  or  64 hex chars (AES-256)">
          <button class="gen-btn" onclick="generate('enc-key','key128')">128-bit</button>
          <button class="gen-btn" onclick="generate('enc-key','key256')">256-bit</button>
        </div>
      </div>

      <div class="field">
        <div class="field-label">
          IV / Nonce (hex)
          <span class="field-hint">blank → auto-generate 96-bit</span>
        </div>
        <div class="key-row">
          <input type="text" id="enc-iv" class="mono-input"
                 placeholder="24 hex chars (96-bit)">
          <button class="gen-btn" onclick="generate('enc-iv','iv')">Generate</button>
        </div>
      </div>

      <button class="btn btn-enc" id="enc-btn" onclick="doEncrypt()">
        <span id="enc-btn-txt">Encrypt</span>
      </button>

      <div class="output-area hidden" id="enc-out">
        <div class="out-field">
          <div class="out-label">
            Ciphertext (hex)
            <button class="copy-btn" onclick="copyEl('enc-r-ct',this)">copy</button>
          </div>
          <div class="out-value" id="enc-r-ct"></div>
        </div>
        <div class="out-field">
          <div class="out-label">
            Authentication Tag (hex)
            <button class="copy-btn" onclick="copyEl('enc-r-tag',this)">copy</button>
          </div>
          <div class="out-value" id="enc-r-tag"></div>
        </div>
        <div class="out-field">
          <div class="out-label">
            Key Used (hex)
            <button class="copy-btn" onclick="copyEl('enc-r-key',this)">copy</button>
          </div>
          <div class="out-value" id="enc-r-key"></div>
        </div>
        <div class="out-field">
          <div class="out-label">
            IV Used (hex)
            <button class="copy-btn" onclick="copyEl('enc-r-iv',this)">copy</button>
          </div>
          <div class="out-value" id="enc-r-iv"></div>
        </div>
        <button class="btn btn-fill" onclick="fillDecrypt()">
          ⇒ Fill Decrypt Panel with these values
        </button>
      </div>

      <div class="err-box hidden" id="enc-err"></div>
    </div>
  </div>

  <!-- ── Decrypt ─────────────────────────────────────────────────────────── -->
  <div class="panel" id="dec-panel">
    <div class="panel-header">
      <div class="ph-icon dec">🔓</div>
      <div>
        <div class="ph-title">Decrypt &amp; Verify</div>
        <div class="ph-sub">ct_memcmp tag verification · plaintext withheld on failure</div>
      </div>
    </div>
    <div class="panel-body">

      <div class="field">
        <div class="field-label">Ciphertext (hex)</div>
        <textarea id="dec-ct" class="mono-input" rows="3"
                  placeholder="Paste ciphertext hex…"></textarea>
      </div>

      <div class="field">
        <div class="field-label">Authentication Tag (hex)</div>
        <input type="text" id="dec-tag" class="mono-input"
               placeholder="32 hex chars">
      </div>

      <div class="field">
        <div class="field-label">AES Key (hex)</div>
        <input type="text" id="dec-key" class="mono-input"
               placeholder="32 or 64 hex chars">
      </div>

      <div class="field">
        <div class="field-label">IV / Nonce (hex)</div>
        <input type="text" id="dec-iv" class="mono-input"
               placeholder="24 hex chars">
      </div>

      <button class="btn btn-dec" id="dec-btn" onclick="doDecrypt(false)">
        <span id="dec-btn-txt">Decrypt &amp; Verify</span>
      </button>
      <button class="btn btn-tamper" id="tamper-btn" onclick="doDecrypt(true)">
        ⚡ Simulate Tamper — flip 1 ciphertext bit, then attempt decrypt
      </button>

      <div class="output-area hidden" id="dec-out">

        <div class="badge-wrap hidden" id="badge-pass">
          <div class="int-badge pass">
            <div class="int-icon">✅</div>
            <div>
              <div class="int-title">AUTHENTICATED</div>
              <div class="int-desc">Tag verified — message is cryptographically intact</div>
            </div>
          </div>
        </div>

        <div class="badge-wrap hidden" id="badge-fail">
          <div class="int-badge fail">
            <div class="int-icon">❌</div>
            <div>
              <div class="int-title">AUTHENTICATION FAILED</div>
              <div class="int-desc">
                Tag mismatch — message was tampered, corrupted, or keys are wrong.
                Plaintext withheld and output buffer zeroed.
              </div>
            </div>
          </div>
        </div>

        <div class="out-field hidden" id="dec-pt-field">
          <div class="out-label">
            Recovered Plaintext
            <button class="copy-btn" onclick="copyEl('dec-r-pt',this)">copy</button>
          </div>
          <div class="out-value plain" id="dec-r-pt"></div>
        </div>

      </div>

      <div class="err-box hidden" id="dec-err"></div>
    </div>
  </div>

</main>

<footer>
  AES-GCM Engine &nbsp;·&nbsp;
  <span>NIST SP 800-38D</span> &nbsp;·&nbsp;
  Constant-time <code>ct_memcmp</code> &nbsp;·&nbsp;
  Counter starts at 2 (J0 reserved for tag mask) &nbsp;·&nbsp;
  Zero-on-failure plaintext isolation
</footer>

<script>
// ── Helpers ─────────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);
const show = id => $(id).classList.remove('hidden');
const hide = id => $(id).classList.add('hidden');
const setText = (id, t) => $(id).textContent = t;

async function generate(fieldId, type) {
  const resp = await fetch('/generate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ type })
  });
  const d = await resp.json();
  if (d.hex) $(fieldId).value = d.hex;
}

function copyEl(srcId, btn) {
  const txt = $(srcId).textContent.trim();
  navigator.clipboard.writeText(txt).then(() => {
    const orig = btn.textContent;
    btn.textContent = 'copied!';
    btn.classList.add('ok');
    setTimeout(() => { btn.textContent = orig; btn.classList.remove('ok'); }, 1400);
  });
}

function setBtn(txtId, html, disabled) {
  $(txtId).innerHTML = html;
  $(txtId).closest('button').disabled = !!disabled;
}

// ── State ────────────────────────────────────────────────────────────────────
let lastResult = null;

// ── Encrypt ──────────────────────────────────────────────────────────────────
async function doEncrypt() {
  hide('enc-out'); hide('enc-err');

  const pt = $('enc-pt').value;
  if (!pt.trim()) { show('enc-err'); setText('enc-err', 'Plaintext is empty.'); return; }

  setBtn('enc-btn-txt', '<span class="spin"></span> Encrypting…', true);

  try {
    const resp = await fetch('/encrypt', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        plaintext: pt,
        key: $('enc-key').value.trim(),
        iv:  $('enc-iv').value.trim()
      })
    });
    const d = await resp.json();

    if (d.error) {
      show('enc-err'); setText('enc-err', d.error);
    } else {
      lastResult = d;
      setText('enc-r-ct',  d.ciphertext);
      setText('enc-r-tag', d.tag);
      setText('enc-r-key', d.key);
      setText('enc-r-iv',  d.iv);
      $('enc-key').value = d.key;
      $('enc-iv').value  = d.iv;
      show('enc-out');
    }
  } catch(e) {
    show('enc-err'); setText('enc-err', 'Server error: ' + e.message);
  } finally {
    setBtn('enc-btn-txt', 'Encrypt', false);
  }
}

// ── Fill Decrypt ─────────────────────────────────────────────────────────────
function fillDecrypt() {
  if (!lastResult) return;
  $('dec-ct').value  = lastResult.ciphertext;
  $('dec-tag').value = lastResult.tag;
  $('dec-key').value = lastResult.key;
  $('dec-iv').value  = lastResult.iv;
  $('dec-panel').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ── Decrypt ──────────────────────────────────────────────────────────────────
async function doDecrypt(tamper) {
  hide('dec-out'); hide('dec-err');
  hide('badge-pass'); hide('badge-fail'); hide('dec-pt-field');

  const ct  = $('dec-ct').value.trim().replace(/\s/g, '');
  const tag = $('dec-tag').value.trim();
  const key = $('dec-key').value.trim();
  const iv  = $('dec-iv').value.trim();

  if (!ct || !tag || !key || !iv) {
    show('dec-err');
    setText('dec-err', 'All four fields are required: ciphertext, tag, key, and IV.');
    return;
  }

  const label = tamper
    ? '<span class="spin"></span> Tampering &amp; verifying…'
    : '<span class="spin"></span> Verifying &amp; decrypting…';
  setBtn('dec-btn-txt', label, true);
  $('tamper-btn').disabled = true;

  try {
    const resp = await fetch('/decrypt', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ciphertext: ct, tag, key, iv, tamper })
    });
    const d = await resp.json();

    show('dec-out');

    if (d.integrity === true) {
      show('badge-pass');
      show('dec-pt-field');
      setText('dec-r-pt', d.plaintext);
    } else if (d.integrity === false) {
      show('badge-fail');
    } else if (d.error) {
      hide('dec-out');
      show('dec-err');
      setText('dec-err', d.error);
    }
  } catch(e) {
    show('dec-err'); setText('dec-err', 'Server error: ' + e.message);
  } finally {
    setBtn('dec-btn-txt', 'Decrypt &amp; Verify', false);
    $('tamper-btn').disabled = false;
  }
}

// Ctrl+Enter to submit
$('enc-pt').addEventListener('keydown', e => {
  if (e.ctrlKey && e.key === 'Enter') doEncrypt();
});

// ── Engine mode badge ─────────────────────────────────────────────────────────
(async () => {
  try {
    const r = await fetch('/status');
    const d = await r.json();
    const el = $('eng-badge');
    if (d.mode === 'binary') {
      el.textContent = 'C++ Engine';
      el.style.cssText = 'background:#003d22;color:#00d97e;border:1px solid #00804a;font-size:.6rem;font-weight:700;padding:.18rem .55rem;border-radius:99px;letter-spacing:.06em;text-transform:uppercase;opacity:1';
    } else if (d.mode === 'python') {
      el.textContent = 'Python / OpenSSL';
      el.style.cssText = 'background:#1e3a5f;color:#93c5fd;border:1px solid #2563eb;font-size:.6rem;font-weight:700;padding:.18rem .55rem;border-radius:99px;letter-spacing:.06em;text-transform:uppercase;opacity:1';
    }
  } catch(_) {}
})();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

def _rm(path):
    try:
        if path and os.path.exists(path):
            os.unlink(path)
    except OSError:
        pass


class AesGcmHandler(http.server.BaseHTTPRequestHandler):

    # ------------------------------------------------------------------
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/status":
            payload = json.dumps(self._status()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        else:
            self.send_response(404)
            self.end_headers()

    # ------------------------------------------------------------------
    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length)
            data = json.loads(raw.decode("utf-8"))

            if   self.path == "/generate": result = self._generate(data)
            elif self.path == "/encrypt":  result = self._encrypt(data)
            elif self.path == "/decrypt":  result = self._decrypt(data)
            else:                          result = {"error": "Unknown endpoint"}

        except Exception as exc:
            result = {"error": str(exc)}

        payload = json.dumps(result).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    # ------------------------------------------------------------------
    def _status(self):
        if USE_BINARY:
            return {"mode": "binary", "engine": BINARY.name}
        elif _PY_CRYPTO_OK:
            return {"mode": "python", "engine": "cryptography (OpenSSL)"}
        else:
            return {"mode": "none", "engine": "no backend available"}

    # ------------------------------------------------------------------
    def _generate(self, data):
        t = data.get("type", "key128")
        if t == "iv":
            return {"hex": secrets.token_hex(12)}   # 96-bit
        if t == "key256":
            return {"hex": secrets.token_hex(32)}   # 256-bit
        return {"hex": secrets.token_hex(16)}        # 128-bit default

    # ------------------------------------------------------------------
    def _encrypt(self, data):
        if USE_BINARY:
            return self._encrypt_binary(data)
        if _PY_CRYPTO_OK:
            return self._encrypt_python(data)
        return {"error": "No crypto backend available. Install 'cryptography' or build the C++ binary."}

    def _encrypt_binary(self, data):
        plaintext = data.get("plaintext", "")
        if not plaintext:
            return {"error": "Plaintext cannot be empty."}

        key_hex = data.get("key", "").strip() or secrets.token_hex(16)
        iv_hex  = data.get("iv",  "").strip() or secrets.token_hex(12)

        if len(key_hex) not in (32, 64):
            return {"error": f"Key must be 32 or 64 hex chars (got {len(key_hex)})."}
        if len(iv_hex) != 24:
            return {"error": f"IV must be 24 hex chars (got {len(iv_hex)})."}

        in_path = out_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
                f.write(plaintext.encode("utf-8"))
                in_path = f.name
            out_path = in_path + ".enc"

            proc = subprocess.run(
                [str(BINARY), "encrypt",
                 "--key", key_hex, "--iv", iv_hex,
                 "--in", in_path, "--out", out_path],
                capture_output=True, text=True, timeout=30
            )

            if proc.returncode != 0:
                msg = (proc.stderr or proc.stdout or "Encryption failed.").strip()
                return {"error": msg}

            tag_hex = ""
            for line in proc.stdout.splitlines():
                if "Auth Tag" in line:
                    m = re.search(r":\s*([0-9a-fA-F]+)", line)
                    if m:
                        tag_hex = m.group(1).strip()
                        break

            if not os.path.exists(out_path):
                return {"error": "Output file was not created — check binary."}

            with open(out_path, "rb") as f:
                ct_bytes = f.read()

            return {
                "ciphertext": ct_bytes.hex(),
                "tag":        tag_hex,
                "key":        key_hex,
                "iv":         iv_hex,
            }
        finally:
            _rm(in_path)
            _rm(out_path)

    # ------------------------------------------------------------------
    def _decrypt(self, data):
        if USE_BINARY:
            return self._decrypt_binary(data)
        if _PY_CRYPTO_OK:
            return self._decrypt_python(data)
        return {"error": "No crypto backend available. Install 'cryptography' or build the C++ binary."}

    def _decrypt_binary(self, data):
        ct_hex  = data.get("ciphertext", "").strip().replace(" ", "").replace("\n", "")
        tag_hex = data.get("tag",        "").strip()
        key_hex = data.get("key",        "").strip()
        iv_hex  = data.get("iv",         "").strip()
        tamper  = bool(data.get("tamper", False))

        missing = [k for k, v in
                   [("ciphertext", ct_hex), ("tag", tag_hex),
                    ("key", key_hex), ("IV", iv_hex)] if not v]
        if missing:
            return {"error": f"Missing fields: {', '.join(missing)}."}
        if len(tag_hex) != 32:
            return {"error": f"Tag must be 32 hex chars (got {len(tag_hex)})."}
        if len(key_hex) not in (32, 64):
            return {"error": f"Key must be 32 or 64 hex chars (got {len(key_hex)})."}
        if len(iv_hex) != 24:
            return {"error": f"IV must be 24 hex chars (got {len(iv_hex)})."}

        try:
            ct_bytes = bytes.fromhex(ct_hex)
        except ValueError:
            return {"error": "Invalid hex string in ciphertext field."}

        if tamper and ct_bytes:
            ct_arr = bytearray(ct_bytes)
            ct_arr[0] ^= 0x01          # flip LSB of byte 0
            ct_bytes = bytes(ct_arr)

        in_path = out_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".enc") as f:
                f.write(ct_bytes)
                in_path = f.name
            out_path = in_path + ".dec"

            proc = subprocess.run(
                [str(BINARY), "decrypt",
                 "--key", key_hex, "--iv", iv_hex, "--tag", tag_hex,
                 "--in", in_path, "--out", out_path],
                capture_output=True, text=True, timeout=30
            )

            if proc.returncode != 0:
                return {"integrity": False}

            if not os.path.exists(out_path):
                return {"error": "Decrypt output file was not created."}

            with open(out_path, "rb") as f:
                plaintext = f.read().decode("utf-8", errors="replace")

            return {"integrity": True, "plaintext": plaintext}

        finally:
            _rm(in_path)
            _rm(out_path)

    # ------------------------------------------------------------------
    # Python / OpenSSL fallback — used when the C++ binary is absent
    # (e.g. cloud deployment).  NIST-identical output: same key, IV, tag.
    # ------------------------------------------------------------------
    def _encrypt_python(self, data):
        plaintext = data.get("plaintext", "")
        if not plaintext:
            return {"error": "Plaintext cannot be empty."}

        key_hex = data.get("key", "").strip() or secrets.token_hex(16)
        iv_hex  = data.get("iv",  "").strip() or secrets.token_hex(12)

        if len(key_hex) not in (32, 64):
            return {"error": f"Key must be 32 or 64 hex chars (got {len(key_hex)})."}
        if len(iv_hex) != 24:
            return {"error": f"IV must be 24 hex chars (got {len(iv_hex)})."}

        try:
            key_b = bytes.fromhex(key_hex)
            iv_b  = bytes.fromhex(iv_hex)
        except ValueError as e:
            return {"error": f"Hex decode error: {e}"}

        # cryptography returns ciphertext || tag (tag is last 16 bytes)
        ct_tag = _AESGCM(key_b).encrypt(iv_b, plaintext.encode("utf-8"), None)
        ct_hex  = ct_tag[:-16].hex()
        tag_hex = ct_tag[-16:].hex()

        return {"ciphertext": ct_hex, "tag": tag_hex, "key": key_hex, "iv": iv_hex}

    def _decrypt_python(self, data):
        ct_hex  = data.get("ciphertext", "").strip().replace(" ", "").replace("\n", "")
        tag_hex = data.get("tag",        "").strip()
        key_hex = data.get("key",        "").strip()
        iv_hex  = data.get("iv",         "").strip()
        tamper  = bool(data.get("tamper", False))

        missing = [k for k, v in
                   [("ciphertext", ct_hex), ("tag", tag_hex),
                    ("key", key_hex), ("IV", iv_hex)] if not v]
        if missing:
            return {"error": f"Missing fields: {', '.join(missing)}."}
        if len(tag_hex) != 32:
            return {"error": f"Tag must be 32 hex chars (got {len(tag_hex)})."}
        if len(key_hex) not in (32, 64):
            return {"error": f"Key must be 32 or 64 hex chars (got {len(key_hex)})."}
        if len(iv_hex) != 24:
            return {"error": f"IV must be 24 hex chars (got {len(iv_hex)})."}

        try:
            ct_bytes  = bytes.fromhex(ct_hex)
            tag_bytes = bytes.fromhex(tag_hex)
            key_bytes = bytes.fromhex(key_hex)
            iv_bytes  = bytes.fromhex(iv_hex)
        except ValueError as e:
            return {"error": f"Hex decode error: {e}"}

        if tamper and ct_bytes:
            arr = bytearray(ct_bytes); arr[0] ^= 0x01; ct_bytes = bytes(arr)

        try:
            # AESGCM.decrypt expects ciphertext || tag concatenated
            plaintext = _AESGCM(key_bytes).decrypt(iv_bytes, ct_bytes + tag_bytes, None)
            return {"integrity": True, "plaintext": plaintext.decode("utf-8", errors="replace")}
        except _InvalidTag:
            return {"integrity": False}

    # ------------------------------------------------------------------
    def log_message(self, fmt, *args):
        # Print a terse one-liner instead of the default verbose format.
        method, path, _ = args[0].split() if args else ("?", "?", "?")
        print(f"  {args[1]}  {method}  {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not USE_BINARY and not _PY_CRYPTO_OK:
        print("\n  ERROR: no crypto backend found.")
        print("  Option A (local):  cmake --build build --config Release")
        print("  Option B (cloud):  pip install cryptography\n")
        sys.exit(1)

    host = "0.0.0.0"
    print(f"\n  AES-GCM Engine  ->  http://localhost:{PORT}")
    print(f"  Crypto backend  ->  {ENGINE_LABEL}")
    print(f"  Listening on       {host}:{PORT}")
    print(f"  Press Ctrl+C to stop\n")

    httpd = http.server.HTTPServer((host, PORT), AesGcmHandler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  Server stopped.")
