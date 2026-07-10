#!/usr/bin/env bash
# Install Qwen3-TTS dependencies on remote CUDA machine
set -euo pipefail

BLOOM="${BLOOM:-/home/debian/bloom}"
VENV="${BLOOM}/venv-tts"

python3 -m venv "$VENV"
"${VENV}/bin/pip" install -U pip wheel

# RTX 3060 — CUDA 12.x wheels
"${VENV}/bin/pip" install torch torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
"${VENV}/bin/pip" install qwen-tts fastapi uvicorn soundfile

# FlashAttention optional but recommended on Ampere+
"${VENV}/bin/pip" install flash-attn --no-build-isolation 2>/dev/null || \
  echo "flash-attn build skipped — TTS will use default attention"

echo "TTS venv ready at $VENV"
