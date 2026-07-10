#!/usr/bin/env bash
set -euo pipefail

BLOOM="${BLOOM:-/home/debian/bloom}"
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export CUDA_VISIBLE_DEVICES=0
export TTS_MODEL="${TTS_MODEL:-Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice}"
export TTS_HOST="${TTS_HOST:-127.0.0.1}"
export TTS_PORT="${TTS_PORT:-8091}"

exec "${BLOOM}/venv-tts/bin/python" "${BLOOM}/scripts/tts_server.py"
