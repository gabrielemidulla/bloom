#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p models

uv run python -m mlx_audio.convert \
  --hf-path bosonai/higgs-audio-v3-tts-4b \
  --mlx-path models/higgs-v3-4bit \
  --model-domain tts \
  --quantize \
  --q-bits 4
