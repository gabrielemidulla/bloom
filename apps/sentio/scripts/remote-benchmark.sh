#!/usr/bin/env bash
# Bloom remote benchmark script — run on debian@192.168.1.118
set -euo pipefail

BLOOM="${BLOOM:-/home/debian/bloom}"
LLAMA_BENCH="${BLOOM}/llama.cpp/build/bin/llama-bench"
MODEL="${BLOOM}/models/qwen3.6/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf"
OUT="${BLOOM}/results/benchmark-$(date +%Y%m%d-%H%M%S).jsonl"
mkdir -p "${BLOOM}/results"

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export CUDA_VISIBLE_DEVICES=0

if [[ ! -f "$MODEL" ]]; then
  echo "Model not found: $MODEL" >&2
  exit 1
fi

echo "Benchmarking ncmoe sweep → $OUT"
for N in 28 30 32 34 36; do
  echo "=== ncmoe=$N ==="
  "$LLAMA_BENCH" \
    -m "$MODEL" \
    -ngl all -ncmoe "$N" \
    -fa on -ctk q8_0 -ctv q8_0 \
    -b 1024 -ub 256 \
    -p 512 -n 128 -r 3 -o json \
    >> "$OUT"
done

echo "Done. Results: $OUT"
