#!/usr/bin/env bash
set -euo pipefail

LLAMA_BIN="${LLAMA_BIN:-/usr/local/bin/llama-server}"
MODEL="${LLAMA_MODEL:-/models/qwen3.6/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf}"
HOST="${LLAMA_HOST:-0.0.0.0}"
PORT="${LLAMA_PORT:-8080}"
PHYS="${LLAMA_THREADS:-8}"
CTX="${LLAMA_CTX:-12288}"

if [[ ! -f "$MODEL" ]]; then
  echo "Model not found: $MODEL" >&2
  echo "Place Qwen3.6-35B-A3B-UD-IQ4_XS.gguf under ./models/qwen3.6/ on the host." >&2
  exit 1
fi

# Fitted via llama-fit-params (-fitt 768) on RTX 3060 12GB + 32GB RAM.
exec "$LLAMA_BIN" \
  -m "$MODEL" \
  --alias qwen3.6-local \
  --host "$HOST" --port "$PORT" \
  --fit off \
  -c "$CTX" \
  -ngl 41 \
  -ot "blk\.24\.ffn_(gate|up|gate_up|down).*=CPU,blk\.25\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.26\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.27\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.28\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.29\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.30\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.31\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.32\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.33\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.34\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.35\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.36\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.37\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.38\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.39\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.40\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU" \
  -np 1 --kv-unified \
  -fa on -ctk q8_0 -ctv q8_0 \
  -b 1024 -ub 256 \
  -t "$PHYS" -tb "$PHYS" \
  --no-mmap --no-mmproj \
  --jinja --reasoning-format deepseek \
  --cache-ram 512 \
  --ctx-checkpoints 8 \
  --checkpoint-min-step 512 \
  --no-context-shift \
  --metrics --no-webui
