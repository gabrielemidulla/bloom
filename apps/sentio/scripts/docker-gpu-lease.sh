#!/usr/bin/env bash
# Exclusive GPU handoff for 12GB-class cards running llama + TTS.
# Usage: ./apps/sentio/scripts/docker-gpu-lease.sh {llama|tts|status|release}
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

ACTION="${1:-status}"
COMPOSE=(docker compose)

case "$ACTION" in
  llama)
    "${COMPOSE[@]}" stop tts 2>/dev/null || true
    "${COMPOSE[@]}" up -d llama
    echo "GPU leased to llama"
    ;;
  tts)
    "${COMPOSE[@]}" stop llama
    sleep 2
    "${COMPOSE[@}" --profile voice up -d tts
    echo "GPU leased to tts"
    ;;
  release)
    "${COMPOSE[@]}" stop tts 2>/dev/null || true
    "${COMPOSE[@]}" up -d llama
    echo "Default: llama active"
    ;;
  status)
    "${COMPOSE[@]}" ps llama tts 2>/dev/null || true
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null || true
    ;;
  *)
    echo "Usage: $0 {llama|tts|status|release}"
    exit 1
    ;;
esac
