# Qwen3-TTS 0.6B — matches remote Sentio voice worker
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "torchaudio==2.6.0+cu124" --index-url https://download.pytorch.org/whl/cu124 \
    && pip install --no-cache-dir \
    qwen-tts fastapi uvicorn soundfile numpy

COPY apps/sentio/scripts/tts_server.py /app/tts_server.py

ENV TTS_MODEL=Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
    TTS_ATTN=sdpa \
    TTS_HOST=0.0.0.0 \
    TTS_PORT=8091 \
    HF_HOME=/models/huggingface

EXPOSE 8091

HEALTHCHECK --interval=15s --timeout=5s --start-period=300s --retries=10 \
  CMD curl -fsS "http://127.0.0.1:8091/health" || exit 1

CMD ["python", "tts_server.py"]
