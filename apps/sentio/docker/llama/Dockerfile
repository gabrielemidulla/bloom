# llama.cpp b9948 — matches remote RTX 3060 deployment
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

ARG LLAMA_CPP_REF=b9948
ARG CMAKE_CUDA_ARCHITECTURES=80;86;89

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
RUN git clone https://github.com/ggml-org/llama.cpp.git . \
    && git checkout "${LLAMA_CPP_REF}"

RUN cmake -B build \
      -DGGML_CUDA=ON \
      -DGGML_CUDA_FA_ALL_QUANTS=ON \
      -DCMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES}" \
      -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --config Release -j "$(nproc)" --target llama-server

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /src/build/bin/llama-server /usr/local/bin/llama-server
COPY apps/sentio/docker/llama/start-llama.sh /usr/local/bin/start-llama.sh
RUN chmod +x /usr/local/bin/start-llama.sh

ENV LLAMA_MODEL=/models/qwen3.6/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf \
    LLAMA_CTX=12288 \
    LLAMA_THREADS=8 \
    LLAMA_HOST=0.0.0.0 \
    LLAMA_PORT=8080

EXPOSE 8080

HEALTHCHECK --interval=15s --timeout=5s --start-period=180s --retries=10 \
  CMD curl -fsS "http://127.0.0.1:8080/health" || exit 1

ENTRYPOINT ["/usr/local/bin/start-llama.sh"]
