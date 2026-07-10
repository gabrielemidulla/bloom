#!/usr/bin/env python3
"""Minimal Qwen3-TTS HTTP server for Bloom Sentio (CUDA)."""

from __future__ import annotations

import io
import os
import wave
from contextlib import asynccontextmanager

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

MODEL_ID = os.environ.get(
    "TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
)
HOST = os.environ.get("TTS_HOST", "127.0.0.1")
PORT = int(os.environ.get("TTS_PORT", "8091"))

_model = None
_sample_rate = 24000


class SpeechRequest(BaseModel):
    input: str = Field(min_length=1)
    voice: str = "Aiden"
    language: str = "English"
    instruct: str | None = None
    stream: bool = False


def _load_model():
    global _model, _sample_rate
    from qwen_tts import Qwen3TTSModel

    _model = Qwen3TTSModel.from_pretrained(
        MODEL_ID,
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation=os.environ.get("TTS_ATTN", "sdpa"),
    )
    _sample_rate = 24000


def _pcm_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    pcm = np.clip(audio, -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
    return buf.getvalue()


def _synthesize(req: SpeechRequest) -> tuple[np.ndarray, int]:
    if _model is None:
        raise RuntimeError("TTS model not loaded")

    instruct = req.instruct or "Speak naturally and clearly."
    wavs, sr = _model.generate_custom_voice(
        text=req.input,
        language=req.language,
        speaker=req.voice,
        instruct=instruct,
    )
    audio = np.asarray(wavs[0], dtype=np.float32)
    return audio, int(sr)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _load_model()
    yield


app = FastAPI(title="Bloom Sentio TTS", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": MODEL_ID}


@app.post("/v1/audio/speech")
def speech(req: SpeechRequest):
    try:
        audio, sr = _synthesize(req)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if req.stream:
        data = _pcm_bytes(audio, sr)

        def stream():
            chunk = 4096
            for i in range(0, len(data), chunk):
                yield data[i : i + chunk]

        return StreamingResponse(stream(), media_type="audio/wav")

    return Response(content=_pcm_bytes(audio, sr), media_type="audio/wav")


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
