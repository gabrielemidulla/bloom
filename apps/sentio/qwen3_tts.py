"""Qwen3-TTS helpers (MLX on Apple Silicon)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from mlx_audio.tts.utils import load_model

from voice_post import SAMPLE_RATE, postprocess_voice

CUSTOM_VOICE_MODEL = "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit"
CLONE_MODEL = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit"

SPEAKERS = (
    "ryan",
    "aiden",
    "dylan",
    "eric",
    "uncle_fu",
    "serena",
    "vivian",
    "ono_anna",
    "sohee",
)

# Verbatim transcript of notebooks/voices/clara.wav (ElevenLabs reference).
DEFAULT_CLARA_REF_TEXT = (
    "So yeah, I was thinking about it earlier. And honestly, it's not that complicated. "
    "You just tell me what you're trying to do, and we'll figure it out from there. "
    "I'm not going to over-explain it. We can just start."
)


class Qwen3TTS:
    def __init__(self, model_id: str | None = None):
        self._model_id = model_id
        self.model = None
        self._mode: str | None = None

        self.speaker = "ryan"
        self.language = "English"
        self.instruct: str | None = None
        self.ref_audio: str | None = None
        self.ref_text: str | None = None

    def _load(self, model_id: str, mode: str) -> None:
        if self.model is not None and self._model_id == model_id and self._mode == mode:
            return
        self.model = load_model(model_id)
        self._model_id = model_id
        self._mode = mode

    def set_voice(
        self,
        *,
        speaker: str = "ryan",
        instruct: str | None = None,
        language: str = "English",
    ) -> None:
        """Preset speaker + style (CustomVoice model)."""
        if speaker.lower() not in SPEAKERS:
            raise ValueError(f"Unknown speaker {speaker!r}. Choose from: {', '.join(SPEAKERS)}")
        self._load(CUSTOM_VOICE_MODEL, "custom")
        self.speaker = speaker.lower()
        self.instruct = instruct
        self.language = language
        self.ref_audio = None
        self.ref_text = None

    def set_voice_clone(
        self,
        ref_audio: str | Path,
        ref_text: str,
    ) -> None:
        """Clone a voice from reference audio (Base model)."""
        path = Path(ref_audio)
        if not path.exists():
            raise FileNotFoundError(path)
        self._load(CLONE_MODEL, "clone")
        self.ref_audio = str(path)
        self.ref_text = ref_text
        self.instruct = None

    def synthesize(
        self,
        text: str,
        *,
        temperature: float = 0.9,
        top_p: float = 1.0,
        repetition_penalty: float = 1.05,
        max_tokens: int = 4096,
        postprocess: bool = True,
        high_boost_db: float = 5.0,
        air_boost_db: float = 2.5,
        compress_threshold_db: float = -20.0,
        compress_ratio: float = 2.5,
        compress_makeup_db: float = 2.0,
    ) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Call set_voice() or set_voice_clone() before synthesize()")

        if self._mode == "clone":
            if not self.ref_audio or not self.ref_text:
                raise RuntimeError("Voice clone not configured")
            result = next(
                self.model.generate(
                    text=text,
                    ref_audio=self.ref_audio,
                    ref_text=self.ref_text,
                    lang_code="en",
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    max_tokens=max_tokens,
                )
            )
        else:
            if self.instruct is None:
                raise RuntimeError("Call set_voice(instruct=...) before synthesize()")
            result = next(
                self.model.generate_custom_voice(
                    text=text,
                    speaker=self.speaker,
                    language=self.language,
                    instruct=self.instruct,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    max_tokens=max_tokens,
                )
            )

        sample_rate = int(getattr(result, "sample_rate", SAMPLE_RATE))
        audio = np.asarray(result.audio, dtype=np.float32)

        if postprocess:
            audio = postprocess_voice(
                audio,
                sample_rate=sample_rate,
                high_boost_db=high_boost_db,
                air_boost_db=air_boost_db,
                compress_threshold_db=compress_threshold_db,
                compress_ratio=compress_ratio,
                compress_makeup_db=compress_makeup_db,
            )

        return audio
