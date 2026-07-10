"""Shared voice post-processing: EQ + compression."""

from __future__ import annotations

import numpy as np
from scipy import signal

SAMPLE_RATE = 24_000


def postprocess_voice(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    *,
    high_boost_db: float = 5.0,
    high_shelf_hz: float = 3200.0,
    air_boost_db: float = 2.5,
    air_hz: float = 6500.0,
    compress_threshold_db: float = -20.0,
    compress_ratio: float = 2.5,
    compress_makeup_db: float = 2.0,
    pitch_semitones: float = 0.0,
    speed_factor: float = 1.0,
) -> np.ndarray:
    """EQ + compression chain for clearer, more present TTS output."""
    if audio.size == 0:
        return audio

    x = audio.astype(np.float64, copy=False)

    if speed_factor > 1.0:
        x = _time_stretch(x.astype(np.float32), speed_factor).astype(np.float64)

    if pitch_semitones:
        x = _pitch_shift(x.astype(np.float32), sample_rate, pitch_semitones).astype(np.float64)

    if high_boost_db:
        x = _high_shelf(x, sample_rate, high_shelf_hz, high_boost_db)

    if air_boost_db:
        x = _high_shelf(x, sample_rate, air_hz, air_boost_db, q=0.5)

    if compress_ratio > 1.0:
        x = _compress(
            x,
            sample_rate,
            threshold_db=compress_threshold_db,
            ratio=compress_ratio,
            makeup_db=compress_makeup_db,
        )

    peak = np.max(np.abs(x))
    if peak > 0.98:
        x = x * (0.98 / peak)

    return x.astype(np.float32)


def _pitch_shift(audio: np.ndarray, sample_rate: int, semitones: float) -> np.ndarray:
    import librosa

    return librosa.effects.pitch_shift(
        audio.astype(np.float32),
        sr=sample_rate,
        n_steps=semitones,
        n_fft=2048,
        hop_length=256,
    )


def _time_stretch(audio: np.ndarray, speed_factor: float) -> np.ndarray:
    import librosa

    return librosa.effects.time_stretch(audio.astype(np.float32), rate=speed_factor)


def _high_shelf(
    audio: np.ndarray,
    sample_rate: int,
    freq_hz: float,
    gain_db: float,
    q: float = 0.707,
) -> np.ndarray:
    a = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * freq_hz / sample_rate
    sin_w0 = np.sin(w0)
    cos_w0 = np.cos(w0)
    alpha = sin_w0 / 2.0 * np.sqrt((a + 1.0 / a) * (1.0 / q - 1.0) + 2.0)
    sqrt_a = np.sqrt(a)

    b0 = a * ((a + 1.0) + (a - 1.0) * cos_w0 + 2.0 * sqrt_a * alpha)
    b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_w0)
    b2 = a * ((a + 1.0) + (a - 1.0) * cos_w0 - 2.0 * sqrt_a * alpha)
    a0 = (a + 1.0) - (a - 1.0) * cos_w0 + 2.0 * sqrt_a * alpha
    a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cos_w0)
    a2 = (a + 1.0) - (a - 1.0) * cos_w0 - 2.0 * sqrt_a * alpha

    b = np.array([b0, b1, b2]) / a0
    a = np.array([1.0, a1 / a0, a2 / a0])
    return signal.lfilter(b, a, audio)


def _compress(
    audio: np.ndarray,
    sample_rate: int,
    *,
    threshold_db: float = -20.0,
    ratio: float = 2.5,
    attack_ms: float = 5.0,
    release_ms: float = 100.0,
    makeup_db: float = 2.0,
) -> np.ndarray:
    threshold = 10.0 ** (threshold_db / 20.0)
    makeup = 10.0 ** (makeup_db / 20.0)
    attack = np.exp(-1.0 / (sample_rate * attack_ms / 1000.0))
    release = np.exp(-1.0 / (sample_rate * release_ms / 1000.0))

    envelope = 0.0
    out = np.empty_like(audio)
    for i, sample in enumerate(audio):
        level = abs(sample)
        coeff = attack if level > envelope else release
        envelope = coeff * envelope + (1.0 - coeff) * level

        if envelope > threshold:
            reduced = threshold + (envelope - threshold) / ratio
            gain = reduced / (envelope + 1e-12)
        else:
            gain = 1.0

        out[i] = sample * gain

    return out * makeup
