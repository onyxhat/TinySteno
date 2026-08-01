"""Whisper transcription module for TinySteno."""

from __future__ import annotations

import platform
from pathlib import Path
from typing import Callable, Protocol, runtime_checkable

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from scipy.signal import resample as scipy_resample


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Transcriber(Protocol):
    """Protocol for audio transcription backends.

    Both local (WhisperTranscriber) and API (ApiTranscriber) backends
    implement this interface so the rest of the pipeline can treat them
    interchangeably.
    """

    def transcribe(
        self,
        audio_path: str,
        diarize: bool = False,
        on_progress: Callable[[float], None] | None = None,
    ) -> dict:
        """Transcribe an audio file and return results.

        Returns dict with keys:
            text (str)            — full transcript
            diarised_text (str | None) — speaker-tagged transcript or None
            duration_seconds (float)   — audio duration
            detected_language (str)    — ISO language code
        """


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def convert_to_16khz_array(data: np.ndarray, sr: int) -> np.ndarray:
    """Convert audio data to 16 kHz mono float32 numpy array.

    Uses scipy FFT-based resampling instead of np.interp to avoid
    large index array allocations and improve accuracy.
    """
    if data.ndim > 1:
        data = data[:, 0]

    data = data.astype(np.float32)

    if sr != 16000:
        num_samples = int(len(data) * 16000 / sr)
        data = scipy_resample(data, num_samples).astype(np.float32)

    return data


# ---------------------------------------------------------------------------
# GPU / device detection
# ---------------------------------------------------------------------------

_MODEL_CACHE: dict[tuple[str, str, str], WhisperModel] = {}


def _detect_device() -> tuple[str, str]:
    """Detect optimal device and compute type for faster-whisper.

    Checks CUDA (via CTranslate2), Apple Silicon Metal, then falls back to CPU.

    Returns (device, compute_type) tuple.
    """
    # CUDA — check through CTranslate2's own device enumeration
    try:
        from ctranslate2 import get_supported_devices

        devices = get_supported_devices()
        if "cuda" in devices:
            return ("cuda", "float16")
    except ImportError:
        pass

    # Apple Silicon — CTranslate2 4+ supports MPS via device="auto"
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return ("auto", "auto")

    # Fallback: CPU
    return ("cpu", "int8")


# ---------------------------------------------------------------------------
# Local transcriber (faster-whisper)
# ---------------------------------------------------------------------------


class WhisperTranscriber:
    """Transcribe audio files using local faster-whisper."""

    def __init__(
        self,
        model_size: str = "small",
        device: str = "auto",
        compute_type: str = "auto",
    ):
        self.model_size = model_size

        # Resolve "auto" to detected hardware once.
        if device == "auto" or compute_type == "auto":
            detected_device, detected_compute = _detect_device()
            if device == "auto":
                device = detected_device
            if compute_type == "auto":
                compute_type = detected_compute

        self.device = device
        self.compute_type = compute_type

        cache_key = (model_size, device, compute_type)
        if cache_key not in _MODEL_CACHE:
            _MODEL_CACHE[cache_key] = WhisperModel(
                model_size, device=device, compute_type=compute_type
            )
        self._model = _MODEL_CACHE[cache_key]

    def transcribe(
        self,
        audio_path: str,
        diarize: bool = False,
        on_progress: Callable[[float], None] | None = None,
    ) -> dict:
        """Transcribe an audio file and return results.

        Returns:
            dict with keys: text, diarised_text, duration_seconds, detected_language
        """
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        # Read original file before converting — conversion drops to mono,
        # so we must check for stereo here to support diarization.
        data, sr = sf.read(str(path))
        is_stereo = data.ndim == 2 and data.shape[1] >= 2

        audio_16k = convert_to_16khz_array(data, sr)
        text, language = self._run_whisper(audio_16k, on_progress=on_progress)
        duration = len(audio_16k) / 16000.0

        diarised_text = None
        if diarize and is_stereo:
            diarised_text = self._diarize(data, sr)

        return {
            "text": text.strip(),
            "diarised_text": diarised_text,
            "duration_seconds": duration,
            "detected_language": language,
        }

    def _run_whisper(
        self,
        audio: np.ndarray,
        on_progress: Callable[[float], None] | None = None,
    ) -> tuple[str, str]:
        """Run Whisper on a 16kHz mono float32 numpy array."""
        duration = len(audio) / 16000.0
        segments, info = self._model.transcribe(audio, beam_size=5)
        parts = []
        for seg in segments:
            parts.append(seg.text)
            if on_progress is not None and duration > 0:
                on_progress(min(seg.start / duration, 1.0))
        if on_progress is not None:
            on_progress(1.0)
        return "".join(parts), info.language

    def _run_whisper_segments(self, audio: np.ndarray) -> list[tuple[float, str]]:
        """Run Whisper and return (start_seconds, text) tuples."""
        segments, _ = self._model.transcribe(audio, beam_size=5)
        return [(seg.start, seg.text.strip()) for seg in segments if seg.text.strip()]

    def _diarize(self, data: np.ndarray, sr: int) -> str | None:
        """Split stereo channels and transcribe each independently."""
        if data.ndim != 2:
            return None

        left_16k = convert_to_16khz_array(data[:, 0].copy(), sr)
        right_16k = convert_to_16khz_array(data[:, 1].copy(), sr)

        left_segs = self._run_whisper_segments(left_16k)
        right_segs = self._run_whisper_segments(right_16k)

        tagged = (
            [("You", start, text) for start, text in left_segs]
            + [("Others", start, text) for start, text in right_segs]
        )
        tagged.sort(key=lambda x: x[1])
        return "\n".join(f"[{speaker}] {text}" for speaker, _, text in tagged)
