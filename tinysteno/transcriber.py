"""Whisper transcription module for TinySteno."""

from __future__ import annotations

import logging
import platform
from pathlib import Path
from typing import Callable, Protocol, runtime_checkable

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from scipy.signal import resample as scipy_resample

from tinysteno.transcript import OTHERS, SAMPLE_RATE, YOU, Segment, assemble, is_silent

logger = logging.getLogger(__name__)


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
            segments (list[Segment])   — time-ordered, speaker-labelled segments
            warnings (list[str])       — problems worth showing the user
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


def split_channels(data: np.ndarray, sr: int) -> dict[str, np.ndarray]:
    """Split audio into 16 kHz mono channels keyed by speaker label.

    One-channel audio gives {"": mono}; two or more give {YOU: left, OTHERS: right}.
    """
    if data.ndim == 1 or data.shape[1] == 1:
        return {"": convert_to_16khz_array(data, sr)}
    return {
        YOU: convert_to_16khz_array(data[:, 0].copy(), sr),
        OTHERS: convert_to_16khz_array(data[:, 1].copy(), sr),
    }


def _scaled_progress(
    on_progress: Callable[[float], None] | None, index: int, count: int
) -> Callable[[float], None] | None:
    """Map one pass's 0-1 progress onto its share of the whole job."""
    if on_progress is None:
        return None
    return lambda value: on_progress((index + value) / count)


# ---------------------------------------------------------------------------
# GPU / device detection
# ---------------------------------------------------------------------------

_MODEL_CACHE: dict[tuple[str, str, str], WhisperModel] = {}


def _detect_device() -> tuple[str, str]:
    """Detect optimal device and compute type for faster-whisper.

    Checks CUDA (via CTranslate2); Apple Silicon gets "auto" (CTranslate2 resolves it to
    the CPU); anything else uses the CPU.

    Returns (device, compute_type) tuple.
    """
    # CUDA — ask CTranslate2 how many CUDA devices it can see
    try:
        from ctranslate2 import get_cuda_device_count

        if get_cuda_device_count() > 0:
            return ("cuda", "float16")
    except Exception:
        logger.debug("CUDA detection failed; assuming no CUDA device.", exc_info=True)

    # Apple Silicon — CTranslate2 has no MPS backend; "auto" resolves to the CPU
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
            dict as built by tinysteno.transcript.assemble
        """
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        data, sr = sf.read(str(path))
        audio = split_channels(data, sr)
        to_decode = [label for label, samples in audio.items() if not is_silent(samples)]

        channels: dict[str, list[Segment]] = {label: [] for label in audio}
        languages: dict[str, str] = {}
        for index, label in enumerate(to_decode):
            channels[label], languages[label] = self._decode(
                audio[label],
                _scaled_progress(on_progress, index, len(to_decode)),
            )
        if on_progress is not None:
            on_progress(1.0)

        language = next(
            (languages[label] for label in (YOU, "", OTHERS) if channels.get(label)),
            "en",
        )
        return assemble(channels, audio, diarize=diarize, language=language)

    def _decode(
        self,
        audio: np.ndarray,
        on_progress: Callable[[float], None] | None = None,
    ) -> tuple[list[Segment], str]:
        """Run one Whisper pass over 16 kHz mono audio; return segments and language."""
        duration = len(audio) / SAMPLE_RATE
        raw_segments, info = self._model.transcribe(audio, beam_size=5)
        segments = []
        for seg in raw_segments:
            text = seg.text.strip()
            if text:
                segments.append(Segment(start=seg.start, end=seg.end, text=text))
            if on_progress is not None and duration > 0:
                on_progress(min(seg.start / duration, 1.0))
        if on_progress is not None:
            on_progress(1.0)
        return segments, info.language
