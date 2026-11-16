"""API-based Whisper transcription for TinySteno.

Uses OpenAI-compatible Whisper API endpoints instead of local faster-whisper.
Suitable for environments where local model loading is impractical or when
higher-quality Whisper models are desired without local GPU.

Privacy note: audio files are uploaded to the configured API server.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf
from openai import OpenAI

from tinysteno.transcriber import split_channels
from tinysteno.transcript import (
    OTHERS, SAMPLE_RATE, SPLIT_DECODE_SHARE, YOU, Segment, TranscriptOptions, assemble,
    is_silent, scale_progress, split_enabled)


class ApiTranscriber:
    """Transcribe audio using OpenAI Whisper API (or compatible)."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        api_key: str = "",
        base_url: str = "",
        model: str = "whisper-1",
        *,
        prompt: str = "",
        options: TranscriptOptions | None = None,
    ) -> None:
        self.model = model
        self.prompt = prompt
        self.options = options
        kwargs: dict = {"api_key": api_key, "timeout": 180.0}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)

    def transcribe(  # pylint: disable=too-many-locals
        self,
        audio_path: str,
        diarize: bool = False,
        on_progress: Callable[[float], None] | None = None,
    ) -> dict:
        """Transcribe an audio file via the Whisper API.

        Every non-silent channel is uploaded once; the result is built by
        tinysteno.transcript.assemble.
        """
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        data, sr = sf.read(str(path), dtype="float32")
        audio = split_channels(data, sr)
        to_decode = [label for label, samples in audio.items() if not is_silent(samples)]

        splitting = split_enabled(self.options, diarize)
        decode_share = SPLIT_DECODE_SHARE if splitting else 1.0
        if on_progress:
            on_progress(0.0)
        channels: dict[str, list[Segment]] = {label: [] for label in audio}
        languages: dict[str, str] = {}
        for index, label in enumerate(to_decode):
            if index and on_progress:
                on_progress(decode_share * index / len(to_decode))
            channels[label], languages[label] = self._decode(audio[label])
        if on_progress:
            on_progress(decode_share)

        language = next(
            (languages[label] for label in (YOU, "", OTHERS) if channels.get(label)),
            "en",
        )
        result = assemble(
            channels, audio, diarize=diarize, language=language, options=self.options,
            on_progress=scale_progress(on_progress, SPLIT_DECODE_SHARE, 1.0))
        if splitting and on_progress:
            on_progress(1.0)
        return result

    def _decode(  # pylint: disable=too-many-locals
            self, audio: np.ndarray) -> tuple[list[Segment], str]:
        """Upload 16 kHz mono audio once; return its segments and language."""
        duration = len(audio) / SAMPLE_RATE

        # Write to a temporary WAV file (API needs a real file).
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
            sf.write(tmp_path, audio, SAMPLE_RATE)

        extra = {"prompt": self.prompt} if self.prompt else {}
        try:
            with open(tmp_path, "rb") as audio_file:
                response = self._client.audio.transcriptions.create(
                    model=self.model,
                    file=audio_file,
                    response_format="verbose_json",
                    **extra,
                )
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        texts: list[str] = []
        segments: list[Segment] = []
        timed = True
        for raw in getattr(response, "segments", None) or []:
            text = str(_field(raw, "text") or "").strip()
            if not text:
                continue
            texts.append(text)
            start = _number(_field(raw, "start"))
            if start is None:
                timed = False
                continue
            end = _number(_field(raw, "end"))
            segments.append(Segment(start, start if end is None else max(end, start), text))
        if not timed:
            segments = []
        if not segments:
            text = str(getattr(response, "text", None) or "").strip() or " ".join(texts)
            if text:
                segments = [Segment(0.0, duration, text)]
        language = getattr(response, "language", None) or "en"
        return segments, language


def _field(item: object, name: str):
    """Read a field from a segment given as a dict or as an object."""
    if isinstance(item, dict):
        return item.get(name)
    return getattr(item, name, None)


def _number(value: object) -> float | None:
    """A finite float from a number or numeric string, else None."""
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None
