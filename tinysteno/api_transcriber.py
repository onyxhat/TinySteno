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
from typing import Callable, NamedTuple

import numpy as np
import soundfile as sf
from openai import BadRequestError, OpenAI, UnprocessableEntityError

from tinysteno.speaker_split import Turn
from tinysteno.transcriber import recorded_gains, split_channels
from tinysteno.transcript import (
    OTHERS, SAMPLE_RATE, SPLIT_DECODE_SHARE, YOU, Segment, TranscriptOptions, assemble,
    audible_channels, scale_progress, split_enabled)

REJECTED_WARNING = "The transcription server rejected speaker separation; channel labels were used."
NO_LABELS_WARNING = (
    "The transcription server returned no speaker labels; channel labels were used.")


class _Decoded(NamedTuple):
    """One upload's result: segments, language, server speaker turns, and server refusal."""
    segments: list[Segment]
    language: str
    turns: list[Turn]
    rejected: bool = False


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
        server_diarize: bool = False,
        timeout: float = 600.0,
    ) -> None:
        self.model = model
        self.prompt = prompt
        self.options = options
        self.server_diarize = server_diarize
        kwargs: dict = {"api_key": api_key, "timeout": timeout}
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
        gains = recorded_gains(str(path), data)
        to_decode = audible_channels(audio, gains)

        splitting = split_enabled(self.options, diarize)
        decode_share = SPLIT_DECODE_SHARE if splitting else 1.0
        if on_progress:
            on_progress(0.0)
        channels: dict[str, list[Segment]] = {label: [] for label in audio}
        languages: dict[str, str] = {}
        # The channel that will carry the server's speaker labels: the remote one if uploaded.
        server_label = None
        if splitting and self.server_diarize and to_decode:
            server_label = OTHERS if OTHERS in to_decode else to_decode[0]
        server_turns: dict[str, list[Turn]] = {}
        rejected = False
        for index, label in enumerate(to_decode):
            if index and on_progress:
                on_progress(decode_share * index / len(to_decode))
            body = self._extra_body() if label == server_label else None
            decoded = self._decode(audio[label], body)
            channels[label], languages[label] = decoded.segments, decoded.language
            server_turns[label] = decoded.turns
            rejected = rejected or decoded.rejected
        if on_progress:
            on_progress(decode_share)

        language = next(
            (languages[label] for label in (YOU, "", OTHERS) if channels.get(label)),
            "en",
        )
        turns_for = None
        if server_label is not None:
            turns_for = _server_turns_for(server_label, server_turns[server_label])
        result = assemble(
            channels, audio, diarize=diarize, language=language, options=self.options,
            turns_for=turns_for,
            on_progress=scale_progress(on_progress, SPLIT_DECODE_SHARE, 1.0), gains=gains)
        if rejected:
            result["warnings"] = [*result.get("warnings", []), REJECTED_WARNING]
        elif server_label is not None and channels[server_label] and not server_turns[server_label]:
            result["warnings"] = [*result.get("warnings", []), NO_LABELS_WARNING]
        if splitting and on_progress:
            on_progress(1.0)
        return result

    def _extra_body(self) -> dict:
        """The non-standard form fields that ask a WhisperX-style server for speaker labels."""
        body: dict = {"diarize": True}
        hint = self.options.max_speakers if self.options else None
        if hint is not None and hint > 1:
            body["max_speakers"] = hint
        return body

    def _upload(self, audio_file, extra_body: dict | None):
        """One transcription request; `extra_body` carries the non-standard fields, if any."""
        extra = {"prompt": self.prompt} if self.prompt else {}
        if extra_body:
            extra["extra_body"] = extra_body
        return self._client.audio.transcriptions.create(
            model=self.model,
            file=audio_file,
            response_format="verbose_json",
            **extra,
        )

    def _decode(  # pylint: disable=too-many-locals
            self, audio: np.ndarray, extra_body: dict | None = None) -> _Decoded:
        """Upload 16 kHz mono audio once; return its segments, language and speaker turns.

        With `extra_body`, a server that answers 400/422 is asked again without it.
        """
        duration = len(audio) / SAMPLE_RATE

        # Write to a temporary WAV file (API needs a real file).
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
            sf.write(tmp_path, audio, SAMPLE_RATE)

        rejected = False
        try:
            try:
                with open(tmp_path, "rb") as audio_file:
                    response = self._upload(audio_file, extra_body)
            except (BadRequestError, UnprocessableEntityError):
                if not extra_body:
                    raise
                rejected = True
                with open(tmp_path, "rb") as audio_file:
                    response = self._upload(audio_file, None)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        texts: list[str] = []
        segments: list[Segment] = []
        turns: list[Turn] = []
        labels: dict[str, int] = {}
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
            start = max(start, 0.0)  # a negative time is a server glitch, not before the audio
            end = _number(_field(raw, "end"))
            end = start if end is None else max(end, start)
            segments.append(Segment(start, end, text))
            speaker = _field(raw, "speaker")
            if extra_body and not rejected and isinstance(speaker, str) and speaker:
                turns.append(Turn(start, end, labels.setdefault(speaker, len(labels))))
        if not timed:
            segments = []
        if not segments:
            turns = []
            text = str(getattr(response, "text", None) or "").strip() or " ".join(texts)
            if text:
                segments = [Segment(0.0, duration, text)]
        language = getattr(response, "language", None) or "en"
        return _Decoded(segments, language, turns, rejected)


def _server_turns_for(server_label: str, turns: list[Turn]) -> Callable:
    """A `turns_for` hook for assemble: the server's turns for the channel uploaded with the
    extra fields, nothing to split for any other channel."""
    def turns_for(label: str, _audio: np.ndarray) -> list[Turn]:
        # assemble names a single-channel recording's channel YOU, though it is keyed "".
        return list(turns) if (label or YOU) == (server_label or YOU) else []
    return turns_for


def _field(item: object, name: str):
    """Read a field from a segment given as a dict or as an object."""
    if isinstance(item, dict):
        return item.get(name)
    return getattr(item, name, None)


def _number(value: object) -> float | None:
    """A finite float from a number or numeric string, else None."""
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None
