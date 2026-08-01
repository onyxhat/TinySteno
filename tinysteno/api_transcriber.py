"""API-based Whisper transcription for TinySteno.

Uses OpenAI-compatible Whisper API endpoints instead of local faster-whisper.
Suitable for environments where local model loading is impractical or when
higher-quality Whisper models are desired without local GPU.

Privacy note: audio files are uploaded to the configured API server.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf
from openai import OpenAI
from openai.types.audio import TranscriptionVerbose

from tinysteno.transcriber import convert_to_16khz_array


class ApiTranscriber:
    """Transcribe audio using OpenAI Whisper API (or compatible)."""

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "",
        model: str = "whisper-1",
    ) -> None:
        self.model = model
        kwargs: dict = {"api_key": api_key, "timeout": 180.0}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)

    def transcribe(
        self,
        audio_path: str,
        diarize: bool = False,
        on_progress: Callable[[float], None] | None = None,
    ) -> dict:
        """Transcribe an audio file via the Whisper API.

        Returns dict with keys: text, diarised_text, duration_seconds,
        detected_language.
        """
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        data, sr = sf.read(str(path), dtype="float32")
        is_stereo = data.ndim == 2 and data.shape[1] == 2

        if is_stereo and diarize:
            return self._transcribe_diarized(data, sr, on_progress)

        if on_progress:
            on_progress(0.0)
        text, duration, language = self._transcribe_array(data, sr)
        if on_progress:
            on_progress(1.0)

        return {
            "text": text,
            "diarised_text": None,
            "duration_seconds": duration,
            "detected_language": language,
        }

    def _transcribe_diarized(
        self,
        data: np.ndarray,
        sr: int,
        on_progress: Callable[[float], None] | None = None,
    ) -> dict:
        """Split stereo channels, transcribe each separately, merge."""
        if on_progress:
            on_progress(0.0)

        # Transcribe left channel (You).
        left_mono = data[:, 0].copy()
        left_text, left_duration, left_language = self._transcribe_array(
            left_mono, sr
        )

        if on_progress:
            on_progress(0.5)

        # Transcribe right channel (Others).
        right_mono = data[:, 1].copy()
        right_text, _, _ = self._transcribe_array(
            right_mono, sr
        )

        if on_progress:
            on_progress(1.0)

        diarised_text = f"[You] {left_text.strip()}\n\n[Others] {right_text.strip()}"
        return {
            "text": f"{left_text.strip()} {right_text.strip()}",
            "diarised_text": diarised_text,
            "duration_seconds": left_duration,
            "detected_language": left_language,
        }

    def _transcribe_array(
        self,
        audio: np.ndarray,
        sr: int,
        _on_progress: Callable[[float], None] | None = None,
    ) -> tuple[str, float, str]:
        """Convert numpy array to WAV temp file and transcribe via API."""
        audio_16k = convert_to_16khz_array(audio, sr)
        duration = len(audio_16k) / 16000.0

        # Write to a temporary WAV file (API needs a real file).
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
            sf.write(tmp_path, audio_16k, 16000)

        try:
            response = self._client.audio.transcriptions.create(
                model=self.model,
                file=open(tmp_path, "rb"),  # noqa: SIM115  # pylint: disable=consider-using-with  # cleanup in finally
                response_format="verbose_json",
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        parsed = self._parse_response(response)
        return parsed["text"], duration, parsed["detected_language"]

    @staticmethod
    def _parse_response(response: TranscriptionVerbose) -> dict:
        """Extract standard result dict from an OpenAI API response."""
        language = response.language if hasattr(response, "language") else ""
        if not language:
            language = "en"

        return {
            "text": response.text or "",
            "diarised_text": None,
            "duration_seconds": response.duration or 0.0,
            "detected_language": language,
        }
