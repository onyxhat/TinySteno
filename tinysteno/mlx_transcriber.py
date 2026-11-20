"""Local transcription on the Apple Silicon GPU through mlx-whisper.

``mlx_whisper`` is an optional extra (``mlx``) that only installs on macOS arm64, so it is
imported inside functions: this module imports on every platform.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from tinysteno.transcriber import LocalTranscriber
from tinysteno.transcript import Segment, TranscriptOptions

# The whisper_model setting's sizes, as mlx-community conversions; other names pass through.
_MLX_MODELS: dict[str, str] = {
    size: f"mlx-community/whisper-{size}-mlx"
    for size in ("tiny", "base", "small", "medium", "large-v3")
}


def mlx_available() -> bool:
    """True when mlx_whisper imports on this machine."""
    try:
        import mlx_whisper  # noqa: F401  # pylint: disable=unused-import
    except Exception:
        return False
    return True


class MlxWhisperTranscriber(LocalTranscriber):
    """Transcribe audio files with mlx-whisper (Whisper on the GPU through Apple MLX).

    mlx-whisper has no beam search and no built-in VAD; the hygiene filters in
    ``transcript.assemble`` still run on its output.
    """

    def __init__(
        self,
        model_size: str = "small",
        *,
        prompt: str = "",
        options: TranscriptOptions | None = None,
    ):
        super().__init__(vad=False, prompt=prompt, options=options)
        self.model = _MLX_MODELS.get(model_size, model_size)

    def _decode(
        self,
        audio: np.ndarray,
        on_progress: Callable[[float], None] | None = None,
    ) -> tuple[list[Segment], str]:
        """Run one mlx-whisper pass over 16 kHz mono audio; return segments and language."""
        import mlx_whisper

        result = mlx_whisper.transcribe(
            audio.astype(np.float32, copy=False),
            path_or_hf_repo=self.model,
            initial_prompt=self.prompt or None,
        )
        segments = []
        for seg in result.get("segments") or []:
            text = str(seg.get("text", "")).strip()
            if text:
                segments.append(
                    Segment(start=float(seg["start"]), end=float(seg["end"]), text=text))
        if on_progress is not None:
            on_progress(1.0)
        return segments, result.get("language") or "en"
