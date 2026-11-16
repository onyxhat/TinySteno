"""Transcript segments and the single assembly path shared by both backends."""
from dataclasses import dataclass, replace
from typing import Iterable

import numpy as np

YOU = "You"
OTHERS = "Others"
SILENCE_RMS = 0.001
SAMPLE_RATE = 16000

_CHANNEL_NAMES = {YOU: "Microphone", OTHERS: "System"}


@dataclass(frozen=True)
class Segment:
    """One recognised span of speech."""

    start: float
    end: float
    text: str
    speaker: str = ""


def is_silent(samples: np.ndarray) -> bool:
    """True when the RMS of the samples is below SILENCE_RMS (empty is silent)."""
    if samples.size == 0:
        return True
    rms = float(np.sqrt(np.mean(np.square(samples.astype(np.float64)))))
    return rms < SILENCE_RMS


def _speaker_rank(speaker: str) -> int:
    if speaker == YOU:
        return 0
    if speaker == OTHERS:
        return 1
    return 2


def interleave(segments: Iterable[Segment]) -> list[Segment]:
    """Sort by start; ties go You, Others, other speakers, then input order."""
    return sorted(segments, key=lambda seg: (seg.start, _speaker_rank(seg.speaker)))


def plain_text(segments: Iterable[Segment]) -> str:
    """Segment texts joined with single spaces, in the order given."""
    return " ".join(seg.text for seg in segments)


def labelled_text(segments: Iterable[Segment]) -> str | None:
    """One `[speaker] text` line per segment; None if there is no speaker at all."""
    segs = list(segments)
    if not any(seg.speaker for seg in segs):
        return None
    return "\n".join(f"[{seg.speaker}] {seg.text}" for seg in segs)


def assemble(
    channels: dict[str, list[Segment]],
    audio: dict[str, np.ndarray],
    *,
    diarize: bool,
    language: str,
) -> dict:
    """Build the transcriber result from per-channel segments and 16 kHz mono audio."""
    stamped = [
        seg if seg.speaker else replace(seg, speaker=label)
        for label, segs in channels.items()
        for seg in segs
    ]
    segments = interleave(stamped)

    warnings = [
        f"{_CHANNEL_NAMES[label]} audio was present but no speech was recognised."
        for label, segs in channels.items()
        if label in _CHANNEL_NAMES and not segs and not is_silent(audio[label])
    ]

    longest = max((len(arr) for arr in audio.values()), default=0)
    return {
        "text": plain_text(segments),
        "diarised_text": labelled_text(segments) if diarize else None,
        "duration_seconds": longest / SAMPLE_RATE,
        "detected_language": language,
        "segments": segments,
        "warnings": warnings,
    }
