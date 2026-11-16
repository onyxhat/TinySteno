"""Transcript segments and the single assembly path shared by both backends."""
from dataclasses import dataclass, replace
from typing import Callable, Iterable

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


@dataclass(frozen=True)
class TranscriptOptions:
    """Switches for the filters applied while assembling a transcript."""

    drop_hallucinations: bool = True
    suppress_bleed: bool = True


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


def _clock(seconds: float) -> str:
    """Zero-padded hh:mm:ss from whole seconds (floored); hours are never truncated."""
    hours, rest = divmod(int(seconds), 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def render_transcript(segments: Iterable[Segment], *, labels: bool, timestamps: bool) -> str:
    """One `[hh:mm:ss] [Speaker] text` line per segment, each prefix only when enabled.

    The speaker prefix is also left out for a segment with no speaker. When no prefix
    applies at all the result is `plain_text`.
    """
    segs = list(segments)
    show_speakers = labels and any(seg.speaker for seg in segs)
    if not timestamps and not show_speakers:
        return plain_text(segs)
    lines = []
    for seg in segs:
        prefix = f"[{_clock(seg.start)}] " if timestamps else ""
        if labels and seg.speaker:
            prefix += f"[{seg.speaker}] "
        lines.append(prefix + seg.text)
    return "\n".join(lines)


def _level_over(audio: dict[str, np.ndarray], label: str, start: float, end: float) -> float:
    """RMS of one channel between two times in seconds (0.0 for an empty range)."""
    samples = audio[label][max(0, int(start * SAMPLE_RATE)):max(0, int(end * SAMPLE_RATE))]
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples.astype(np.float64)))))


def assemble(
    channels: dict[str, list[Segment]],
    audio: dict[str, np.ndarray],
    *,
    diarize: bool,
    language: str,
    options: TranscriptOptions | None = None,
) -> dict:
    """Build the transcriber result from per-channel segments and 16 kHz mono audio."""
    from tinysteno.hygiene import drop_hallucinations, suppress_bleed

    options = options or TranscriptOptions()
    stamped = {
        label: [seg if seg.speaker else replace(seg, speaker=label) for seg in segs]
        for label, segs in channels.items()
    }

    warnings = [
        f"{_CHANNEL_NAMES[label]} audio was present but no speech was recognised."
        for label, segs in channels.items()
        if label in _CHANNEL_NAMES and not segs and not is_silent(audio[label])
    ]

    def level(label: str, start: float, end: float) -> float:
        return _level_over(audio, label, start, end)

    def channel_level(label: str) -> Callable[[float, float], float]:
        return lambda start, end: level(label, start, end)

    if options.drop_hallucinations:
        stamped = {
            label: drop_hallucinations(
                sorted(segs, key=lambda s: s.start),
                rms=(lambda start, end, label=label: level(label, start, end))
                if label in audio else None)
            for label, segs in stamped.items()
        }
    if options.suppress_bleed and YOU in stamped and OTHERS in stamped:
        stamped[YOU], stamped[OTHERS] = suppress_bleed(
            stamped[YOU], stamped[OTHERS], channel_level(YOU), channel_level(OTHERS))

    segments = interleave(seg for segs in stamped.values() for seg in segs)
    longest = max((len(arr) for arr in audio.values()), default=0)
    return {
        "text": plain_text(segments),
        "diarised_text": labelled_text(segments) if diarize else None,
        "duration_seconds": longest / SAMPLE_RATE,
        "detected_language": language,
        "segments": segments,
        "warnings": warnings,
    }
