"""Transcript segments and the single assembly path shared by both backends."""
import json
import logging
import os
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

logger = logging.getLogger(__name__)

YOU = "You"
OTHERS = "Others"
SILENCE_RMS = 0.001
SAMPLE_RATE = 16000
SIDECAR_VERSION = 1

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


def sidecar_path(wav_path: str) -> Path:
    """Where the saved transcript of a recording lives: `Meeting-1.transcript.json`."""
    return Path(wav_path).with_suffix(".transcript.json")


def save_transcript(wav_path: str, result: dict) -> None:
    """Write the transcriber result beside the recording, atomically.

    A result without a segment list is not saved. A failure to write is logged and ignored.
    """
    segments = result.get("segments")
    if not isinstance(segments, list):
        return
    payload = {
        "version": SIDECAR_VERSION,
        "detected_language": result.get("detected_language", ""),
        "duration_seconds": result.get("duration_seconds", 0.0),
        "warnings": list(result.get("warnings") or []),
        "segments": [
            {"start": seg.start, "end": seg.end, "text": seg.text, "speaker": seg.speaker}
            for seg in segments
        ],
    }
    target = sidecar_path(wav_path)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=target.parent, prefix=target.name + ".",
            suffix=".tmp", delete=False,
        ) as handle:
            temp_name = handle.name
            json.dump(payload, handle, ensure_ascii=False)
        os.replace(temp_name, target)
    except OSError as err:
        logger.warning("Could not save the transcript to %s: %s", target, err)
        if temp_name:
            try:
                os.unlink(temp_name)
            except OSError:
                pass


def load_transcript(wav_path: str) -> dict | None:
    """The saved transcriber result for a recording, or None when it cannot be trusted.

    None when the sidecar is missing, older than the WAV, of another version or unreadable,
    and when the WAV itself is missing (there is nothing to compare against).
    """
    path = sidecar_path(wav_path)
    try:
        if path.stat().st_mtime < Path(wav_path).stat().st_mtime:
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["version"] != SIDECAR_VERSION:
            return None
        segments = [
            Segment(float(item["start"]), float(item["end"]), str(item["text"]),
                    str(item.get("speaker", "")))
            for item in data["segments"]
        ]
        return {
            "text": plain_text(segments),
            "diarised_text": labelled_text(segments),
            "duration_seconds": float(data["duration_seconds"]),
            "detected_language": str(data["detected_language"]),
            "segments": segments,
            "warnings": [str(text) for text in data["warnings"]],
        }
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as err:
        if not isinstance(err, FileNotFoundError):
            logger.info("Ignoring saved transcript %s: %s", path, err)
        return None
