"""Transcript segments and the single assembly path shared by both backends."""
import json
import logging
import math
import os
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from tinysteno.speaker_split import (
    MIN_AUDIO_SECONDS, DiarizerUnavailable, Turn, apply_policy, label_segments)

logger = logging.getLogger(__name__)

YOU = "You"
OTHERS = "Others"
SILENCE_RMS = 0.001
SAMPLE_RATE = 16000
SIDECAR_VERSION = 2

_CHANNEL_NAMES = {YOU: "Microphone", OTHERS: "System"}
_WARN_SPLIT_FAILED = "Speaker separation failed; channel labels were used."
# Share of a transcriber's progress spent decoding when a speaker split follows.
SPLIT_DECODE_SHARE = 0.7


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
    speaker_split: bool = False
    max_speakers: int | None = None


def split_enabled(options: TranscriptOptions | None, diarize: bool) -> bool:
    """True when a speaker split will run: it is on, and the labels it affects are shown."""
    return bool(options and options.speaker_split and diarize)


def scale_progress(
    on_progress: Callable[[float], None] | None, low: float, high: float
) -> Callable[[float], None] | None:
    """Map a 0-1 fraction onto the `low`-`high` part of the job's progress."""
    if on_progress is None:
        return None
    return lambda value: on_progress(low + (high - low) * value)


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


def _default_turns(on_progress: Callable[[float], None] | None):
    """A `turns_for` that runs the optional sherpa-onnx diarizer (imported only when used)."""
    def turns_for(_label: str, samples: np.ndarray) -> list[Turn]:
        from tinysteno.sherpa_diarizer import diarize as run_diarizer
        return run_diarizer(samples, on_progress=on_progress)
    return turns_for


def _split_target(stamped: dict[str, list[Segment]]) -> str | None:
    """The channel to split: the remote side, else the microphone, else a mono file."""
    for label in (OTHERS, YOU, ""):
        if stamped.get(label):
            return label
    return None


def _split_speakers(
    stamped: dict[str, list[Segment]],
    audio: dict[str, np.ndarray],
    options: TranscriptOptions,
    turns_for: Callable[[str, np.ndarray], list[Turn]],
    warnings: list[str],
) -> None:
    """Relabel one channel's segments by speaker, in place; keep them as they are on failure."""
    key = _split_target(stamped)
    if key is None or key not in audio or options.max_speakers == 1:
        return
    if len(audio[key]) / SAMPLE_RATE < MIN_AUDIO_SECONDS:
        return
    base = key or YOU
    try:
        turns = apply_policy(turns_for(base, audio[key]), max_speakers=options.max_speakers)
        if turns is None:
            return
        segments = [seg if seg.speaker else replace(seg, speaker=base) for seg in stamped[key]]
        relabelled = label_segments(segments, turns, base)
    except DiarizerUnavailable as err:
        warnings.append(str(err))
        return
    except Exception:
        logger.warning("Speaker separation failed", exc_info=True)
        warnings.append(_WARN_SPLIT_FAILED)
        return
    stamped[key] = relabelled


def assemble(  # pylint: disable=too-many-arguments
    channels: dict[str, list[Segment]],
    audio: dict[str, np.ndarray],
    *,
    diarize: bool,
    language: str,
    options: TranscriptOptions | None = None,
    turns_for: Callable[[str, np.ndarray], list[Turn]] | None = None,
    on_progress: Callable[[float], None] | None = None,
) -> dict:
    """Build the transcriber result from per-channel segments and 16 kHz mono audio.

    With `options.speaker_split` and `diarize` on, the remote channel (else the microphone) is
    split into speakers after the filters; any failure keeps the plain channel labels.
    `turns_for(label, audio)` supplies the raw turns; `on_progress` receives the diarizer's
    0-1 fraction.
    """
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

    if split_enabled(options, diarize):
        _split_speakers(stamped, audio, options, turns_for or _default_turns(on_progress),
                        warnings)

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
    """Where the saved transcript of a recording lives: `Meeting-1.wav.transcript.json`."""
    return Path(str(wav_path) + ".transcript.json")


def save_transcript(wav_path: str, result: dict, fingerprint: dict) -> None:
    """Write the transcriber result beside the recording, atomically.

    Nothing is saved for a result without a segment list or a recording that does not exist.
    Any failure is logged and ignored: saving never fails the run.
    """
    target = sidecar_path(wav_path)
    temp_name = None
    try:
        segments = result.get("segments")
        if not isinstance(segments, list) or not Path(wav_path).is_file():
            return
        payload = {
            "version": SIDECAR_VERSION,
            "fingerprint": fingerprint,
            "detected_language": result.get("detected_language", ""),
            "duration_seconds": result.get("duration_seconds", 0.0),
            "warnings": list(result.get("warnings") or []),
            "segments": [
                {"start": seg.start, "end": seg.end, "text": seg.text, "speaker": seg.speaker}
                for seg in segments
            ],
        }
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=target.parent, prefix=target.name + ".",
            suffix=".tmp", delete=False,
        ) as handle:
            temp_name = handle.name
            json.dump(payload, handle, ensure_ascii=False, allow_nan=False)
        os.replace(temp_name, target)
    except Exception as err:
        logger.warning("Could not save the transcript to %s: %s", target, err)
    finally:
        if temp_name and os.path.exists(temp_name):
            try:
                os.unlink(temp_name)
            except OSError:
                pass


def _is_number(value: object) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _segment_from(item: object) -> Segment:
    """A Segment from one saved entry; ValueError when it is not well formed."""
    if not isinstance(item, dict):
        raise ValueError("segment is not an object")
    start, end = item.get("start"), item.get("end")
    text, speaker = item.get("text"), item.get("speaker")
    if not (_is_number(start) and _is_number(end) and 0 <= start <= end):
        raise ValueError("bad segment times")
    if not isinstance(text, str) or not isinstance(speaker, str):
        raise ValueError("bad segment text or speaker")
    return Segment(float(start), float(end), text, speaker)


def _parse_sidecar(data: object, fingerprint: dict) -> dict | None:
    """The result a saved sidecar holds, None for a stale one; ValueError if malformed."""
    if not isinstance(data, dict) or data.get("version") != SIDECAR_VERSION:
        return None
    if data.get("fingerprint") != fingerprint:
        return None
    raw_segments, warnings = data.get("segments"), data.get("warnings")
    language, duration = data.get("detected_language"), data.get("duration_seconds")
    if not isinstance(raw_segments, list):
        raise ValueError("segments is not a list")
    if not isinstance(warnings, list) or not all(isinstance(w, str) for w in warnings):
        raise ValueError("warnings is not a list of strings")
    if not isinstance(language, str) or not _is_number(duration) or duration < 0:
        raise ValueError("bad language or duration")
    segments = [_segment_from(item) for item in raw_segments]
    return {
        "text": plain_text(segments),
        "diarised_text": labelled_text(segments),
        "duration_seconds": float(duration),
        "detected_language": language,
        "segments": segments,
        "warnings": list(warnings),
    }


def load_transcript(wav_path: str, fingerprint: dict) -> dict | None:
    """The saved transcriber result for a recording, or None when it cannot be trusted.

    None when the recording or sidecar is missing, the sidecar is older than the recording,
    of another version, made under a different fingerprint, or malformed.
    """
    path = sidecar_path(wav_path)
    try:
        if path.stat().st_mtime < Path(wav_path).stat().st_mtime:
            return None
        return _parse_sidecar(json.loads(path.read_text(encoding="utf-8")), fingerprint)
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as err:
        logger.info("Ignoring saved transcript %s: %s", path, err)
        return None
