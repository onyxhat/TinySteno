"""Decide whether a diarizer's speaker split can be trusted, and relabel segments with it.

Raw diarizer output on real meetings invents phantom speakers from short noises and sometimes
splits one voice in two, so the answer is "keep the channel label" whenever in doubt. This
module is pure policy: it has no dependency on any diarization library.

The clustering-threshold ramp, the phantom-cluster and dominance rules are ported from
OpenWhispr's ``diarizationPolicy.js`` (MIT). Keeping the base channel label on the dominant
voice is ported from StenoAI (MIT).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from functools import partial
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # transcript.py imports this module, so avoid a runtime cycle
    from tinysteno.transcript import Segment

MIN_AUDIO_SECONDS = 30.0
_THRESHOLD_LOW = 0.6
_THRESHOLD_HIGH = 0.7
_RAMP_START_SECONDS = 900.0
_RAMP_END_SECONDS = 3600.0
_PHANTOM_MIN_SECONDS = 1.0
_PHANTOM_MIN_SHARE = 0.10
_PHANTOM_MEAN_TURN_SECONDS = 3.0
_DOMINANCE_SHARE = 0.93
_NEAREST_TURN_SECONDS = 2.0


class DiarizerUnavailable(RuntimeError):
    """Speaker separation cannot run; the message is meant to be shown to the user."""


@dataclass(frozen=True)
class Turn:
    """One span of a single diarized speaker."""
    start: float
    end: float
    speaker: int

    @property
    def duration(self) -> float:
        """Length of the turn in seconds."""
        return self.end - self.start


def cluster_threshold(duration_seconds: float) -> float:
    """Clustering threshold: 0.6 up to 15 minutes, rising linearly to 0.7 at 60 minutes."""
    if duration_seconds <= _RAMP_START_SECONDS:
        return _THRESHOLD_LOW
    if duration_seconds >= _RAMP_END_SECONDS:
        return _THRESHOLD_HIGH
    fraction = (duration_seconds - _RAMP_START_SECONDS) / (_RAMP_END_SECONDS - _RAMP_START_SECONDS)
    return _THRESHOLD_LOW + fraction * (_THRESHOLD_HIGH - _THRESHOLD_LOW)


def _speech_by_speaker(turns: list[Turn]) -> dict[int, float]:
    totals: dict[int, float] = defaultdict(float)
    for turn in turns:
        totals[turn.speaker] += turn.duration
    return dict(totals)


def _drop_phantoms(turns: list[Turn]) -> list[Turn]:
    totals = _speech_by_speaker(turns)
    everything = sum(totals.values())
    counts: dict[int, int] = defaultdict(int)
    for turn in turns:
        counts[turn.speaker] += 1
    phantoms = {
        speaker for speaker, total in totals.items()
        if total < _PHANTOM_MIN_SECONDS
        or (total / everything < _PHANTOM_MIN_SHARE
            and total / counts[speaker] < _PHANTOM_MEAN_TURN_SECONDS)
    }
    return [t for t in turns if t.speaker not in phantoms]


def _gap(first: Turn, second: Turn) -> float:
    return max(0.0, first.start - second.end, second.start - first.end)


def _fold(turns: list[Turn], max_speakers: int) -> list[Turn]:
    totals = _speech_by_speaker(turns)
    first_start: dict[int, float] = {}
    for turn in turns:
        first_start[turn.speaker] = min(turn.start, first_start.get(turn.speaker, turn.start))
    ranked = sorted(totals, key=lambda s: (-totals[s], first_start[s], s))
    if len(ranked) <= max_speakers:
        return turns
    kept = set(ranked[:max_speakers])
    anchors = sorted((t for t in turns if t.speaker in kept), key=lambda t: (t.start, t.end))
    folded = []
    for turn in turns:
        if turn.speaker in kept:
            folded.append(turn)
            continue
        nearest = min(anchors, key=partial(_gap, turn))  # min keeps the earliest on ties
        folded.append(replace(turn, speaker=nearest.speaker))
    return folded


def apply_policy(turns: list[Turn], *, max_speakers: int | None = None) -> list[Turn] | None:
    """Clean up diarizer turns, or return None to keep the plain channel labels."""
    cap = None if max_speakers is None else max(1, max_speakers)
    remaining = _drop_phantoms([t for t in turns if t.end > t.start])
    if cap is not None and remaining:
        remaining = _fold(remaining, cap)
    if not remaining or cap == 1:
        return None
    totals = _speech_by_speaker(remaining)
    if len(totals) < 2 or max(totals.values()) / sum(totals.values()) >= _DOMINANCE_SHARE:
        return None
    first_start: dict[int, float] = {}
    for turn in remaining:
        first_start[turn.speaker] = min(turn.start, first_start.get(turn.speaker, turn.start))
    ranked = sorted(totals, key=lambda s: (-totals[s], first_start[s]))
    renumber = {old: new for new, old in enumerate(ranked)}
    out = [replace(t, speaker=renumber[t.speaker]) for t in remaining]
    return sorted(out, key=lambda t: (t.start, t.end))


def _speaker_for(segment: Segment, turns: list[Turn]) -> int:
    overlap: dict[int, float] = defaultdict(float)
    for turn in turns:
        shared = min(segment.end, turn.end) - max(segment.start, turn.start)
        if shared > 0:
            overlap[turn.speaker] += shared
    if overlap:
        return min(overlap, key=lambda s: (-overlap[s], s))
    middle = (segment.start + segment.end) / 2

    def distance(turn: Turn) -> float:
        return abs((turn.start + turn.end) / 2 - middle)

    nearest = min(turns, key=distance, default=None)
    if nearest is not None and distance(nearest) <= _NEAREST_TURN_SECONDS:
        return nearest.speaker
    return 0


def label_segments(segments: list[Segment], turns: list[Turn], base_label: str) -> list[Segment]:
    """Give each segment on the base channel the speaker of the turn it overlaps most."""
    out = []
    for segment in segments:
        if segment.speaker != base_label:
            out.append(segment)
            continue
        speaker = _speaker_for(segment, turns)
        label = base_label if speaker == 0 else f"Speaker {speaker + 1}"
        out.append(replace(segment, speaker=label))
    return out
