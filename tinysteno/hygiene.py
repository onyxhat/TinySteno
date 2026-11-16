"""Remove lines Whisper invents when given silence or noise."""
import logging
import re
from collections import Counter

from tinysteno.transcript import Segment

logger = logging.getLogger(__name__)

_CREDIT_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(subtitles?|captions?|transcri(bed|ption)|translated)\s+(provided\s+)?by\b",
        r"\bthanks?(\s+you)?\s+for\s+watching\b",
        r"\bplease\s+subscribe\b",
        r"\bamara\.org\b",
    )
]
_STUCK_RUN = 5
_SHORT_WORDS = 5
_SHORT_REPEATS = 3
_ISOLATION_GAP = 10.0


def _normalise(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", "", text.lower()).split())


def _log_removed(reason: str, seg: Segment) -> None:
    logger.info("Dropped %s: speaker=%s start=%.1f text=%r", reason, seg.speaker, seg.start,
                seg.text)


def _drop_credits(segments: list[Segment]) -> list[Segment]:
    kept = []
    for seg in segments:
        if any(p.search(seg.text) for p in _CREDIT_PATTERNS):
            _log_removed("caption credit", seg)
        else:
            kept.append(seg)
    return kept


def _drop_stuck_runs(segments: list[Segment]) -> list[Segment]:
    kept = []
    i = 0
    while i < len(segments):
        norm = _normalise(segments[i].text)
        j = i + 1
        while norm and j < len(segments) and _normalise(segments[j].text) == norm:
            j += 1
        kept.append(segments[i])
        if j - i >= _STUCK_RUN:
            for dup in segments[i + 1:j]:
                _log_removed("stuck repeat", dup)
        else:
            kept.extend(segments[i + 1:j])
        i = j
    return kept


def _drop_stranded_repeats(segments: list[Segment]) -> list[Segment]:
    norms = [_normalise(s.text) for s in segments]
    counts = Counter(n for n in norms if n)
    kept = []
    for i, seg in enumerate(segments):
        norm = norms[i]
        candidate = (norm and len(norm.split()) <= _SHORT_WORDS
                     and counts[norm] >= _SHORT_REPEATS)
        if candidate:
            before = seg.start - segments[i - 1].end if i > 0 else _ISOLATION_GAP
            after = segments[i + 1].start - seg.end if i + 1 < len(segments) else _ISOLATION_GAP
            if before >= _ISOLATION_GAP and after >= _ISOLATION_GAP:
                _log_removed("stranded repeat", seg)
                continue
        kept.append(seg)
    return kept


def drop_hallucinations(segments: list[Segment]) -> list[Segment]:
    """Remove caption credits, stuck repeats and stranded short repeats from one channel.

    Takes the channel's segments in time order and returns a new list; the input is untouched.
    """
    return _drop_stranded_repeats(_drop_stuck_runs(_drop_credits(segments)))
