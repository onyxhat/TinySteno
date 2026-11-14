"""Recording session timeout: a pure countdown that never sleeps or spawns threads."""

import time
from collections.abc import Callable

DEFAULT_SESSION_TIMEOUT_MINUTES = 75
TIMEOUT_WARNING_SECONDS = 60
EXTEND_SECONDS = 900


class SessionTimer:
    """Tracks elapsed recording time against an optional limit.

    The timer only reads the injected clock, so callers decide when to poll it.
    """

    def __init__(self, limit_seconds: float | None,
                 clock: Callable[[], float] = time.monotonic):
        self._limit = limit_seconds
        self._clock = clock
        self._started: float | None = None

    def start(self) -> None:
        self._started = self._clock()

    def cancel(self) -> None:
        """Remove the limit; elapsed time keeps counting."""
        self._limit = None

    def extend(self, seconds: float = EXTEND_SECONDS) -> None:
        """Add time to the limit. No effect once the limit has been cancelled."""
        if self._limit is not None:
            self._limit += seconds

    @property
    def elapsed(self) -> float:
        if self._started is None:
            return 0
        return self._clock() - self._started

    @property
    def limit_seconds(self) -> float | None:
        return self._limit

    @property
    def remaining(self) -> float | None:
        if self._limit is None:
            return None
        return max(0, self._limit - self.elapsed)

    @property
    def expired(self) -> bool:
        return self.remaining == 0

    @property
    def in_warning(self) -> bool:
        return self.remaining is not None and self.remaining <= TIMEOUT_WARNING_SECONDS


def resolve_timeout_seconds(config: dict, persona_minutes: float | None = None,
                            cli_minutes: float | None = None) -> float | None:
    """Pick the timeout in seconds: CLI flag, then persona, then config, then default.

    Returns None when the chosen value is 0, which disables the timeout.
    """
    minutes = next(
        value for value in (
            cli_minutes,
            persona_minutes,
            config.get("session_timeout_minutes"),
            DEFAULT_SESSION_TIMEOUT_MINUTES,
        )
        if value is not None
    )
    return None if minutes == 0 else minutes * 60


def format_clock(seconds: float) -> str:
    """Format a duration as MM:SS, or H:MM:SS from one hour."""
    total = max(0, int(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"
