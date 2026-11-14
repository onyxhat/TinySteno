"""Tests for the recording session timer."""
import pytest

from tinysteno.session import (
    DEFAULT_SESSION_TIMEOUT_MINUTES, SessionTimer, format_clock, resolve_timeout_seconds,
)


class FakeClock:
    """Manually advanced monotonic clock."""
    def __init__(self):
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _started(limit):
    clock = FakeClock()
    timer = SessionTimer(limit, clock=clock)
    timer.start()
    return timer, clock


def test_counts_down_and_expires():
    timer, clock = _started(600)
    clock.now += 599
    assert timer.elapsed == 599 and timer.remaining == 1 and not timer.expired
    clock.now += 5
    assert timer.remaining == 0 and timer.expired


def test_no_limit_never_expires():
    timer, clock = _started(None)
    clock.now += 10 ** 6
    assert timer.remaining is None and not timer.expired and not timer.in_warning


def test_warning_window_is_last_60_seconds():
    timer, clock = _started(600)
    clock.now += 539
    assert not timer.in_warning
    clock.now += 1
    assert timer.in_warning


def test_cancel_removes_limit_and_keeps_elapsed():
    timer, clock = _started(600)
    clock.now += 590
    timer.cancel()
    clock.now += 100
    assert timer.limit_seconds is None and not timer.expired and timer.elapsed == 690


def test_extend_adds_15_minutes_by_default():
    timer, clock = _started(600)
    clock.now += 590
    timer.extend()
    assert timer.remaining == 910 and not timer.in_warning


def test_extend_after_cancel_stays_unlimited():
    timer, _ = _started(600)
    timer.cancel()
    timer.extend()
    assert timer.limit_seconds is None


def test_before_start_nothing_has_elapsed():
    timer = SessionTimer(600, clock=FakeClock())
    assert timer.elapsed == 0 and timer.remaining == 600 and not timer.expired


def test_default_is_75_minutes():
    assert DEFAULT_SESSION_TIMEOUT_MINUTES == 75
    assert resolve_timeout_seconds({}) == 4500


@pytest.mark.parametrize("config, persona, cli, expected", [
    ({"session_timeout_minutes": 90}, None, None, 5400),     # global config
    ({"session_timeout_minutes": 0.5}, None, None, 30),
    ({"session_timeout_minutes": 0}, None, None, None),      # 0 disables
    ({"session_timeout_minutes": 90}, 30, None, 1800),       # persona overrides global
    ({}, 30, None, 1800),                                    # persona overrides default
    ({"session_timeout_minutes": 90}, 0, None, None),        # persona can disable
    ({"session_timeout_minutes": 0}, 45, None, 2700),        # persona can re-enable
    ({"session_timeout_minutes": 90}, 30, 10, 600),          # flag overrides persona
    ({"session_timeout_minutes": 90}, 30, 0, None),          # flag can disable
])
def test_resolve_timeout_precedence(config, persona, cli, expected):
    assert resolve_timeout_seconds(config, persona, cli) == expected


@pytest.mark.parametrize("seconds, text", [
    (0, "00:00"), (59.9, "00:59"), (3599, "59:59"), (3600, "1:00:00"), (4663, "1:17:43"),
])
def test_format_clock(seconds, text):
    assert format_clock(seconds) == text
