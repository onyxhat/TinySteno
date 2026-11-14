"""Recording screen: live clock and meters, abortable timeout, stop/discard/quit paths."""
# pylint: disable=redefined-outer-name
import threading

import pytest

from tinysteno.personas import PersonaInvalidError, PersonaNotFoundError
from tinysteno.tui.app import TinyStenoApp
from tinysteno.tui.home import HomeScreen
from tinysteno.tui.recording import RecordingScreen
from tests.tui_fakes import FakeClock, FakeRecorder, make_deps


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def rec(tmp_path):
    return FakeRecorder(tmp_path)


@pytest.fixture
def deps(tmp_path, clock, rec):
    return make_deps(tmp_path, clock=clock, make_recorder=lambda: rec)


@pytest.fixture
def app(deps):
    return TinyStenoApp({"persona": "default", "session_timeout_minutes": 120}, deps)


async def _recording(pilot, app):
    await pilot.press("enter")
    await app.workers.wait_for_complete()
    await pilot.pause()
    assert isinstance(app.screen, RecordingScreen)
    return app.screen


def _capture(app):
    """Record (message, severity) for every notification, and count bells."""
    notes: list[tuple[str, str]] = []
    bells: list[None] = []
    app.notify = lambda message, **kw: notes.append((message, kw.get("severity", "information")))
    app.bell = lambda: bells.append(None)
    return notes, bells


def _release_later(gate: threading.Event, delay: float = 0.2) -> None:
    timer = threading.Timer(delay, gate.set)
    timer.daemon = True
    timer.start()


# ── tests from the brief ─────────────────────────────────────────────────────


async def test_enter_stops_and_hands_wav_to_processing(app, rec):
    async with app.run_test() as pilot:
        await _recording(pilot, app)
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert rec.calls == ["start", "stop"]
        assert ("process", rec.wav_path, "default") in app.events


async def test_tick_shows_elapsed_and_countdown(app, clock):
    async with app.run_test() as pilot:
        screen = await _recording(pilot, app)
        clock.now += 2537                       # 42:17 in, 1:17:43 left of 2:00:00
        screen.tick()
        await pilot.pause()
        assert str(screen.query_one("#elapsed").value) == "42:17"
        assert "1:17:43" in str(screen.query_one("#timeout-label").content)
        assert not screen.has_class("-warning")


async def test_warning_window_then_auto_stop_processes_recording(app, clock, rec):
    async with app.run_test() as pilot:
        screen = await _recording(pilot, app)
        clock.now += 7200 - 30
        screen.tick()
        await pilot.pause()
        assert screen.has_class("-warning") and "stop" not in rec.calls
        clock.now += 31
        screen.tick()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert rec.calls == ["start", "stop"]
        assert ("process", rec.wav_path, "default") in app.events


async def test_k_cancels_timeout_so_recording_outlives_the_limit(app, clock, rec):
    async with app.run_test() as pilot:
        screen = await _recording(pilot, app)
        clock.now += 7200 - 30
        screen.tick()
        await pilot.press("k")
        clock.now += 3600
        screen.tick()
        await pilot.pause()
        assert "stop" not in rec.calls and not screen.has_class("-warning")
        assert "No time limit" in str(screen.query_one("#timeout-label").content)


async def test_plus_extends_by_15_minutes(app, clock):
    async with app.run_test() as pilot:
        screen = await _recording(pilot, app)
        clock.now += 7200 - 30
        await pilot.press("plus")
        screen.tick()
        await pilot.pause()
        assert screen.timer.remaining == 930 and not screen.has_class("-warning")


async def test_stop_pressed_twice_stops_once(app, rec):
    async with app.run_test() as pilot:
        await _recording(pilot, app)
        await pilot.press("s", "s")
        await app.workers.wait_for_complete()
        assert rec.calls.count("stop") == 1


async def test_discard_needs_confirmation(app, rec):
    async with app.run_test() as pilot:
        await _recording(pilot, app)
        await pilot.press("d", "n")
        await pilot.pause()
        assert "discard" not in rec.calls and isinstance(app.screen, RecordingScreen)
        await pilot.press("d", "y")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert rec.calls == ["start", "discard"] and isinstance(app.screen, HomeScreen)


async def test_start_failure_returns_home_with_error(app, rec):
    rec.start_error = RuntimeError("No audio input devices found")
    async with app.run_test() as pilot:
        notes, _ = _capture(app)
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert isinstance(app.screen, HomeScreen)
        assert ("No audio input devices found", "error") in notes


async def test_quit_while_recording_saves_wav_and_reports_path(app, rec):
    async with app.run_test() as pilot:
        await _recording(pilot, app)
        await pilot.press("q")
        await pilot.pause()
    assert rec.calls == ["start", "stop"]
    assert rec.wav_path in app.exit_message and "tinysteno process" in app.exit_message


async def test_each_stream_has_its_own_rolling_meter(app, rec):
    async with app.run_test() as pilot:
        screen = await _recording(pilot, app)
        for mic, system in [(0.1, 0.9), (0.2, 0.8), (0.3, 0.7)]:
            rec.mic_level, rec.system_level = mic, system
            screen.tick()
        await pilot.pause()
        assert list(screen.query_one("#mic-level").data)[-3:] == [0.1, 0.2, 0.3]
        assert list(screen.query_one("#system-level").data)[-3:] == [0.9, 0.8, 0.7]
        for _ in range(200):
            screen.tick()
        assert len(screen.query_one("#mic-level").data) == 120


async def test_without_system_audio_the_system_meter_is_replaced_by_a_label(app, rec):
    rec.has_loopback = False
    async with app.run_test() as pilot:
        screen = await _recording(pilot, app)
        screen.tick()
        await pilot.pause()
        assert not screen.query_one("#system-level").display
        assert screen.query_one("#system-unavailable").display
        assert screen.query_one("#mic-level").display


async def test_persona_timeout_overrides_global_in_the_tui(tmp_path, clock, rec):
    deps = make_deps(tmp_path, clock=clock, make_recorder=lambda: rec,
                     persona_timeouts={"default": 30})
    app = TinyStenoApp({"persona": "default", "session_timeout_minutes": 120}, deps)
    async with app.run_test() as pilot:
        screen = await _recording(pilot, app)
        assert screen.timer.limit_seconds == 1800


async def test_timeout_flag_overrides_persona_in_the_tui(tmp_path, clock, rec):
    deps = make_deps(tmp_path, clock=clock, make_recorder=lambda: rec,
                     persona_timeouts={"default": 30})
    app = TinyStenoApp({"persona": "default"}, deps, timeout_override=5)
    async with app.run_test() as pilot:
        screen = await _recording(pilot, app)
        assert screen.timer.limit_seconds == 300


async def test_invalid_persona_never_starts_the_recorder(tmp_path, rec):
    def broken(_slug):
        raise PersonaInvalidError("session_timeout_minutes must be a number >= 0")
    deps = make_deps(tmp_path, make_recorder=lambda: rec, load_persona=broken)
    app = TinyStenoApp({"persona": "default"}, deps)
    async with app.run_test() as pilot:
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert isinstance(app.screen, HomeScreen) and rec.calls == []


async def test_zero_timeout_config_means_no_limit(deps, clock, rec):
    app = TinyStenoApp({"persona": "default", "session_timeout_minutes": 0}, deps)
    async with app.run_test() as pilot:
        screen = await _recording(pilot, app)
        clock.now += 10 ** 6
        screen.tick()
        assert "stop" not in rec.calls


# ── edge cases resolved by the controller ────────────────────────────────────


async def test_autostart_with_missing_persona_lands_on_home_with_error(tmp_path, rec):
    def missing(slug):
        raise PersonaNotFoundError(f"Persona '{slug}' not found")
    deps = make_deps(tmp_path, make_recorder=lambda: rec, load_persona=missing)
    app = TinyStenoApp({"persona": "default"}, deps, autostart=True, persona="nope")
    async with app.run_test() as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert isinstance(app.screen, HomeScreen) and rec.calls == []
    assert app.exit_message is None


async def test_autostart_persona_error_is_notified(tmp_path, rec, monkeypatch):
    notes: list[tuple[str, str]] = []
    monkeypatch.setattr(
        TinyStenoApp, "notify",
        lambda self, message, **kw: notes.append((message, kw.get("severity"))),
    )

    def missing(slug):
        raise PersonaNotFoundError(f"Persona '{slug}' not found")
    deps = make_deps(tmp_path, make_recorder=lambda: rec, load_persona=missing)
    app = TinyStenoApp({"persona": "default"}, deps, autostart=True, persona="nope")
    async with app.run_test() as pilot:
        await pilot.pause()
        assert isinstance(app.screen, HomeScreen)
    assert any("nope" in message and severity == "error" for message, severity in notes)


async def test_stop_keys_while_starting_are_ignored_with_a_notice(app, rec):
    # A second Enter from Home must not end the recording the moment it starts.
    rec.start_gate = threading.Event()
    async with app.run_test() as pilot:
        await pilot.press("enter")
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, RecordingScreen)
        assert "Starting" in str(screen.query_one("#rec-indicator").content)
        notes, _ = _capture(app)
        await pilot.press("s", "enter")
        await pilot.pause()
        assert notes == [("Still starting…", "information")] * 2
        rec.start_gate.set()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert rec.calls == ["start"] and isinstance(app.screen, RecordingScreen)
        assert "REC" in str(screen.query_one("#rec-indicator").content)
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert rec.calls == ["start", "stop"]
        assert app.events.count(("process", rec.wav_path, "default")) == 1


async def test_expiry_and_key_on_the_same_tick_stop_once(app, clock, rec):
    async with app.run_test() as pilot:
        screen = await _recording(pilot, app)
        clock.now += 7200
        screen.tick()
        screen.action_stop()
        screen.tick()
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert rec.calls == ["start", "stop"]
        assert app.events.count(("process", rec.wav_path, "default")) == 1


async def test_quit_while_starting_still_saves_the_wav(app, rec):
    rec.start_gate = threading.Event()
    async with app.run_test() as pilot:
        await pilot.press("enter")
        await pilot.pause()
        _release_later(rec.start_gate)
        await pilot.press("q")
        await pilot.pause()
    assert rec.calls == ["start", "stop"]
    assert rec.wav_path in app.exit_message
    assert not any(event[0] == "process" for event in app.events)


async def test_quit_while_stop_in_flight_saves_once_and_skips_processing(app, rec):
    rec.stop_gate = threading.Event()
    async with app.run_test() as pilot:
        await _recording(pilot, app)
        await pilot.press("s")
        _release_later(rec.stop_gate)
        await pilot.press("q")
        await pilot.pause()
    assert rec.calls == ["start", "stop"]
    assert rec.wav_path in app.exit_message
    assert not any(event[0] == "process" for event in app.events)


async def test_quit_when_stop_fails_reports_the_failure_and_exits_1(app, rec, tmp_path):
    rec.stop_error = RuntimeError("disk full")
    rec.raw_paths = (tmp_path / "mic.f32", tmp_path / "system.f32")
    async with app.run_test() as pilot:
        await _recording(pilot, app)
        await pilot.press("q")
        await pilot.pause()
    assert "Could not save the recording: disk full" in app.exit_message
    assert all(str(p) in app.exit_message for p in rec.raw_paths)
    assert "Saved recording" not in app.exit_message
    assert app.return_code == 1


async def test_quit_while_failing_stop_in_flight_reports_the_failure_and_exits_1(app, rec,
                                                                                  tmp_path):
    rec.stop_error = RuntimeError("disk full")
    rec.raw_paths = (tmp_path / "mic.f32",)
    rec.stop_gate = threading.Event()
    async with app.run_test() as pilot:
        await _recording(pilot, app)
        await pilot.press("s")
        _release_later(rec.stop_gate)
        await pilot.press("q")
        await pilot.pause()
    assert rec.calls == ["start", "stop"]
    assert "Could not save the recording: disk full" in app.exit_message
    assert str(rec.raw_paths[0]) in app.exit_message
    assert "Saved recording" not in app.exit_message
    assert app.return_code == 1


async def test_quit_when_start_fails_claims_nothing(app, rec):
    rec.start_gate = threading.Event()
    rec.start_error = RuntimeError("No audio input devices found")
    async with app.run_test() as pilot:
        await pilot.press("enter")
        await pilot.pause()
        _release_later(rec.start_gate)
        await pilot.press("q")
        await pilot.pause()
    assert rec.calls == ["start"]
    assert app.exit_message is None


async def test_quit_with_no_audio_captured_claims_nothing(app, rec):
    rec.stop_result = False
    async with app.run_test() as pilot:
        await _recording(pilot, app)
        await pilot.press("q")
        await pilot.pause()
    assert rec.calls == ["start", "stop"]
    assert app.exit_message is None
    assert app.return_code == 0


async def test_ctrl_q_with_discard_dialog_open_saves_the_wav(app, rec):
    # The modal swallows plain "q"; ctrl+q is a priority binding and reaches the app.
    async with app.run_test() as pilot:
        await _recording(pilot, app)
        await pilot.press("d")
        await pilot.pause()
        await pilot.press("ctrl+q")
        await pilot.pause()
    assert rec.calls == ["start", "stop"]
    assert rec.wav_path in app.exit_message


async def test_no_audio_captured_returns_home_with_error(app, rec):
    rec.stop_result = False
    async with app.run_test() as pilot:
        await _recording(pilot, app)
        notes, _ = _capture(app)
        await pilot.press("s")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert isinstance(app.screen, HomeScreen)
        assert ("No audio was captured.", "error") in notes
        assert not any(event[0] == "process" for event in app.events)


async def test_stop_error_is_notified_with_the_raw_audio_location(app, rec, tmp_path):
    raw = tmp_path / "mic.f32"
    try:
        raise OSError("disk full")
    except OSError as cause:
        rec.stop_error = RuntimeError(
            f"Failed to save recording: disk full. Raw audio kept at {raw}"
        )
        rec.stop_error.__cause__ = cause
    rec.raw_paths = (raw,)
    async with app.run_test() as pilot:
        await _recording(pilot, app)
        notes, _ = _capture(app)
        await pilot.press("s")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert isinstance(app.screen, HomeScreen) and app.is_running
        assert notes == [
            (f"Could not save the recording: disk full. Raw audio kept at {raw}", "error")
        ]
        assert not any(event[0] == "process" for event in app.events)


async def test_discard_error_is_notified_and_returns_home(app, rec):
    rec.discard_error = OSError("device busy")
    async with app.run_test() as pilot:
        await _recording(pilot, app)
        notes, _ = _capture(app)
        await pilot.press("d", "y")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert isinstance(app.screen, HomeScreen) and app.is_running
        assert any("device busy" in m and s == "error" for m, s in notes)


async def test_discard_notifies(app, rec):
    async with app.run_test() as pilot:
        await _recording(pilot, app)
        notes, _ = _capture(app)
        await pilot.press("d", "y")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert ("Recording discarded.", "information") in notes
        assert rec.calls == ["start", "discard"]


async def test_warning_notifies_and_rings_once_when_entering_the_window(app, clock):
    async with app.run_test() as pilot:
        screen = await _recording(pilot, app)
        notes, bells = _capture(app)
        clock.now += 7200 - 59
        screen.tick()
        clock.now += 10
        screen.tick()
        await pilot.pause()
        warnings = [n for n in notes if n[1] == "warning"]
        assert warnings == [("Auto-stopping in 60s. Press K to keep recording.", "warning")]
        assert len(bells) == 1


async def test_k_and_plus_notify(app):
    async with app.run_test() as pilot:
        await _recording(pilot, app)
        notes, _ = _capture(app)
        await pilot.press("plus")
        await pilot.press("k")
        await pilot.press("plus")
        await pilot.pause()
        assert [m for m, _ in notes] == [
            "Extended by 15 minutes.",
            "Timeout cancelled. Recording until you stop.",
            "No time limit is set.",
        ]


async def test_no_time_limit_hides_the_bar(deps):
    app = TinyStenoApp({"persona": "default", "session_timeout_minutes": 0}, deps)
    async with app.run_test() as pilot:
        screen = await _recording(pilot, app)
        screen.tick()
        await pilot.pause()
        assert "No time limit" in str(screen.query_one("#timeout-label").content)
        assert not screen.query_one("#timeout-bar").display


async def test_tick_is_a_no_op_when_not_recording(app, rec):
    RecordingScreen("Sync", "default").tick()   # never mounted
    async with app.run_test() as pilot:
        screen = await _recording(pilot, app)
        await pilot.press("s")
        await app.workers.wait_for_complete()
        await pilot.pause()
        screen.tick()
        assert rec.calls == ["start", "stop"]
