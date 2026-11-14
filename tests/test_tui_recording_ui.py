"""Recording screen: discard dialog keys, footer labels, system-audio reason, live smoke run."""
# pylint: disable=redefined-outer-name
import pytest
from textual.widgets import Footer

from tinysteno.tui.app import TinyStenoApp
from tinysteno.tui.recording import ConfirmDiscard, RecordingScreen
from tests.tui_fakes import FakeClock, FakeRecorder, make_deps

CONFIG = {"persona": "default", "session_timeout_minutes": 120}


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def rec(tmp_path):
    return FakeRecorder(tmp_path)


@pytest.fixture
def app(tmp_path, clock, rec):
    return TinyStenoApp(CONFIG, make_deps(tmp_path, clock=clock, make_recorder=lambda: rec))


async def _recording(pilot, app):
    await pilot.press("enter")
    await app.workers.wait_for_complete()
    await pilot.pause()
    assert isinstance(app.screen, RecordingScreen)
    return app.screen


def _shown_bindings(screen) -> dict[str, str]:
    """key_display (or key) -> description, for what the footer shows on this screen."""
    return {
        active.binding.key_display or key: active.binding.description
        for key, active in screen.active_bindings.items()
        if active.binding.show
    }


async def test_discard_dialog_has_its_own_footer_and_key_hints(app):
    async with app.run_test() as pilot:
        await _recording(pilot, app)
        await pilot.press("d")
        await pilot.pause()
        dialog = app.screen
        assert isinstance(dialog, ConfirmDiscard)
        assert dialog.query(Footer)
        shown = _shown_bindings(dialog)
        assert shown["y"] == "Discard" and shown["n"] == "Keep recording"
        assert "Stop" not in shown.values()


async def test_plain_q_in_the_discard_dialog_saves_and_quits(app, rec):
    async with app.run_test() as pilot:
        await _recording(pilot, app)
        await pilot.press("d")
        await pilot.pause()
        await pilot.press("q")
        await pilot.pause()
    assert rec.calls == ["start", "stop"]
    assert rec.wav_path in app.exit_message and "tinysteno process" in app.exit_message
    assert app.return_code == 0


async def test_extend_key_label_reads_plus_15_min(app):
    async with app.run_test() as pilot:
        screen = await _recording(pilot, app)
        assert _shown_bindings(screen)["+"] == "15 min"


async def test_system_audio_failure_reason_is_shown_once(app, rec):
    rec.has_loopback = False
    rec.loopback_error = "Screen Recording permission denied"
    notes = []
    async with app.run_test() as pilot:
        app.notify = lambda message, **kw: notes.append((message, kw.get("severity")))
        screen = await _recording(pilot, app)
        screen.tick()
        screen.tick()
        await pilot.pause()
    assert notes == [
        ("System audio unavailable: Screen Recording permission denied", "warning")
    ]


async def test_no_reason_no_system_audio_notice(app, rec):
    rec.has_loopback = False
    notes = []
    async with app.run_test() as pilot:
        app.notify = lambda message, **kw: notes.append((message, kw.get("severity")))
        await _recording(pilot, app)
    assert not notes


async def test_live_updates_smoke_pulse_and_warning_transition(tmp_path, clock, rec):
    deps = make_deps(tmp_path, clock=clock, make_recorder=lambda: rec, live_updates=True)
    app = TinyStenoApp(CONFIG, deps)
    async with app.run_test() as pilot:
        screen = await _recording(pilot, app)
        await pilot.pause(1.0)                 # the 0.1 s ticker and at least one pulse
        clock.now += 7200 - 30                 # into the last minute
        await pilot.pause(0.6)
        assert screen.has_class("-warning")
        assert len(screen.query_one("#mic-level").data) >= 5
        assert screen.query_one("#rec-indicator").styles.opacity < 1.0  # it has pulsed
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
    assert rec.calls == ["start", "stop"]


async def test_live_updates_refresh_the_meters_without_manual_ticks(tmp_path, clock, rec):
    deps = make_deps(tmp_path, clock=clock, make_recorder=lambda: rec, live_updates=True)
    app = TinyStenoApp({"persona": "default"}, deps)
    async with app.run_test() as pilot:
        screen = await _recording(pilot, app)
        await pilot.pause(0.5)
        assert len(screen.query_one("#mic-level").data) >= 3
