"""No UI path loses a recording: crashes, closed terminals and kill signals still save it."""
# pylint: disable=redefined-outer-name
import os
import signal

import pytest

from tinysteno.tui.app import TinyStenoApp
from tinysteno.tui.recording import RecordingScreen
from tests.tui_fakes import FakeClock, FakeRecorder, make_deps

CONFIG = {"persona": "default", "session_timeout_minutes": 120}
EXIT_SIGNALS = [getattr(signal, n) for n in ("SIGHUP", "SIGTERM") if hasattr(signal, n)]


class LevelsBreak(FakeRecorder):
    """A recorder whose level read blows up inside the UI's tick() once ``explode`` is set."""

    explode = False

    @property
    def mic_level(self) -> float:
        """Fail like a bug in a UI callback would."""
        if self.explode:
            raise RuntimeError("meter exploded")
        return 0.1

    @mic_level.setter
    def mic_level(self, _value) -> None:
        pass


async def _crash_in_tick(app, rec):
    with pytest.raises(RuntimeError, match="meter exploded"):
        async with app.run_test() as pilot:
            screen = await _recording(pilot, app)
            rec.explode = True
            app.call_later(screen.tick)        # a UI callback raises: Textual exits
            await pilot.pause()


@pytest.fixture
def rec(tmp_path):
    return FakeRecorder(tmp_path)


@pytest.fixture
def app(tmp_path, rec):
    return TinyStenoApp(CONFIG, make_deps(tmp_path, clock=FakeClock(), make_recorder=lambda: rec))


async def _recording(pilot, app):
    await pilot.press("enter")
    await app.workers.wait_for_complete()
    await pilot.pause()
    assert isinstance(app.screen, RecordingScreen)
    return app.screen


async def test_crash_in_tick_while_recording_still_saves_the_wav(tmp_path):
    rec = LevelsBreak(tmp_path)
    app = TinyStenoApp(CONFIG, make_deps(tmp_path, make_recorder=lambda: rec))
    await _crash_in_tick(app, rec)
    assert rec.calls == ["start"]              # nothing in the UI stopped it
    message, code = app.final_result()
    assert rec.calls == ["start", "stop"]
    assert message == (
        f"Saved recording: {rec.wav_path}\nProcess it with: tinysteno process {rec.wav_path}"
    )
    assert code == 1                           # the app did crash
    assert app.final_result() == (message, 1)  # asking again gives the same answer
    assert rec.calls == ["start", "stop"]      # and the safety net stopped it only once


async def test_crash_with_a_failing_save_reports_the_raw_audio(tmp_path):
    rec = LevelsBreak(tmp_path)
    rec.stop_error = RuntimeError("disk full")
    rec.raw_paths = (tmp_path / "mic.f32",)
    app = TinyStenoApp(CONFIG, make_deps(tmp_path, make_recorder=lambda: rec))
    await _crash_in_tick(app, rec)
    message, code = app.final_result()
    assert message == (
        f"Could not save the recording: disk full. Raw audio kept at {rec.raw_paths[0]}"
    )
    assert code == 1


async def test_safety_net_leaves_a_settled_recording_alone(app, rec):
    async with app.run_test() as pilot:
        await _recording(pilot, app)
        await pilot.press("d", "y")            # discarded on purpose
        await app.workers.wait_for_complete()
        await pilot.pause()
    assert app.final_result() == (None, 0)
    assert rec.calls == ["start", "discard"]


async def test_safety_net_does_not_double_report_a_quit(app, rec):
    async with app.run_test() as pilot:
        await _recording(pilot, app)
        await pilot.press("q")
        await pilot.pause()
    message, code = app.final_result()
    assert rec.calls == ["start", "stop"] and code == 0
    assert message.count("Saved recording") == 1


@pytest.mark.parametrize("sig", EXIT_SIGNALS, ids=lambda s: s.name)
async def test_exit_signal_while_recording_saves_like_q(app, rec, sig):
    previous = signal.getsignal(sig)
    async with app.run_test() as pilot:
        await _recording(pilot, app)
        assert sig in app.exit_signals         # else the kill below would end the test run
        os.kill(os.getpid(), sig)
        await pilot.pause(0.2)
        await app.workers.wait_for_complete()
        await pilot.pause()
    assert rec.calls == ["start", "stop"]
    assert rec.wav_path in app.exit_message and "tinysteno process" in app.exit_message
    assert app.final_result()[1] == 0
    assert signal.getsignal(sig) == previous   # handler restored on exit


async def test_exit_signal_on_home_just_quits(app, rec):
    async with app.run_test() as pilot:
        await pilot.pause()
        app._on_exit_signal()                  # pylint: disable=protected-access
        await pilot.pause()
    assert not app.is_running and rec.calls == []
    assert app.exit_message is None
