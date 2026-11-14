"""What the app says on the way out reflects everything that happened to this session's audio."""
# pylint: disable=redefined-outer-name
import os
import signal
import threading

import pytest

from tinysteno.pipeline import PipelineError, Stage
from tinysteno.tui import run_tui
from tinysteno.tui.app import TinyStenoApp
from tinysteno.tui.guard import RecorderGuard
from tinysteno.tui.home import HomeScreen
from tinysteno.tui.processing import ProcessingScreen
from tinysteno.tui.recording import RecordingScreen
from tests.tui_fakes import FakeClock, FakeRecorder, make_deps

CONFIG = {"persona": "default", "session_timeout_minutes": 120}


def _saved_message(wav) -> str:
    return f"Saved recording: {wav}\nProcess it with: tinysteno process {wav}"


async def _settle(app, pilot):
    await app.workers.wait_for_complete()
    await pilot.pause()


async def _record(pilot, app):
    await pilot.press("enter")
    await _settle(app, pilot)
    assert isinstance(app.screen, RecordingScreen)


async def test_quit_from_home_after_a_failed_save_still_says_where_the_audio_is(tmp_path):
    rec = FakeRecorder(tmp_path)
    rec.stop_error = RuntimeError("disk full")
    rec.raw_paths = (tmp_path / "mic.f32",)
    app = TinyStenoApp(CONFIG, make_deps(tmp_path, clock=FakeClock(), make_recorder=lambda: rec))
    async with app.run_test() as pilot:
        await _record(pilot, app)
        await pilot.press("enter")               # stop: the save fails, back to Home
        await _settle(app, pilot)
        assert isinstance(app.screen, HomeScreen)
        await pilot.press("q")
        await pilot.pause()
    message, code = app.final_result()
    kept = tmp_path / "mic.f32"
    assert message == f"Could not save the recording: disk full. Raw audio kept at {kept}"
    assert code == 1


async def test_quit_from_home_after_leaving_a_failed_processing_run(tmp_path):
    def fake(wav_path, name, persona, ts, reporter):  # pylint: disable=unused-argument
        raise PipelineError(Stage.SUMMARIZE) from ConnectionError("endpoint down")

    rec = FakeRecorder(tmp_path)
    deps = make_deps(tmp_path, clock=FakeClock(), make_recorder=lambda: rec, run_pipeline=fake)
    app = TinyStenoApp(CONFIG, deps)
    async with app.run_test() as pilot:
        await _record(pilot, app)
        await pilot.press("enter")
        await _settle(app, pilot)
        await _settle(app, pilot)
        assert isinstance(app.screen, ProcessingScreen)
        await pilot.press("n")                   # give up on it for now: back to Home
        await pilot.pause()
        assert isinstance(app.screen, HomeScreen)
        await pilot.press("q")
        await pilot.pause()
    assert app.final_result() == (_saved_message(rec.wav_path), 0)


async def test_a_processed_recording_is_not_mentioned_on_exit(tmp_path):
    rec = FakeRecorder(tmp_path)
    app = TinyStenoApp(CONFIG, make_deps(tmp_path, clock=FakeClock(), make_recorder=lambda: rec))
    async with app.run_test() as pilot:
        await _record(pilot, app)
        await pilot.press("enter")
        await _settle(app, pilot)
        await _settle(app, pilot)
        await pilot.press("q")
        await pilot.pause()
    assert app.final_result() == (None, 0)


async def test_a_retry_that_succeeds_clears_the_unprocessed_recording(tmp_path):
    attempts = []
    deps = make_deps(tmp_path)
    succeed = deps.run_pipeline

    def flaky(wav_path, name, persona, ts, reporter):
        attempts.append(1)
        if len(attempts) == 1:
            raise PipelineError(Stage.SUMMARIZE) from ConnectionError("endpoint down")
        return succeed(wav_path, name, persona, ts, reporter)

    deps.run_pipeline = flaky
    wav = tmp_path / "Old.wav"
    wav.write_bytes(b"")
    app = TinyStenoApp(CONFIG, deps)
    async with app.run_test() as pilot:
        app.process_file(str(wav), "default")
        await _settle(app, pilot)
        await pilot.press("r")
        await _settle(app, pilot)
        await pilot.press("q")
        await pilot.pause()
    assert len(attempts) == 2
    assert app.final_result() == (None, 0)


async def test_crash_while_processing_still_says_where_the_recording_is(tmp_path):
    gate = threading.Event()

    def slow(wav_path, name, persona, ts, reporter):  # pylint: disable=unused-argument
        gate.wait(5)
        raise RuntimeError("abandoned")

    def boom():
        raise RuntimeError("ui bug")

    rec = FakeRecorder(tmp_path)
    deps = make_deps(tmp_path, clock=FakeClock(), make_recorder=lambda: rec, run_pipeline=slow)
    app = TinyStenoApp(CONFIG, deps)
    try:
        with pytest.raises(RuntimeError, match="ui bug"):
            async with app.run_test() as pilot:
                await _record(pilot, app)
                await pilot.press("enter")
                await _settle(app, pilot)
                await pilot.pause()
                assert isinstance(app.screen, ProcessingScreen)
                app.call_later(boom)             # a UI callback raises: Textual exits
                await pilot.pause()
    finally:
        gate.set()
    message, _code = app.final_result()
    assert message == _saved_message(rec.wav_path)


async def test_recent_recording_cannot_be_processed_without_personas(tmp_path):
    (tmp_path / "Old.wav").write_bytes(b"")
    app = TinyStenoApp(CONFIG, make_deps(tmp_path, list_personas=list))
    async with app.run_test() as pilot:
        app.screen.query_one("#recent").focus()
        await pilot.press("enter")
        await pilot.pause()
        assert not app.events and isinstance(app.screen, HomeScreen)


@pytest.mark.skipif(not hasattr(signal, "SIGTERM"), reason="needs POSIX signals")
def test_kill_signal_during_the_crash_save_waits_for_the_save(tmp_path, monkeypatch, capsys):
    order = []
    rec = FakeRecorder(tmp_path)
    save = rec.stop

    def slow_stop():
        os.kill(os.getpid(), signal.SIGTERM)     # arrives while the WAV is being written
        for _ in range(1000):                    # give a pending Python handler every chance
            pass
        order.append("saved")
        return save()

    rec.stop = slow_stop

    def crashing_run(self):
        self.recorder_guard = RecorderGuard(rec, rec.wav_path)
        raise RuntimeError("ui bug")

    monkeypatch.setattr(TinyStenoApp, "run", crashing_run)
    previous = signal.signal(signal.SIGTERM, lambda *_: order.append("signal delivered"))
    try:
        with pytest.raises(RuntimeError, match="ui bug"):
            run_tui(CONFIG, make_deps(tmp_path))
    finally:
        signal.signal(signal.SIGTERM, previous)
    assert order == ["saved", "signal delivered"]
    assert f"Saved recording: {rec.wav_path}" in capsys.readouterr().err
