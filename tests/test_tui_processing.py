"""Processing screen: stage checklist, no-speech, failure, retry and quit paths."""
import threading
import time

import pytest

from tinysteno.personas import PersonaInvalidError
from tinysteno.pipeline import PipelineError, PipelineResult, Stage
from tinysteno.tui.app import TinyStenoApp
from tinysteno.tui.home import HomeScreen
from tinysteno.tui.processing import ProcessingScreen
from tests.tui_fakes import make_deps

CONFIG = {"persona": "default", "session_timeout_minutes": 120}
RESULT = PipelineResult(
    wav_path="x.wav", note_path="Sync.md", title="Sync", duration_seconds=61.0, no_speech=False
)


def _wav(tmp_path):
    wav = tmp_path / "Meeting.wav"
    wav.write_bytes(b"")
    return wav


def _text(screen, selector):
    return str(screen.query_one(selector).content)


async def _settle(app, pilot):
    await app.workers.wait_for_complete()
    await pilot.pause()


async def test_success_marks_all_stages_done_and_shows_result(tmp_path):
    def fake(wav, name, persona, ts, reporter):  # pylint: disable=unused-argument
        for stage in Stage:
            reporter.stage_started(stage)
            if stage is Stage.TRANSCRIBE:
                reporter.stage_progress(stage, 0.5)
            reporter.stage_finished(stage)
        return RESULT

    app = TinyStenoApp(CONFIG, make_deps(tmp_path, run_pipeline=fake))
    async with app.run_test() as pilot:
        app.process_file(str(_wav(tmp_path)), "default")
        await _settle(app, pilot)
        for stage in Stage:
            assert app.screen.query_one(f"#stage-{stage.value}").has_class("-done")
        assert ("result", RESULT) in app.events


async def test_progress_fraction_drives_the_transcription_bar(tmp_path):
    gate = threading.Event()

    def fake(wav, name, persona, ts, reporter):  # pylint: disable=unused-argument
        reporter.stage_started(Stage.TRANSCRIBE)
        reporter.stage_progress(Stage.TRANSCRIBE, 0.61)
        gate.wait(5)
        return RESULT

    app = TinyStenoApp(CONFIG, make_deps(tmp_path, run_pipeline=fake))
    async with app.run_test() as pilot:
        app.process_file(str(_wav(tmp_path)), "default")
        for _ in range(50):
            await pilot.pause(0.02)
            if app.screen.query_one("#transcribe-bar").progress == pytest.approx(0.61):
                break
        screen = app.screen
        assert screen.query_one("#stage-transcribe").has_class("-active")
        assert screen.query_one("#transcribe-bar").progress == pytest.approx(0.61)
        assert screen.query_one("#stage-summarize").has_class("-pending")
        gate.set()
        await _settle(app, pilot)


async def test_no_speech_keeps_audio_and_offers_home(tmp_path):
    wav = _wav(tmp_path)

    def fake(wav_path, name, persona, ts, reporter):  # pylint: disable=unused-argument
        reporter.stage_started(Stage.TRANSCRIBE)
        reporter.stage_finished(Stage.TRANSCRIBE)
        return PipelineResult(str(wav), None, None, 3.0, True)

    app = TinyStenoApp(CONFIG, make_deps(tmp_path, run_pipeline=fake))
    async with app.run_test() as pilot:
        app.process_file(str(wav), "default")
        await _settle(app, pilot)
        screen = app.screen
        assert "No speech detected" in _text(screen, "#error")
        assert str(wav) in _text(screen, "#error")
        assert screen.query_one("#stage-summarize").has_class("-skipped")
        assert not any(event[0] == "result" for event in app.events)
        await pilot.press("r")  # nothing to retry
        await pilot.press("n")
        await pilot.pause()
        assert isinstance(app.screen, HomeScreen)
        assert sum(isinstance(s, HomeScreen) for s in app.screen_stack) == 1


async def test_failure_shows_stage_cause_and_wav_then_retry_succeeds(tmp_path):
    wav = _wav(tmp_path)
    attempts = []

    def fake(wav_path, name, persona, ts, reporter):  # pylint: disable=unused-argument
        attempts.append(1)
        if len(attempts) == 1:
            reporter.stage_started(Stage.TRANSCRIBE)
            reporter.stage_finished(Stage.TRANSCRIBE)
            reporter.stage_started(Stage.SUMMARIZE)
            raise PipelineError(Stage.SUMMARIZE) from ConnectionError("endpoint down")
        return RESULT

    app = TinyStenoApp(CONFIG, make_deps(tmp_path, run_pipeline=fake))
    async with app.run_test() as pilot:
        app.process_file(str(wav), "default")
        await _settle(app, pilot)
        screen = app.screen
        error = _text(screen, "#error")
        assert "Summarizing failed" in error and "endpoint down" in error and str(wav) in error
        assert screen.query_one("#stage-summarize").has_class("-failed")
        await pilot.press("r")
        await _settle(app, pilot)
        assert ("result", RESULT) in app.events and len(attempts) == 2


async def test_retry_resets_rows_and_error_while_running(tmp_path):
    gate = threading.Event()
    attempts = []

    def fake(wav_path, name, persona, ts, reporter):  # pylint: disable=unused-argument
        attempts.append(1)
        if len(attempts) == 1:
            raise PipelineError(Stage.TRANSCRIBE) from RuntimeError("boom")
        reporter.stage_started(Stage.TRANSCRIBE)
        gate.wait(5)
        return RESULT

    app = TinyStenoApp(CONFIG, make_deps(tmp_path, run_pipeline=fake))
    async with app.run_test() as pilot:
        app.process_file(str(_wav(tmp_path)), "default")
        await _settle(app, pilot)
        await pilot.press("r")
        for _ in range(50):
            await pilot.pause(0.02)
            if app.screen.query_one("#stage-transcribe").has_class("-active"):
                break
        screen = app.screen
        assert not screen.query_one("#stage-transcribe").has_class("-failed")
        assert not screen.query_one("#error").display
        await pilot.press("n")  # ignored while running
        assert isinstance(app.screen, ProcessingScreen)
        gate.set()
        await _settle(app, pilot)


async def test_persona_load_failure_is_shown_not_raised(tmp_path):
    wav = _wav(tmp_path)

    def bad_load(slug):
        raise PersonaInvalidError("bad schema")

    app = TinyStenoApp(CONFIG, make_deps(tmp_path, load_persona=bad_load))
    async with app.run_test() as pilot:
        # Home also loads the persona (for its hint); that failure is already handled there.
        app.process_file(str(wav), "default")
        await _settle(app, pilot)
        screen = app.screen
        assert isinstance(screen, ProcessingScreen)
        assert "bad schema" in _text(screen, "#error")
        assert str(wav) in _text(screen, "#error")
        assert not any(screen.query_one(f"#stage-{s.value}").has_class("-failed") for s in Stage)


async def test_missing_wav_notifies_and_stays_home(tmp_path):
    app = TinyStenoApp(CONFIG, make_deps(tmp_path))
    async with app.run_test() as pilot:
        notes = []
        app.notify = lambda message, **kw: notes.append((message, kw.get("severity")))
        app.process_file(str(tmp_path / "gone.wav"), "default")
        await pilot.pause()
        assert isinstance(app.screen, HomeScreen)
        assert notes and notes[0][1] == "error"


async def test_quit_during_pipeline_saves_message_and_exits_promptly(tmp_path):
    gate = threading.Event()
    wav = _wav(tmp_path)

    def fake(wav_path, name, persona, ts, reporter):  # pylint: disable=unused-argument
        reporter.stage_started(Stage.TRANSCRIBE)
        gate.wait(10)
        reporter.stage_finished(Stage.TRANSCRIBE)  # arrives after the app has exited
        return RESULT

    app = TinyStenoApp(CONFIG, make_deps(tmp_path, run_pipeline=fake))
    thread = None
    started = time.monotonic()
    try:
        async with app.run_test() as pilot:
            app.process_file(str(wav), "default")
            await pilot.pause()
            thread = app.screen.pipeline_thread
            await pilot.press("q")
            await pilot.pause()
        assert time.monotonic() - started < 2
        assert thread.is_alive()  # still running: exit did not wait for it
        assert str(wav) in app.exit_message and "tinysteno process" in app.exit_message
        assert app.return_code == 0
    finally:
        gate.set()
        if thread is not None:
            thread.join(5)
