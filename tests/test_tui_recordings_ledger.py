"""Recordings: size and date formats, and the ledger's busy answer through every way a run ends."""
import threading

import pytest

from tinysteno.pipeline import PipelineError, PipelineResult, Stage
from tinysteno.tui.app import TinyStenoApp
from tinysteno.tui.home import HomeScreen
from tinysteno.tui.ledger import SessionLedger
from tinysteno.tui.processing import ProcessingScreen
from tinysteno.tui.recordings import format_date, format_size, shorten
from tinysteno.tui.result import ResultScreen
from tests.tui_fakes import make_deps
from tests.tui_recordings_helpers import CONFIG, make_wavs

# ── pure formatting ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("size, text", [
    (0, "0 B"), (1023, "1023 B"), (1024, "1.0 KB"), (1536, "1.5 KB"),
    (1024 * 1024, "1.0 MB"), (int(12.34 * 1024 * 1024), "12.3 MB"),
    (3 * 1024 ** 3, "3.0 GB"), (5 * 1024 ** 4, "5120.0 GB"),
])
def test_sizes_are_1024_based_with_one_decimal_from_kb(size, text):
    assert format_size(size) == text


def test_dates_are_minute_precision():
    from datetime import datetime  # pylint: disable=import-outside-toplevel
    assert format_date(datetime(2026, 3, 4, 9, 5, 59)) == "2026-03-04 09:05"


# ── the ledger's busy answer, and every way a run ends ───────────────────────


def test_ledger_knows_which_wav_is_running():
    ledger = SessionLedger()
    assert not ledger.is_busy("/r/a.wav")
    ledger.begin_run("/r/a.wav")
    assert ledger.is_busy("/r/a.wav") and not ledger.is_busy("/r/b.wav")
    assert ledger.is_busy("/r/../r/a.wav")  # the same file under another spelling
    ledger.end_run("/r/a.wav")
    assert not ledger.is_busy("/r/a.wav")
    ledger.end_run("/r/a.wav")  # ending twice is harmless


def test_a_run_that_began_twice_is_busy_until_it_has_ended_as_often():
    ledger = SessionLedger()
    ledger.begin_run("a.wav")
    ledger.begin_run("a.wav")
    ledger.end_run("a.wav")
    assert ledger.is_busy("a.wav")
    ledger.end_run("a.wav")
    assert not ledger.is_busy("a.wav")


async def test_a_running_pipeline_makes_its_wav_busy_and_success_frees_it(tmp_path):
    gate = threading.Event()
    wav = make_wavs(tmp_path, ["A.wav"])[0]
    base = make_deps(tmp_path)

    def pipeline(wav_path, *args, **kwargs):
        gate.wait(5)
        return base.run_pipeline(wav_path, *args, **kwargs)

    app = TinyStenoApp(CONFIG, make_deps(tmp_path, run_pipeline=pipeline))
    async with app.run_test() as pilot:
        app.process_file(str(wav), "1on1")
        await pilot.pause()
        assert isinstance(app.screen, ProcessingScreen)
        assert app.ledger.is_busy(str(wav))
        gate.set()
        for _ in range(20):
            await pilot.pause()
            if isinstance(app.screen, ResultScreen):
                break
        assert isinstance(app.screen, ResultScreen)
        assert not app.ledger.is_busy(str(wav))


async def test_failure_frees_the_wav_and_a_retry_makes_it_busy_again(tmp_path):
    gate = threading.Event()
    calls = []
    wav = make_wavs(tmp_path, ["A.wav"])[0]

    def pipeline(wav_path, name, persona, timestamp, reporter, **kwargs):  # pylint: disable=unused-argument
        calls.append(wav_path)
        if len(calls) == 1:
            raise PipelineError(Stage.TRANSCRIBE, "boom")
        gate.wait(5)
        return PipelineResult(wav_path=wav_path, note_path=None, title="", no_speech=True,
                              duration_seconds=0.0)

    app = TinyStenoApp(CONFIG, make_deps(tmp_path, run_pipeline=pipeline))
    async with app.run_test() as pilot:
        app.process_file(str(wav), "1on1")
        for _ in range(20):
            await pilot.pause()
            if app.screen.state.value == "failed":
                break
        assert app.screen.state.value == "failed"
        assert not app.ledger.is_busy(str(wav))
        await pilot.press("r")
        await pilot.pause()
        assert app.ledger.is_busy(str(wav))
        gate.set()
        for _ in range(20):
            await pilot.pause()
            if app.screen.state.value == "no_speech":
                break
        assert app.screen.state.value == "no_speech"
        assert not app.ledger.is_busy(str(wav))


async def test_leaving_a_failed_run_leaves_the_wav_free(tmp_path):
    wav = make_wavs(tmp_path, ["A.wav"])[0]

    def pipeline(*args, **kwargs):  # pylint: disable=unused-argument
        raise RuntimeError("no")

    app = TinyStenoApp(CONFIG, make_deps(tmp_path, run_pipeline=pipeline))
    async with app.run_test() as pilot:
        app.process_file(str(wav), "1on1")
        for _ in range(20):
            await pilot.pause()
            if app.screen.state.value == "failed":
                break
        await pilot.press("n")
        await pilot.pause()
        assert isinstance(app.screen, HomeScreen)
        assert not app.ledger.is_busy(str(wav))


def test_forget_drops_a_wav_whatever_its_spelling_and_keeps_failures():
    ledger = SessionLedger()
    ledger.saved("/r/../r/a.wav")
    ledger.saved("/r/b.wav")
    ledger.save_failed("Could not save the recording: disk full")
    ledger.forget("/r/a.wav")
    ledger.forget("/r/never-seen.wav")
    message, failed = ledger.summary()
    text = str(message)
    assert "a.wav" not in text and "b.wav" in text and "disk full" in text and failed


def test_shorten_keeps_the_start_and_the_end_of_a_long_name():
    assert shorten("short.wav") == "short.wav"
    one = shorten("Weekly-sync-with-the-platform-team-2026-03-04-0905.wav")
    two = shorten("Weekly-sync-with-the-platform-team-2026-03-04-0915.wav")
    assert len(one) == 28 and one != two
    assert one.startswith("Weekly-sync") and one.endswith("0905.wav") and "…" in one
