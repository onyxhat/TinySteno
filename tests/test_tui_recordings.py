"""Recordings screen: list, process, transcribe again, delete (and what guards a delete)."""
import os
import threading
from pathlib import Path

import pytest
from rich.text import Text
from textual.widgets import DataTable, Select, Static

from tinysteno.pipeline import PipelineError, PipelineResult, Stage
from tinysteno.tui.app import TinyStenoApp
from tinysteno.tui.confirm import ConfirmScreen
from tinysteno.tui.guard import RecorderGuard
from tinysteno.tui.home import HomeScreen, NO_PERSONAS
from tinysteno.tui.ledger import SessionLedger
from tinysteno.tui.processing import ProcessingScreen
from tinysteno.tui.recordings import RecordingsScreen, format_date, format_size
from tinysteno.tui.result import ResultScreen
from tests.tui_fakes import FakeRecorder, make_deps
from tests.tui_settings_helpers import notes

CONFIG = {"persona": "1on1", "session_timeout_minutes": 120}
BASE_TIME = 1_700_000_000


def make_wav(directory: Path, name: str, index: int, sidecars: bool) -> Path:
    """One WAV whose mtime rises with ``index``."""
    wav = directory / name
    wav.write_bytes(b"RIFF" * 10)
    os.utime(wav, (BASE_TIME + index * 60, BASE_TIME + index * 60))
    if sidecars:
        Path(str(wav) + ".transcript.json").write_text("{}", encoding="utf-8")
        Path(str(wav) + ".levels.json").write_text("{}", encoding="utf-8")
    return wav


def make_wavs(directory: Path, names, sidecars=False) -> list[Path]:
    """Create WAVs whose mtime rises with their position in ``names``."""
    return [make_wav(directory, name, i, sidecars) for i, name in enumerate(names)]


async def settle(pilot, screen) -> None:
    """Wait until the screen has no read or delete in flight."""
    for _ in range(40):
        await pilot.pause()
        if screen._load is None and screen._delete is None:  # pylint: disable=protected-access
            await pilot.pause()
            return
    raise AssertionError("the screen never settled")


async def open_recordings(app, pilot) -> RecordingsScreen:
    """Open the Recordings screen from Home and wait for the table to load."""
    app.open_destination("recordings")
    screen = app.screen
    assert isinstance(screen, RecordingsScreen)
    await settle(pilot, screen)
    return screen


def table_of(screen) -> DataTable:
    """The recordings table."""
    return screen.query_one("#recordings", DataTable)


def listed(screen) -> list[str]:
    """The recording column, top to bottom."""
    table = table_of(screen)
    return [str(table.get_row_at(i)[0]) for i in range(table.row_count)]


def cursor_name(screen) -> str:
    """The file name under the cursor."""
    table = table_of(screen)
    return str(table.get_row_at(table.cursor_row)[0])


def move_to(screen, name) -> None:
    """Put the cursor on the row of a file name."""
    table = table_of(screen)
    table.move_cursor(row=listed(screen).index(name))


def note(screen) -> str:
    """The line above the table."""
    return str(screen.query_one("#recordings-note", Static).content)


async def confirm(pilot, screen) -> None:
    """Press the Delete button of the open dialog and let the delete finish."""
    await pilot.click("#confirm")
    await settle(pilot, screen)


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


# ── listing ──────────────────────────────────────────────────────────────────


async def test_all_recordings_are_listed_newest_first(tmp_path):
    files = [f"m{i:02d}.wav" for i in range(12)]
    make_wavs(tmp_path, files)
    (tmp_path / "notes.txt").write_text("x")
    app = TinyStenoApp(CONFIG, make_deps(tmp_path))
    async with app.run_test() as pilot:
        screen = await open_recordings(app, pilot)
        table = table_of(screen)
        assert [str(c.label) for c in table.columns.values()] == [
            "Recording", "Date", "Size", "Transcript"]
        assert listed(screen) == list(reversed(files))
        assert note(screen) == ""


async def test_cells_show_date_size_and_transcript_state(tmp_path):
    make_wavs(tmp_path, ["a.wav", "b.wav"])
    Path(str(tmp_path / "b.wav") + ".transcript.json").write_text("{}", encoding="utf-8")
    app = TinyStenoApp(CONFIG, make_deps(tmp_path))
    async with app.run_test() as pilot:
        screen = await open_recordings(app, pilot)
        table = table_of(screen)
        first, second = ([str(c) for c in table.get_row_at(i)] for i in range(2))
        assert first[0] == "b.wav" and first[2] == "40 B" and first[3] == "saved"
        assert second[0] == "a.wav" and second[3] == "—"
        assert len(first[1]) == 16 and first[1][4] == "-" and first[1][13] == ":"


async def test_empty_folder_message_names_the_folder(tmp_path):
    folder = tmp_path / "empty"
    folder.mkdir()
    app = TinyStenoApp(CONFIG, make_deps(tmp_path, recordings_dir=folder))
    async with app.run_test() as pilot:
        screen = await open_recordings(app, pilot)
        assert "No recordings yet" in note(screen) and str(folder) in note(screen)
        assert table_of(screen).row_count == 0


async def test_a_listing_error_is_shown_with_an_empty_table(tmp_path, monkeypatch):
    make_wavs(tmp_path, ["a.wav"])

    def broken(directory):
        raise PermissionError(f"cannot read {directory}")

    monkeypatch.setattr("tinysteno.tui.recordings.list_recordings", broken)
    app = TinyStenoApp(CONFIG, make_deps(tmp_path))
    async with app.run_test() as pilot:
        screen = await open_recordings(app, pilot)
        assert note(screen).startswith("Recordings cannot be listed: ")
        assert "cannot read" in note(screen)
        assert table_of(screen).row_count == 0


async def test_the_screen_shows_loading_until_the_read_ends(tmp_path, monkeypatch):
    gate = threading.Event()
    real = __import__("tinysteno.recordings", fromlist=["list_recordings"]).list_recordings

    def slow(directory):
        gate.wait(5)
        return real(directory)

    monkeypatch.setattr("tinysteno.tui.recordings.list_recordings", slow)
    make_wavs(tmp_path, ["a.wav"])
    app = TinyStenoApp(CONFIG, make_deps(tmp_path))
    async with app.run_test() as pilot:
        app.open_destination("recordings")
        await pilot.pause()
        assert note(app.screen) == "Loading…"
        gate.set()
        await settle(pilot, app.screen)
        assert listed(app.screen) == ["a.wav"] and note(app.screen) == ""


async def test_a_name_with_markup_is_listed_literally_and_can_be_deleted(tmp_path):
    make_wavs(tmp_path, ["[red]x[bold].wav", "keep.wav"])
    app = TinyStenoApp(CONFIG, make_deps(tmp_path))
    async with app.run_test() as pilot:
        screen = await open_recordings(app, pilot)
        assert "[red]x[bold].wav" in listed(screen)
        assert all(isinstance(c, Text) for c in table_of(screen).get_row_at(0))
        move_to(screen, "[red]x[bold].wav")
        await pilot.press("d")
        await pilot.pause()
        assert "[red]x[bold].wav" in str(app.screen.query_one("#confirm-title").content)
        await confirm(pilot, screen)
        assert not (tmp_path / "[red]x[bold].wav").exists()
        assert (tmp_path / "keep.wav").exists()
        assert ("Deleted [red]x[bold].wav", "information") in notes(app)


async def test_refresh_key_reads_the_folder_again(tmp_path):
    make_wavs(tmp_path, ["a.wav"])
    app = TinyStenoApp(CONFIG, make_deps(tmp_path))
    async with app.run_test() as pilot:
        screen = await open_recordings(app, pilot)
        make_wavs(tmp_path, ["a.wav", "b.wav"])
        await pilot.press("r")
        await settle(pilot, screen)
        assert listed(screen) == ["b.wav", "a.wav"]


async def test_after_a_settings_change_the_screen_lists_the_new_folder(tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    make_wavs(tmp_path, ["old.wav"])
    make_wavs(other, ["new.wav"])
    deps = make_deps(tmp_path)
    app = TinyStenoApp(CONFIG, deps)
    async with app.run_test() as pilot:
        screen = await open_recordings(app, pilot)
        assert listed(screen) == ["old.wav"]
        deps.recordings_dir = other
        await pilot.press("r")
        await settle(pilot, screen)
        assert listed(screen) == ["new.wav"]


async def test_returning_from_a_processed_recording_refreshes_the_table(tmp_path):
    make_wavs(tmp_path, ["a.wav"])
    app = TinyStenoApp(CONFIG, make_deps(tmp_path))
    async with app.run_test() as pilot:
        screen = await open_recordings(app, pilot)
        await pilot.press("enter")
        for _ in range(20):
            await pilot.pause()
            if isinstance(app.screen, ResultScreen):
                break
        assert isinstance(app.screen, ResultScreen)
        make_wavs(tmp_path, ["a.wav", "b.wav"])
        await pilot.press("n")
        await settle(pilot, screen)
        assert app.screen is screen and listed(screen) == ["b.wav", "a.wav"]


# ── processing ───────────────────────────────────────────────────────────────


async def test_the_persona_select_offers_the_personas_with_the_default_chosen(tmp_path):
    app = TinyStenoApp(CONFIG, make_deps(tmp_path))
    async with app.run_test() as pilot:
        screen = await open_recordings(app, pilot)
        select = screen.query_one("#persona", Select)
        assert select.value == "1on1"


async def test_enter_processes_with_the_chosen_persona(tmp_path):
    wavs = make_wavs(tmp_path, ["a.wav", "b.wav"])
    app = TinyStenoApp(CONFIG, make_deps(tmp_path))
    async with app.run_test() as pilot:
        screen = await open_recordings(app, pilot)
        screen.query_one("#persona", Select).value = "default"
        move_to(screen, "a.wav")
        table_of(screen).focus()
        await pilot.press("enter")
        await pilot.pause()
        assert ("process", str(wavs[0]), "default") in app.events
        assert isinstance(app.screen, (ProcessingScreen, ResultScreen))


async def test_t_requests_a_new_transcription(tmp_path):
    wav = make_wavs(tmp_path, ["a.wav"], sidecars=True)[0]
    deps = make_deps(tmp_path)
    app = TinyStenoApp(CONFIG, deps)
    async with app.run_test() as pilot:
        await open_recordings(app, pilot)
        await pilot.press("t")
        for _ in range(20):
            await pilot.pause()
            if deps.pipeline_kwargs:
                break
        assert ("process", str(wav), "1on1") in app.events
        assert deps.pipeline_kwargs == [{"retranscribe": True}]


async def test_t_on_a_row_without_a_transcript_acts_like_enter(tmp_path):
    make_wavs(tmp_path, ["a.wav"])
    deps = make_deps(tmp_path)
    app = TinyStenoApp(CONFIG, deps)
    async with app.run_test() as pilot:
        await open_recordings(app, pilot)
        await pilot.press("t")
        for _ in range(20):
            await pilot.pause()
            if deps.pipeline_kwargs:
                break
        assert deps.pipeline_kwargs == [{}]


async def test_no_personas_disables_processing_but_not_delete(tmp_path):
    wav = make_wavs(tmp_path, ["a.wav"], sidecars=True)[0]
    deps = make_deps(tmp_path, personas=[])
    app = TinyStenoApp(CONFIG, deps)
    async with app.run_test() as pilot:
        screen = await open_recordings(app, pilot)
        for key in ("enter", "t"):
            await pilot.press(key)
            await pilot.pause()
            assert isinstance(app.screen, RecordingsScreen)
        assert not app.events
        assert (NO_PERSONAS, "error") in notes(app)
        await pilot.press("d")
        await pilot.pause()
        await confirm(pilot, screen)
        assert not wav.exists()


async def test_nothing_selected_does_nothing(tmp_path):
    app = TinyStenoApp(CONFIG, make_deps(tmp_path))
    async with app.run_test() as pilot:
        await open_recordings(app, pilot)
        for key in ("enter", "t", "d"):
            await pilot.press(key)
            await pilot.pause()
        assert isinstance(app.screen, RecordingsScreen)
        assert not app.events and not notes(app)


# ── delete ───────────────────────────────────────────────────────────────────


async def test_delete_asks_and_cancel_keeps_the_file(tmp_path):
    wav = make_wavs(tmp_path, ["a.wav"], sidecars=True)[0]
    app = TinyStenoApp(CONFIG, make_deps(tmp_path))
    async with app.run_test() as pilot:
        screen = await open_recordings(app, pilot)
        await pilot.press("d")
        await pilot.pause()
        dialog = app.screen
        assert isinstance(dialog, ConfirmScreen)
        assert dialog.question == (
            "Delete a.wav? The audio and its saved transcript are removed. "
            "Notes in your vault are kept.")
        assert app.focused.id == "cancel"
        await pilot.click("#cancel")
        await settle(pilot, screen)
        assert wav.exists() and Path(str(wav) + ".transcript.json").exists()
        assert listed(screen) == ["a.wav"]


async def test_escape_in_the_dialog_deletes_nothing(tmp_path):
    wav = make_wavs(tmp_path, ["a.wav"])[0]
    app = TinyStenoApp(CONFIG, make_deps(tmp_path))
    async with app.run_test() as pilot:
        screen = await open_recordings(app, pilot)
        await pilot.press("d")
        await pilot.pause()
        await pilot.press("escape")
        await settle(pilot, screen)
        assert wav.exists() and app.screen is screen


async def test_enter_on_the_focused_cancel_button_deletes_nothing(tmp_path):
    wav = make_wavs(tmp_path, ["a.wav"])[0]
    app = TinyStenoApp(CONFIG, make_deps(tmp_path))
    async with app.run_test() as pilot:
        screen = await open_recordings(app, pilot)
        await pilot.press("d")
        await pilot.pause()
        await pilot.press("enter")
        await settle(pilot, screen)
        assert wav.exists()


async def test_delete_removes_audio_and_sidecars_and_keeps_others(tmp_path):
    wavs = make_wavs(tmp_path, ["a.wav", "b.wav", "c.wav"], sidecars=True)
    a, b, c = wavs
    app = TinyStenoApp(CONFIG, make_deps(tmp_path))
    async with app.run_test() as pilot:
        screen = await open_recordings(app, pilot)
        move_to(screen, "b.wav")
        await pilot.press("d")
        await pilot.pause()
        await confirm(pilot, screen)
        assert not b.exists()
        assert not Path(str(b) + ".transcript.json").exists()
        assert not Path(str(b) + ".levels.json").exists()
        for kept in (a, c):
            assert kept.exists() and Path(str(kept) + ".transcript.json").exists()
        assert listed(screen) == ["c.wav", "a.wav"]
        assert ("Deleted b.wav", "information") in notes(app)
        assert cursor_name(screen) == "a.wav"  # the row that followed the deleted one


async def test_the_path_deleted_is_the_one_selected_when_d_was_pressed(tmp_path):
    a, b = make_wavs(tmp_path, ["a.wav", "b.wav"])
    app = TinyStenoApp(CONFIG, make_deps(tmp_path))
    async with app.run_test() as pilot:
        screen = await open_recordings(app, pilot)
        move_to(screen, "a.wav")
        await pilot.press("d")
        await pilot.pause()
        make_wavs(tmp_path, ["a.wav", "b.wav", "z.wav"])  # the list changes under the dialog
        screen.action_refresh()
        await pilot.pause()
        move_to(screen, "z.wav")
        await confirm(pilot, screen)
        assert not a.exists() and b.exists() and (tmp_path / "z.wav").exists()


async def test_delete_failure_is_reported_and_row_stays(tmp_path):
    wav = make_wavs(tmp_path, ["a.wav"])[0]

    def failing(path):
        raise OSError("disk is read-only")

    app = TinyStenoApp(CONFIG, make_deps(tmp_path, delete_recording=failing))
    async with app.run_test() as pilot:
        screen = await open_recordings(app, pilot)
        await pilot.press("d")
        await pilot.pause()
        await confirm(pilot, screen)
        assert ("Could not delete a.wav: disk is read-only", "error") in notes(app)
        assert wav.exists() and listed(screen) == ["a.wav"]


async def test_a_refused_path_is_reported_as_a_failure(tmp_path):
    make_wavs(tmp_path, ["a.wav"])

    def refusing(path):
        raise ValueError("Not a .wav file")

    app = TinyStenoApp(CONFIG, make_deps(tmp_path, delete_recording=refusing))
    async with app.run_test() as pilot:
        screen = await open_recordings(app, pilot)
        await pilot.press("d")
        await pilot.pause()
        await confirm(pilot, screen)
        assert ("Could not delete a.wav: Not a .wav file", "error") in notes(app)


async def test_a_failure_after_the_audio_went_still_rereads_the_list(tmp_path):
    wav = make_wavs(tmp_path, ["a.wav", "b.wav"])[0]

    def half(path):
        Path(path).unlink()
        raise OSError("could not remove the transcript")

    app = TinyStenoApp(CONFIG, make_deps(tmp_path, delete_recording=half))
    async with app.run_test() as pilot:
        screen = await open_recordings(app, pilot)
        move_to(screen, "a.wav")
        await pilot.press("d")
        await pilot.pause()
        await confirm(pilot, screen)
        assert not wav.exists() and listed(screen) == ["b.wav"]
        assert any(m.startswith("Could not delete a.wav:") for m, _ in notes(app))


async def test_file_removed_outside_the_app_is_reported_not_a_crash(tmp_path):
    a, b = make_wavs(tmp_path, ["a.wav", "b.wav"])
    app = TinyStenoApp(CONFIG, make_deps(tmp_path))
    async with app.run_test() as pilot:
        screen = await open_recordings(app, pilot)
        move_to(screen, "a.wav")
        await pilot.press("d")
        await pilot.pause()
        a.unlink()
        await confirm(pilot, screen)
        assert ("a.wav is no longer there", "warning") in notes(app)
        assert listed(screen) == ["b.wav"] and b.exists()


async def test_cannot_delete_the_recording_being_processed(tmp_path):
    wav = make_wavs(tmp_path, ["a.wav"])[0]
    app = TinyStenoApp(CONFIG, make_deps(tmp_path))
    async with app.run_test() as pilot:
        screen = await open_recordings(app, pilot)
        app.ledger.begin_run(str(wav))
        await pilot.press("d")
        await pilot.pause()
        assert app.screen is screen
        assert ("a.wav is being processed", "error") in notes(app)
        assert wav.exists()


async def test_a_run_that_began_while_the_dialog_was_open_stops_the_delete(tmp_path):
    wav = make_wavs(tmp_path, ["a.wav"])[0]
    app = TinyStenoApp(CONFIG, make_deps(tmp_path))
    async with app.run_test() as pilot:
        screen = await open_recordings(app, pilot)
        await pilot.press("d")
        await pilot.pause()
        app.ledger.begin_run(str(wav))
        await confirm(pilot, screen)
        assert wav.exists()
        assert ("a.wav is being processed", "error") in notes(app)


async def test_cannot_delete_the_recording_in_progress(tmp_path):
    wav = make_wavs(tmp_path, ["a.wav"])[0]
    app = TinyStenoApp(CONFIG, make_deps(tmp_path))
    async with app.run_test() as pilot:
        screen = await open_recordings(app, pilot)
        app.recorder_guard = RecorderGuard(FakeRecorder(tmp_path), str(wav))
        await pilot.press("d")
        await pilot.pause()
        assert app.screen is screen and wav.exists()
        assert ("a.wav is being recorded", "error") in notes(app)


async def test_a_second_d_while_a_delete_runs_is_ignored(tmp_path):
    gate = threading.Event()
    calls = []
    _, b = make_wavs(tmp_path, ["a.wav", "b.wav"])

    def slow(path):
        calls.append(path)
        gate.wait(5)
        Path(path).unlink()
        return [path]

    app = TinyStenoApp(CONFIG, make_deps(tmp_path, delete_recording=slow))
    async with app.run_test() as pilot:
        screen = await open_recordings(app, pilot)
        move_to(screen, "b.wav")
        await pilot.press("d")
        await pilot.pause()
        await pilot.click("#confirm")
        await pilot.pause()
        for key in ("d", "enter", "t"):
            await pilot.press(key)
            await pilot.pause()
        assert app.screen is screen and not app.events
        gate.set()
        await settle(pilot, screen)
        assert calls == [b]
        assert listed(screen) == ["a.wav"]


async def test_deleting_a_symlinked_wav_is_refused_and_both_survive(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    target = make_wavs(outside, ["target.wav"], sidecars=True)[0]
    folder = tmp_path / "rec"
    folder.mkdir()
    link = folder / "link.wav"
    link.symlink_to(target)
    deps = make_deps(tmp_path, recordings_dir=folder)
    deps.delete_recording = lambda wav: __import__(
        "tinysteno.recordings", fromlist=["x"]).delete_recording(folder, wav)
    app = TinyStenoApp(CONFIG, deps)
    async with app.run_test() as pilot:
        screen = await open_recordings(app, pilot)
        assert listed(screen) == ["link.wav"]
        await pilot.press("d")
        await pilot.pause()
        await confirm(pilot, screen)
        assert link.is_symlink() and target.exists()
        assert Path(str(target) + ".transcript.json").exists()
        assert any(m.startswith("Could not delete link.wav:") and s == "error"
                   for m, s in notes(app))


async def test_the_screen_has_no_unsaved_changes_hook_and_no_closing_attribute():
    screen = RecordingsScreen()
    assert not hasattr(screen, "has_unsaved_changes")
    assert screen._closing is False  # pylint: disable=protected-access  # Textual's, untouched


# ── Home hint and layout ─────────────────────────────────────────────────────


async def test_home_label_hints_at_the_screen_only_when_there_are_more(tmp_path):
    make_wavs(tmp_path, [f"m{i}.wav" for i in range(8)])
    app = TinyStenoApp(CONFIG, make_deps(tmp_path))
    async with app.run_test():
        assert str(app.screen.query_one("#recent-label").content) == "Recent recordings"
    make_wavs(tmp_path, [f"m{i}.wav" for i in range(9)])
    app = TinyStenoApp(CONFIG, make_deps(tmp_path))
    async with app.run_test():
        assert str(app.screen.query_one("#recent-label").content) == (
            "Recent recordings (m → Recordings for all)")
        assert app.screen.query_one("#recent").option_count == 8


async def test_fits_80x24_with_long_names(tmp_path):
    make_wavs(tmp_path, ["a-very-long-meeting-name-" + "x" * 60 + f"{i}.wav" for i in range(6)])
    app = TinyStenoApp(CONFIG, make_deps(tmp_path))
    async with app.run_test(size=(80, 24)) as pilot:
        screen = await open_recordings(app, pilot)
        table = table_of(screen)
        assert table.max_scroll_x == 0
        footer = screen.query_one("Footer")
        assert table.region.bottom <= footer.region.y
        assert screen.query_one("#persona").region.right <= 80
