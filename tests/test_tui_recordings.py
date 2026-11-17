"""Recordings screen: listing, processing and transcribing again; Home's hint."""
import threading
from pathlib import Path

from rich.text import Text
from textual.widgets import Select

from tinysteno.recordings import list_recordings
from tinysteno.tui.app import TinyStenoApp
from tinysteno.tui.home import NO_PERSONAS
from tinysteno.tui.personas import PersonasScreen
from tinysteno.tui.processing import ProcessingScreen
from tinysteno.tui.recordings import RecordingsScreen
from tinysteno.tui.result import ResultScreen
from tests.tui_fakes import make_deps
from tests.tui_recordings_helpers import (
    CONFIG, confirm, listed, make_wavs, move_to, note, open_recordings, settle, table_of)
from tests.tui_settings_helpers import notes

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
    real = list_recordings

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


async def test_long_names_that_differ_only_at_the_end_stay_distinguishable(tmp_path):
    base = "Weekly-sync-with-the-platform-team-2026-03-04-"
    make_wavs(tmp_path, [base + "0905.wav", base + "0915.wav"])
    app = TinyStenoApp(CONFIG, make_deps(tmp_path))
    async with app.run_test() as pilot:
        screen = await open_recordings(app, pilot)
        shown = listed(screen)
        assert len(set(shown)) == 2
        assert shown[0].endswith("0915.wav") and shown[1].endswith("0905.wav")


async def test_the_persona_select_follows_a_restore_made_in_the_personas_screen(tmp_path):
    make_wavs(tmp_path, ["a.wav"])
    deps = make_deps(tmp_path, personas=[])
    app = TinyStenoApp(CONFIG, deps)
    async with app.run_test() as pilot:
        screen = await open_recordings(app, pilot)
        await pilot.press("enter")
        await pilot.pause()
        assert app.screen is screen and not app.events
        assert (NO_PERSONAS, "error") in notes(app)
        app.open_destination("personas")
        await pilot.pause()
        assert isinstance(app.screen, PersonasScreen)
        deps.personas[:] = ["1on1", "default"]  # restored there
        await pilot.press("escape")
        await settle(pilot, screen)
        assert app.screen is screen
        assert screen.query_one("#persona", Select).value == "1on1"
        await pilot.press("enter")
        for _ in range(20):
            await pilot.pause()
            if isinstance(app.screen, (ProcessingScreen, ResultScreen)):
                break
        assert ("process", str(tmp_path / "a.wav"), "1on1") in app.events
