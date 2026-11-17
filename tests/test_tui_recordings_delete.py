"""Recordings screen: deleting, and everything that guards a delete."""
import threading
from pathlib import Path

import pytest

from tinysteno.pipeline import PipelineError, Stage
from tinysteno.recordings import delete_recording
from tinysteno.tui.app import TinyStenoApp
from tinysteno.tui.confirm import ConfirmScreen
from tinysteno.tui.guard import RecorderGuard
from tinysteno.tui.home import HomeScreen
from tinysteno.tui.recordings import RecordingsScreen
from tests.tui_fakes import FakeRecorder, make_deps
from tests.tui_recordings_helpers import (
    CONFIG, confirm, cursor_name, listed, make_wavs, move_to, note, open_recordings, settle)
from tests.tui_settings_helpers import notes

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
    deps.delete_recording = lambda wav: delete_recording(folder, wav)
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


async def test_the_screen_has_no_unsaved_changes_hook():
    assert not hasattr(RecordingsScreen(), "has_unsaved_changes")



# ── a delete always reports, and the session ledger follows it ───────────────


def gated_delete(gate, fail=None):
    """A delete that waits for ``gate``, then removes the WAV (or fails with ``fail``)."""
    def run(path):
        gate.wait(5)
        if fail is not None:
            raise fail
        Path(path).unlink()
        return [path]
    return run


def failing_delete(path):
    """A delete that removes nothing."""
    raise OSError(f"cannot delete {path}")


async def test_deleting_a_wav_whose_processing_failed_drops_it_from_the_exit_message(tmp_path):
    wav = make_wavs(tmp_path, ["A.wav"])[0]

    def failing(*args, **kwargs):  # pylint: disable=unused-argument
        raise PipelineError(Stage.TRANSCRIBE, "boom")

    app = TinyStenoApp(CONFIG, make_deps(tmp_path, run_pipeline=failing))
    async with app.run_test() as pilot:
        screen = await open_recordings(app, pilot)
        await pilot.press("enter")
        for _ in range(20):
            await pilot.pause()
            if app.screen.state.value == "failed":
                break
        assert str(wav) in str(app.ledger.summary()[0])
        await pilot.press("n")
        await pilot.pause()
        assert app.screen is screen
        await settle(pilot, screen)
        await pilot.press("d")
        await pilot.pause()
        await confirm(pilot, screen)
        assert not wav.exists()
        assert app.ledger.summary() == (None, False)


async def test_a_delete_error_after_the_wav_went_still_forgets_it(tmp_path):
    wav = make_wavs(tmp_path, ["A.wav"])[0]

    def half(path):
        Path(path).unlink()
        raise OSError("transcript stuck")

    app = TinyStenoApp(CONFIG, make_deps(tmp_path, delete_recording=half))
    async with app.run_test() as pilot:
        screen = await open_recordings(app, pilot)
        app.ledger.saved(str(wav))
        await pilot.press("d")
        await pilot.pause()
        await confirm(pilot, screen)
        assert app.ledger.summary() == (None, False)


async def test_a_failed_delete_keeps_the_wav_in_the_exit_message(tmp_path):
    wav = make_wavs(tmp_path, ["A.wav"])[0]
    app = TinyStenoApp(CONFIG, make_deps(tmp_path, delete_recording=failing_delete))
    async with app.run_test() as pilot:
        screen = await open_recordings(app, pilot)
        app.ledger.saved(str(wav))
        await pilot.press("d")
        await pilot.pause()
        await confirm(pilot, screen)
        assert str(wav) in str(app.ledger.summary()[0])


@pytest.mark.parametrize("fail", [None, OSError("disk is read-only")])
async def test_escape_during_a_delete_is_ignored_and_the_outcome_is_reported(tmp_path, fail):
    gate = threading.Event()
    make_wavs(tmp_path, ["a.wav"])
    deps = make_deps(tmp_path, delete_recording=gated_delete(gate, fail))
    app = TinyStenoApp(CONFIG, deps)
    async with app.run_test() as pilot:
        screen = await open_recordings(app, pilot)
        await pilot.press("d")
        await pilot.pause()
        await pilot.click("#confirm")
        await pilot.pause()
        assert note(screen) == "Deleting a.wav…"
        await pilot.press("escape")
        await pilot.pause()
        assert app.screen is screen
        gate.set()
        await settle(pilot, screen)
        expected = "Deleted a.wav" if fail is None else "Could not delete a.wav: disk is read-only"
        assert expected in [m for m, _ in notes(app)]
        await pilot.press("escape")
        await pilot.pause()
        assert isinstance(app.screen, HomeScreen)


async def test_a_palette_destination_cannot_be_opened_during_a_delete(tmp_path):
    gate = threading.Event()
    make_wavs(tmp_path, ["a.wav"])
    app = TinyStenoApp(CONFIG, make_deps(tmp_path, delete_recording=gated_delete(gate)))
    async with app.run_test() as pilot:
        screen = await open_recordings(app, pilot)
        await pilot.press("d")
        await pilot.pause()
        await pilot.click("#confirm")
        await pilot.pause()
        app.open_destination("personas")
        await pilot.pause()
        assert app.screen is screen
        assert "Personas" not in [c.title for c in app.get_system_commands(app.screen)]
        gate.set()
        await settle(pilot, screen)
        app.open_destination("checks")
        await pilot.pause()
        assert app.screen is not screen


async def test_quitting_during_a_delete_waits_for_it_and_the_summary_is_right(tmp_path):
    gate = threading.Event()
    wav = make_wavs(tmp_path, ["a.wav"])[0]
    app = TinyStenoApp(CONFIG, make_deps(tmp_path, delete_recording=gated_delete(gate)))
    async with app.run_test() as pilot:
        screen = await open_recordings(app, pilot)
        app.ledger.saved(str(wav))
        await pilot.press("d")
        await pilot.pause()
        await pilot.click("#confirm")
        await pilot.pause()
        threading.Timer(0.3, gate.set).start()
        await pilot.press("q")
        for _ in range(40):
            await pilot.pause(0.05)
            if app.exit_message is not None or not app.is_running:
                break
        assert not wav.exists()
        assert app.exit_message is None  # nothing left to tell the user
        assert screen._delete is None  # pylint: disable=protected-access
