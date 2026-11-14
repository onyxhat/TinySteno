"""Home screen behaviour."""
import os
import subprocess
import sys

from textual.widgets import Footer

from tinysteno.tui.app import TinyStenoApp
from tinysteno.tui.home import HomeScreen, recent_recordings
from tests.tui_fakes import make_deps

CONFIG = {"persona": "1on1", "session_timeout_minutes": 120}


async def test_record_button_has_focus_and_enter_starts_recording(tmp_path):
    app = TinyStenoApp(CONFIG, make_deps(tmp_path))
    async with app.run_test() as pilot:
        assert app.focused.id == "record"
        await pilot.press("enter")
        await pilot.pause()
        assert ("record", None, "1on1") in app.events


async def test_typed_name_is_passed_and_r_does_not_fire_inside_input(tmp_path):
    app = TinyStenoApp(CONFIG, make_deps(tmp_path))
    async with app.run_test() as pilot:
        app.screen.query_one("#name").focus()
        await pilot.press(*"Retro")          # contains "r": must type, not record
        assert not app.events
        await pilot.press("enter")            # Enter in the name field starts recording
        await pilot.pause()
        assert ("record", "Retro", "1on1") in app.events


async def test_unknown_configured_persona_falls_back_to_first_available(tmp_path):
    app = TinyStenoApp({"persona": "gone"}, make_deps(tmp_path))
    async with app.run_test() as pilot:
        await pilot.press("r")
        await pilot.pause()
        assert ("record", None, "default") in app.events


async def test_selecting_recent_recording_processes_it(tmp_path):
    wav = tmp_path / "Old.wav"
    wav.write_bytes(b"")
    app = TinyStenoApp(CONFIG, make_deps(tmp_path))
    async with app.run_test() as pilot:
        app.screen.query_one("#recent").focus()
        await pilot.press("enter")
        await pilot.pause()
        assert ("process", str(wav), "1on1") in app.events


async def test_timeout_hint_follows_the_selected_persona(tmp_path):
    deps = make_deps(tmp_path, persona_timeouts={"1on1": 30, "default": 0})
    app = TinyStenoApp({"persona": "1on1"}, deps)
    async with app.run_test() as pilot:
        hint = app.screen.query_one("#timeout-hint")
        assert "30 min" in str(hint.content)
        app.screen.query_one("#persona").value = "default"
        await pilot.pause()
        assert "No time limit" in str(hint.content)


async def test_timeout_hint_shows_global_default_of_75(tmp_path):
    app = TinyStenoApp({"persona": "default"}, make_deps(tmp_path))
    async with app.run_test():
        assert "75 min" in str(app.screen.query_one("#timeout-hint").content)


async def test_q_quits_cleanly(tmp_path):
    app = TinyStenoApp(CONFIG, make_deps(tmp_path))
    async with app.run_test() as pilot:
        await pilot.press("q")
    assert app.return_code in (0, None)


async def test_autostart_begins_recording_immediately(tmp_path):
    app = TinyStenoApp(
        CONFIG, make_deps(tmp_path), autostart=True, name="Sync", persona="default",
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.events == [("record", "Sync", "default")]


def test_recent_recordings_newest_first_wav_only(tmp_path):
    for i, name in enumerate(["a.wav", "b.wav", "notes.md"]):
        path = tmp_path / name
        path.write_bytes(b"")
        os.utime(path, (i, i))
    assert [p.name for p in recent_recordings(tmp_path)] == ["b.wav", "a.wav"]
    assert recent_recordings(tmp_path / "missing") == []


def test_importing_main_does_not_import_textual():
    check = (
        "import sys, tinysteno.main; "
        "assert not any(m == 'textual' or m.startswith('textual.') for m in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", check], check=True)


async def test_unknown_persona_flag_is_reported_not_silently_replaced(tmp_path):
    app = TinyStenoApp(CONFIG, make_deps(tmp_path), persona="typo")
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.screen.query_one("#persona").value == "1on1"   # the configured persona
        notes = [(n.message, n.severity) for n in app._notifications]  # pylint: disable=protected-access
        assert notes == [("Unknown persona 'typo'; using '1on1' instead.", "error")]


async def test_no_personas_shows_a_hint_and_disables_record(tmp_path):
    app = TinyStenoApp(CONFIG, make_deps(tmp_path, list_personas=list))
    async with app.run_test() as pilot:
        screen = app.screen
        assert isinstance(screen, HomeScreen)
        hint = screen.query_one("#timeout-hint")
        assert str(hint.content) == "No personas found. Run tinysteno setup."
        assert hint.has_class("error")
        assert screen.query_one("#record").disabled
        await pilot.press("r")
        screen.query_one("#name").focus()
        await pilot.press("enter")
        await pilot.pause()
        assert not app.events and isinstance(app.screen, HomeScreen)


async def test_home_fits_80x24_with_five_recent_recordings(tmp_path):
    for i in range(5):
        (tmp_path / f"Meeting-2026111{i}-0930.wav").write_bytes(b"")
    app = TinyStenoApp(CONFIG, make_deps(tmp_path))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        screen = app.screen
        panel = screen.query_one("#panel")
        recent = screen.query_one("#recent")
        footer = screen.query_one(Footer)
        assert screen.max_scroll_y == 0 and panel.max_scroll_y == 0
        assert recent.max_scroll_y == 0 and recent.option_count == 5
        assert recent.region.bottom <= footer.region.y
        assert app.focused.id == "record"
        assert screen.query_one("#record").region.height == 1
