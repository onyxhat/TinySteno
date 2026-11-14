"""Home screen behaviour."""
import os

from tinysteno.tui.app import TinyStenoApp
from tinysteno.tui.home import recent_recordings
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
    import importlib
    import sys
    saved = {k: v for k, v in sys.modules.items()
             if k == "textual" or k.startswith(("textual.", "tinysteno.tui", "tinysteno.main"))}
    for key in saved:
        del sys.modules[key]
    try:
        importlib.import_module("tinysteno.main")
        assert not any(k == "textual" or k.startswith("textual.") for k in sys.modules)
    finally:
        for key in [k for k in sys.modules
                    if k == "textual" or k.startswith(("textual.", "tinysteno.tui",
                                                       "tinysteno.main"))]:
            del sys.modules[key]
        sys.modules.update(saved)
