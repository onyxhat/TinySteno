"""Command palette contents of the app."""
from pathlib import Path

from tinysteno.tui.app import TinyStenoApp
from tests.tui_fakes import make_deps

CONFIG = {"persona": "1on1", "session_timeout_minutes": 120}


async def test_command_palette_has_no_screenshot_command(tmp_path):
    app = TinyStenoApp(CONFIG, make_deps(tmp_path))
    async with app.run_test():
        titles = [c.title for c in app.get_system_commands(app.screen)]
        assert "Screenshot" not in titles
        assert {"Theme", "Quit", "Keys"} <= set(titles)


async def test_screenshot_action_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # Textual saves screenshots to ~/Downloads: redirect home so a regression lands in tmp_path.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    app = TinyStenoApp(CONFIG, make_deps(tmp_path))
    async with app.run_test() as pilot:
        await app.run_action("screenshot")
        await pilot.pause()
    assert not list(tmp_path.rglob("*.svg"))
