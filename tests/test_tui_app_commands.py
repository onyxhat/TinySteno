"""Command palette contents of the app, and proof that nothing in it saves a screenshot."""
import asyncio
import inspect
from pathlib import Path

import pytest
from textual.app import App

from tinysteno.tui.app import TinyStenoApp
from tests.tui_fakes import make_deps

CONFIG = {"persona": "1on1", "session_timeout_minutes": 120}

# Every command the app exposes on the Home screen. A new or renamed Textual default fails
# test_palette_titles_are_pinned, so someone has to decide whether it belongs in the app.
# "Settings" is the app's own: one command per registered Menu destination.
EXPECTED_TITLES = {"Theme", "Quit", "Keys", "Maximize", "Settings"}


@pytest.fixture(autouse=True)
def _contain_writes(tmp_path, monkeypatch):
    """Textual saves screenshots under ~/Downloads and cwd: keep any write inside tmp_path."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)


def _record_screenshots(app, monkeypatch) -> list:
    """Replace every screenshot-saving entry point with a recorder."""
    calls: list = []
    for name in ("deliver_screenshot", "save_screenshot"):
        monkeypatch.setattr(app, name, lambda *a, _n=name, **k: calls.append(_n))
    return calls


async def test_command_palette_has_no_screenshot_command(tmp_path):
    app = TinyStenoApp(CONFIG, make_deps(tmp_path))
    async with app.run_test():
        titles = [c.title for c in app.get_system_commands(app.screen)]
        assert "Screenshot" not in titles
        assert {"Theme", "Quit", "Keys"} <= set(titles)


async def test_palette_titles_are_pinned(tmp_path):
    app = TinyStenoApp(CONFIG, make_deps(tmp_path))
    async with app.run_test():
        titles = {c.title for c in app.get_system_commands(app.screen)}
        assert titles == EXPECTED_TITLES


async def test_screenshot_action_saves_nothing(tmp_path, monkeypatch):
    app = TinyStenoApp(CONFIG, make_deps(tmp_path))
    calls = _record_screenshots(app, monkeypatch)
    async with app.run_test() as pilot:
        await app.run_action("screenshot")
        await pilot.pause()
    assert not calls
    assert not list(tmp_path.rglob("*.svg"))


async def test_no_palette_command_saves_a_screenshot(tmp_path, monkeypatch):
    app = TinyStenoApp(CONFIG, make_deps(tmp_path))
    calls: list = []
    # export_screenshot is what deliver_screenshot calls first; save_screenshot is the other
    # way to write one. Neither may run, whatever route a command takes.
    for name in ("save_screenshot", "export_screenshot"):
        monkeypatch.setattr(app, name, lambda *a, _n=name, **k: calls.append(_n))
    async with app.run_test() as pilot:
        base = len(app.screen_stack)
        # Textual's own list, unfiltered, so its Screenshot command is included.
        for command in list(App.get_system_commands(app, app.screen)):
            if command.title == "Quit":      # exits the app; covered by its own tests
                continue
            result = command.callback()
            if inspect.isawaitable(result):
                await result
            await pilot.pause()
            while len(app.screen_stack) > base:   # e.g. the theme picker
                app.pop_screen()
                await pilot.pause()
        await asyncio.sleep(0.3)             # the Screenshot command fires on a 0.1 s timer
        await pilot.pause()
    assert not calls
    assert not list(tmp_path.rglob("*.svg"))
