"""A second activation queued behind the first never closes the screen below.

Textual's ``Screen.dismiss()`` pops whatever screen is on top, so a second Enter, Back or
press that arrives before the first has been handled must be ignored by a screen that is
already leaving.
"""
import pytest
from textual import events
from textual.widgets import Button

from tinysteno.tui.app import TinyStenoApp
from tinysteno.tui.checks import ChecksScreen
from tinysteno.tui.confirm import ConfirmScreen
from tinysteno.tui.home import HomeScreen
from tinysteno.tui.menu import MenuScreen
from tinysteno.tui.settings import SettingsScreen
from tests.tui_fakes import make_deps

CONFIG = {"persona": "default", "session_timeout_minutes": 120}
DESTINATIONS = ("settings", "setup", "personas", "recordings", "checks")


def send_together(app, *keys: str) -> None:
    """Queue key events for the app without letting it handle any of them in between."""
    for key in keys:
        event = events.Key(key, None)
        event.set_sender(app)
        app._driver.send_message(event)  # pylint: disable=protected-access


async def _settle(pilot):
    for _ in range(5):
        await pilot.pause()


def _stack(app) -> list[type]:
    return [type(s) for s in app.screen_stack[1:]]   # under the first: Textual's blank Screen


async def _open(app, pilot, dest: str):
    app.open_destination(dest)
    await _settle(pilot)
    await app.workers.wait_for_complete()
    await _settle(pilot)
    return app.screen


async def test_two_enters_in_the_menu_open_one_destination_over_home(tmp_path):
    app = TinyStenoApp(CONFIG, make_deps(tmp_path))
    async with app.run_test() as pilot:
        await pilot.press("m")
        await _settle(pilot)
        assert isinstance(app.screen, MenuScreen)
        send_together(app, "enter", "enter")
        await _settle(pilot)
        assert _stack(app) == [HomeScreen, SettingsScreen]


async def test_two_escapes_in_the_menu_close_only_the_menu(tmp_path):
    app = TinyStenoApp(CONFIG, make_deps(tmp_path))
    async with app.run_test() as pilot:
        await pilot.press("m")
        await _settle(pilot)
        send_together(app, "escape", "escape")
        await _settle(pilot)
        assert _stack(app) == [HomeScreen]


async def test_two_closes_queued_in_the_menu_close_only_the_menu(tmp_path):
    app = TinyStenoApp(CONFIG, make_deps(tmp_path))
    async with app.run_test() as pilot:
        await pilot.press("m")
        await _settle(pilot)
        menu = app.screen
        menu.action_close()
        menu.action_close()
        await _settle(pilot)
        assert _stack(app) == [HomeScreen]


@pytest.mark.parametrize("button", ["cancel", "confirm"])
async def test_two_presses_in_the_confirm_dialog_close_only_the_dialog(tmp_path, button):
    app = TinyStenoApp(CONFIG, make_deps(tmp_path))
    answers: list = []
    async with app.run_test() as pilot:
        await _open(app, pilot, "checks")
        app.push_screen(ConfirmScreen("Sure?", "Yes"), answers.append)
        await _settle(pilot)
        widget = app.screen.query_one(f"#{button}", Button)
        widget.press()
        widget.press()
        await _settle(pilot)
        assert _stack(app) == [HomeScreen, ChecksScreen]
        assert answers == [button == "confirm"]


async def test_two_enters_in_the_confirm_dialog_close_only_the_dialog(tmp_path):
    app = TinyStenoApp(CONFIG, make_deps(tmp_path))
    answers: list = []
    async with app.run_test() as pilot:
        await _open(app, pilot, "checks")
        app.push_screen(ConfirmScreen("Sure?", "Yes"), answers.append)
        await _settle(pilot)
        app.screen.query_one("#confirm", Button).focus()
        await _settle(pilot)
        send_together(app, "enter", "enter")
        await _settle(pilot)
        assert _stack(app) == [HomeScreen, ChecksScreen]
        assert answers == [True]


async def test_two_cancels_queued_in_the_confirm_dialog_close_only_the_dialog(tmp_path):
    app = TinyStenoApp(CONFIG, make_deps(tmp_path))
    answers: list = []
    async with app.run_test() as pilot:
        await _open(app, pilot, "checks")
        app.push_screen(ConfirmScreen("Sure?", "Yes"), answers.append)
        await _settle(pilot)
        dialog = app.screen
        dialog.action_cancel()
        dialog.action_cancel()
        await _settle(pilot)
        assert _stack(app) == [HomeScreen, ChecksScreen]
        assert answers == [False]


@pytest.mark.parametrize("dest", DESTINATIONS)
async def test_two_escapes_on_a_management_screen_go_back_to_home_once(tmp_path, dest):
    app = TinyStenoApp(CONFIG, make_deps(tmp_path))
    async with app.run_test() as pilot:
        await _open(app, pilot, dest)
        send_together(app, "escape", "escape")
        await _settle(pilot)
        assert _stack(app) == [HomeScreen]


@pytest.mark.parametrize("dest", DESTINATIONS)
async def test_two_back_actions_queued_on_a_management_screen_leave_once(tmp_path, dest):
    app = TinyStenoApp(CONFIG, make_deps(tmp_path))
    async with app.run_test() as pilot:
        screen = await _open(app, pilot, dest)
        back = screen.action_skip if dest == "setup" else screen.action_back
        back()
        back()
        await _settle(pilot)
        assert _stack(app) == [HomeScreen]


@pytest.mark.parametrize("dest,button", [
    ("settings", "back"), ("setup", "skip"), ("checks", "back"),
])
async def test_two_presses_of_back_leave_once(tmp_path, dest, button):
    app = TinyStenoApp(CONFIG, make_deps(tmp_path))
    async with app.run_test() as pilot:
        screen = await _open(app, pilot, dest)
        widget = screen.query_one(f"#{button}", Button)
        widget.press()
        widget.press()
        await _settle(pilot)
        assert _stack(app) == [HomeScreen]
