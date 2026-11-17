"""Quitting asks when any open screen, not only the top one, has unsaved changes."""
import pytest

from tinysteno.tui.app import TinyStenoApp
from tinysteno.tui.checks import ChecksScreen
from tinysteno.tui.confirm import ConfirmScreen
from tinysteno.tui.recordings import RecordingsScreen
from tinysteno.tui.settings import SettingsScreen
from tinysteno.tui.setup import SetupScreen
from tests.tui_fakes import make_deps
from tests.tui_settings_helpers import CONFIG, field, open_settings


def _record_exits(app) -> list:
    calls: list = []
    real_exit = app.exit

    def exit_(*args, **kwargs):
        calls.append(kwargs)
        real_exit(*args, **kwargs)
    app.exit = exit_
    return calls


async def _settle(pilot):
    for _ in range(3):
        await pilot.pause()


async def _edit_settings_then_open(app, pilot, dest: str):
    """Settings with an unsaved edit, then another destination over it (as the palette
    opens it)."""
    settings = await open_settings(app, pilot)
    field(settings, "model").value = "edited"
    app.open_destination(dest)
    await _settle(pilot)
    await app.workers.wait_for_complete()
    await _settle(pilot)
    assert not isinstance(app.screen, SettingsScreen)
    return app.screen


@pytest.mark.parametrize("dest,kind", [("checks", ChecksScreen),
                                       ("recordings", RecordingsScreen)])
async def test_q_asks_when_a_screen_below_has_unsaved_changes(tmp_path, dest, kind):
    app = TinyStenoApp(CONFIG, make_deps(tmp_path))
    exits = _record_exits(app)
    async with app.run_test() as pilot:
        top = await _edit_settings_then_open(app, pilot, dest)
        assert isinstance(top, kind)
        await pilot.press("q")
        await _settle(pilot)
        assert isinstance(app.screen, ConfirmScreen)
        assert str(app.screen.query_one("#confirm-title").content) == (
            "Quit and discard unsaved changes?")
        await pilot.press("enter")              # focus starts on Cancel
        await _settle(pilot)
        assert app.screen is top and not exits
        await pilot.press("q")
        await _settle(pilot)
        app.screen.query_one("#confirm").focus()
        await pilot.press("enter")
        await _settle(pilot)
        assert exits


async def test_q_asks_for_a_reopened_setup_guide_under_another_screen(tmp_path):
    app = TinyStenoApp(CONFIG, make_deps(tmp_path))
    exits = _record_exits(app)
    async with app.run_test() as pilot:
        app.open_destination("setup")
        await _settle(pilot)
        assert isinstance(app.screen, SetupScreen)
        field(app.screen, "model").value = "edited"
        app.open_destination("checks")
        await _settle(pilot)
        await app.run_action("quit")
        await _settle(pilot)
        assert isinstance(app.screen, ConfirmScreen) and not exits


async def test_q_does_not_ask_when_no_open_screen_has_changes(tmp_path):
    app = TinyStenoApp(CONFIG, make_deps(tmp_path))
    exits = _record_exits(app)
    async with app.run_test() as pilot:
        await open_settings(app, pilot)
        app.open_destination("checks")
        await _settle(pilot)
        await app.run_action("quit")
        await _settle(pilot)
        assert exits


async def test_an_exit_signal_never_asks_even_with_changes_below(tmp_path):
    app = TinyStenoApp(CONFIG, make_deps(tmp_path))
    exits = _record_exits(app)
    async with app.run_test() as pilot:
        await _edit_settings_then_open(app, pilot, "checks")
        app._on_exit_signal()                   # pylint: disable=protected-access
        await _settle(pilot)
        assert exits and not isinstance(app.screen, ConfirmScreen)
