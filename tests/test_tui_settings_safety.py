"""Settings screen hardening: failures never crash or leak, no double saves, quitting asks."""
import pytest
from textual.widgets import Button

from tinysteno.tui.app import TinyStenoApp
from tinysteno.tui.confirm import ConfirmScreen
from tinysteno.tui.deps import SettingsNotApplied
from tinysteno.tui.forms import SettingsForm
from tinysteno.tui.home import HomeScreen
from tests.tui_fakes import make_deps
from tests.tui_settings_helpers import CONFIG, field, form_error, notes, open_settings, save

SECRET = "sk-stored-SECRET"


def _record_exits(app) -> list:
    calls: list = []
    real_exit = app.exit

    def exit_(*args, **kwargs):
        calls.append(kwargs)
        real_exit(*args, **kwargs)
    app.exit = exit_
    return calls


async def test_personas_that_cannot_be_listed_do_not_crash_or_leak(tmp_path, capsys):
    failing = []

    def personas():
        if failing:
            raise OSError("no personas folder")
        return ["default", "1on1"]
    deps = make_deps(tmp_path, list_personas=personas,
                     settings={"api_key": SECRET, "persona": "1on1"})
    app = TinyStenoApp(CONFIG, deps)
    async with app.run_test() as pilot:
        failing.append(True)                    # Home has listed them; Settings cannot
        screen = await open_settings(app, pilot)
        assert form_error(screen) == "Personas cannot be listed: no personas folder"
        select = field(screen, "persona")
        options = [(str(p), v) for p, v in select._options]  # pylint: disable=protected-access
        assert options == [("1on1", "1on1")]
        field(screen, "model").value = "m2"
        await save(pilot)
        assert deps.saved == [{"model": "m2"}]
    out, err = capsys.readouterr()
    assert SECRET not in out + err + repr(notes(app)) + repr(app.events)


async def test_unexpected_error_in_an_action_is_shown_not_a_crash(tmp_path, monkeypatch):
    def broken(self):
        raise RuntimeError(SECRET)
    monkeypatch.setattr(SettingsForm, "changes", broken)
    deps = make_deps(tmp_path, settings={"api_key": SECRET})
    app = TinyStenoApp(CONFIG, deps)
    async with app.run_test() as pilot:
        screen = await open_settings(app, pilot)
        await save(pilot)
        assert app.screen is screen and app.is_running
        assert form_error(screen) == "Unexpected error: RuntimeError"
        await pilot.press("escape")             # has_changes fails too: still no crash
        await pilot.pause()
        assert app.is_running


async def test_two_queued_save_presses_save_once_and_stop_at_home(tmp_path):
    deps = make_deps(tmp_path)
    app = TinyStenoApp(CONFIG, deps)
    async with app.run_test() as pilot:
        screen = await open_settings(app, pilot)
        field(screen, "model").value = "m2"
        button = screen.query_one("#save", Button)
        button.press()
        button.press()
        await pilot.pause()
        await pilot.pause()
        assert deps.saved == [{"model": "m2"}]
        assert isinstance(app.screen, HomeScreen) and len(app.screen_stack) == 2


async def test_two_back_actions_leave_only_once(tmp_path):
    app = TinyStenoApp(CONFIG, make_deps(tmp_path))
    async with app.run_test() as pilot:
        screen = await open_settings(app, pilot)
        screen.action_back()
        screen.action_back()
        await pilot.pause()
        assert isinstance(app.screen, HomeScreen) and len(app.screen_stack) == 2


async def test_quit_with_unsaved_changes_asks_and_cancel_stays(tmp_path):
    app = TinyStenoApp(CONFIG, make_deps(tmp_path))
    exits = _record_exits(app)
    async with app.run_test() as pilot:
        screen = await open_settings(app, pilot)
        field(screen, "model").value = "m2"
        field(screen, "auto_title").focus()
        await pilot.press("q")
        await pilot.pause()
        assert isinstance(app.screen, ConfirmScreen)
        assert str(app.screen.query_one("#confirm-title").content) == (
            "Quit and discard unsaved changes?")
        await pilot.press("escape")
        await pilot.pause()
        assert app.screen is screen and not exits
        await pilot.press("ctrl+q")
        await pilot.pause()
        app.screen.query_one("#confirm").focus()
        await pilot.press("enter")
        await pilot.pause()
        assert exits


async def test_quit_without_unsaved_changes_does_not_ask(tmp_path):
    app = TinyStenoApp(CONFIG, make_deps(tmp_path))
    exits = _record_exits(app)
    async with app.run_test() as pilot:
        await open_settings(app, pilot)
        await app.run_action("quit")
        await pilot.pause()
        assert exits


async def test_exit_signal_with_unsaved_changes_quits_at_once(tmp_path):
    app = TinyStenoApp(CONFIG, make_deps(tmp_path))
    exits = _record_exits(app)
    async with app.run_test() as pilot:
        screen = await open_settings(app, pilot)
        field(screen, "model").value = "m2"
        app._on_exit_signal()                   # pylint: disable=protected-access
        await pilot.pause()
        assert exits and not isinstance(app.screen, ConfirmScreen)


QUOTING = ('while scanning a simple key\n  in "<unicode string>", line 2, column 1:\n'
           '    api_key sk-quoted-SECRET\n    ^')


@pytest.mark.parametrize("error", [
    ValueError(QUOTING), OSError(QUOTING), RuntimeError(QUOTING), SettingsNotApplied(QUOTING),
])
async def test_no_save_error_shows_quoted_file_content(tmp_path, error):
    deps = make_deps(tmp_path, save_error=error)
    app = TinyStenoApp(CONFIG, deps)
    async with app.run_test() as pilot:
        screen = await open_settings(app, pilot)
        field(screen, "model").value = "m2"
        await save(pilot)
        shown = repr(notes(app)) + (form_error(screen) if app.screen is screen else "")
        assert "SECRET" not in shown and "line 2, column 1" in shown
