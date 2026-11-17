"""First-run Setup guide: the essential settings, shown once, re-openable from the Menu."""
import pytest
from textual.widgets import Input, Static

from tinysteno.doctor import Check
from tinysteno.settings import SETTINGS
from tinysteno.tui.app import TinyStenoApp
from tinysteno.tui.checks import ChecksScreen
from tinysteno.tui.confirm import ConfirmScreen
from tinysteno.tui.deps import SettingsNotApplied
from tinysteno.tui.home import HomeScreen
from tinysteno.tui.setup import NO_PERSONAS_NOTE, SetupScreen
from tests.tui_fakes import make_deps

CONFIG = {"persona": "1on1", "session_timeout_minutes": 120}
ESSENTIAL = [s.key for s in SETTINGS if s.essential]


async def _settle(pilot):
    for _ in range(3):
        await pilot.pause()


def _error(screen) -> str:
    return str(screen.query_one("#form-error", Static).content)


async def test_first_run_opens_the_guide_over_home(tmp_path):
    app = TinyStenoApp(CONFIG, make_deps(tmp_path, first_run=True))
    async with app.run_test() as pilot:
        await _settle(pilot)
        assert isinstance(app.screen, SetupScreen)
        assert isinstance(app.screen_stack[-2], HomeScreen)
        assert app.screen_stack.count(app.screen) == 1


async def test_not_first_run_goes_straight_to_home(tmp_path):
    app = TinyStenoApp(CONFIG, make_deps(tmp_path))
    async with app.run_test() as pilot:
        await _settle(pilot)
        assert isinstance(app.screen, HomeScreen)


async def test_now_skips_the_guide(tmp_path):
    app = TinyStenoApp(CONFIG, make_deps(tmp_path, first_run=True), autostart=True)
    async with app.run_test() as pilot:
        await _settle(pilot)
        assert not isinstance(app.screen, SetupScreen)
        assert not any(isinstance(s, SetupScreen) for s in app.screen_stack)
        assert app.events and app.events[0][0] == "record"


async def test_guide_shows_only_essential_settings(tmp_path):
    app = TinyStenoApp(CONFIG, make_deps(tmp_path, first_run=True))
    async with app.run_test() as pilot:
        await _settle(pilot)
        shown = [w.id[4:] for w in app.screen.query("#setup-form *")
                 if w.id and w.id.startswith("set-")]
        assert shown == ESSENTIAL
        assert app.screen.query_one("#finish").label.plain == "Save and continue"
        assert app.screen.query_one("#skip").label.plain == "Skip for now"


async def test_finish_on_a_file_lacking_them_saves_every_essential_key_then_checks(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    # The fake file is empty, so every essential key is lacking and is written; a file that
    # holds them gets only the changes (tests/test_tui_setup_finish.py).
    deps = make_deps(tmp_path, first_run=True, checks=[Check("Python", True)])
    app = TinyStenoApp(CONFIG, deps)
    async with app.run_test() as pilot:
        await _settle(pilot)
        app.screen.query_one("#set-obsidian_vault", Input).value = str(vault)
        await pilot.click("#finish")
        await _settle(pilot)
        assert len(deps.saved) == 1
        assert set(deps.saved[0]) == set(ESSENTIAL)
        assert deps.saved[0]["obsidian_vault"] == str(vault)
        assert isinstance(app.screen, ChecksScreen)
        await pilot.press("escape")
        await _settle(pilot)
        assert isinstance(app.screen, HomeScreen)
        assert not any(isinstance(s, SetupScreen) for s in app.screen_stack)


async def test_skip_writes_nothing_and_shows_home(tmp_path):
    deps = make_deps(tmp_path, first_run=True)
    app = TinyStenoApp(CONFIG, deps)
    async with app.run_test() as pilot:
        await _settle(pilot)
        await pilot.click("#skip")
        await _settle(pilot)
        assert isinstance(app.screen, HomeScreen)
        assert not deps.saved


async def test_escape_returns_to_home_without_writing(tmp_path):
    deps = make_deps(tmp_path, first_run=True)
    app = TinyStenoApp(CONFIG, deps)
    async with app.run_test() as pilot:
        await _settle(pilot)
        await pilot.press("escape")
        await _settle(pilot)
        assert isinstance(app.screen, HomeScreen)
        assert not deps.saved


async def test_invalid_vault_is_refused(tmp_path):
    deps = make_deps(tmp_path, first_run=True)
    app = TinyStenoApp(CONFIG, deps)
    async with app.run_test() as pilot:
        await _settle(pilot)
        app.screen.query_one("#set-obsidian_vault", Input).value = ""
        await pilot.click("#finish")
        await _settle(pilot)
        assert isinstance(app.screen, SetupScreen)
        assert not deps.saved
        assert "Obsidian vault" in _error(app.screen)
        assert app.focused is not None and app.focused.id == "set-obsidian_vault"


async def test_save_value_error_is_shown_and_nothing_is_left_half_done(tmp_path):
    deps = make_deps(tmp_path, first_run=True, save_error=ValueError("vault is not a folder"))
    app = TinyStenoApp(CONFIG, deps)
    async with app.run_test() as pilot:
        await _settle(pilot)
        app.screen.query_one("#set-obsidian_vault", Input).value = "x"
        await pilot.click("#finish")
        await _settle(pilot)
        assert isinstance(app.screen, SetupScreen)
        assert "vault is not a folder" in _error(app.screen)


async def test_save_os_error_is_shown(tmp_path):
    deps = make_deps(tmp_path, first_run=True, save_error=PermissionError("read-only"))
    app = TinyStenoApp(CONFIG, deps)
    async with app.run_test() as pilot:
        await _settle(pilot)
        app.screen.query_one("#set-obsidian_vault", Input).value = "x"
        await pilot.click("#finish")
        await _settle(pilot)
        assert isinstance(app.screen, SetupScreen)
        assert "read-only" in _error(app.screen)


async def test_not_applied_warns_and_continues_to_checks(tmp_path):
    deps = make_deps(tmp_path, first_run=True, save_error=SettingsNotApplied("odd"))
    app = TinyStenoApp(CONFIG, deps)
    async with app.run_test() as pilot:
        await _settle(pilot)
        app.screen.query_one("#set-obsidian_vault", Input).value = "x"
        await pilot.click("#finish")
        await _settle(pilot)
        assert isinstance(app.screen, ChecksScreen)
        assert any(n.severity == "warning" for n in app._notifications)  # pylint: disable=protected-access


async def test_guide_can_be_reopened_from_the_menu(tmp_path):
    app = TinyStenoApp(CONFIG, make_deps(tmp_path))
    async with app.run_test() as pilot:
        await pilot.press("m")
        await pilot.pause()
        await pilot.press("down", "enter")      # Settings, Setup guide
        await _settle(pilot)
        assert isinstance(app.screen, SetupScreen)


async def test_quitting_from_the_first_run_guide_never_asks(tmp_path):
    app = TinyStenoApp(CONFIG, make_deps(tmp_path, first_run=True))
    async with app.run_test() as pilot:
        await _settle(pilot)
        app.screen.query_one("#set-obsidian_vault", Input).value = "edited"
        assert app.screen.has_unsaved_changes() is False


async def test_reopened_guide_reports_real_unsaved_changes(tmp_path):
    app = TinyStenoApp(CONFIG, make_deps(tmp_path))
    async with app.run_test() as pilot:
        app.open_destination("setup")
        await _settle(pilot)
        assert app.screen.has_unsaved_changes() is False
        app.screen.query_one("#set-obsidian_vault", Input).value = "edited"
        assert app.screen.has_unsaved_changes() is True


async def test_no_personas_still_completable_with_a_note(tmp_path):
    vault = tmp_path / "v"
    vault.mkdir()
    deps = make_deps(tmp_path, first_run=True, list_personas=lambda: [])
    app = TinyStenoApp(CONFIG, deps)
    async with app.run_test() as pilot:
        await _settle(pilot)
        assert str(app.screen.query_one("#setup-no-personas", Static).content) == NO_PERSONAS_NOTE
        assert NO_PERSONAS_NOTE == ("No personas found. After setup, open the menu and choose "
                                    "Personas to restore the built-in ones.")
        app.screen.query_one("#set-obsidian_vault", Input).value = str(vault)
        await pilot.click("#finish")
        await _settle(pilot)
        assert isinstance(app.screen, ChecksScreen)
        assert len(deps.saved) == 1


async def test_a_guide_reopened_in_a_first_run_session_reports_real_changes(tmp_path):
    app = TinyStenoApp(CONFIG, make_deps(tmp_path, first_run=True))
    async with app.run_test() as pilot:
        await _settle(pilot)
        await pilot.press("escape")             # skip the automatic guide
        await _settle(pilot)
        app.open_destination("setup")           # reopened from the Menu, same session
        await _settle(pilot)
        assert isinstance(app.screen, SetupScreen)
        assert app.screen.has_unsaved_changes() is False
        app.screen.query_one("#set-obsidian_vault", Input).value = "edited"
        assert app.screen.has_unsaved_changes() is True


@pytest.mark.parametrize("leave", ["escape", "skip"])
async def test_leaving_a_reopened_guide_with_edits_asks_first(tmp_path, leave):
    deps = make_deps(tmp_path)
    app = TinyStenoApp(CONFIG, deps)
    async with app.run_test() as pilot:
        app.open_destination("setup")
        await _settle(pilot)
        guide = app.screen
        guide.query_one("#set-obsidian_vault", Input).value = "edited"
        await (pilot.press("escape") if leave == "escape" else pilot.click("#skip"))
        await _settle(pilot)
        assert isinstance(app.screen, ConfirmScreen)
        assert str(app.screen.query_one("#confirm-title").content) == "Discard unsaved changes?"
        await pilot.press("enter")              # Cancel
        await _settle(pilot)
        assert app.screen is guide
        await (pilot.press("escape") if leave == "escape" else pilot.click("#skip"))
        await _settle(pilot)
        app.screen.query_one("#confirm").focus()
        await pilot.press("enter")              # Discard
        await _settle(pilot)
        assert isinstance(app.screen, HomeScreen)
        assert not deps.saved


async def test_leaving_a_reopened_guide_without_edits_does_not_ask(tmp_path):
    app = TinyStenoApp(CONFIG, make_deps(tmp_path))
    async with app.run_test() as pilot:
        app.open_destination("setup")
        await _settle(pilot)
        await pilot.press("escape")
        await _settle(pilot)
        assert isinstance(app.screen, HomeScreen)


@pytest.mark.parametrize("leave", ["escape", "skip"])
async def test_leaving_the_first_run_guide_with_edits_never_asks(tmp_path, leave):
    deps = make_deps(tmp_path, first_run=True)
    app = TinyStenoApp(CONFIG, deps)
    async with app.run_test() as pilot:
        await _settle(pilot)
        app.screen.query_one("#set-obsidian_vault", Input).value = "edited"
        await (pilot.press("escape") if leave == "escape" else pilot.click("#skip"))
        await _settle(pilot)
        assert isinstance(app.screen, HomeScreen)
        assert not deps.saved


async def test_the_checks_shown_after_finish_are_not_opened_twice(tmp_path):
    deps = make_deps(tmp_path, first_run=True, checks=[Check("Python", True)])
    app = TinyStenoApp(CONFIG, deps)
    async with app.run_test() as pilot:
        await _settle(pilot)
        app.screen.query_one("#set-obsidian_vault", Input).value = "x"
        await pilot.click("#finish")
        await _settle(pilot)
        checks = app.screen
        assert isinstance(checks, ChecksScreen)
        app.open_destination("checks")          # as the palette does
        await _settle(pilot)
        assert app.screen is checks
        assert sum(isinstance(s, ChecksScreen) for s in app.screen_stack) == 1
        assert [type(s) for s in app.screen_stack[1:]] == [HomeScreen, ChecksScreen]
