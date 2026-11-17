"""Setup guide: Save and continue writes only what the user chose, plus what the file lacks."""
from textual.widgets import Button, Input, Static

from tinysteno.config import DEFAULT_CONFIG
from tinysteno.tui.app import TinyStenoApp
from tinysteno.tui.checks import ChecksScreen
from tinysteno.tui.home import HomeScreen
from tinysteno.tui.personas import PersonasScreen
from tinysteno.tui.setup import SetupScreen
from tests.tui_fakes import make_deps
from tests.tui_settings_helpers import notes

CONFIG = {"persona": "default", "session_timeout_minutes": 120}
SECRET = "sk-test-DO-NOT-SHOW-123"


async def _settle(pilot):
    for _ in range(3):
        await pilot.pause()


def _full_file() -> dict:
    return dict(DEFAULT_CONFIG)


async def _open_guide(app, pilot) -> SetupScreen:
    app.open_destination("setup")
    await _settle(pilot)
    assert isinstance(app.screen, SetupScreen)
    return app.screen


async def test_a_full_default_file_with_nothing_touched_sends_nothing(tmp_path):
    settings = _full_file()
    deps = make_deps(tmp_path, settings=settings)
    app = TinyStenoApp(CONFIG, deps)
    async with app.run_test() as pilot:
        await _open_guide(app, pilot)
        await pilot.click("#finish")
        await _settle(pilot)
        assert not deps.saved
        assert settings == _full_file()
        assert isinstance(app.screen, ChecksScreen)
        assert ("Settings saved", "information") not in notes(app)


async def test_a_first_run_file_lacking_an_essential_key_gets_it(tmp_path):
    settings = _full_file()
    del settings["model"]
    deps = make_deps(tmp_path, settings=settings, first_run=True)
    app = TinyStenoApp(CONFIG, deps)
    async with app.run_test() as pilot:
        await _settle(pilot)
        assert isinstance(app.screen, SetupScreen)
        app.screen.query_one("#set-base_url", Input).value = "http://llm.test/v1"
        await pilot.click("#finish")
        await _settle(pilot)
        assert deps.saved == [{"base_url": "http://llm.test/v1", "model": "llama3.2:3b"}]
        assert isinstance(app.screen, ChecksScreen)


async def test_a_key_changed_in_the_file_while_the_guide_is_open_is_kept(tmp_path):
    settings = _full_file()
    deps = make_deps(tmp_path, settings=settings)
    app = TinyStenoApp(CONFIG, deps)
    async with app.run_test() as pilot:
        await _open_guide(app, pilot)
        settings["model"] = "edited-by-hand"          # e.g. in an editor, meanwhile
        settings["api_key"] = "hand-key"
        await pilot.click("#finish")
        await _settle(pilot)
        assert not deps.saved
        assert settings["model"] == "edited-by-hand"
        assert settings["api_key"] == "hand-key"


async def test_a_key_removed_from_the_file_meanwhile_is_written_from_the_form(tmp_path):
    settings = _full_file()
    deps = make_deps(tmp_path, settings=settings)
    app = TinyStenoApp(CONFIG, deps)
    async with app.run_test() as pilot:
        await _open_guide(app, pilot)
        del settings["output_folder"]
        await pilot.click("#finish")
        await _settle(pilot)
        assert deps.saved == [{"output_folder": "meetings"}]


async def test_a_default_persona_set_while_the_guide_is_open_is_not_reverted(tmp_path):
    settings = _full_file()
    deps = make_deps(tmp_path, settings=settings)
    app = TinyStenoApp(CONFIG, deps)
    async with app.run_test() as pilot:
        await _open_guide(app, pilot)
        app.open_destination("personas")       # as the palette does
        await _settle(pilot)
        await app.workers.wait_for_complete()
        await _settle(pilot)
        assert isinstance(app.screen, PersonasScreen)
        await pilot.press("down", "d")         # the second row: 1on1
        await _settle(pilot)
        assert settings["persona"] == "1on1"
        await pilot.press("escape")
        await _settle(pilot)
        assert isinstance(app.screen, SetupScreen)
        await pilot.click("#finish")
        await _settle(pilot)
        assert [s.get("persona") for s in deps.saved] == ["1on1"]
        assert settings["persona"] == "1on1"
        assert isinstance(app.screen, ChecksScreen)


async def test_the_file_cannot_be_read_at_finish_writes_nothing(tmp_path):
    settings = _full_file()
    state = {"broken": False}

    def load():
        if state["broken"]:
            raise ValueError("Config file is not valid YAML (line 3, column 1): /x/config.yaml")
        return dict(settings)

    deps = make_deps(tmp_path, load_settings=load)
    app = TinyStenoApp(CONFIG, deps)
    async with app.run_test() as pilot:
        screen = await _open_guide(app, pilot)
        screen.query_one("#set-model", Input).value = "other"
        state["broken"] = True
        await pilot.click("#finish")
        await _settle(pilot)
        assert not deps.saved
        assert app.screen is screen
        error = str(screen.query_one("#form-error", Static).content)
        assert error.startswith("Config file cannot be read: Config file is not valid YAML")


async def test_an_unreadable_file_on_opening_shows_no_form_and_cannot_be_saved(tmp_path):
    def load():
        raise ValueError("Config file is not valid YAML (line 2, column 5): /x/config.yaml")

    deps = make_deps(tmp_path, load_settings=load, first_run=True)
    app = TinyStenoApp(CONFIG, deps)
    async with app.run_test() as pilot:
        await _settle(pilot)
        screen = app.screen
        assert isinstance(screen, SetupScreen)
        assert not screen.query("#setup-form")
        assert not screen.query("Input")
        assert str(screen.query_one("#load-error", Static).content) == (
            "Config file cannot be read: Config file is not valid YAML (line 2, column 5): "
            "/x/config.yaml")
        path = str(screen.query_one("#config-path", Static).content)
        assert path.startswith("File: ") and path.endswith("config.yaml")
        assert screen.query_one("#finish", Button).disabled
        await pilot.press("ctrl+s")
        screen.action_finish()
        await _settle(pilot)
        assert not deps.saved
        assert app.screen is screen
        await pilot.click("#skip")
        await _settle(pilot)
        assert isinstance(app.screen, HomeScreen)
        assert not deps.saved


async def test_a_persona_listing_failure_does_not_hide_the_form(tmp_path):
    state = {"broken": False}

    def personas():
        if state["broken"]:
            raise PermissionError("no access")
        return ["default"]

    deps = make_deps(tmp_path, settings=_full_file(), list_personas=personas)
    app = TinyStenoApp(CONFIG, deps)
    async with app.run_test() as pilot:
        state["broken"] = True
        screen = await _open_guide(app, pilot)
        assert screen.query("#setup-form")
        assert not screen.query("#load-error")
        error = str(screen.query_one("#form-error", Static).content)
        assert error == "Personas cannot be listed: no access"


async def _type_secret_and_finish(app, pilot):
    await _settle(pilot)
    app.screen.query_one("#set-api_key", Input).value = SECRET
    await pilot.click("#finish")
    await _settle(pilot)


async def test_a_typed_secret_is_shown_nowhere_after_saving(tmp_path):
    deps = make_deps(tmp_path, first_run=True)
    app = TinyStenoApp(CONFIG, deps)
    async with app.run_test() as pilot:
        await _type_secret_and_finish(app, pilot)
        assert deps.saved and deps.saved[0]["api_key"] == SECRET
        assert isinstance(app.screen, ChecksScreen)
        assert all(SECRET not in message for message, _ in notes(app))
        assert SECRET not in repr(app.events)


async def test_a_typed_secret_is_shown_nowhere_when_saving_fails(tmp_path):
    deps = make_deps(tmp_path, first_run=True, save_error=ValueError("cannot save"))
    app = TinyStenoApp(CONFIG, deps)
    async with app.run_test() as pilot:
        await _type_secret_and_finish(app, pilot)
        assert isinstance(app.screen, SetupScreen)
        assert str(app.screen.query_one("#form-error", Static).content) == "cannot save"
        assert all(SECRET not in message for message, _ in notes(app))
        assert SECRET not in repr(app.events)
