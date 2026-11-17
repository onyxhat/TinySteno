"""The Menu: opened with m on Home or from the command palette, never during a recording."""
import threading

from textual.command import CommandPalette
from textual.widgets import OptionList

from tinysteno.tui.app import TinyStenoApp
from tinysteno.tui.confirm import ConfirmScreen
from tinysteno.tui.home import HomeScreen
from tinysteno.tui.menu import DESTINATION_ORDER, MenuScreen
from tinysteno.tui.processing import ProcessingScreen
from tinysteno.tui.recording import RecordingScreen
from tinysteno.tui.settings import SettingsScreen
from tests.tui_fakes import make_deps

CONFIG = {"persona": "1on1", "session_timeout_minutes": 120}


async def test_m_opens_the_menu_and_escape_closes_it(tmp_path):
    app = TinyStenoApp(CONFIG, make_deps(tmp_path))
    async with app.run_test() as pilot:
        await pilot.press("m")
        await pilot.pause()
        assert isinstance(app.screen, MenuScreen)
        menu = app.screen.query_one("#menu", OptionList)
        ids = [menu.get_option_at_index(i).id for i in range(menu.option_count)]
        assert ids == [d for d in DESTINATION_ORDER if d in app.destinations]
        assert "settings" in ids
        await pilot.press("escape")
        await pilot.pause()
        assert isinstance(app.screen, HomeScreen)
        assert not app.events


async def test_menu_lists_destinations_in_the_fixed_order_with_their_labels(tmp_path):
    app = TinyStenoApp(CONFIG, make_deps(tmp_path))
    labels = {"settings": "Settings", "setup": "Setup guide", "personas": "Personas",
              "recordings": "Recordings", "checks": "System check"}
    assert DESTINATION_ORDER == tuple(labels)
    # Later tasks register these; the menu must keep the fixed order whatever was added first.
    for dest in ("checks", "personas"):
        app.destinations[dest] = (labels[dest], SettingsScreen)
    async with app.run_test() as pilot:
        await pilot.press("m")
        await pilot.pause()
        menu = app.screen.query_one("#menu", OptionList)
        options = [menu.get_option_at_index(i) for i in range(menu.option_count)]
        assert [o.id for o in options] == ["settings", "personas", "checks"]
        assert [str(o.prompt) for o in options] == ["Settings", "Personas", "System check"]


async def test_choosing_settings_in_the_menu_opens_it(tmp_path):
    app = TinyStenoApp(CONFIG, make_deps(tmp_path))
    async with app.run_test() as pilot:
        await pilot.press("m")
        await pilot.pause()
        await pilot.press("enter")          # the highlighted first entry: Settings
        await pilot.pause()
        assert isinstance(app.screen, SettingsScreen)


async def test_menu_does_not_open_while_typing_in_the_name_field(tmp_path):
    app = TinyStenoApp(CONFIG, make_deps(tmp_path))
    async with app.run_test() as pilot:
        app.screen.query_one("#name").focus()
        await pilot.press(*"mm")
        await pilot.pause()
        assert isinstance(app.screen, HomeScreen)
        assert app.screen.query_one("#name").value == "mm"


async def test_palette_lists_the_destinations(tmp_path):
    app = TinyStenoApp(CONFIG, make_deps(tmp_path))
    async with app.run_test():
        titles = {c.title for c in app.get_system_commands(app.screen)}
        assert "Settings" in titles
        assert {label for label, _ in app.destinations.values()} <= titles


async def test_palette_command_opens_the_destination(tmp_path):
    app = TinyStenoApp(CONFIG, make_deps(tmp_path))
    async with app.run_test() as pilot:
        await pilot.press("ctrl+p")
        await pilot.pause()
        assert isinstance(app.screen, CommandPalette)
        await pilot.press(*"Settings")
        await pilot.pause(0.3)
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, SettingsScreen)


async def test_menu_is_not_reachable_from_recording(tmp_path):
    app = TinyStenoApp(CONFIG, make_deps(tmp_path))
    async with app.run_test() as pilot:
        await pilot.press("r")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert isinstance(app.screen, RecordingScreen)
        assert not app.check_action("open_menu", ())
        await app.run_action("open_menu")
        app.open_destination("settings")
        await pilot.pause()
        assert isinstance(app.screen, RecordingScreen)
        titles = {c.title for c in app.get_system_commands(app.screen)}
        assert not titles & {label for label, _ in app.destinations.values()}


async def test_menu_is_not_reachable_from_processing(tmp_path):
    gate = threading.Event()

    def slow(*args, **kwargs):  # pylint: disable=unused-argument
        gate.wait(5)
        raise RuntimeError("stopped by the test")

    wav = tmp_path / "Old.wav"
    wav.write_bytes(b"")
    app = TinyStenoApp(CONFIG, make_deps(tmp_path, run_pipeline=slow))
    async with app.run_test() as pilot:
        try:
            app.process_file(str(wav), "default")
            await pilot.pause()
            assert isinstance(app.screen, ProcessingScreen)
            assert not app.check_action("open_menu", ())
            await app.run_action("open_menu")
            app.open_destination("settings")
            await pilot.pause()
            assert isinstance(app.screen, ProcessingScreen)
            titles = {c.title for c in app.get_system_commands(app.screen)}
            assert "Settings" not in titles
        finally:
            gate.set()
        await app.workers.wait_for_complete()
        await pilot.pause()


async def test_menu_is_not_reachable_from_a_dialog(tmp_path):
    app = TinyStenoApp(CONFIG, make_deps(tmp_path))
    async with app.run_test() as pilot:
        app.push_screen(ConfirmScreen("Delete x?", "Delete"))
        await pilot.pause()
        assert not app.check_action("open_menu", ())
        assert "Settings" not in {c.title for c in app.get_system_commands(app.screen)}
        app.open_destination("settings")
        await pilot.pause()
        assert isinstance(app.screen, ConfirmScreen)


async def test_a_destination_already_open_is_not_opened_again(tmp_path):
    app = TinyStenoApp(CONFIG, make_deps(tmp_path))
    async with app.run_test() as pilot:
        app.open_destination("settings")
        await pilot.pause()
        first = app.screen
        assert isinstance(first, SettingsScreen)
        app.open_destination("settings")
        await pilot.pause()
        assert app.screen is first
        assert sum(isinstance(s, SettingsScreen) for s in app.screen_stack) == 1


async def test_unregistered_destination_is_ignored(tmp_path):
    app = TinyStenoApp(CONFIG, make_deps(tmp_path))
    async with app.run_test() as pilot:
        app.open_destination("nowhere")
        await pilot.pause()
        assert isinstance(app.screen, HomeScreen)


async def test_confirm_starts_on_cancel_and_escape_cancels(tmp_path):
    app = TinyStenoApp(CONFIG, make_deps(tmp_path))
    answers: list = []
    async with app.run_test() as pilot:
        app.push_screen(ConfirmScreen("Delete [b]x[/b]?", "Delete"), answers.append)
        await pilot.pause()
        assert app.focused.id == "cancel"
        assert "[b]x[/b]" in str(app.screen.query_one("#confirm-title").content)
        await pilot.press("escape")
        await pilot.pause()
        app.push_screen(ConfirmScreen("Delete x?", "Delete"), answers.append)
        await pilot.pause()
        await pilot.press("enter")              # Enter activates the focused button: Cancel
        await pilot.pause()
        app.push_screen(ConfirmScreen("Delete x?", "Delete"), answers.append)
        await pilot.pause()
        app.screen.query_one("#confirm").focus()
        await pilot.press("enter")
        await pilot.pause()
        assert answers == [False, False, True]
        assert isinstance(app.screen, HomeScreen)
