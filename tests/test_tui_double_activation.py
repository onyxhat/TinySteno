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
from tinysteno.tui.recording import RecordingScreen
from tinysteno.tui.settings import SettingsScreen
from tests.tui_fakes import FakeRecorder, make_deps

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


async def test_two_enters_on_a_recent_recording_start_one_run(tmp_path):
    (tmp_path / "Old.wav").write_bytes(b"")
    deps = make_deps(tmp_path)
    app = TinyStenoApp(CONFIG, deps)
    async with app.run_test() as pilot:
        app.screen.query_one("#recent").focus()
        await _settle(pilot)
        send_together(app, "enter", "enter")
        await _settle(pilot)
        await app.workers.wait_for_complete()
        await _settle(pilot)
        assert len(deps.pipeline_kwargs) == 1
        assert [e[0] for e in app.events].count("process") == 1


async def test_process_file_is_refused_while_that_recording_is_being_processed(tmp_path):
    wav = tmp_path / "Old.wav"
    wav.write_bytes(b"")
    deps = make_deps(tmp_path)
    app = TinyStenoApp(CONFIG, deps)
    async with app.run_test() as pilot:
        app.ledger.begin_run(str(wav))           # e.g. a run the app still has going
        app.process_file(str(wav), "default")
        await _settle(pilot)
        assert isinstance(app.screen, HomeScreen)
        assert not deps.pipeline_kwargs and not app.events


def _counting_recorders(tmp_path) -> tuple[list, dict]:
    """A make_recorder that keeps every FakeRecorder it makes."""
    made: list = []

    def make():
        made.append(FakeRecorder(tmp_path))
        return made[-1]
    return made, {"make_recorder": make}


async def _one_recording_then_quit(app, pilot, made) -> None:
    await _settle(pilot)
    await app.workers.wait_for_complete()
    await _settle(pilot)
    assert [type(s) for s in app.screen_stack[1:]] == [HomeScreen, RecordingScreen]
    assert len(made) == 1 and made[0].calls == ["start"]
    assert [e[0] for e in app.events].count("record") == 1
    await pilot.press("q")
    await _settle(pilot)


def _assert_stopped_and_named(app, made) -> None:
    """The one recorder was stopped on quit and the exit summary names its WAV."""
    assert made[0].calls == ["start", "stop"]
    assert made[0].wav_path in (app.exit_message or "")


@pytest.mark.parametrize("keys", [("enter", "enter"), ("enter", "r")],
                         ids=["two-enters-in-the-name-box", "enter-and-r-on-record"])
async def test_two_queued_starts_begin_one_recording(tmp_path, keys):
    made, overrides = _counting_recorders(tmp_path)
    app = TinyStenoApp(CONFIG, make_deps(tmp_path, **overrides))
    async with app.run_test() as pilot:
        if keys[0] == "enter":
            app.screen.query_one("#name").focus()
            await _settle(pilot)
        send_together(app, *keys)
        await _one_recording_then_quit(app, pilot, made)
    _assert_stopped_and_named(app, made)


@pytest.mark.parametrize("route", ["record-button", "r-action"])
async def test_two_starts_before_either_is_handled_begin_one_recording(tmp_path, route):
    made, overrides = _counting_recorders(tmp_path)
    app = TinyStenoApp(CONFIG, make_deps(tmp_path, **overrides))
    async with app.run_test() as pilot:
        home = app.screen
        await _settle(pilot)
        for _ in range(2):
            if route == "record-button":
                home.query_one("#record", Button).press()
            else:
                home.action_record()
        await _one_recording_then_quit(app, pilot, made)
    _assert_stopped_and_named(app, made)


async def test_r_then_a_queued_enter_starts_one_recording_and_enter_stops_it(tmp_path):
    # r is resolved first and pushes the Recording screen, so the queued Enter is that
    # screen's Stop: still one recorder, started and stopped once, never a second recording.
    made, overrides = _counting_recorders(tmp_path)
    app = TinyStenoApp(CONFIG, make_deps(tmp_path, **overrides))
    async with app.run_test() as pilot:
        send_together(app, "r", "enter")
        await _settle(pilot)
        await app.workers.wait_for_complete()
        await _settle(pilot)
        assert len(made) == 1 and made[0].calls == ["start", "stop"]
        assert [e[0] for e in app.events].count("record") == 1


async def test_a_start_queued_behind_now_begins_no_second_recording(tmp_path):
    made, overrides = _counting_recorders(tmp_path)
    app = TinyStenoApp(CONFIG, make_deps(tmp_path, **overrides), autostart=True)
    async with app.run_test() as pilot:
        home = app.screen_stack[1]
        assert isinstance(home, HomeScreen)
        send_together(app, "r")
        await _settle(pilot)
        home.action_record()                    # a key that reached Home all the same
        await _one_recording_then_quit(app, pilot, made)
    _assert_stopped_and_named(app, made)
