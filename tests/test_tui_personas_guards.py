"""Personas that cannot be listed never crash a screen; a restore always reports its outcome."""
import threading

from textual.widgets import Button, Select, Static

from tinysteno.tui.app import TinyStenoApp
from tinysteno.tui.confirm import ConfirmScreen
from tinysteno.tui.home import HomeScreen
from tinysteno.tui.personas import PersonasScreen
from tinysteno.tui.recordings import RecordingsScreen
from tests.tui_fakes import make_deps
from tests.tui_settings_helpers import notes

CONFIG = {"persona": "default", "session_timeout_minutes": 75}


async def _settle(pilot, times: int = 3):
    for _ in range(times):
        await pilot.pause()


def _failing_personas(state: dict):
    def personas():
        if state["broken"]:
            raise PermissionError("Permission denied: personas")
        return ["default", "1on1"]
    return personas


# ── listing the personas fails ───────────────────────────────────────────────────────

async def test_home_starts_when_the_personas_cannot_be_listed(tmp_path):
    state = {"broken": True}
    app = TinyStenoApp(CONFIG, make_deps(tmp_path, list_personas=_failing_personas(state)))
    async with app.run_test() as pilot:
        await _settle(pilot)
        assert isinstance(app.screen, HomeScreen)
        hint = str(app.screen.query_one("#timeout-hint", Static).content)
        assert hint == "Personas cannot be listed: Permission denied: personas"
        assert app.screen.query_one("#record", Button).disabled
        state["broken"] = False                 # fixed meanwhile: Home follows on return
        app.open_destination("checks")
        await _settle(pilot)
        await pilot.press("escape")
        await _settle(pilot)
        assert app.screen.query_one("#persona", Select).value == "default"
        assert not app.screen.query_one("#record", Button).disabled
        hint = str(app.screen.query_one("#timeout-hint", Static).content)
        assert not hint.startswith("Personas cannot be listed")


async def test_recordings_opens_when_the_personas_cannot_be_listed(tmp_path):
    state = {"broken": False}
    app = TinyStenoApp(CONFIG, make_deps(tmp_path, list_personas=_failing_personas(state)))
    async with app.run_test() as pilot:
        state["broken"] = True                  # Home has listed them; Recordings cannot
        app.open_destination("recordings")
        await _settle(pilot)
        await app.workers.wait_for_complete()
        await _settle(pilot)
        screen = app.screen
        assert isinstance(screen, RecordingsScreen)
        error = screen.query_one("#recordings-personas-error", Static)
        assert error.display
        assert str(error.content) == "Personas cannot be listed: Permission denied: personas"
        state["broken"] = False
        app.open_destination("checks")
        await _settle(pilot)
        await pilot.press("escape")
        await _settle(pilot)
        assert app.screen is screen
        assert not screen.query_one("#recordings-personas-error", Static).display
        assert screen.query_one("#persona", Select).value == "default"


# ── a restore blocks leaving until it has reported ───────────────────────────────────

async def _open_personas(app, pilot) -> PersonasScreen:
    app.open_destination("personas")
    await _settle(pilot)
    await app.workers.wait_for_complete()
    await _settle(pilot)
    assert isinstance(app.screen, PersonasScreen)
    return app.screen


async def _start_restore(app, pilot) -> None:
    await pilot.press("b")
    await pilot.pause()
    assert isinstance(app.screen, ConfirmScreen)
    app.screen.query_one("#confirm", Button).press()
    await pilot.pause()


async def test_back_and_the_menu_wait_for_a_restore_that_then_reports_its_failure(tmp_path):
    gate = threading.Event()
    deps = make_deps(tmp_path, seed_gate=gate, seed_error=OSError("disk full"))
    app = TinyStenoApp(CONFIG, deps)
    async with app.run_test(notifications=True) as pilot:
        screen = await _open_personas(app, pilot)
        await _start_restore(app, pilot)
        assert screen.navigation_blocked()
        await pilot.press("escape")
        await pilot.pause()
        screen.action_back()
        app.open_destination("checks")          # as the palette would
        await pilot.pause()
        assert app.screen is screen
        assert not app.check_action("open_menu", ())
        assert "System check" not in [c.title for c in app.get_system_commands(app.screen)]
        gate.set()
        for _ in range(40):
            await pilot.pause(0.05)
            if not screen.navigation_blocked():
                break
        await _settle(pilot)
        assert ("Could not restore personas: disk full", "error") in notes(app)
        await pilot.press("escape")
        await _settle(pilot)
        assert isinstance(app.screen, HomeScreen)
