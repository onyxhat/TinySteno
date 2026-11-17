"""Personas screen: validity, default, restore of the built-ins, and Home following the list."""
import threading

from rich.text import Text
from textual.widgets import Button, DataTable, Static

from tinysteno.personas import PersonaInvalidError, PersonaNotFoundError
from tinysteno.tui.app import TinyStenoApp
from tinysteno.tui.confirm import ConfirmScreen
from tinysteno.tui.deps import SettingsNotApplied
from tinysteno.tui.home import HomeScreen, NO_PERSONAS
from tinysteno.tui.personas import PersonasScreen
from tests.tui_fakes import make_deps
from tests.tui_settings_helpers import notes

CONFIG = {"persona": "default", "session_timeout_minutes": 75}
CONFIRM = ("Restore the built-in personas? Your changes to them will be overwritten; "
           "personas you added are kept.")


async def open_personas(app, pilot) -> PersonasScreen:
    """Open the Personas screen from Home and wait for the table to load."""
    app.open_destination("personas")
    await pilot.pause()
    await pilot.pause()
    assert isinstance(app.screen, PersonasScreen)
    return app.screen


def rows(screen) -> list[list[str]]:
    """Each table row as plain strings."""
    table = screen.query_one("#personas", DataTable)
    return [[str(cell) for cell in table.get_row_at(i)] for i in range(table.row_count)]


def note(screen) -> str:
    """The line above the table."""
    return str(screen.query_one("#personas-note", Static).content)


def move_to(screen, slug) -> None:
    """Put the cursor on a persona's row."""
    table = screen.query_one("#personas", DataTable)
    table.move_cursor(row=table.get_row_index(slug))


async def test_rows_show_validity_default_and_timeout(tmp_path):
    deps = make_deps(
        tmp_path, personas=["default", "1on1", "rca", "bad", "gone"],
        persona_timeouts={"1on1": 30, "rca": 0},
        persona_errors={"bad": PersonaInvalidError("Persona 'bad': schema is\nbroken"),
                        "gone": PersonaNotFoundError("no such persona")},
    )
    app = TinyStenoApp(CONFIG, deps)
    async with app.run_test() as pilot:
        screen = await open_personas(app, pilot)
        table = screen.query_one("#personas", DataTable)
        assert [str(c.label) for c in table.columns.values()] == [
            "Persona", "Name", "Auto-stop", "Status"]
        assert rows(screen) == [
            ["default", "Default", "global", "default"],
            ["1on1", "Default", "30 min", "ok"],
            ["rca", "Default", "off", "ok"],
            ["bad", "", "", "Invalid: Persona 'bad': schema is broken"],
            ["gone", "", "", "Invalid: no such persona"],
        ]


async def test_stored_default_wins_over_the_session_config(tmp_path):
    deps = make_deps(tmp_path, settings={"persona": "1on1"})
    app = TinyStenoApp(CONFIG, deps)
    async with app.run_test() as pilot:
        screen = await open_personas(app, pilot)
        assert [r[3] for r in rows(screen)] == ["ok", "default"]


async def test_a_default_that_does_not_exist_is_named_above_the_table(tmp_path):
    app = TinyStenoApp(CONFIG, make_deps(tmp_path, settings={"persona": "ghost"}))
    async with app.run_test() as pilot:
        screen = await open_personas(app, pilot)
        assert note(screen) == 'The default persona "ghost" was not found.'
        assert [r[3] for r in rows(screen)] == ["ok", "ok"]


async def test_cells_are_literal_and_status_is_capped(tmp_path):
    long = "x" * 300
    deps = make_deps(tmp_path, personas=["a"], persona_errors={
        "a": PersonaInvalidError(f"[bold]{long}")})
    app = TinyStenoApp(CONFIG, deps)
    async with app.run_test() as pilot:
        screen = await open_personas(app, pilot)
        status = rows(screen)[0][3]
        assert status.startswith("Invalid: [bold]xxx") and len(status) <= len("Invalid: ") + 120
        assert all(isinstance(c, Text) for c in screen.query_one(
            "#personas", DataTable).get_row_at(0))


async def test_any_other_error_marks_only_that_row(tmp_path):
    deps = make_deps(tmp_path, personas=["a", "b"], persona_errors={"b": KeyError("secret")})
    app = TinyStenoApp(CONFIG, deps)
    async with app.run_test() as pilot:
        screen = await open_personas(app, pilot)
        assert [r[3] for r in rows(screen)] == ["ok", "Invalid: KeyError"]


async def test_a_list_that_cannot_be_read_is_reported_and_restore_still_works(tmp_path):
    deps = make_deps(tmp_path)

    def broken():
        raise OSError("disk gone")
    app = TinyStenoApp(CONFIG, deps)
    async with app.run_test() as pilot:
        deps.list_personas = broken      # Home has listed them; only this screen fails
        screen = await open_personas(app, pilot)
        assert note(screen) == "Personas cannot be listed: disk gone"
        assert rows(screen) == []
        await pilot.press("b")
        await pilot.pause()
        assert isinstance(app.screen, ConfirmScreen)


async def test_d_sets_the_default_and_marks_it(tmp_path):
    deps = make_deps(tmp_path, personas=["default", "1on1", "rca"])
    app = TinyStenoApp(CONFIG, deps)
    async with app.run_test() as pilot:
        screen = await open_personas(app, pilot)
        move_to(screen, "rca")
        await pilot.press("d")
        await pilot.pause()
        assert deps.saved == [{"persona": "rca"}]
        assert [r[3] for r in rows(screen)] == ["ok", "ok", "default"]
        assert ("Default persona: rca", "information") in notes(app)


async def test_d_on_an_invalid_persona_saves_nothing(tmp_path):
    deps = make_deps(tmp_path, persona_errors={"1on1": PersonaInvalidError("broken")})
    app = TinyStenoApp(CONFIG, deps)
    async with app.run_test() as pilot:
        screen = await open_personas(app, pilot)
        move_to(screen, "1on1")
        await pilot.press("d")
        await pilot.pause()
        assert not deps.saved
        assert ("Cannot use an invalid persona as the default", "error") in notes(app)
        assert rows(screen)[0][3] == "default"


async def test_save_failure_is_reported_and_default_unchanged(tmp_path):
    deps = make_deps(tmp_path, save_error=OSError("read-only file"))
    app = TinyStenoApp(CONFIG, deps)
    async with app.run_test() as pilot:
        screen = await open_personas(app, pilot)
        move_to(screen, "1on1")
        await pilot.press("d")
        await pilot.pause()
        assert ("Could not set the default persona: read-only file", "error") in notes(app)
        assert [r[3] for r in rows(screen)] == ["default", "ok"]


async def test_a_value_error_is_reported_the_same_way(tmp_path):
    deps = make_deps(tmp_path, save_error=ValueError("bad value"))
    app = TinyStenoApp(CONFIG, deps)
    async with app.run_test() as pilot:
        screen = await open_personas(app, pilot)
        move_to(screen, "1on1")
        await pilot.press("d")
        await pilot.pause()
        assert ("Could not set the default persona: bad value", "error") in notes(app)


async def test_saved_but_not_applied_warns_and_marks_the_row(tmp_path):
    deps = make_deps(tmp_path, save_error=SettingsNotApplied("locked"))
    app = TinyStenoApp(CONFIG, deps)
    async with app.run_test() as pilot:
        screen = await open_personas(app, pilot)
        move_to(screen, "1on1")
        await pilot.press("d")
        await pilot.pause()
        assert [r[3] for r in rows(screen)] == ["ok", "default"]
        assert ("Settings saved, but they could not be applied to this session: locked. "
                "Restart TinySteno to use them.", "warning") in notes(app)


async def test_restore_asks_first_and_cancel_does_nothing(tmp_path):
    calls = []
    deps = make_deps(tmp_path)
    inner = deps.seed_personas
    deps.seed_personas = lambda: calls.append(1) or inner()
    app = TinyStenoApp(CONFIG, deps)
    async with app.run_test() as pilot:
        screen = await open_personas(app, pilot)
        await pilot.press("b")
        await pilot.pause()
        assert isinstance(app.screen, ConfirmScreen)
        assert app.screen.question == CONFIRM and app.screen.confirm_label == "Restore"
        app.screen.query_one("#cancel", Button).press()
        await pilot.pause()
        assert app.screen is screen and not calls


async def test_restore_runs_and_reports(tmp_path):
    deps = make_deps(tmp_path, personas=[], seed_makes=["default", "rca"],
                     seed_result={"copied": ["default", "rca"], "skipped": []})
    app = TinyStenoApp(CONFIG, deps)
    async with app.run_test() as pilot:
        screen = await open_personas(app, pilot)
        await pilot.press("b")
        await pilot.pause()
        app.screen.query_one("#confirm", Button).press()
        await pilot.pause()
        await pilot.pause()
        assert ("Restored: default, rca", "information") in notes(app)
        assert [r[0] for r in rows(screen)] == ["default", "rca"]


async def test_restore_with_nothing_copied_says_so(tmp_path):
    app = TinyStenoApp(CONFIG, make_deps(tmp_path))
    async with app.run_test() as pilot:
        await open_personas(app, pilot)
        await pilot.press("b")
        await pilot.pause()
        app.screen.query_one("#confirm", Button).press()
        await pilot.pause()
        await pilot.pause()
        assert ("Nothing to restore", "information") in notes(app)


async def test_restore_failure_is_reported(tmp_path):
    app = TinyStenoApp(CONFIG, make_deps(tmp_path, seed_error=PermissionError("not allowed")))
    async with app.run_test() as pilot:
        await open_personas(app, pilot)
        await pilot.press("b")
        await pilot.pause()
        app.screen.query_one("#confirm", Button).press()
        await pilot.pause()
        await pilot.pause()
        assert ("Could not restore personas: not allowed", "error") in notes(app)


async def test_keys_are_ignored_while_restoring_and_late_results_are_discarded(tmp_path):
    gate = threading.Event()
    deps = make_deps(tmp_path, seed_gate=gate, seed_result={"copied": ["x"], "skipped": []})
    app = TinyStenoApp(CONFIG, deps)
    async with app.run_test() as pilot:
        screen = await open_personas(app, pilot)
        await pilot.press("b")
        await pilot.pause()
        app.screen.query_one("#confirm", Button).press()
        await pilot.pause()
        assert note(screen) == "Restoring…"
        await pilot.press("d", "b")
        await pilot.pause()
        assert not deps.saved and app.screen is screen
        await pilot.press("escape")          # leave while the restore is still running
        await pilot.pause()
        assert isinstance(app.screen, HomeScreen)
        gate.set()
        await pilot.pause()
        await pilot.pause()
        assert not [n for n in notes(app) if n[0].startswith("Restored")]


async def test_empty_list_offers_restore_and_home_enables_record_afterwards(tmp_path):
    deps = make_deps(tmp_path, personas=[], seed_makes=["default", "1on1"],
                     seed_result={"copied": ["default", "1on1"], "skipped": []})
    app = TinyStenoApp(CONFIG, deps)
    async with app.run_test() as pilot:
        home = app.screen
        assert home.query_one("#record").disabled
        screen = await open_personas(app, pilot)
        assert note(screen) == "No personas found. Press b to restore the built-in ones."
        await pilot.press("b")
        await pilot.pause()
        app.screen.query_one("#confirm", Button).press()
        await pilot.pause()
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        assert app.screen is home
        assert not home.query_one("#record").disabled
        assert home.query_one("#persona").value == "default"
        assert str(home.query_one("#timeout-hint").content).startswith("Auto-stop")


async def test_home_keeps_the_choice_or_falls_back_when_the_list_changes(tmp_path):
    deps = make_deps(tmp_path, personas=["default", "1on1", "rca"])
    app = TinyStenoApp(CONFIG, deps)
    async with app.run_test() as pilot:
        home = app.screen
        home.query_one("#persona").value = "rca"
        await pilot.pause()
        deps.personas.append("new")
        await open_personas(app, pilot)
        await pilot.press("escape")
        await pilot.pause()
        assert home.query_one("#persona").value == "rca"      # kept
        deps.personas.remove("rca")
        await open_personas(app, pilot)
        await pilot.press("escape")
        await pilot.pause()
        assert home.query_one("#persona").value == "default"  # the configured default
        deps.personas[:] = ["only"]
        await open_personas(app, pilot)
        await pilot.press("escape")
        await pilot.pause()
        assert home.query_one("#persona").value == "only"     # the first one there is
        deps.personas[:] = []
        await open_personas(app, pilot)
        await pilot.press("escape")
        await pilot.pause()
        assert home.query_one("#record").disabled
        assert str(home.query_one("#timeout-hint").content) == NO_PERSONAS


async def test_fits_80x24_and_scrolls_a_long_status(tmp_path):
    deps = make_deps(tmp_path, persona_errors={"1on1": PersonaInvalidError("y" * 400)})
    app = TinyStenoApp(CONFIG, deps)
    async with app.run_test(size=(80, 24)) as pilot:
        screen = await open_personas(app, pilot)
        table = screen.query_one("#personas", DataTable)
        assert table.region.right <= 80 and table.region.bottom <= 24
        assert table.virtual_size.width > table.size.width   # scrolls sideways
        assert screen.max_scroll_x == 0 and screen.max_scroll_y == 0


async def test_escape_goes_back_and_the_menu_lists_personas(tmp_path):
    app = TinyStenoApp(CONFIG, make_deps(tmp_path))
    async with app.run_test() as pilot:
        assert app.destinations["personas"][0] == "Personas"
        screen = await open_personas(app, pilot)
        assert not hasattr(screen, "has_unsaved_changes")
        await pilot.press("escape")
        await pilot.pause()
        assert isinstance(app.screen, HomeScreen)
