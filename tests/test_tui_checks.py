"""System check screen: the results of `tinysteno test`, run in a worker, with a summary."""
import threading

from textual.widgets import Static

from tinysteno.doctor import Check
from tinysteno.tui.app import TinyStenoApp
from tinysteno.tui.checks import ChecksScreen
from tinysteno.tui.home import HomeScreen
from tests.tui_fakes import make_deps

CONFIG = {"persona": "1on1", "session_timeout_minutes": 120}


async def _open(app, pilot) -> ChecksScreen:
    app.open_destination("checks")
    await pilot.pause()
    await pilot.pause()
    assert isinstance(app.screen, ChecksScreen)
    return app.screen


def _rows(screen) -> list[str]:
    return [str(w.content) for w in screen.query(".check-row").results(Static)]


def _text(screen, query) -> str:
    return str(screen.query_one(query, Static).content)


async def test_checks_are_listed_with_marks_and_summary(tmp_path):
    checks = [
        Check("Python", True),
        Check("Vault", False, "folder missing"),
        Check("Diarizer", False, "pip install 'tinysteno[diarize]'", required=False),
    ]
    app = TinyStenoApp(CONFIG, make_deps(tmp_path, checks=checks))
    async with app.run_test() as pilot:
        screen = await _open(app, pilot)
        rows = _rows(screen)
        assert len(rows) == 3
        assert rows[0].startswith("✓") and "Python" in rows[0]
        assert rows[1].startswith("✗") and "Vault" in rows[1] and "folder missing" in rows[1]
        assert rows[2].startswith("!")
        assert "pip install 'tinysteno[diarize]'" in rows[2]
        assert _text(screen, "#checks-summary") == "1 problem(s) found"


async def test_optional_failures_do_not_count_as_problems(tmp_path):
    checks = [Check("Python", True), Check("A", False, "x", required=False),
              Check("B", False, "y", required=False)]
    app = TinyStenoApp(CONFIG, make_deps(tmp_path, checks=checks))
    async with app.run_test() as pilot:
        screen = await _open(app, pilot)
        assert _text(screen, "#checks-summary") == "All checks passed"
        assert _text(screen, "#checks-note") == "2 optional item(s) need attention"


async def test_all_passed_has_no_note(tmp_path):
    app = TinyStenoApp(CONFIG, make_deps(tmp_path, checks=[Check("Python", True)]))
    async with app.run_test() as pilot:
        screen = await _open(app, pilot)
        assert _text(screen, "#checks-summary") == "All checks passed"
        assert not screen.query_one("#checks-note").display


async def test_markup_in_details_is_shown_verbatim(tmp_path):
    checks = [Check("[bold]Odd[/]", False, "run [red]this[/red] now")]
    app = TinyStenoApp(CONFIG, make_deps(tmp_path, checks=checks))
    async with app.run_test() as pilot:
        screen = await _open(app, pilot)
        assert "[bold]Odd[/]" in _rows(screen)[0]
        assert "[red]this[/red]" in _rows(screen)[0]


async def test_run_again_reruns(tmp_path):
    calls = []

    def run():
        calls.append(1)
        return [Check("Python", len(calls) > 1, "old" if len(calls) == 1 else "")]

    app = TinyStenoApp(CONFIG, make_deps(tmp_path, run_checks=run))
    async with app.run_test() as pilot:
        screen = await _open(app, pilot)
        assert _text(screen, "#checks-summary") == "1 problem(s) found"
        await pilot.press("r")
        await pilot.pause()
        await pilot.pause()
        assert len(calls) == 2
        assert _text(screen, "#checks-summary") == "All checks passed"


async def test_running_is_shown_and_r_is_ignored_meanwhile(tmp_path):
    gate = threading.Event()
    calls = []

    def run():
        calls.append(1)
        gate.wait(5)
        return [Check("Python", True)]

    app = TinyStenoApp(CONFIG, make_deps(tmp_path, run_checks=run))
    async with app.run_test() as pilot:
        app.open_destination("checks")
        await pilot.pause()
        screen = app.screen
        assert _text(screen, "#checks-summary") == "Running checks…"
        await pilot.press("r")
        await pilot.pause()
        gate.set()
        await pilot.pause()
        await pilot.pause()
        assert len(calls) == 1
        assert _text(screen, "#checks-summary") == "All checks passed"


async def test_a_check_that_raises_is_shown_as_a_failure_not_a_crash(tmp_path):
    def run():
        raise RuntimeError("disk\nexploded")

    app = TinyStenoApp(CONFIG, make_deps(tmp_path, run_checks=run))
    async with app.run_test() as pilot:
        screen = await _open(app, pilot)
        rows = _rows(screen)
        assert len(rows) == 1
        assert rows[0] == "✗ System check: RuntimeError: disk exploded"
        assert _text(screen, "#checks-summary") == "1 problem(s) found"


async def test_a_run_finishing_after_the_screen_was_left_is_discarded(tmp_path):
    gate = threading.Event()

    def run():
        gate.wait(5)
        return [Check("Python", True)]

    app = TinyStenoApp(CONFIG, make_deps(tmp_path, run_checks=run))
    async with app.run_test() as pilot:
        app.open_destination("checks")
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        assert isinstance(app.screen, HomeScreen)
        gate.set()
        await pilot.pause()
        await pilot.pause()
        assert isinstance(app.screen, HomeScreen)


async def test_escape_goes_back_to_home(tmp_path):
    app = TinyStenoApp(CONFIG, make_deps(tmp_path, checks=[Check("Python", True)]))
    async with app.run_test() as pilot:
        await _open(app, pilot)
        await pilot.press("escape")
        await pilot.pause()
        assert isinstance(app.screen, HomeScreen)
