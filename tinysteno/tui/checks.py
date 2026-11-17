"""System check screen: the checks `tinysteno test` runs, with a summary."""

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Static
from textual.worker import Worker, WorkerState

from tinysteno.doctor import Check

RUNNING = "Running checks…"


def _one_line(text: object) -> str:
    return " ".join(str(text).split())


def _mark(check: Check) -> str:
    if check.ok:
        return "✓"
    return "✗" if check.required else "!"


def _level(check: Check) -> str:
    if check.ok:
        return "ok"
    return "fail" if check.required else "warn"


def _row_text(check: Check) -> str:
    detail = f": {_one_line(check.detail)}" if check.detail else ""
    return f"{_mark(check)} {_one_line(check.name)}{detail}"


class ChecksScreen(Screen):
    """Runs ``deps.run_checks()`` in a thread worker when it opens and on ``r``.

    Rows and details are shown literally (markup off): a detail is often a command to copy.
    """

    SUB_TITLE = "System check"
    BINDINGS = [
        Binding("r", "run_again", "Run again"),
        Binding("escape", "back", "Back"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._run: Worker | None = None   # the run in progress, None when idle

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="checks-panel"):
            yield Static(RUNNING, id="checks-summary", markup=False)
            yield Static("", id="checks-note", markup=False)
            yield VerticalScroll(id="checks-list")
            with Horizontal(id="checks-buttons"):
                yield Button("Run again (r)", id="run-again", compact=True)
                yield Button("Back (esc)", id="back", compact=True)
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#checks-note").display = False
        self.action_run_again()

    def action_run_again(self) -> None:
        """Run the checks again; ignored while a run is in progress."""
        if self._run is not None:
            return
        self.query_one("#checks-summary", Static).update(RUNNING)
        self.query_one("#checks-note").display = False
        # A thread: the checks probe audio devices and import heavy modules.
        self._run = self.run_worker(
            self.app.deps.run_checks, thread=True, exit_on_error=False, group="checks"
        )

    def action_back(self) -> None:
        """Leave; a run still going finishes unseen."""
        self.dismiss()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if event.button.id == "run-again":
            self.action_run_again()
        elif event.button.id == "back":
            self.action_back()

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        """Show the finished run; a run that was cancelled with the screen shows nothing."""
        if event.worker is not self._run:
            return
        state = event.state
        if state == WorkerState.SUCCESS:
            self._show(list(event.worker.result or []))
        elif state == WorkerState.ERROR:
            error = event.worker.error
            reason = _one_line(error) or "no details"
            name = type(error).__name__
            self._show([Check("System check", False, f"{name}: {reason}")])
        elif state == WorkerState.CANCELLED:
            self._run = None

    def _show(self, checks: list[Check]) -> None:
        self._run = None
        panel = self.query_one("#checks-list", VerticalScroll)
        panel.remove_children()
        panel.mount_all(
            Static(_row_text(c), classes=f"check-row {_level(c)}", markup=False) for c in checks
        )
        problems = sum(1 for c in checks if not c.ok and c.required)
        optional = sum(1 for c in checks if not c.ok and not c.required)
        summary = f"{problems} problem(s) found" if problems else "All checks passed"
        self.query_one("#checks-summary", Static).update(summary)
        note = self.query_one("#checks-note", Static)
        note.update(f"{optional} optional item(s) need attention" if optional and not problems
                    else "")
        note.display = bool(optional and not problems)
