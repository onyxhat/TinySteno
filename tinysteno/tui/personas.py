"""Personas screen: which personas exist and are valid, the default, restoring the built-ins."""

from dataclasses import dataclass

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Static
from textual.worker import Worker, WorkerState

from tinysteno.personas import Persona, PersonaInvalidError, PersonaNotFoundError
from tinysteno.tui.confirm import ConfirmScreen
from tinysteno.tui.deps import SettingsNotApplied
from tinysteno.tui.formscreen import readable_reason

LOADING = "Loading…"
RESTORING = "Restoring…"
NO_PERSONAS = "No personas found. Press b to restore the built-in ones."
RESTORE_QUESTION = ("Restore the built-in personas? Your changes to them will be overwritten; "
                    "personas you added are kept.")
STATUS_LIMIT = 120
COLUMNS = ("Persona", "Name", "Auto-stop", "Status")


def _one_line(text: object, limit: int = STATUS_LIMIT) -> str:
    line = " ".join(str(text).split())
    return line if len(line) <= limit else line[: limit - 1] + "…"


@dataclass
class _Row:
    slug: str
    persona: Persona | None = None
    error: str | None = None     # why it is invalid; None for a valid persona


@dataclass
class _Listing:
    """What one load found: the rows, why listing failed, and the stored default."""
    rows: list[_Row]
    list_error: str | None
    default: str | None


def _auto_stop(persona: Persona | None) -> str:
    if persona is None:
        return ""
    minutes = persona.session_timeout_minutes
    if minutes is None:
        return "global"
    return "off" if minutes == 0 else f"{minutes:g} min"


class PersonasScreen(Screen):
    """A table of the personas; ``d`` makes one the default, ``b`` restores the built-ins.

    Reading the personas and restoring run in thread workers; a result that arrives after
    the screen was left is discarded. Cells are shown literally (markup off).
    """

    SUB_TITLE = "Personas"
    BINDINGS = [
        Binding("d", "set_default", "Set as default"),
        Binding("b", "restore", "Restore built-ins"),
        Binding("escape", "back", "Back"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._rows: list[_Row] = []
        self._default: str | None = None
        self._list_error: str | None = None
        self._load: Worker | None = None       # the read in progress, None when idle
        self._restore: Worker | None = None    # the restore in progress, None when idle
        self._leaving = False                  # (not `_closing`: Textual's message pump)

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="personas-panel"):
            yield Static(LOADING, id="personas-note", markup=False)
            yield DataTable(id="personas", cursor_type="row", zebra_stripes=True)
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#personas", DataTable)
        for label in COLUMNS:
            table.add_column(label)
        table.focus()
        self._reload()

    @property
    def _busy(self) -> bool:
        return self._load is not None or self._restore is not None

    # ── reading ──────────────────────────────────────────────────────────────

    def _reload(self) -> None:
        self._load = self.run_worker(
            self._read, thread=True, exit_on_error=False, group="personas-load"
        )
        self._show_rows()

    def _read(self) -> _Listing:
        deps = self.app.deps
        try:
            slugs = list(deps.list_personas())
        except Exception as err:
            slugs, list_error = [], readable_reason(err) or type(err).__name__
        else:
            list_error = None
        try:
            stored = deps.load_settings().get("persona")
        except Exception:
            stored = None
        default = stored if isinstance(stored, str) else self.app.config.get("persona")
        return _Listing([self._read_one(slug) for slug in slugs], list_error, default)

    def _read_one(self, slug: str) -> _Row:
        try:
            return _Row(slug, persona=self.app.deps.load_persona(slug))
        except (PersonaInvalidError, PersonaNotFoundError) as err:
            return _Row(slug, error=_one_line(err) or type(err).__name__)
        except Exception as err:
            return _Row(slug, error=type(err).__name__)

    # ── showing ──────────────────────────────────────────────────────────────

    def _status(self, row: _Row) -> str:
        parts = []
        if row.slug == self._default:
            parts.append("default")
        parts.append("ok" if row.error is None else f"Invalid: {row.error}")
        return ", ".join(parts) if row.error is not None else parts[0]

    def _notice(self) -> str:
        if self._restore is not None:
            return RESTORING
        if self._load is not None:
            return LOADING
        if self._list_error is not None:
            return f"Personas cannot be listed: {self._list_error}"
        if not self._rows:
            return NO_PERSONAS
        if self._default is not None and self._default not in {r.slug for r in self._rows}:
            return f'The default persona "{self._default}" was not found.'
        return ""

    def _show_rows(self) -> None:
        table = self.query_one("#personas", DataTable)
        keep = table.ordered_rows[table.cursor_row].key.value if table.row_count else None
        table.clear()
        for row in self._rows:
            persona = row.persona
            cells = (row.slug, persona.name if persona else "", _auto_stop(persona),
                     self._status(row))
            table.add_row(*(Text(cell) for cell in cells), key=row.slug)
        if keep is not None and any(r.slug == keep for r in self._rows):
            table.move_cursor(row=table.get_row_index(keep))
        note = self.query_one("#personas-note", Static)
        text = self._notice()
        note.update(text)
        note.display = bool(text)

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        """Show a finished read or restore; one that was cancelled with the screen shows
        nothing."""
        worker = event.worker
        if worker is self._load:
            self._read_finished(worker, event.state)
        elif worker is self._restore:
            self._restore_finished(worker, event.state)

    def _read_finished(self, worker: Worker, state: WorkerState) -> None:
        if state == WorkerState.SUCCESS:
            listing: _Listing = worker.result
            self._rows, self._list_error = listing.rows, listing.list_error
            self._default = listing.default
        elif state == WorkerState.ERROR:
            self._rows, self._list_error = [], type(worker.error).__name__
        elif state != WorkerState.CANCELLED:
            return
        self._load = None
        if not self._leaving and self.is_attached:
            self._show_rows()

    def _restore_finished(self, worker: Worker, state: WorkerState) -> None:
        if state not in (WorkerState.SUCCESS, WorkerState.ERROR, WorkerState.CANCELLED):
            return
        self._restore = None
        if self._leaving or not self.is_attached or state == WorkerState.CANCELLED:
            return
        if state == WorkerState.ERROR:
            reason = readable_reason(worker.error) or type(worker.error).__name__
            self.app.notify(f"Could not restore personas: {reason}", severity="error",
                            markup=False)
            self._show_rows()
            return
        copied = list((worker.result or {}).get("copied", []))
        self.app.notify(f"Restored: {', '.join(copied)}" if copied else "Nothing to restore",
                        markup=False)
        self._reload()

    # ── actions ──────────────────────────────────────────────────────────────

    def _can_act(self) -> bool:
        return not self._leaving and self.is_active and not self._busy

    def action_back(self) -> None:
        """Leave; a read or restore still going finishes unseen."""
        if self._leaving or not self.is_active:
            return
        self._leaving = True
        self.dismiss()

    def action_set_default(self) -> None:
        """Make the persona under the cursor the default (a valid one only)."""
        table = self.query_one("#personas", DataTable)
        if not self._can_act() or not table.row_count:
            return
        row = self._rows[table.cursor_row]
        app = self.app
        if row.error is not None:
            app.notify("Cannot use an invalid persona as the default", severity="error")
            return
        try:
            app.deps.save_settings({"persona": row.slug})
        except SettingsNotApplied as err:
            self._default = row.slug
            self._show_rows()
            app.notify(
                "Settings saved, but they could not be applied to this session: "
                f"{readable_reason(err)}. Restart TinySteno to use them.",
                severity="warning", markup=False,
            )
            return
        except Exception as err:
            reason = readable_reason(err) or type(err).__name__
            app.notify(f"Could not set the default persona: {reason}", severity="error",
                       markup=False)
            return
        self._default = row.slug
        self._show_rows()
        app.notify(f"Default persona: {row.slug}", markup=False)

    def action_restore(self) -> None:
        """Ask, then restore the built-in personas (overwriting the user's edits)."""
        if not self._can_act():
            return
        self.app.push_screen(ConfirmScreen(RESTORE_QUESTION, "Restore"), self._restore_answered)

    def _restore_answered(self, confirmed: bool | None) -> None:
        if not confirmed or not self._can_act():
            return
        self._restore = self.run_worker(
            self.app.deps.seed_personas, thread=True, exit_on_error=False,
            group="personas-restore",
        )
        self._show_rows()
