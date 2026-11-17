"""Recordings screen: every recording, to process, transcribe again, or delete."""

import asyncio
import os
from datetime import datetime
from functools import partial
from pathlib import Path

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Label, Select, Static
from textual.worker import Worker, WorkerState

from tinysteno.recordings import RecordingInfo, list_recordings
from tinysteno.tui.confirm import ConfirmScreen
from tinysteno.tui.formscreen import readable_reason
from tinysteno.tui.home import NO_PERSONAS, initial_persona
from tinysteno.tui.leaving import LeaveOnceMixin

LOADING = "Loading…"
COLUMNS = ("Recording", "Date", "Size", "Transcript")
NAME_WIDTH = 28
SAVED, NOT_SAVED = "saved", "—"
_UNITS = ("B", "KB", "MB", "GB")


def format_size(size: int) -> str:
    """``512 B``, ``1.5 KB``, ``12.3 MB``: 1024-based, one decimal from KB up."""
    if size < 1024:
        return f"{size} B"
    value = float(size)
    for unit in _UNITS[1:-1]:
        value /= 1024
        if value < 1024:
            return f"{value:.1f} {unit}"
    return f"{value / 1024:.1f} {_UNITS[-1]}"


def shorten(name: str, width: int = NAME_WIDTH) -> str:
    """``name`` cut in the middle to ``width`` characters: recorder names end in a timestamp,
    so both the start and the end stay readable."""
    if len(name) <= width:
        return name
    head = (width - 1) // 2
    return f"{name[:head]}…{name[len(name) - (width - 1 - head):]}"


def format_date(moment: datetime) -> str:
    """``2026-03-04 09:05``."""
    return moment.strftime("%Y-%m-%d %H:%M")


def _delete_question(name: str) -> str:
    return (f"Delete {name}? The audio and its saved transcript are removed. "
            "Notes in your vault are kept.")


class RecordingsScreen(LeaveOnceMixin, Screen):  # pylint: disable=too-many-instance-attributes
    """A table of the recordings folder; ``enter`` processes, ``t`` transcribes again, ``d``
    deletes (after asking).

    Reading the folder and deleting run in thread workers; a result that arrives after the
    screen was left is not shown. A delete always acts on the file that was selected when
    ``d`` was pressed. Cells are shown literally (markup off).
    """

    SUB_TITLE = "Recordings"
    BINDINGS = [
        Binding("enter", "process", "Process"),
        Binding("t", "retranscribe", "Transcribe again"),
        Binding("d", "delete", "Delete"),
        Binding("r", "refresh", "Refresh"),
        Binding("escape", "back", "Back"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._items: list[RecordingInfo] = []
        self._personas: list[str] = []
        self._list_error: str | None = None
        self._load: Worker | None = None     # the read in progress, None when idle
        self._reload_wanted = False          # a refresh was asked for while a read ran
        self._delete: Worker | None = None   # the delete in progress, None when idle
        self._deleting: Path | None = None   # the file it deletes, for the message

    def compose(self) -> ComposeResult:
        app = self.app
        self._personas = personas = list(app.deps.list_personas())
        yield Header()
        with Vertical(id="recordings-panel"):
            with Horizontal(classes="field"):
                yield Label("Persona", classes="field-label")
                yield self._persona_select(personas, initial_persona(app.config, personas))
            yield Static(LOADING, id="recordings-note", markup=False)
            yield DataTable(id="recordings", cursor_type="row", zebra_stripes=True)
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#recordings", DataTable)
        for label in COLUMNS:
            table.add_column(label, width=NAME_WIDTH if label == COLUMNS[0] else None)
        table.focus()

    async def on_screen_resume(self) -> None:
        """Shown first, or back from Processing / Result / Personas: follow the persona list
        and read the folder again."""
        await self._follow_persona_list()
        self.action_refresh()

    async def _follow_persona_list(self) -> None:
        """Rebuild the persona Select when the personas changed (restored, added, removed):
        keep the current choice if it still exists, else the configured default."""
        try:
            personas = list(self.app.deps.list_personas())
        except Exception:
            return  # cannot tell: keep what is shown
        if personas == self._personas:
            return
        current = self.query_one("#persona", Select).value
        initial = current if current in personas else initial_persona(self.app.config, personas)
        self._personas = personas
        old = self.query_one("#persona", Select)
        parent = old.parent
        await old.remove()
        await parent.mount(self._persona_select(personas, initial))

    @staticmethod
    def _persona_select(personas: list[str], initial: object) -> Select:
        return Select(
            [(slug, slug) for slug in personas], value=initial,
            allow_blank=not personas, prompt="No personas", id="persona",
        )

    # ── reading ──────────────────────────────────────────────────────────────

    def action_refresh(self) -> None:
        """Read the recordings folder again (the folder named in the settings right now)."""
        if self._leaving:
            return
        if self._load is not None:
            self._reload_wanted = True
            return
        directory = self.app.deps.recordings_dir
        self._load = self.run_worker(
            partial(list_recordings, Path(directory)), thread=True, exit_on_error=False,
            group="recordings-load",
        )
        self._show_rows()

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        """Show a finished read or delete; one cancelled with the screen shows nothing."""
        worker, state = event.worker, event.state
        if state not in (WorkerState.SUCCESS, WorkerState.ERROR, WorkerState.CANCELLED):
            return
        if worker is self._load:
            self._read_finished(worker, state)
        elif worker is self._delete:
            self._delete_finished(worker, state)

    def _read_finished(self, worker: Worker, state: WorkerState) -> None:
        self._load = None
        if state == WorkerState.CANCELLED or self._leaving or not self.is_attached:
            return
        if state == WorkerState.SUCCESS:
            self._items, self._list_error = list(worker.result), None
        else:
            self._items = []
            self._list_error = readable_reason(worker.error) or type(worker.error).__name__
        if self._reload_wanted:
            self._reload_wanted = False
            self.action_refresh()
        else:
            self._show_rows()

    # ── showing ──────────────────────────────────────────────────────────────

    def _notice(self) -> str:
        if self._delete is not None and self._deleting is not None:
            return f"Deleting {self._deleting.name}…"
        if self._load is not None:
            return LOADING
        if self._list_error is not None:
            return f"Recordings cannot be listed: {self._list_error}"
        if not self._items:
            return f"No recordings yet in {self.app.deps.recordings_dir}"
        return ""

    def _show_rows(self) -> None:
        table = self.query_one("#recordings", DataTable)
        if self._load is None:  # while a read runs the old rows stay, with the note over them
            self._fill(table)
        note = self.query_one("#recordings-note", Static)
        text = self._notice()
        note.update(text)
        note.display = bool(text)

    def _fill(self, table: DataTable) -> None:
        keep, at = None, table.cursor_row
        if table.row_count:
            keep = table.ordered_rows[table.cursor_row].key.value
        table.clear()
        for info in self._items:
            has = SAVED if info.has_transcript else NOT_SAVED
            cells = (
                Text(shorten(info.path.name), no_wrap=True),
                Text(format_date(info.modified)), Text(format_size(info.size_bytes)), Text(has),
            )
            table.add_row(*cells, key=str(info.path))
        if table.row_count:
            keys = [row.key.value for row in table.ordered_rows]
            # The same file keeps the cursor; a deleted one hands it to the row that followed.
            table.move_cursor(row=keys.index(keep) if keep in keys else min(at, len(keys) - 1))

    # ── actions ──────────────────────────────────────────────────────────────

    def _can_act(self) -> bool:
        return self._may_leave() and self._delete is None

    def _selected(self) -> RecordingInfo | None:
        table = self.query_one("#recordings", DataTable)
        if not table.row_count:
            return None
        key = table.ordered_rows[table.cursor_row].key.value
        return next((info for info in self._items if str(info.path) == key), None)

    def _in_use(self, path: Path) -> str | None:
        """Why this WAV cannot be touched right now, or None."""
        if self.app.ledger.is_busy(str(path)):
            return f"{path.name} is being processed"
        guard = self.app.recorder_guard
        if guard is not None and os.path.abspath(guard.wav_path) == os.path.abspath(path):
            return f"{path.name} is being recorded"
        return None

    def navigation_blocked(self) -> bool:
        """True while a delete runs: its outcome is always reported here, so no leaving."""
        return self._delete is not None

    async def wait_for_pending(self, timeout: float = 5.0) -> None:
        """Let a running delete finish (for a few seconds at most), so quitting leaves the
        session ledger right about it."""
        worker = self._delete
        if worker is None:
            return
        try:
            await asyncio.wait_for(worker.wait(), timeout)
        except Exception:  # the delete failed (reported below) or took too long
            pass
        if self._delete is worker and worker.is_finished:
            self._finish_delete(worker, worker.state)

    def action_back(self) -> None:
        """Leave; a read still going finishes unseen. Not while a delete runs."""
        if not self.navigation_blocked():
            self._leave_once()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Enter on the table (it takes the key before the screen's binding does)."""
        event.stop()
        self.action_process()

    def action_process(self) -> None:
        """Process the recording under the cursor with the chosen persona."""
        self._process(retranscribe=False)

    def action_retranscribe(self) -> None:
        """Transcribe it again, ignoring the saved transcript (plain processing without one)."""
        info = self._selected()
        self._process(retranscribe=info is not None and info.has_transcript)

    def _process(self, *, retranscribe: bool) -> None:
        info = self._selected()
        if not self._can_act() or info is None:
            return
        slug = self.query_one("#persona", Select).value
        if not isinstance(slug, str):
            self.app.notify(NO_PERSONAS, severity="error")
            return
        reason = self._in_use(info.path)
        if reason is not None:
            self.app.notify(reason, severity="error", markup=False)
            return
        self.app.process_file(str(info.path), slug, retranscribe=retranscribe)

    def action_delete(self) -> None:
        """Ask, then delete the recording under the cursor and its saved transcript."""
        info = self._selected()
        if not self._can_act() or info is None:
            return
        reason = self._in_use(info.path)
        if reason is not None:
            self.app.notify(reason, severity="error", markup=False)
            return
        self.app.push_screen(
            ConfirmScreen(_delete_question(info.path.name), "Delete"),
            partial(self._delete_answered, info.path),
        )

    def _delete_answered(self, path: Path, confirmed: bool | None) -> None:
        # ``path`` is what was selected when d was pressed, whatever the cursor says now.
        if not confirmed or not self._can_act():
            return
        reason = self._in_use(path)
        if reason is not None:
            self.app.notify(reason, severity="error", markup=False)
            return
        self._deleting = path
        self._delete = self.run_worker(
            partial(self.app.deps.delete_recording, path), thread=True, exit_on_error=False,
            group="recordings-delete",
        )
        self._show_rows()

    def _delete_finished(self, worker: Worker, state: WorkerState) -> None:
        if worker is self._delete:
            self._finish_delete(worker, state)

    def _finish_delete(self, worker: Worker, state: WorkerState) -> None:
        self._delete = None
        path = self._deleting
        if state == WorkerState.CANCELLED or path is None:
            return
        # Whatever the outcome says, if the audio is gone the exit message must not name it.
        if not os.path.lexists(path):
            self.app.ledger.forget(str(path))
        name, notify = path.name, self.app.notify
        if state == WorkerState.SUCCESS:
            notify(f"Deleted {name}", markup=False)
        elif isinstance(worker.error, FileNotFoundError):
            notify(f"{name} is no longer there", severity="warning", markup=False)
        else:
            reason = readable_reason(worker.error) or type(worker.error).__name__
            notify(f"Could not delete {name}: {reason}", severity="error", markup=False)
        # Re-read either way: a sidecar can fail after the audio was already removed.
        self.action_refresh()
