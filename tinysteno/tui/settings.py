"""Settings screen: edit ~/.tinysteno/config.yaml from the terminal interface."""

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Footer, Header, Static

from tinysteno.config import config_path
from tinysteno.settings import SETTINGS
from tinysteno.tui.confirm import ConfirmScreen
from tinysteno.tui.formscreen import FormScreen, readable_reason, unexpected
from tinysteno.tui.forms import SettingsForm


class SettingsScreen(FormScreen):
    """Every editable setting in one form; ctrl+s saves only what changed.

    Every handler turns an unexpected error into a line under the form instead of a crash.
    """

    SUB_TITLE = "Settings"
    BINDINGS = [
        Binding("ctrl+s", "save", "Save"),
        Binding("escape", "back", "Back"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._load_error: str | None = None
        self._notice = ""       # shown under the form when the screen opens

    def _build_form(self) -> None:
        deps = self.app.deps
        try:
            stored = deps.load_settings()
        except Exception as err:  # ValueError (bad YAML, not a mapping), OSError, anything
            self._load_error = readable_reason(err)
            return
        try:
            personas: list[str] | None = list(deps.list_personas())
        except Exception as err:
            personas = None
            self._notice = f"Personas cannot be listed: {readable_reason(err)}"
        try:
            self._form = SettingsForm(
                SETTINGS, stored, personas, overridden=deps.overridden_keys, id="settings-form"
            )
        except Exception as err:
            self._notice = unexpected(err)

    def compose(self) -> ComposeResult:
        self._build_form()
        yield Header()
        with Vertical(id="settings-panel"):
            if self._form is not None:
                yield self._form
            elif self._load_error is not None:
                yield Static(f"Config file cannot be read: {self._load_error}",
                             id="load-error", markup=False)
                yield Static(f"File: {config_path()}", id="config-path", markup=False)
            error = Static(self._notice, id="form-error", markup=False)
            error.display = bool(self._notice)
            yield error
            with Horizontal(id="settings-buttons"):
                yield Button("Save (ctrl+s)", id="save", variant="primary", compact=True,
                             disabled=self._form is None)
                yield Button("Back (esc)", id="back", compact=True)
        yield Footer()

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        if action == "save" and self._form is None:
            return None  # shown disabled: there is nothing to save
        return True

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if event.button.id == "save":
            self.action_save()
        elif event.button.id == "back":
            self.action_back()

    def has_unsaved_changes(self) -> bool:
        """True when leaving now would lose edits (the app asks before quitting)."""
        if self._form is None or self._leaving:
            return False
        try:
            return self._form.has_changes()
        except Exception:
            return True  # cannot tell: ask rather than lose edits

    # ── saving ───────────────────────────────────────────────────────────────

    def action_save(self) -> None:
        """Validate the changed fields and write them; leave the screen once saved."""
        if self._form is None or not self._can_act():
            return
        try:
            self._save(self._form)
        except Exception as err:
            self._show_error(unexpected(err))

    def _save(self, form: SettingsForm) -> None:
        changes = self._collect(form, form.changes)
        if changes is None:
            return
        if not changes:
            self._close()
            return
        if self._write(changes) is not None:
            self._close()

    # ── leaving ──────────────────────────────────────────────────────────────

    def action_back(self) -> None:
        """Leave; with unsaved changes, ask first."""
        if not self._can_act():
            return
        try:
            if not self.has_unsaved_changes():
                self._close()
                return
            self.app.push_screen(
                ConfirmScreen("Discard unsaved changes?", "Discard"), self._discard_answered
            )
        except Exception as err:
            self._show_error(unexpected(err))

    def _discard_answered(self, confirmed: bool | None) -> None:
        if confirmed and self._can_act():
            self._close()
