"""Settings screen: edit ~/.tinysteno/config.yaml from the terminal interface."""

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Static

from tinysteno.config import config_path
from tinysteno.settings import SETTINGS
from tinysteno.tui.confirm import ConfirmScreen
from tinysteno.tui.deps import SettingsNotApplied
from tinysteno.tui.forms import FormError, SettingsForm


def readable_reason(err: Exception) -> str:
    """The error's text without quoted file content.

    A YAML error quotes the offending line of the file (indented four spaces, with a ``^``
    under it), and that line may hold an API key. config.py no longer passes that text on;
    this is a second layer for any other error that might.
    """
    lines = [line for line in str(err).splitlines() if not line.startswith("    ")]
    return "\n".join(line.rstrip(":") if line.startswith("  in ") else line for line in lines)


def _unexpected(err: Exception) -> str:
    return f"Unexpected error: {type(err).__name__}"


class SettingsScreen(Screen):
    """Every editable setting in one form; ctrl+s saves only what changed.

    Stored values live only inside the form: no handler keeps them in a local variable
    where an unexpected error could print them, and every handler turns such an error into
    a line under the form instead of a crash.
    """

    SUB_TITLE = "Settings"
    BINDINGS = [
        Binding("ctrl+s", "save", "Save"),
        Binding("escape", "back", "Back"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._form: SettingsForm | None = None
        self._load_error: str | None = None
        self._notice = ""       # shown under the form when the screen opens
        self._leaving = False   # dismiss() has been called: act on nothing more
        # (not `_closing`: Textual's message pump uses that name)

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
            self._notice = _unexpected(err)

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

    def _show_error(self, message: str) -> None:
        error = self.query_one("#form-error", Static)
        error.update(message)
        error.display = bool(message)

    def _can_act(self) -> bool:
        """Only the top screen that is not already closing handles Save and Back (queued
        presses arriving after the first save must not pop the screen below)."""
        return not self._leaving and self.is_active

    def _close(self) -> None:
        self._leaving = True
        self.dismiss()

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
            self._show_error(_unexpected(err))

    def _save(self, form: SettingsForm) -> None:
        try:
            changes = form.changes()
        except FormError as err:
            self._show_error(err.message)
            form.focus_field(err.key)
            return
        if not changes:
            self._close()
            return
        try:
            self.app.deps.save_settings(changes)
        except SettingsNotApplied as err:
            self._saved(changes)
            self.app.notify(
                "Settings saved, but they could not be applied to this session: "
                f"{readable_reason(err)}. Restart TinySteno to use them.",
                severity="warning", markup=False,
            )
            self._close()
            return
        except ValueError as err:
            self._show_error(readable_reason(err))
            return
        except Exception as err:  # OSError, or anything else: keep the edits on screen
            self._show_error(f"Could not save settings: {readable_reason(err)}")
            return
        self._saved(changes)
        if "tui_theme" in changes:
            self.app.use_theme(changes["tui_theme"])
        self.app.notify("Settings saved")
        self._close()

    def _saved(self, changes: dict) -> None:
        # Key names only: a value may be an API key.
        self.app.events.append(("settings_saved", tuple(sorted(changes))))

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
            self._show_error(_unexpected(err))

    def _discard_answered(self, confirmed: bool | None) -> None:
        if confirmed and self._can_act():
            self._close()
