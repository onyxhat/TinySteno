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
    under it), and that line may hold an API key.
    """
    lines = [line for line in str(err).splitlines() if not line.startswith("    ")]
    return "\n".join(line.rstrip(":") if line.startswith("  in ") else line for line in lines)


class SettingsScreen(Screen):
    """Every editable setting in one form; ctrl+s saves only what changed."""

    SUB_TITLE = "Settings"
    BINDINGS = [
        Binding("ctrl+s", "save", "Save"),
        Binding("escape", "back", "Back"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._form: SettingsForm | None = None

    def compose(self) -> ComposeResult:
        deps = self.app.deps
        try:
            stored = deps.load_settings()
            load_error = None
        except Exception as err:  # ValueError (bad YAML, not a mapping), OSError, anything
            stored, load_error = {}, readable_reason(err)
        yield Header()
        with Vertical(id="settings-panel"):
            if load_error is None:
                self._form = SettingsForm(
                    SETTINGS, stored, deps.list_personas(),
                    overridden=deps.overridden_keys, id="settings-form",
                )
                yield self._form
            else:
                yield Static(f"Config file cannot be read: {load_error}", id="load-error",
                             markup=False)
                yield Static(f"File: {config_path()}", id="config-path", markup=False)
            yield Static("", id="form-error", markup=False)
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

    def action_save(self) -> None:
        """Validate the changed fields and write them; leave the screen once saved."""
        form = self._form
        if form is None:
            return
        try:
            changes = form.changes()
        except FormError as err:
            self._show_error(err.message)
            form.focus_field(err.key)
            return
        if not changes:
            self.dismiss()
            return
        try:
            self.app.deps.save_settings(changes)
        except SettingsNotApplied as err:
            self._saved(changes)
            self.app.notify(
                "Settings saved, but they could not be applied to this session: "
                f"{err}. Restart TinySteno to use them.",
                severity="warning", markup=False,
            )
            self.dismiss()
            return
        except ValueError as err:
            self._show_error(str(err))
            return
        except Exception as err:  # OSError, or anything else: keep the edits on screen
            self._show_error(f"Could not save settings: {err}")
            return
        self._saved(changes)
        if "tui_theme" in changes:
            self.app.use_theme(changes["tui_theme"])
        self.app.notify("Settings saved")
        self.dismiss()

    def _saved(self, changes: dict) -> None:
        # Key names only: a value may be an API key.
        self.app.events.append(("settings_saved", tuple(sorted(changes))))

    def action_back(self) -> None:
        """Leave; with unsaved changes, ask first."""
        if self._form is None or not self._form.has_changes():
            self.dismiss()
            return

        def discard(confirmed: bool | None) -> None:
            if confirmed:
                self.dismiss()

        self.app.push_screen(ConfirmScreen("Discard unsaved changes?", "Discard"), discard)
