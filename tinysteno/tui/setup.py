"""First-run Setup guide: the essential settings in one short form."""

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Footer, Header, Static

from tinysteno.config import config_path
from tinysteno.settings import SETTINGS
from tinysteno.tui.formscreen import FormScreen, readable_reason, unexpected
from tinysteno.tui.forms import SettingsForm

INTRO = "Welcome to TinySteno. These few settings are all you need to get started."
NO_PERSONAS_NOTE = (
    "No personas found. After setup, open the menu and choose Personas to restore the "
    "built-in ones."
)
_ESSENTIAL = tuple(s for s in SETTINGS if s.essential)


class SetupScreen(FormScreen):
    """The essential settings. Finish saves what the user changed, plus any essential setting
    the file still lacks (so a first-run file holds the user's answers), and shows the System
    check; Skip and escape leave without writing."""

    SUB_TITLE = "Setup"
    BINDINGS = [
        Binding("ctrl+s", "finish", "Save and continue"),
        Binding("escape", "skip", "Skip"),
    ]

    def __init__(self, automatic: bool = False) -> None:
        """``automatic`` is the instance the app opens by itself on first run: it never asks
        before Skip or quit. One opened from the Menu asks like the Settings screen."""
        super().__init__()
        self._automatic = automatic
        self._load_error: str | None = None   # the config file could not be read
        self._notice = ""                     # anything else worth saying under the form
        self._no_personas = False

    def _build_form(self) -> None:
        deps = self.app.deps
        try:
            stored = deps.load_settings()
        except Exception as err:
            # A file that cannot be read: no form, so nothing built on defaults can be
            # written over the real values once the file is repaired.
            self._load_error = readable_reason(err)
            return
        try:
            personas: list[str] | None = list(deps.list_personas())
        except Exception as err:
            personas = None
            self._notice = f"Personas cannot be listed: {readable_reason(err)}"
        self._no_personas = personas == []
        try:
            self._form = SettingsForm(
                _ESSENTIAL, stored, personas, overridden=deps.overridden_keys, flat=True,
                id="setup-form",
            )
        except Exception as err:
            self._notice = unexpected(err)

    def compose(self) -> ComposeResult:
        self._build_form()
        yield Header()
        with Vertical(id="setup-panel"):
            yield Static(INTRO, id="setup-intro", markup=False)
            if self._no_personas:
                yield Static(NO_PERSONAS_NOTE, id="setup-no-personas", markup=False)
            if self._form is not None:
                yield self._form
            elif self._load_error is not None:
                yield Static(f"Config file cannot be read: {self._load_error}",
                             id="load-error", markup=False)
                yield Static(f"File: {config_path()}", id="config-path", markup=False)
            error = Static(self._notice, id="form-error", markup=False)
            error.display = bool(self._notice)
            yield error
            with Horizontal(id="setup-buttons"):
                yield Button("Save and continue", id="finish", variant="primary", compact=True,
                             disabled=self._form is None)
                yield Button("Skip for now", id="skip", compact=True)
        yield Footer()

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        if action == "finish" and self._form is None:
            return None
        return True

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if event.button.id == "finish":
            self.action_finish()
        elif event.button.id == "skip":
            self.action_skip()

    def has_unsaved_changes(self) -> bool:
        """Never for the automatic first-run instance (quitting from it must not ask);
        otherwise the form's real state."""
        if self._automatic:
            return False
        return super().has_unsaved_changes()

    def action_skip(self) -> None:
        """Leave without writing anything; a guide opened from the Menu asks first when it
        has unsaved changes."""
        if self._automatic:
            if self._can_act():
                self._close()
            return
        self._leave_asking()

    def action_finish(self) -> None:
        """Save what changed and what the file lacks, then show the System check."""
        if self._form is None or not self._can_act():
            return
        try:
            values = self._values_to_save(self._form)
            if values is None or (values and self._write(values) is None):
                return
        except Exception as err:
            self._show_error(unexpected(err))
            return
        # Through open_destination, so the palette cannot stack a second System check.
        app = self.app
        self._close()
        app.open_destination("checks")

    def _values_to_save(self, form: SettingsForm) -> dict | None:
        """The user's changes, plus the shown value of each essential setting the stored file
        lacks right now (it is read again: it may have changed since the guide opened). None
        after showing why nothing can be saved."""
        changes = self._collect(form, form.changes)
        if changes is None:
            return None
        try:
            stored_keys = frozenset(self.app.deps.load_settings())
        except Exception as err:
            self._show_error(f"Config file cannot be read: {readable_reason(err)}")
            return None
        lacking = [s.key for s in _ESSENTIAL if s.key not in stored_keys and s.key not in changes]
        shown = self._collect(form, lambda: form.read(lacking)) if lacking else {}
        if shown is None:
            return None
        changes.update((key, value) for key, value in shown.items() if value is not None)
        return {s.key: changes[s.key] for s in _ESSENTIAL if s.key in changes}
