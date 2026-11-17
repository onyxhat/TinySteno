"""First-run Setup guide: the essential settings in one short form."""

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Footer, Header, Static

from tinysteno.settings import SETTINGS
from tinysteno.tui.checks import ChecksScreen
from tinysteno.tui.formscreen import FormScreen, readable_reason, unexpected
from tinysteno.tui.forms import SettingsForm

INTRO = "Welcome to TinySteno. These few settings are all you need to get started."
NO_PERSONAS_NOTE = (
    "No personas found. After setup, open the menu and choose Personas to restore the "
    "built-in ones."
)
_ESSENTIAL = tuple(s for s in SETTINGS if s.essential)


class SetupScreen(FormScreen):
    """The essential settings. Finish saves every one of them (so a first-run file holds the
    user's answers) and shows the System check; Skip and escape leave without writing."""

    SUB_TITLE = "Setup"
    BINDINGS = [
        Binding("ctrl+s", "finish", "Save and continue"),
        Binding("escape", "skip", "Skip"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._notice = ""
        self._no_personas = False

    def _build_form(self) -> None:
        deps = self.app.deps
        try:
            stored = deps.load_settings()
        except Exception as err:
            # A file that cannot be read: show the guide on the defaults; saving will then
            # report the real reason and leave the file untouched.
            stored = {}
            self._notice = f"Config file cannot be read: {readable_reason(err)}"
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
        """Never on first run (quitting from the guide must not ask); otherwise the form's
        real state."""
        if self._form is None or self._leaving or self.app.deps.first_run:
            return False
        try:
            return self._form.has_changes()
        except Exception:
            return True

    def action_skip(self) -> None:
        """Leave without writing anything."""
        if self._can_act():
            self._close()

    def action_finish(self) -> None:
        """Save every essential setting, then show the System check."""
        if self._form is None or not self._can_act():
            return
        try:
            values = self._collect(self._form, self._form.read)
            if values is None or self._write(values) is None:
                return
        except Exception as err:
            self._show_error(unexpected(err))
            return
        self._leaving = True
        self.app.switch_screen(ChecksScreen())
