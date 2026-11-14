"""Home screen: pick a name and persona, start recording, or reopen a recording."""

from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Input, Label, OptionList, Select, Static
from textual.widgets.option_list import Option

from tinysteno.personas import PersonaInvalidError, PersonaNotFoundError

NO_RECORDINGS = "No recordings yet"


def recent_recordings(directory: Path, limit: int = 8) -> list[Path]:
    """Newest-first WAV files in a directory (empty when it does not exist)."""
    if not directory.is_dir():
        return []
    wavs = sorted(directory.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
    return wavs[:limit]


class HomeScreen(Screen):
    """Landing screen."""

    BINDINGS = [("r", "record", "Record")]

    def __init__(self, persona: str | None = None) -> None:
        super().__init__()
        self._persona_arg = persona
        self._recent: list[Path] = []

    def compose(self) -> ComposeResult:
        app = self.app
        personas = app.deps.list_personas()
        wanted = self._persona_arg or app.config.get("persona")
        initial = wanted if wanted in personas else (personas[0] if personas else Select.BLANK)
        self._recent = recent_recordings(app.deps.recordings_dir)
        yield Header()
        with Vertical(id="panel"):
            yield Label("Meeting name (optional)")
            yield Input(placeholder="Untitled meeting", id="name")
            yield Label("Persona")
            yield Select(
                [(slug, slug) for slug in personas], value=initial, allow_blank=False,
                id="persona",
            )
            yield Static("", id="timeout-hint")
            yield Button("Record", id="record", variant="success")
            yield Label("Recent recordings")
            yield OptionList(*self._recent_options(), id="recent")
        yield Footer()

    def _recent_options(self) -> list[Option]:
        if self._recent:
            return [Option(p.name) for p in self._recent]
        return [Option(NO_RECORDINGS, disabled=True)]

    def on_mount(self) -> None:
        self.query_one("#record", Button).focus()
        self._update_hint()

    def on_screen_resume(self) -> None:
        """Back from another screen: show recordings made meanwhile, Record ready."""
        recent = recent_recordings(self.app.deps.recordings_dir)
        if recent != self._recent:
            self._recent = recent
            listing = self.query_one("#recent", OptionList)
            listing.clear_options()
            listing.add_options(self._recent_options())
        self.query_one("#record", Button).focus()

    def _slug(self) -> str:
        value = self.query_one("#persona", Select).value
        return value if isinstance(value, str) else self.app.config.get("persona", "default")

    def _update_hint(self) -> None:
        hint = self.query_one("#timeout-hint", Static)
        try:
            persona = self.app.deps.load_persona(self._slug())
        except (PersonaNotFoundError, PersonaInvalidError) as err:
            hint.add_class("error")
            hint.update(f"Persona error: {err}")
            return
        hint.remove_class("error")
        limit = self.app.timeout_seconds_for(persona)
        hint.update("No time limit" if limit is None else f"Auto-stop after {limit / 60:g} min")

    def _start(self) -> None:
        name = self.query_one("#name", Input).value.strip()
        self.app.start_recording(name or None, self._slug())

    def action_record(self) -> None:
        """Start recording with the current name and persona."""
        self._start()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "record":
            self._start()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "name":
            self._start()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "persona":
            self._update_hint()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id == "recent" and self._recent:
            self.app.process_file(str(self._recent[event.option_index]), self._slug())
