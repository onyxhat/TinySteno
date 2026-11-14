"""The TinySteno Textual application shell."""

from textual.app import App

from tinysteno.personas import Persona
from tinysteno.session import resolve_timeout_seconds
from tinysteno.tui.deps import TuiDeps
from tinysteno.tui.home import HomeScreen


class TinyStenoApp(App):
    """Record, process and review meetings from the terminal."""

    CSS_PATH = "styles.tcss"
    TITLE = "TinySteno"
    BINDINGS = [("q", "quit", "Quit")]

    def __init__(  # pylint: disable=too-many-arguments
        self,
        config: dict,
        deps: TuiDeps,
        *,
        autostart: bool = False,
        name: str | None = None,
        persona: str | None = None,
        timeout_override: float | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.deps = deps
        self.autostart = autostart
        self.initial_name = name
        self.initial_persona = persona
        self.timeout_override = timeout_override
        self.exit_message: str | None = None
        self.events: list[tuple] = []

    def on_mount(self) -> None:
        theme = self.config.get("tui_theme")
        if theme in self.available_themes:
            self.theme = theme
        self.push_screen(HomeScreen(persona=self.initial_persona))
        if self.autostart:
            slug = self.initial_persona or self.config.get("persona", "default")
            self.start_recording(self.initial_name, slug)

    def timeout_seconds_for(self, persona: Persona) -> float | None:
        """Auto-stop limit in seconds for a persona, or None for no limit."""
        return resolve_timeout_seconds(
            self.config, persona.session_timeout_minutes, self.timeout_override
        )

    def start_recording(self, name: str | None, persona_slug: str) -> None:
        """Begin a recording session."""
        self.events.append(("record", name, persona_slug))

    def process_file(self, wav_path: str, persona_slug: str) -> None:
        """Process an existing recording."""
        self.events.append(("process", wav_path, persona_slug))
