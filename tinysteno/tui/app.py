"""The TinySteno Textual application shell."""

import os
import shlex
from datetime import datetime
from pathlib import Path

from textual.app import App

from tinysteno.personas import Persona
from tinysteno.session import resolve_timeout_seconds
from tinysteno.tui.deps import TuiDeps
from tinysteno.pipeline import PipelineResult
from tinysteno.tui.home import HomeScreen
from tinysteno.tui.processing import ProcessingScreen
from tinysteno.tui.recording import RecordingScreen
from tinysteno.tui.result import ResultScreen


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
            # After Home has mounted, so a failed start can fall back to a ready Home screen.
            self.call_after_refresh(self.start_recording, self.initial_name, slug)

    def timeout_seconds_for(self, persona: Persona) -> float | None:
        """Auto-stop limit in seconds for a persona, or None for no limit."""
        return resolve_timeout_seconds(
            self.config, persona.session_timeout_minutes, self.timeout_override
        )

    def start_recording(self, name: str | None, persona_slug: str) -> None:
        """Begin a recording session."""
        self.events.append(("record", name, persona_slug))
        self.push_screen(RecordingScreen(name, persona_slug))

    def finish_recording(
        self, wav_path: str, name: str | None, persona_slug: str, started_at: datetime
    ) -> None:
        """Hand a successfully stopped recording to processing."""
        self.events.append(("process", wav_path, persona_slug))
        self.switch_screen(ProcessingScreen(wav_path, name, persona_slug, started_at))

    def process_file(self, wav_path: str, persona_slug: str) -> None:
        """Process an existing recording."""
        self.events.append(("process", wav_path, persona_slug))
        try:
            started_at = datetime.fromtimestamp(os.path.getmtime(wav_path))
        except OSError:
            self.notify(f"Recording not found: {Path(wav_path).name}", severity="error",
                        markup=False)
            return
        self.push_screen(ProcessingScreen(wav_path, None, persona_slug, started_at))

    def show_result(self, result: PipelineResult) -> None:
        """Show the finished note."""
        self.events.append(("result", result))
        self.switch_screen(ResultScreen(result))

    async def action_quit(self) -> None:
        """Quit; a recording in progress is stopped and saved first, never lost.

        If saving fails, the app exits with code 1 and the error as its exit message.
        """
        recording = next(
            (s for s in reversed(self.screen_stack) if isinstance(s, RecordingScreen)), None
        )
        if recording is not None:
            wav = await recording.stop_for_quit()
            if recording.save_error:
                self.exit_message = recording.save_error
                self.exit(return_code=1)
                return
            if wav:
                self.exit_message = (
                    f"Saved recording: {wav}\n"
                    f"Process it with: tinysteno process {shlex.quote(wav)}"
                )
        processing = next(
            (s for s in reversed(self.screen_stack) if isinstance(s, ProcessingScreen)), None
        )
        if processing is not None and processing.running:
            # The pipeline thread cannot be cancelled; the WAV stays on disk.
            self.exit_message = processing.quit_message()
        self.exit()
