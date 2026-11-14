"""The TinySteno Textual application shell."""

import asyncio
import os
import signal
from datetime import datetime
from pathlib import Path

from textual.app import App
from textual.binding import Binding

from tinysteno.personas import Persona
from tinysteno.session import resolve_timeout_seconds
from tinysteno.tui.deps import TuiDeps
from tinysteno.tui.guard import RecorderGuard
from tinysteno.pipeline import PipelineResult
from tinysteno.tui.home import HomeScreen
from tinysteno.tui.ledger import SessionLedger
from tinysteno.tui.processing import ProcessingScreen
from tinysteno.tui.recording import RecordingScreen
from tinysteno.tui.result import ResultScreen


class TinyStenoApp(App):
    """Record, process and review meetings from the terminal."""

    CSS_PATH = "styles.tcss"
    TITLE = "TinySteno"
    BINDINGS = [
        ("q", "quit", "Quit"),
        Binding("escape", "dismiss_notices", "Dismiss notices", show=False),
    ]

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
        # What became of each recording; every exit message is read from here.
        self.ledger = SessionLedger()
        # The started recorder until the UI has reported what became of it; run_tui's safety
        # net stops it if the app ends any other way (crash, closed terminal).
        self.recorder_guard: RecorderGuard | None = None
        self.exit_signals: list[signal.Signals] = []  # signals routed to action_quit
        self._previous_signal_handlers: dict[signal.Signals, object] = {}

    def on_mount(self) -> None:
        self._install_signal_handlers()
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

    # ── closing the terminal (SIGHUP) or being killed (SIGTERM) saves like q ────

    def _install_signal_handlers(self) -> None:
        loop = asyncio.get_running_loop()
        for name in ("SIGHUP", "SIGTERM"):
            sig = getattr(signal, name, None)
            if sig is None:
                continue
            previous = signal.getsignal(sig)
            try:
                loop.add_signal_handler(sig, self._on_exit_signal)
            except (NotImplementedError, RuntimeError, ValueError):
                continue  # no signal support here (Windows, or not the main thread)
            self._previous_signal_handlers[sig] = previous
            self.exit_signals.append(sig)

    def _on_exit_signal(self) -> None:
        self.call_later(self.action_quit)

    def on_unmount(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in self.exit_signals:
            loop.remove_signal_handler(sig)
            previous = self._previous_signal_handlers.pop(sig)
            if previous is not None:
                signal.signal(sig, previous)
        self.exit_signals = []

    def hold_recorder(self, guard: RecorderGuard) -> None:
        """Track a started recorder (called from the start worker: one attribute store)."""
        self.recorder_guard = guard

    def release_recorder(self, guard: RecorderGuard) -> None:
        """The UI has settled this recorder and told the user what became of it."""
        if self.recorder_guard is guard:
            self.recorder_guard = None

    def final_result(self) -> tuple[str | None, int]:
        """Exit message and code once the app has stopped, saving a recorder nothing stopped."""
        guard, self.recorder_guard = self.recorder_guard, None
        if guard is not None:
            note, failed = guard.final_message()
            if failed:
                self.ledger.save_failed(note)
            elif note:
                self.ledger.saved(guard.wav_path)
        message, failed = self.ledger.summary()
        return message, 1 if failed else (self.return_code or 0)

    # ── navigation ───────────────────────────────────────────────────────────

    def start_recording(self, name: str | None, persona_slug: str) -> None:
        """Begin a recording session."""
        self.events.append(("record", name, persona_slug))
        self.push_screen(RecordingScreen(name, persona_slug))

    def finish_recording(
        self, wav_path: str, name: str | None, persona_slug: str, started_at: datetime
    ) -> None:
        """Hand a successfully stopped recording to processing."""
        self.events.append(("process", wav_path, persona_slug))
        self.ledger.saved(wav_path)
        self._replace_topmost(
            RecordingScreen, ProcessingScreen(wav_path, name, persona_slug, started_at)
        )

    def process_file(self, wav_path: str, persona_slug: str) -> None:
        """Process an existing recording."""
        self.events.append(("process", wav_path, persona_slug))
        try:
            started_at = datetime.fromtimestamp(os.path.getmtime(wav_path))
        except OSError:
            self.notify(f"Recording not found: {Path(wav_path).name}", severity="error",
                        markup=False)
            return
        self.ledger.saved(wav_path)
        self.push_screen(ProcessingScreen(wav_path, None, persona_slug, started_at))

    def show_result(self, result: PipelineResult) -> None:
        """Show the finished note."""
        self.events.append(("result", result))
        self._replace_topmost(ProcessingScreen, ResultScreen(result))

    def _replace_topmost(self, kind: type, new_screen) -> None:
        """Replace the topmost screen of ``kind`` (closing any dialog above it, such as the
        command palette); with none in the stack, replace the current screen."""
        target = next((s for s in reversed(self.screen_stack) if isinstance(s, kind)), None)
        while target is not None and self.screen is not target:
            self.pop_screen()
        self.switch_screen(new_screen)

    def action_dismiss_notices(self) -> None:
        """Clear every notice on screen (a failed-save notice stays until this or a click)."""
        self.clear_notifications()

    async def action_quit(self) -> None:
        """Quit; a recording in progress is stopped and saved first, never lost.

        The exit message comes from the ledger: every WAV that has no note yet (a running
        pipeline cannot be cancelled, so that one included) and every save that failed, in
        which case the app exits with code 1.
        """
        recording = next(
            (s for s in reversed(self.screen_stack) if isinstance(s, RecordingScreen)), None
        )
        if recording is not None:
            await recording.stop_for_quit()
        message, failed = self.ledger.summary()
        self.exit_message = message
        self.exit(return_code=1 if failed else 0)
