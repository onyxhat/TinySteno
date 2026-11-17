"""The TinySteno Textual application shell."""

import asyncio
import os
import signal
import weakref
from collections.abc import Callable, Iterable
from datetime import datetime
from functools import partial
from pathlib import Path

from textual.app import App, SystemCommand
from textual.binding import Binding
from textual.screen import ModalScreen, Screen

from tinysteno.personas import Persona
from tinysteno.session import resolve_timeout_seconds
from tinysteno.tui.checks import ChecksScreen
from tinysteno.tui.confirm import ConfirmScreen
from tinysteno.tui.deps import TuiDeps
from tinysteno.tui.guard import RecorderGuard
from tinysteno.pipeline import PipelineResult
from tinysteno.tui.home import HomeScreen
from tinysteno.tui.ledger import SessionLedger
from tinysteno.tui.menu import DESTINATION_ORDER, MenuScreen
from tinysteno.tui.personas import PersonasScreen
from tinysteno.tui.processing import ProcessingScreen
from tinysteno.tui.recording import RecordingScreen
from tinysteno.tui.result import ResultScreen
from tinysteno.tui.settings import SettingsScreen
from tinysteno.tui.setup import SetupScreen

# While one of these is open, a recording or a pipeline run is (or was just) in progress: the
# Menu and its destinations stay closed, so settings never change under a running session.
_BUSY_SCREENS = (RecordingScreen, ProcessingScreen, ResultScreen)


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
        # Menu destinations: id -> (label, screen factory). The Menu and the command palette
        # list them in DESTINATION_ORDER; each management screen registers itself here.
        self.destinations: dict[str, tuple[str, Callable[[], Screen]]] = {
            "settings": ("Settings", SettingsScreen),
            "setup": ("Setup guide", SetupScreen),
            "personas": ("Personas", PersonasScreen),
            "checks": ("System check", ChecksScreen),
        }
        self._opened: dict[str, weakref.ref[Screen]] = {}  # the last screen of each destination
        self._asking_to_quit = False

    def on_mount(self) -> None:
        self._install_signal_handlers()
        self.use_theme(self.config.get("tui_theme"))
        self.push_screen(HomeScreen(persona=self.initial_persona))
        if self.deps.first_run and not self.autostart:
            # Over Home, so Skip, escape and the checks' Back all land on a working Home.
            self.call_after_refresh(self.open_destination, "setup")
        if self.autostart:
            slug = self.initial_persona or self.config.get("persona", "default")
            # After Home has mounted, so a failed start can fall back to a ready Home screen.
            self.call_after_refresh(self.start_recording, self.initial_name, slug)

    def use_theme(self, name: object) -> None:
        """Switch to the named theme when it is one of the available themes."""
        if isinstance(name, str) and name in self.available_themes:
            self.theme = name

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
        # Never held up by a dialog: a closing terminal or a kill quits at once.
        self.call_later(self._quit_now)

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

    def process_file(
        self, wav_path: str, persona_slug: str, *, retranscribe: bool = False
    ) -> None:
        """Process an existing recording."""
        self.events.append(("process", wav_path, persona_slug))
        try:
            started_at = datetime.fromtimestamp(os.path.getmtime(wav_path))
        except OSError:
            self.notify(f"Recording not found: {Path(wav_path).name}", severity="error",
                        markup=False)
            return
        self.ledger.saved(wav_path)
        self.push_screen(
            ProcessingScreen(wav_path, None, persona_slug, started_at, retranscribe))

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

    # ── the Menu ─────────────────────────────────────────────────────────────

    def _menu_entries(self) -> list[tuple[str, str]]:
        """(destination id, label) for each registered destination, in DESTINATION_ORDER."""
        rank = {dest: i for i, dest in enumerate(DESTINATION_ORDER)}
        ordered = sorted(self.destinations, key=lambda dest: rank.get(dest, len(rank)))
        return [(dest, self.destinations[dest][0]) for dest in ordered]

    def menu_available(self, screen: Screen | None = None) -> bool:
        """True when ``screen`` (default: the current one) may open the Menu: Home, or a
        management screen opened from Home; never a dialog, and never while a recording,
        processing or result screen is open."""
        screen = self.screen if screen is None else screen
        stack = self.screen_stack
        if isinstance(screen, ModalScreen) or any(isinstance(s, _BUSY_SCREENS) for s in stack):
            return False
        return any(isinstance(s, HomeScreen) for s in stack)

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        if action == "open_menu":
            return self.menu_available()
        return True

    def action_open_menu(self) -> None:
        """Show the Menu (only where menu_available allows it)."""
        if not self.menu_available():
            return
        self.push_screen(MenuScreen(self._menu_entries()), self._menu_chosen)

    def _menu_chosen(self, dest: str | None) -> None:
        if dest is not None:
            self.open_destination(dest)

    def open_destination(self, dest: str) -> None:
        """Open a registered destination's screen; ignored where the Menu is unavailable."""
        if dest not in self.destinations or not self.menu_available():
            return
        opened = self._opened.get(dest)
        if opened is not None and opened() in self.screen_stack:
            return  # already open (perhaps under another management screen)
        _, factory = self.destinations[dest]
        screen = factory()
        self._opened[dest] = weakref.ref(screen)
        self.push_screen(screen)

    def get_system_commands(self, screen: Screen) -> Iterable[SystemCommand]:
        """Command palette entries: Textual's defaults minus Screenshot, plus the app's own:
        one per Menu destination, where the Menu is available."""
        for command in super().get_system_commands(screen):
            if command.title != "Screenshot":
                yield command
        if self.menu_available(screen):
            for dest, label in self._menu_entries():
                yield SystemCommand(label, f"Open {label}", partial(self.open_destination, dest))

    def action_screenshot(self, filename: str | None = None, path: str | None = None) -> None:
        """Do nothing: screenshots are made by scripts/tui_screenshots.py, which calls
        save_screenshot directly."""

    def deliver_screenshot(  # pylint: disable=arguments-differ
        self,
        filename: str | None = None,
        path: str | None = None,
        time_format: str | None = None,
    ) -> str | None:
        """Do nothing: the palette command and the action both end here, so neither saves."""
        return None

    def action_dismiss_notices(self) -> None:
        """Clear every notice on screen (a failed-save notice stays until this or a click)."""
        self.clear_notifications()

    async def action_quit(self) -> None:
        """Quit (q, ctrl+q, the palette); ask first when the screen has unsaved changes.

        A screen with edits that quitting would lose implements ``has_unsaved_changes() ->
        bool``; the topmost screen that is not a dialog is asked. Exit signals skip this.
        """
        if self._asking_to_quit:
            return
        screen = next(
            (s for s in reversed(self.screen_stack) if not isinstance(s, ModalScreen)), None
        )
        has_unsaved = getattr(screen, "has_unsaved_changes", None)
        if callable(has_unsaved) and has_unsaved():
            self._asking_to_quit = True
            self.push_screen(
                ConfirmScreen("Quit and discard unsaved changes?", "Quit"), self._quit_answered
            )
            return
        await self._quit_now()

    def _quit_answered(self, confirmed: bool | None) -> None:
        self._asking_to_quit = False
        if confirmed:
            self.call_later(self._quit_now)

    async def _quit_now(self) -> None:
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
