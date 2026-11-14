"""Recording screen: elapsed clock, one level meter per stream, and an abortable timeout."""

import asyncio
from collections import deque
from datetime import datetime

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen, Screen
from textual.widgets import Button, Digits, Footer, Header, Label, ProgressBar, Sparkline, Static

from tinysteno.personas import PersonaInvalidError, PersonaNotFoundError
from tinysteno.session import SessionTimer, format_clock

REFRESH_SECONDS = 0.1
METER_SAMPLES = 120  # 12 seconds of history at the 0.1 s refresh
PULSE_SECONDS = 0.7
TRANSITION_SECONDS = 0.45  # a little over the panel's 400 ms CSS transition
WARNING_TEXT = "Auto-stopping in 60s. Press K to keep recording."

# Lifecycle: loading -> starting -> recording -> stopping|discarding -> done.
# Every transition happens on the UI thread; recorder calls run in thread workers.
_LOADING = "loading"
_STARTING = "starting"
_RECORDING = "recording"
_STOPPING = "stopping"
_DISCARDING = "discarding"
_DONE = "done"


def _call_ui(app: App, callback, *args) -> None:
    """Run ``callback(*args)`` on the UI thread, unless the app has already shut down."""
    if app.is_running:
        app.call_from_thread(callback, *args)


class ConfirmDiscard(ModalScreen[bool]):
    """Ask before throwing a recording away. Dismisses with True to discard."""

    BINDINGS = [
        Binding("y", "dismiss(True)", "Discard"),
        Binding("n,escape", "dismiss(False)", "Keep recording"),
    ]
    AUTO_FOCUS = "#keep"

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm-box"):
            yield Label("Discard this recording?", id="confirm-title")
            yield Static("The audio will be deleted and cannot be recovered.")
            with Horizontal(id="confirm-buttons"):
                yield Button("Keep recording (n)", id="keep", variant="primary")
                yield Button("Discard (y)", id="discard", variant="error")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "discard")


class RecordingScreen(Screen):  # pylint: disable=too-many-instance-attributes
    """The screen a user sits on for the whole meeting."""

    BINDINGS = [
        Binding("enter,space,s", "stop", "Stop"),
        Binding("k", "keep", "Keep recording"),
        Binding("plus", "extend", "+15 min", key_display="+"),
        Binding("d", "discard", "Discard"),
    ]

    def __init__(self, name: str | None, persona_slug: str) -> None:
        # ``name`` is the meeting name; Screen's own ``name`` (a DOM name) is left unset.
        super().__init__()
        self.meeting_name = name
        self.persona_slug = persona_slug
        self.timer: SessionTimer | None = None
        self.recorder = None
        self.wav_path: str | None = None
        self.started_at: datetime | None = None
        self.save_error: str | None = None  # set when recorder.stop() raised
        self._phase = _LOADING
        self._stop_pending = False
        self._quitting = False
        self._saved = False
        self._warned = False
        self._ticker = None
        self._settled = asyncio.Event()  # set once the recorder can no longer change state
        self._mic_samples: deque[float] = deque(maxlen=METER_SAMPLES)
        self._system_samples: deque[float] = deque(maxlen=METER_SAMPLES)

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="rec-panel"):
            yield Static("Starting…", id="rec-indicator")
            yield Digits("00:00", id="elapsed")
            with Horizontal(classes="meter"):
                yield Label("Mic", classes="meter-label")
                yield Sparkline([], summary_function=max, id="mic-level")
            with Horizontal(classes="meter"):
                yield Label("System", classes="meter-label")
                yield Sparkline([], summary_function=max, id="system-level")
                yield Static("System audio unavailable", id="system-unavailable")
            yield Static("", id="timeout-label")
            yield ProgressBar(show_percentage=False, show_eta=False, id="timeout-bar")
        yield Footer()

    def on_mount(self) -> None:
        if self._phase != _LOADING:  # quit arrived before the screen mounted
            return
        self.query_one("#rec-panel").border_title = self.meeting_name or "Recording"
        deps = self.app.deps
        try:
            persona = deps.load_persona(self.persona_slug)
        except (PersonaNotFoundError, PersonaInvalidError) as err:
            self._give_up(f"Persona error: {err}")
            return
        self.timer = SessionTimer(self.app.timeout_seconds_for(persona), clock=deps.clock)
        self._update_timeout()
        if deps.live_updates and self.app.animation_level != "none":
            # Not chained via on_complete: the short idle gap lets the animator settle,
            # which Pilot-driven key presses wait for.
            self.set_interval(PULSE_SECONDS + 0.1, self._pulse)
        self._phase = _STARTING
        self._start_recorder(self.app, deps.make_recorder)

    # ── refresh ──────────────────────────────────────────────────────────────

    def tick(self) -> None:
        """One UI refresh from the timer and the recorder; a no-op unless recording."""
        if self._phase != _RECORDING:
            return
        self.query_one("#elapsed", Digits).update(format_clock(self.timer.elapsed))
        self._sample_levels()
        self._update_timeout()
        if self.timer.expired:
            self.action_stop()

    def _sample_levels(self) -> None:
        recorder = self.recorder
        self._mic_samples.append(float(recorder.mic_level))
        self.query_one("#mic-level", Sparkline).data = list(self._mic_samples)
        system = self.query_one("#system-level", Sparkline)
        loopback = bool(recorder.has_loopback)
        system.display = loopback
        self.query_one("#system-unavailable").display = not loopback
        if loopback:
            self._system_samples.append(float(recorder.system_level))
            system.data = list(self._system_samples)

    def _update_timeout(self) -> None:
        timer = self.timer
        remaining = timer.remaining
        label = self.query_one("#timeout-label", Static)
        progress = self.query_one("#timeout-bar", ProgressBar)
        if remaining is None:
            label.update("No time limit")
            progress.display = False
        else:
            label.update(f"Auto-stop in {format_clock(remaining)}")
            progress.display = True
            progress.update(total=timer.limit_seconds, progress=timer.elapsed)
        warning = timer.in_warning
        if warning != self.has_class("-warning"):
            self.set_class(warning, "-warning")
            # Textual does not repaint children when an ancestor's background transitions,
            # so repaint the panel once the 400 ms CSS transition has finished.
            self.set_timer(TRANSITION_SECONDS, self._repaint_panel)
        if warning and not self._warned and not timer.expired:
            self.app.notify(WARNING_TEXT, severity="warning")
            self.app.bell()
        self._warned = warning

    def _repaint_panel(self) -> None:
        for widget in self.query_one("#rec-panel").query("*"):
            widget.notify_style_update()  # drop styles cached with the old background
            widget.refresh()

    def _pulse(self) -> None:
        indicator = self.query_one("#rec-indicator")
        target = 1.0 if indicator.styles.opacity < 0.6 else 0.25
        indicator.styles.animate("opacity", target, duration=PULSE_SECONDS)

    def _set_status(self, text: str) -> None:
        self.query_one("#rec-indicator", Static).update(text)

    # ── key actions ──────────────────────────────────────────────────────────

    def action_stop(self) -> None:
        """Stop and hand the recording to processing. Safe to call any number of times."""
        if self._phase == _STARTING:
            self._stop_pending = True
            self._set_status("Stopping…")
        elif self._phase == _RECORDING:
            self._begin_stop()

    def action_keep(self) -> None:
        """Cancel the timeout: record until the user stops."""
        if self._phase not in (_STARTING, _RECORDING):
            return
        self.timer.cancel()
        self.app.notify("Timeout cancelled. Recording until you stop.")
        self._update_timeout()

    def action_extend(self) -> None:
        """Push the timeout back by 15 minutes."""
        if self._phase not in (_STARTING, _RECORDING):
            return
        if self.timer.limit_seconds is None:
            self.app.notify("No time limit is set.")
            return
        self.timer.extend()
        self.app.notify("Extended by 15 minutes.")
        self._update_timeout()

    def action_discard(self) -> None:
        """Ask, then throw the recording away."""
        if self._phase == _RECORDING:
            self.app.push_screen(ConfirmDiscard(), self._on_discard_answer)

    def _on_discard_answer(self, confirmed: bool | None) -> None:
        if not confirmed or self._phase != _RECORDING:
            return
        self._phase = _DISCARDING
        self._stop_ticker()
        self._set_status("Discarding…")
        self._discard_recorder(self.app, self.recorder)

    async def stop_for_quit(self) -> str | None:
        """Stop without processing (the app is quitting); return the saved WAV path, if any.

        When ``stop()`` raised, this returns None and ``save_error`` says what happened.
        """
        self._quitting = True
        if self._phase == _LOADING:
            self._phase = _DONE
            self._settled.set()
        self.action_stop()
        await self._settled.wait()
        return self.wav_path if self._saved else None

    # ── recorder lifecycle (thread workers report back via _call_ui) ─────────

    @work(thread=True)
    def _start_recorder(self, app: App, make_recorder) -> None:
        try:
            recorder = make_recorder()
            wav_path = recorder.start(self.meeting_name)
        except Exception as err:
            _call_ui(app, self._on_start_failed, str(err) or type(err).__name__)
            return
        _call_ui(app, self._on_started, recorder, wav_path)

    def _on_started(self, recorder, wav_path: str) -> None:
        self.recorder = recorder
        self.wav_path = wav_path
        self.timer.start()
        self.started_at = datetime.now()
        self._phase = _RECORDING
        self._set_status("● REC")
        if self._stop_pending:
            self._begin_stop()
            return
        if self.app.deps.live_updates:
            self._ticker = self.set_interval(REFRESH_SECONDS, self.tick)
        self.tick()

    def _on_start_failed(self, message: str) -> None:
        self._give_up(message)

    def _begin_stop(self) -> None:
        self._phase = _STOPPING
        self._stop_ticker()
        self._set_status("Saving…")
        self._stop_recorder(self.app, self.recorder)

    @work(thread=True)
    def _stop_recorder(self, app: App, recorder) -> None:
        try:
            saved = bool(recorder.stop())
        except Exception as err:
            _call_ui(app, self._on_stop_failed, err)
            return
        _call_ui(app, self._on_stopped, saved)

    def _on_stopped(self, saved: bool) -> None:
        self._finish(saved=saved)
        if self._quitting:
            return
        if saved:
            self._close_dialogs()
            self.app.finish_recording(
                self.wav_path, self.meeting_name, self.persona_slug, self.started_at
            )
        else:
            self.app.notify("No audio was captured.", severity="error")
            self._leave()

    def _on_stop_failed(self, err: Exception) -> None:
        self.save_error = (
            f"Could not save the recording: {err}. "
            f"Any audio that was written is at {self.wav_path}"
        )
        self._finish(saved=False)
        self.app.notify(self.save_error, severity="error", markup=False)
        self._leave()

    @work(thread=True)
    def _discard_recorder(self, app: App, recorder) -> None:
        try:
            recorder.discard()
        except Exception as err:
            _call_ui(app, self._on_discard_failed, err)
            return
        _call_ui(app, self._on_discarded)

    def _on_discarded(self) -> None:
        self._finish(saved=False)
        self.app.notify("Recording discarded.")
        self._leave()

    def _on_discard_failed(self, err: Exception) -> None:
        self._finish(saved=False)
        self.app.notify(f"Could not discard the recording: {err}", severity="error",
                        markup=False)
        self._leave()

    # ── helpers ──────────────────────────────────────────────────────────────

    def _give_up(self, message: str) -> None:
        """The recorder never started: say why and go back to Home."""
        self._finish(saved=False)
        self.app.notify(message, severity="error", markup=False)
        self.call_after_refresh(self._leave)  # may run from on_mount: let mounting finish

    def _finish(self, *, saved: bool) -> None:
        self._phase = _DONE
        self._saved = saved
        self._stop_ticker()
        self._settled.set()

    def _stop_ticker(self) -> None:
        if self._ticker is not None:
            self._ticker.stop()
            self._ticker = None

    def _close_dialogs(self) -> None:
        """Pop any dialog (e.g. ConfirmDiscard) still open on top of this screen."""
        app = self.app
        while self in app.screen_stack and app.screen is not self:
            app.pop_screen()

    def _leave(self) -> None:
        """Return to the screen underneath (Home), unless the app is quitting."""
        if self._quitting or self not in self.app.screen_stack:
            return
        self._close_dialogs()
        self.app.pop_screen()
