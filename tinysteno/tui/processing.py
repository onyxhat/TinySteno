"""Processing screen: per-stage progress for transcribe -> summarize -> title/tags -> export."""

import asyncio
import threading
from enum import Enum
from datetime import datetime

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Footer, Header, ProgressBar, Static

from tinysteno.pipeline import PipelineError, PipelineResult, Stage

SPINNER_SECONDS = 0.1
SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
STATES = ("-pending", "-active", "-done", "-skipped", "-failed")
GLYPHS = {"-pending": "·", "-done": "✓", "-skipped": "–", "-failed": "✗"}
LABELS = {
    Stage.TRANSCRIBE: "Transcribing",
    Stage.SUMMARIZE: "Summarizing",
    Stage.METADATA: "Title & tags",
    Stage.EXPORT: "Exporting note",
}


def _call_ui(app: App, callback, *args) -> None:
    """Run ``callback(*args)`` on the UI thread; silently drop it once the app is shutting down."""
    if not app.is_running:
        return
    try:
        app.call_from_thread(callback, *args)
    except RuntimeError:  # the app stopped between the check and the call
        pass


class _UiReporter:
    """PipelineReporter that marshals every callback onto the UI thread."""

    def __init__(self, app: App, screen: "ProcessingScreen") -> None:
        self._app = app
        self._screen = screen

    def stage_started(self, stage: Stage) -> None:
        """A stage began running."""
        _call_ui(self._app, self._screen.set_state, stage, "-active")

    def stage_progress(self, stage: Stage, fraction: float) -> None:
        """A running stage reports completion as a fraction."""
        _call_ui(self._app, self._screen.set_progress, stage, fraction)

    def stage_finished(self, stage: Stage) -> None:
        """A stage completed."""
        _call_ui(self._app, self._screen.set_state, stage, "-done")

    def stage_skipped(self, stage: Stage) -> None:
        """A stage does not apply."""
        _call_ui(self._app, self._screen.set_state, stage, "-skipped")


class Run(Enum):
    """How the pipeline run on this screen stands.

    pending ─▶ running ─▶ done        (a note was exported; the Result screen takes over)
                  ├─────▶ no_speech   (nothing to export; the WAV is kept)
                  └─────▶ failed ──r──▶ running
    """

    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    NO_SPEECH = "no_speech"
    FAILED = "failed"


class ProcessingScreen(Screen):  # pylint: disable=too-many-instance-attributes
    """Runs the pipeline on a WAV in a background thread and shows where it is."""

    BINDINGS = [
        Binding("r", "retry", "Retry"),
        Binding("n", "home", "Home"),
    ]

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self, wav_path: str, name: str | None, persona_slug: str, timestamp: datetime,
        retranscribe: bool = False,
    ) -> None:
        # ``name`` is the meeting name; Screen's own ``name`` (a DOM name) is left unset.
        super().__init__()
        self.wav_path = wav_path
        self.meeting_name = name
        self.persona_slug = persona_slug
        self.timestamp = timestamp
        self.retranscribe = retranscribe  # applies to the first run only
        self.state = Run.PENDING
        self.pipeline_thread: threading.Thread | None = None
        self._frame = 0
        self._spinner = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="proc-panel"):
            for stage in Stage:
                yield Static("", id=f"stage-{stage.value}", classes="stage -pending")
                if stage is Stage.TRANSCRIBE:
                    yield ProgressBar(
                        total=1.0, show_eta=False, id="transcribe-bar", classes="stage-bar"
                    )
            yield Static("", id="error", markup=False)
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#proc-panel").border_title = self.meeting_name or "Processing"
        if self.app.deps.live_updates:
            self._spinner = self.set_interval(SPINNER_SECONDS, self._spin)
        self._reset()
        self._start()

    # ── row state ────────────────────────────────────────────────────────────

    def _row(self, stage: Stage) -> Static:
        return self.query_one(f"#stage-{stage.value}", Static)

    def _render_row(self, stage: Stage) -> None:
        row = self._row(stage)
        state = next(s for s in STATES if row.has_class(s))
        glyph = SPINNER_FRAMES[self._frame] if state == "-active" else GLYPHS[state]
        row.update(f"{glyph} {LABELS[stage]}")

    def set_state(self, stage: Stage, state: str) -> None:
        """Mark a stage's row with exactly one state class."""
        row = self._row(stage)
        row.remove_class(*STATES)
        row.add_class(state)
        if stage is Stage.TRANSCRIBE:
            meter = self.query_one("#transcribe-bar", ProgressBar)
            meter.display = state in ("-active", "-done")
            if state == "-done":
                meter.update(progress=1.0)
        self._render_row(stage)

    def set_progress(self, stage: Stage, fraction: float) -> None:
        """Drive the transcription bar (the only stage that reports a fraction)."""
        if stage is Stage.TRANSCRIBE:
            self.query_one("#transcribe-bar", ProgressBar).update(
                progress=max(0.0, min(1.0, fraction))
            )

    def _spin(self) -> None:
        self._frame = (self._frame + 1) % len(SPINNER_FRAMES)
        for stage in Stage:
            if self._row(stage).has_class("-active"):
                self._render_row(stage)

    def _reset(self) -> None:
        for stage in Stage:
            self.set_state(stage, "-pending")
        self.query_one("#transcribe-bar", ProgressBar).update(total=1.0, progress=0.0)
        error = self.query_one("#error", Static)
        error.update("")
        error.display = False

    # ── pipeline run ─────────────────────────────────────────────────────────

    @property
    def running(self) -> bool:
        """True while the pipeline thread is at work."""
        return self.state is Run.RUNNING

    def _start(self) -> None:
        self.state = Run.RUNNING
        self.app.ledger.begin_run(self.wav_path)
        self.refresh_bindings()
        loop = asyncio.get_running_loop()
        outcome: asyncio.Future = loop.create_future()
        # A daemon thread, not a Textual thread worker: the pipeline cannot be cancelled, and
        # neither app shutdown nor interpreter exit may wait for it.
        self.pipeline_thread = threading.Thread(
            target=self._pipeline_main,
            args=(self.app, self.app.deps, loop, outcome),
            name="tinysteno-pipeline",
            daemon=True,
        )
        self.pipeline_thread.start()
        self._await_outcome(outcome)

    def _pipeline_main(self, app: App, deps, loop, outcome: asyncio.Future) -> None:
        try:
            persona = deps.load_persona(self.persona_slug)
            # Only the first run may transcribe again; the keyword is passed only when set.
            extra = {"retranscribe": True} if self.retranscribe else {}
            self.retranscribe = False
            result = deps.run_pipeline(
                self.wav_path, self.meeting_name, persona, self.timestamp,
                _UiReporter(app, self), **extra,
            )
            report = (result, None)
        except Exception as err:
            report = (None, err)

        def deliver() -> None:
            if not outcome.done():
                outcome.set_result(report)

        try:
            loop.call_soon_threadsafe(deliver)
        except RuntimeError:  # the loop is closed: the app has already exited
            pass

    @work
    async def _await_outcome(self, outcome: asyncio.Future) -> None:
        """Wait (without blocking the loop) for the pipeline thread, then show its outcome."""
        result, err = await outcome
        if err is not None:
            self._on_failed(err)
        else:
            self._on_done(result)

    def _on_done(self, result: PipelineResult) -> None:
        self.app.ledger.end_run(self.wav_path)
        if result.no_speech or not result.note_path:
            self.state = Run.NO_SPEECH
            self.refresh_bindings()
            self._on_no_speech(result)
            return
        self.state = Run.DONE
        self.refresh_bindings()
        self.app.ledger.processed(self.wav_path)
        self.app.show_result(result)

    def _on_no_speech(self, result: PipelineResult) -> None:
        if self._row(Stage.TRANSCRIBE).has_class("-active"):
            self.set_state(Stage.TRANSCRIBE, "-done")
        for stage in list(Stage)[1:]:
            self.set_state(stage, "-skipped")
        lines = [f"No speech detected. Audio kept at {self.wav_path}", *result.warnings]
        self._show_error("\n".join(lines))
        if result.reused_transcript:
            self.app.notify("Used the saved transcript for this recording.", markup=False)

    def _on_failed(self, err: Exception) -> None:
        self.app.ledger.end_run(self.wav_path)
        self.state = Run.FAILED
        self.refresh_bindings()
        if isinstance(err, PipelineError):
            self.set_state(err.stage, "-failed")
            cause = err.__cause__ or err
            text = f"{LABELS[err.stage]} failed: {cause}"
        else:
            for stage in Stage:  # no stage to blame: stop any row still spinning
                if self._row(stage).has_class("-active"):
                    self.set_state(stage, "-pending")
            text = f"Processing failed: {err}"
        warnings = err.warnings if isinstance(err, PipelineError) else ()
        self._show_error("\n".join([
            text, *warnings, f"Audio kept at {self.wav_path}", "Press r to retry or n for home."]))

    def _show_error(self, text: str) -> None:
        error = self.query_one("#error", Static)
        error.update(text)
        error.display = True

    # ── actions ──────────────────────────────────────────────────────────────

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """Show (and allow) r only after a failure and n only once the pipeline has ended."""
        if action == "retry":
            return self.state is Run.FAILED
        if action == "home":
            return self.state in (Run.FAILED, Run.NO_SPEECH)
        return super().check_action(action, parameters)

    def action_retry(self) -> None:
        """Run the pipeline again from the top, after a failure."""
        if self.state is not Run.FAILED:
            return
        self._reset()
        self._start()

    def action_home(self) -> None:
        """Back to Home, once the pipeline is no longer running."""
        if self.state in (Run.FAILED, Run.NO_SPEECH):
            self.app.pop_screen()
