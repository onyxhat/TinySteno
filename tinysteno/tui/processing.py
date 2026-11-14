"""Processing screen: per-stage progress for transcribe -> summarize -> title/tags -> export."""

import shlex
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


class ProcessingScreen(Screen):  # pylint: disable=too-many-instance-attributes
    """Runs the pipeline on a WAV in a background thread and shows where it is."""

    BINDINGS = [
        Binding("r", "retry", "Retry"),
        Binding("n", "home", "Home"),
    ]

    def __init__(
        self, wav_path: str, name: str | None, persona_slug: str, timestamp: datetime
    ) -> None:
        # ``name`` is the meeting name; Screen's own ``name`` (a DOM name) is left unset.
        super().__init__()
        self.wav_path = wav_path
        self.meeting_name = name
        self.persona_slug = persona_slug
        self.timestamp = timestamp
        self.running = False
        self._failed = False
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

    def _start(self) -> None:
        self.running = True
        self._failed = False
        self._run_pipeline(self.app, self.app.deps)

    @work(thread=True)
    def _run_pipeline(self, app: App, deps) -> None:
        try:
            persona = deps.load_persona(self.persona_slug)
            result = deps.run_pipeline(
                self.wav_path, self.meeting_name, persona, self.timestamp,
                _UiReporter(app, self),
            )
        except Exception as err:
            _call_ui(app, self._on_failed, err)
            return
        _call_ui(app, self._on_done, result)

    def _on_done(self, result: PipelineResult) -> None:
        self.running = False
        if result.no_speech or not result.note_path:
            self._on_no_speech()
            return
        self.app.show_result(result)

    def _on_no_speech(self) -> None:
        if self._row(Stage.TRANSCRIBE).has_class("-active"):
            self.set_state(Stage.TRANSCRIBE, "-done")
        for stage in list(Stage)[1:]:
            self.set_state(stage, "-skipped")
        self._show_error(f"No speech detected. Audio kept at {self.wav_path}")

    def _on_failed(self, err: Exception) -> None:
        self.running = False
        self._failed = True
        if isinstance(err, PipelineError):
            self.set_state(err.stage, "-failed")
            cause = err.__cause__ or err
            text = f"{LABELS[err.stage]} failed: {cause}"
        else:
            for stage in Stage:  # no stage to blame: stop any row still spinning
                if self._row(stage).has_class("-active"):
                    self.set_state(stage, "-pending")
            text = f"Processing failed: {err}"
        self._show_error(f"{text}\nAudio kept at {self.wav_path}\nPress r to retry or n for home.")

    def _show_error(self, text: str) -> None:
        error = self.query_one("#error", Static)
        error.update(text)
        error.display = True

    # ── actions ──────────────────────────────────────────────────────────────

    def action_retry(self) -> None:
        """Run the pipeline again from the top, after a failure."""
        if self.running or not self._failed:
            return
        self._reset()
        self._start()

    def action_home(self) -> None:
        """Back to Home, once the pipeline is no longer running."""
        if not self.running:
            self.app.pop_screen()

    def quit_message(self) -> str:
        """The exit message for quitting while the pipeline is still running."""
        return (
            f"Saved recording: {self.wav_path}\n"
            f"Process it with: tinysteno process {shlex.quote(self.wav_path)}"
        )
