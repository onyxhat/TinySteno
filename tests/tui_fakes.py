"""Test doubles for TUI tests."""
import threading
from pathlib import Path

from tinysteno.personas import Persona
from tinysteno.pipeline import PipelineResult
from tinysteno.recordings import delete_recording
from tinysteno.tui.deps import TuiDeps

NOTE_TEXT = "---\ntitle: Sync\n---\n# Sync\n\nWe met.\n"


class FakeClock:
    """Manually advanced monotonic clock."""
    def __init__(self):
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class FakeRecorder:  # pylint: disable=too-many-instance-attributes
    """AudioRecorder-shaped fake; safe to call from worker threads.

    ``start_gate`` / ``stop_gate`` (threading.Event) make start() / stop() block until set,
    to simulate slow hardware. ``stop_result=False`` simulates "no audio captured";
    ``stop_error`` / ``discard_error`` are raised from stop() / discard(). ``raw_paths`` are
    the raw files a failed save keeps; ``loopback_error`` is why system audio is unavailable.
    """

    def __init__(self, tmp_path: Path, start_error: Exception | None = None):
        self.wav_path = str(tmp_path / "Meeting.wav")
        self.start_error = start_error
        self.stop_error: Exception | None = None
        self.discard_error: Exception | None = None
        self.stop_result = True
        self.start_gate: threading.Event | None = None
        self.stop_gate: threading.Event | None = None
        self.calls: list[str] = []
        self.mic_level = 0.3
        self.system_level = 0.6
        self.has_loopback = True
        self.raw_paths: tuple[Path, ...] = ()
        self.loopback_error: str | None = None

    def start(self, name=None) -> str:  # pylint: disable=unused-argument
        """Record the call and return the WAV path, or raise start_error."""
        self.calls.append("start")
        if self.start_gate is not None:
            self.start_gate.wait(5)
        if self.start_error is not None:
            raise self.start_error
        return self.wav_path

    def stop(self) -> bool:
        """Write an empty WAV at the path and report success (or stop_result/stop_error)."""
        self.calls.append("stop")
        if self.stop_gate is not None:
            self.stop_gate.wait(5)
        if self.stop_error is not None:
            raise self.stop_error
        if self.stop_result:
            Path(self.wav_path).write_bytes(b"")
        return self.stop_result

    def discard(self) -> None:
        """Record the call, or raise discard_error."""
        self.calls.append("discard")
        if self.discard_error is not None:
            raise self.discard_error


def _persona(slug: str, timeout: float | None) -> Persona:
    return Persona(
        slug=slug,
        name="Default",
        description="desc",
        schema={"summary": {"type": "string", "description": "summary"}},
        system_prompt="You are a test assistant.",
        template="{{ title }}",
        template_path=Path("/fake/template.md"),
        tags=[],
        session_timeout_minutes=timeout,
    )


class _Fakes:  # pylint: disable=too-many-instance-attributes
    """In-memory settings, personas-restore and pipeline fakes behind make_deps."""

    def __init__(self, tmp_path: Path, options: dict):
        self.settings = options.pop("settings", {})
        self.save_error = options.pop("save_error", None)  # any exception, e.g. SettingsNotApplied
        self.checks = options.pop("checks", [])
        self.seed_result = options.pop("seed_result", {"copied": [], "skipped": []})
        self.seed_error = options.pop("seed_error", None)
        self.persona_timeouts = options.pop("persona_timeouts", {})
        self.persona_errors = options.pop("persona_errors", {})  # slug -> exception to raise
        self.persona_names = options.pop("persona_names", {})  # slug -> any value for name
        self.personas = list(options.pop("personas", ["default", "1on1"]))
        self.seed_makes = options.pop("seed_makes", None)  # the list once personas are restored
        self.seed_gate = options.pop("seed_gate", None)    # threading.Event: seeding waits on it
        self.saved: list[dict] = []
        self.pipeline_kwargs: list[dict] = []
        self.note = tmp_path / "Sync.md"
        self.note.write_text(NOTE_TEXT)

    def pipeline(  # pylint: disable=unused-argument,too-many-arguments,too-many-positional-arguments
        self, wav_path, name, persona, timestamp, reporter, **kwargs
    ):
        """Record the keyword arguments of each run and return a finished result."""
        self.pipeline_kwargs.append(kwargs)
        return PipelineResult(
            wav_path=wav_path, note_path=str(self.note), title="Sync",
            duration_seconds=61.0, no_speech=False,
        )

    def save(self, changes):
        """Merge changes into the settings (None removes a key) or raise save_error."""
        if self.save_error is not None:
            raise self.save_error
        self.saved.append(dict(changes))
        for key, value in changes.items():
            if value is None:
                self.settings.pop(key, None)
            else:
                self.settings[key] = value

    def load(self, slug):
        """The fake persona, or raise the error set for this slug."""
        if slug in self.persona_errors:
            raise self.persona_errors[slug]
        persona = _persona(slug, self.persona_timeouts.get(slug))
        if slug in self.persona_names:
            persona.name = self.persona_names[slug]
        return persona

    def seed(self):
        """Return seed_result (and apply seed_makes), or raise seed_error."""
        if self.seed_gate is not None:
            self.seed_gate.wait(5)
        if self.seed_makes is not None and self.seed_error is None:
            self.personas[:] = self.seed_makes
        if self.seed_error is not None:
            raise self.seed_error
        return self.seed_result


def make_deps(tmp_path: Path, **overrides) -> TuiDeps:
    """Build TuiDeps backed by fakes; any TuiDeps field can be overridden.

    Extra keywords: settings, save_error, checks, seed_result, seed_error, persona_timeouts,
    persona_errors, persona_names, personas, seed_makes, seed_gate.
    The returned deps also carries ``saved`` (each save's changes), ``pipeline_kwargs`` (the
    keyword arguments of each pipeline call) and ``retranscribes`` (the value each run saw).
    """
    fakes = _Fakes(tmp_path, overrides)
    fields = {
        "make_recorder": lambda: FakeRecorder(tmp_path),
        "run_pipeline": fakes.pipeline,
        "list_personas": lambda: list(fakes.personas),
        "load_persona": fakes.load,
        "recordings_dir": tmp_path,
        "clock": FakeClock(),
        # Tests drive RecordingScreen.tick() by hand; a real-time interval would race them.
        "live_updates": False,
        "load_settings": lambda: dict(fakes.settings),
        "save_settings": fakes.save,
        "run_checks": lambda: list(fakes.checks),
        "seed_personas": fakes.seed,
        "delete_recording": lambda wav: delete_recording(tmp_path, wav),
        "first_run": False,
    }
    fields.update(overrides)
    deps = TuiDeps(**fields)
    deps.saved = fakes.saved  # type: ignore[attr-defined]
    deps.persona_errors = fakes.persona_errors  # type: ignore[attr-defined]
    deps.personas = fakes.personas  # type: ignore[attr-defined]  # mutate to change the list
    deps.pipeline_kwargs = fakes.pipeline_kwargs  # type: ignore[attr-defined]
    return deps
