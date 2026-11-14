"""Test doubles for TUI tests."""
from pathlib import Path

from tinysteno.personas import Persona
from tinysteno.pipeline import PipelineResult
from tinysteno.tui.deps import TuiDeps

NOTE_TEXT = "---\ntitle: Sync\n---\n# Sync\n\nWe met.\n"


class FakeClock:
    """Manually advanced monotonic clock."""
    def __init__(self):
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class FakeRecorder:
    """AudioRecorder-shaped fake; safe to call from worker threads."""

    def __init__(self, tmp_path: Path, start_error: Exception | None = None):
        self.wav_path = str(tmp_path / "Meeting.wav")
        self.start_error = start_error
        self.calls: list[str] = []
        self.mic_level = 0.3
        self.system_level = 0.6
        self.has_loopback = True

    def start(self, name=None) -> str:  # pylint: disable=unused-argument
        """Record the call and return the WAV path, or raise start_error."""
        self.calls.append("start")
        if self.start_error is not None:
            raise self.start_error
        return self.wav_path

    def stop(self) -> bool:
        """Write an empty WAV at the path and report success."""
        self.calls.append("stop")
        Path(self.wav_path).write_bytes(b"")
        return True

    def discard(self) -> None:
        """Record the call."""
        self.calls.append("discard")


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


def make_deps(tmp_path: Path, **overrides) -> TuiDeps:
    """Build TuiDeps backed by fakes; any TuiDeps field can be overridden."""
    persona_timeouts = overrides.pop("persona_timeouts", {})
    note = tmp_path / "Sync.md"
    note.write_text(NOTE_TEXT)

    def fake_pipeline(wav_path, name, persona, timestamp, reporter):  # pylint: disable=unused-argument
        return PipelineResult(
            wav_path=wav_path, note_path=str(note), title="Sync",
            duration_seconds=61.0, no_speech=False,
        )

    fields = {
        "make_recorder": lambda: FakeRecorder(tmp_path),
        "run_pipeline": fake_pipeline,
        "list_personas": lambda: ["default", "1on1"],
        "load_persona": lambda slug: _persona(slug, persona_timeouts.get(slug)),
        "recordings_dir": tmp_path,
        "clock": FakeClock(),
    }
    fields.update(overrides)
    return TuiDeps(**fields)
