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


def make_deps(tmp_path: Path, **overrides) -> TuiDeps:
    """Build TuiDeps backed by fakes; any TuiDeps field can be overridden."""
    persona_timeouts = overrides.pop("persona_timeouts", {})
    settings = overrides.pop("settings", {})
    save_error = overrides.pop("save_error", None)
    checks = overrides.pop("checks", [])
    seed_result = overrides.pop("seed_result", {"copied": [], "skipped": []})
    seed_error = overrides.pop("seed_error", None)
    saved: list[dict] = []
    retranscribes: list[bool] = []
    note = tmp_path / "Sync.md"
    note.write_text(NOTE_TEXT)

    def fake_pipeline(  # pylint: disable=unused-argument,too-many-arguments,too-many-positional-arguments
        wav_path, name, persona, timestamp, reporter, *, retranscribe=False
    ):
        retranscribes.append(retranscribe)
        return PipelineResult(
            wav_path=wav_path, note_path=str(note), title="Sync",
            duration_seconds=61.0, no_speech=False,
        )

    def fake_save(changes):
        if save_error is not None:
            raise save_error
        saved.append(dict(changes))
        for key, value in changes.items():
            if value is None:
                settings.pop(key, None)
            else:
                settings[key] = value

    def fake_seed():
        if seed_error is not None:
            raise seed_error
        return seed_result

    fields = {
        "make_recorder": lambda: FakeRecorder(tmp_path),
        "run_pipeline": fake_pipeline,
        "list_personas": lambda: ["default", "1on1"],
        "load_persona": lambda slug: _persona(slug, persona_timeouts.get(slug)),
        "recordings_dir": tmp_path,
        "clock": FakeClock(),
        # Tests drive RecordingScreen.tick() by hand; a real-time interval would race them.
        "live_updates": False,
        "load_settings": lambda: dict(settings),
        "save_settings": fake_save,
        "run_checks": lambda: list(checks),
        "seed_personas": fake_seed,
        "delete_recording": lambda wav: delete_recording(tmp_path, wav),
        "first_run": False,
    }
    fields.update(overrides)
    deps = TuiDeps(**fields)
    deps.saved = saved  # type: ignore[attr-defined]
    deps.retranscribes = retranscribes  # type: ignore[attr-defined]
    return deps
