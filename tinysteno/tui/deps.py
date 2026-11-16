"""Injected dependencies for the TUI, so screens never import the CLI module."""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from tinysteno.doctor import Check
from tinysteno.personas import Persona
from tinysteno.pipeline import PipelineResult


@dataclass
class TuiDeps:
    """Everything the TUI needs from the outside world."""

    # AudioRecorder-shaped: start(name) -> str, stop() -> bool, discard(),
    # mic_level, system_level, has_loopback
    make_recorder: Callable[[], Any]
    # run_pipeline(wav_path, name, persona, timestamp, reporter, *, retranscribe=False)
    run_pipeline: Callable[..., PipelineResult]
    list_personas: Callable[[], list[str]]
    load_persona: Callable[[str], Persona]
    recordings_dir: Path
    clock: Callable[[], float] = time.monotonic
    # When False, screens never refresh themselves on a real-time interval (no 0.1 s tick,
    # no pulse); tests drive RecordingScreen.tick() by hand against a fake clock.
    live_updates: bool = True
    # Stored contents of the config file (unexpanded), for the Settings screen.
    load_settings: Callable[[], dict] = dict
    # Validate and write the given changes (None removes a key) and apply them to the live
    # configuration; raises ValueError / OSError and changes nothing.
    save_settings: Callable[[dict], None] = lambda changes: None
    run_checks: Callable[[], list[Check]] = list
    # Restore the built-in personas; returns {"copied": [...], "skipped": [...]}.
    seed_personas: Callable[[], dict[str, list[str]]] = (
        lambda: {"copied": [], "skipped": []}
    )
    # Delete a recording (and its sidecar files); returns the paths removed.
    delete_recording: Callable[[Path], list[Path]] = lambda wav: []
    # True when no config file existed at launch.
    first_run: bool = False
