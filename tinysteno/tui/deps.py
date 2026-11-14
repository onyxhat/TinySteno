"""Injected dependencies for the TUI, so screens never import the CLI module."""

import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from tinysteno.personas import Persona
from tinysteno.pipeline import PipelineReporter, PipelineResult


@dataclass
class TuiDeps:
    """Everything the TUI needs from the outside world."""

    # AudioRecorder-shaped: start(name) -> str, stop() -> bool, discard(),
    # mic_level, system_level, has_loopback
    make_recorder: Callable[[], Any]
    run_pipeline: Callable[[str, str | None, Persona, datetime, PipelineReporter], PipelineResult]
    list_personas: Callable[[], list[str]]
    load_persona: Callable[[str], Persona]
    recordings_dir: Path
    clock: Callable[[], float] = time.monotonic
