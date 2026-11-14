#!/usr/bin/env python3
"""Regenerate the TUI screenshots used in README.md.

Drives the real TinySteno Textual app headlessly with stand-in dependencies (no audio
hardware, no transcription, no LLM) and saves one SVG per screen.

Usage:
    uv run python scripts/tui_screenshots.py [output_dir]    # default: docs/images
"""

import asyncio
import contextlib
import math
import os
import sys
import tempfile
import threading
from pathlib import Path

from tinysteno.personas import Persona
from tinysteno.pipeline import PipelineResult, Stage
from tinysteno.tui.app import TinyStenoApp
from tinysteno.tui.deps import TuiDeps
from tinysteno.tui.processing import ProcessingScreen
from tinysteno.tui.recording import RecordingScreen
from tinysteno.tui.result import ResultScreen

SIZE = (84, 26)
MEETING = "Weekly sync"
ELAPSED_SECONDS = 42 * 60 + 17
CONFIG = {"persona": "default", "session_timeout_minutes": 75}
PERSONAS = ["default", "1on1", "rca", "irm", "sprint", "kickoff", "executive-summary"]
VAULT = Path("Obsidian/Vault/meetings")  # relative: the script runs inside a temp directory
# (who is talking, for how many 0.1 s ticks): 120 ticks fill both meters' 12-second history.
TURNS = [
    ("system", 14), ("none", 4), ("mic", 22), ("system", 18), ("none", 3),
    ("mic", 16), ("system", 12), ("mic", 20), ("none", 3), ("mic", 8),
]
RECENT = [
    ("Design-review-20261113-1502.wav", 1),
    ("Meeting-20261112-0930.wav", 2),
    ("1on1-Priya-20261110-1400.wav", 4),
]
NOTE = """---
title: Weekly sync
date: 2026-11-14 15:02
duration: 00:42:17
tags: [meeting, launch]
---
# Weekly sync

## Overview

The team reviewed launch readiness for the mobile release. Crash rate is under target,
but the payments flow still fails intermittently on older Android versions, so the date
holds only if that fix lands by Wednesday.

## Key points

- Ship on the 21st if the payments fix is verified by Wednesday; otherwise slip a week.
- The onboarding redesign is dropped from this release.

## Action items

- Verify the payments fix on Android 11 and 12 by Wednesday (Priya)
- Draft the release notes without the onboarding section (Marcus)
"""


class DemoClock:
    """Manually advanced monotonic clock."""

    def __init__(self) -> None:
        self.started = 1000.0
        self.now = self.started

    def __call__(self) -> float:
        return self.now


class DemoRecorder:
    """AudioRecorder-shaped stand-in that records nothing."""

    def __init__(self, recordings_dir: Path) -> None:
        self.wav_path = str(recordings_dir / "Weekly-sync-20261114-1502.wav")
        self.mic_level = 0.0
        self.system_level = 0.0
        self.has_loopback = True

    def start(self, _name=None) -> str:
        """Pretend to start capturing."""
        return self.wav_path

    def stop(self) -> bool:
        """Pretend to save the recording."""
        Path(self.wav_path).write_bytes(b"")
        return True

    def discard(self) -> None:
        """Nothing to throw away."""


def _persona(slug: str) -> Persona:
    return Persona(
        slug=slug, name=slug, description="", schema={}, system_prompt="", template="",
        template_path=Path("template.md"), tags=[],
    )


def _make_deps(clock: DemoClock, recorder: DemoRecorder, gate: threading.Event) -> TuiDeps:
    recordings = VAULT / "audio"
    note = VAULT / "Weekly sync (2026-11-14).md"
    note.write_text(NOTE, encoding="utf-8")

    def pipeline(wav_path, _name, _persona_obj, _timestamp, reporter) -> PipelineResult:
        reporter.stage_started(Stage.TRANSCRIBE)
        reporter.stage_progress(Stage.TRANSCRIBE, 1.0)
        reporter.stage_finished(Stage.TRANSCRIBE)
        reporter.stage_started(Stage.SUMMARIZE)
        gate.wait(30)  # held here while the Processing screenshot is taken
        for stage in (Stage.SUMMARIZE, Stage.METADATA, Stage.EXPORT):
            if stage is not Stage.SUMMARIZE:
                reporter.stage_started(stage)
            reporter.stage_finished(stage)
        return PipelineResult(
            wav_path=wav_path, note_path=str(note), title=MEETING,
            duration_seconds=float(ELAPSED_SECONDS), no_speech=False,
        )

    return TuiDeps(
        make_recorder=lambda: recorder,
        run_pipeline=pipeline,
        list_personas=lambda: PERSONAS,
        load_persona=_persona,
        recordings_dir=recordings,
        clock=clock,
        live_updates=False,  # the script advances the clock and ticks by hand
    )


def _seed_recordings(recordings: Path) -> None:
    recordings.mkdir(parents=True, exist_ok=True)
    for filename, days_ago in RECENT:
        path = recordings / filename
        path.write_bytes(b"")
        stamp = 1_800_000_000 - days_ago * 86_400
        os.utime(path, (stamp, stamp))


def _voice(tick: int, speaking: bool) -> float:
    """A speech-like level for one tick: a syllable swell while speaking, else near silence."""
    if not speaking:
        return 0.02
    return 0.3 + 0.6 * abs(math.sin(tick * 0.9)) * (0.6 + 0.4 * math.sin(tick * 0.23))


def _fill_meters(screen: RecordingScreen, recorder: DemoRecorder, clock: DemoClock) -> None:
    """Play a short two-way conversation into both meters, 0.1 s per tick."""
    tick = 0
    for turn, length in TURNS:
        for _ in range(length):
            tick += 1
            recorder.mic_level = _voice(tick, turn == "mic")
            recorder.system_level = _voice(tick + 7, turn == "system")
            clock.now += 0.1
            screen.tick()


async def _wait_for(pilot, condition) -> None:
    for _ in range(200):
        await pilot.pause(0.02)
        if condition():
            return
    raise RuntimeError("the app did not reach the expected state")


async def _start_recording(pilot, app: TinyStenoApp) -> RecordingScreen:
    app.screen.query_one("#name").value = MEETING
    await pilot.press("r")
    await app.workers.wait_for_complete()
    await pilot.pause()
    screen = app.screen
    assert isinstance(screen, RecordingScreen)
    return screen


@contextlib.contextmanager
def _demo_app():
    """A TinyStenoApp on stand-in dependencies, run from a throwaway directory.

    Yields (app, clock, recorder, gate). Paths are relative to that directory so the
    screenshots show a vault-like path instead of a temp folder.
    """
    previous = Path.cwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        clock, gate = DemoClock(), threading.Event()
        try:
            _seed_recordings(VAULT / "audio")
            recorder = DemoRecorder(VAULT / "audio")
            yield TinyStenoApp(CONFIG, _make_deps(clock, recorder, gate)), clock, recorder, gate
        finally:
            gate.set()
            os.chdir(previous)


async def _capture_journey(out: Path) -> None:
    """Home → Recording → Processing → Result."""
    with _demo_app() as (app, clock, recorder, gate):
        async with app.run_test(size=SIZE) as pilot:
            await pilot.pause()
            app.save_screenshot("tui-home.svg", path=str(out))

            screen = await _start_recording(pilot, app)
            _fill_meters(screen, recorder, clock)
            clock.now = clock.started + ELAPSED_SECONDS
            screen.tick()
            await pilot.pause()
            app.save_screenshot("tui-recording.svg", path=str(out))

            await pilot.press("enter")
            await _wait_for(
                pilot,
                lambda: isinstance(app.screen, ProcessingScreen)
                and app.screen.query("#stage-summarize.-active"),
            )
            app.save_screenshot("tui-processing.svg", path=str(out))

            gate.set()
            await _wait_for(pilot, lambda: isinstance(app.screen, ResultScreen))
            await app.workers.wait_for_complete()
            await pilot.pause()
            app.save_screenshot("tui-result.svg", path=str(out))


async def _capture_warning(out: Path) -> None:
    """The Recording screen inside the last minute before auto-stop."""
    with _demo_app() as (app, clock, recorder, _gate):
        async with app.run_test(size=SIZE) as pilot:
            screen = await _start_recording(pilot, app)
            _fill_meters(screen, recorder, clock)
            clock.now = clock.started + 75 * 60 - 45
            screen.tick()
            await pilot.pause(0.3)
            app.save_screenshot("tui-recording-warning.svg", path=str(out))


def main() -> None:
    """Write the screenshots into the output directory."""
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/images").resolve()
    out.mkdir(parents=True, exist_ok=True)
    asyncio.run(_capture_journey(out))
    asyncio.run(_capture_warning(out))
    for path in sorted(out.glob("tui-*.svg")):
        print(path)


if __name__ == "__main__":
    main()
