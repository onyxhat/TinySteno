#!/usr/bin/env python3
"""Regenerate the TUI screenshots used in README.md.

Drives the real TinySteno Textual app headlessly with stand-in dependencies (no audio
hardware, no transcription, no LLM) and saves one SVG per screen. It never reads or writes
the real ~/.tinysteno or a real recordings folder: every stand-in file lives in a temporary
directory, and the settings, personas and recordings the screens show are made up here.

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
from datetime import datetime
from pathlib import Path

from tinysteno.doctor import Check
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
# What the Personas screen lists: the built-ins plus one persona of the user's own.
LISTED_PERSONAS = [*PERSONAS, "weekly-review"]
PERSONA_NAMES = {
    "default": "Default meeting", "1on1": "One-on-one", "rca": "Root cause analysis",
    "irm": "Incident review", "sprint": "Sprint planning", "kickoff": "Project kickoff",
    "executive-summary": "Executive summary", "weekly-review": "Weekly review",
}
PERSONA_TIMEOUTS = {"1on1": 45, "executive-summary": 30, "kickoff": 0}  # others: the global limit
# The config file's own contents as the Settings screen shows them. The key is a placeholder
# (the field is masked anyway) and the paths are made up.
STORED_SETTINGS = {
    "obsidian_vault": "~/Obsidian/Vault", "output_folder": "meetings", "persona": "default",
    "base_url": "http://localhost:11434/v1", "model": "llama3.2:3b",
    "api_key": "sk-placeholder-not-a-real-key", "transcription_backend": "local",
    "whisper_model": "small", "session_timeout_minutes": 75, "tui_theme": "tokyo-night",
}
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
    ("Sprint-planning-20261109-1000.wav", 5),
    ("Incident-review-20261106-1630.wav", 8),
]
SMALL_SIZE = (80, 24)  # a standard terminal: Home must fit without scrolling
SETTINGS_SIZE = (84, 40)  # tall enough to show the first sections of the form
# Recordings whose transcript was saved, and how big each audio file looks (bytes).
TRANSCRIBED = {"Design-review-20261113-1502.wav", "Meeting-20261112-0930.wav",
               "Incident-review-20261106-1630.wav"}
SIZES = [31_400_000, 18_900_000, 52_300_000, 44_100_000, 27_800_000]
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
        slug=slug, name=PERSONA_NAMES.get(slug, slug), description="", schema={},
        system_prompt="", template="", template_path=Path("template.md"), tags=[],
        session_timeout_minutes=PERSONA_TIMEOUTS.get(slug),
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
        list_personas=lambda: LISTED_PERSONAS,
        load_persona=_persona,
        recordings_dir=recordings,
        clock=clock,
        live_updates=False,  # the script advances the clock and ticks by hand
        load_settings=lambda: dict(STORED_SETTINGS),
        save_settings=lambda changes: None,  # nothing is ever written
        run_checks=lambda: [Check("System check", True)],
        seed_personas=lambda: {"copied": [], "skipped": []},
        delete_recording=lambda wav: [],  # nothing is ever deleted
    )


def _seed_recordings(recordings: Path) -> None:
    recordings.mkdir(parents=True, exist_ok=True)
    for (filename, _days_ago), size in zip(RECENT, SIZES):
        path = recordings / filename
        path.write_bytes(b"")
        os.truncate(path, size)  # a sparse file: it only has to look this big
        if filename in TRANSCRIBED:
            Path(f"{path}.transcript.json").write_text("{}", encoding="utf-8")
        # The date in the name (YYYYMMDD-HHMM), so the screenshot does not depend on the clock
        # or time zone of the machine that made it.
        stamp = datetime.strptime(filename.rsplit("-", 2)[-2] + filename[-8:-4], "%Y%m%d%H%M")
        os.utime(path, (stamp.timestamp(), stamp.timestamp()))


def _voice(tick: int, speaking: bool) -> float:
    """A speech-like RMS level for one tick: syllables between about -36 and -8 dBFS while
    speaking, else the input's noise floor (below the meter's -60 dBFS)."""
    if not speaking:
        return 0.0008
    return 0.015 + 0.38 * abs(math.sin(tick * 0.9)) * (0.6 + 0.4 * math.sin(tick * 0.23))


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
        # run_test() disables toasts unless asked; this screenshot is about the warning one.
        async with app.run_test(size=SIZE, notifications=True) as pilot:
            screen = await _start_recording(pilot, app)
            _fill_meters(screen, recorder, clock)
            clock.now = clock.started + 75 * 60 - 45
            screen.tick()
            await pilot.pause(0.3)
            app.save_screenshot("tui-recording-warning.svg", path=str(out))


async def _capture_small_home(out: Path) -> None:
    """Home on a standard 80x24 terminal with five recent recordings."""
    with _demo_app() as (app, _clock, _recorder, _gate):
        async with app.run_test(size=SMALL_SIZE) as pilot:
            await pilot.pause()
            app.save_screenshot("tui-home-80x24.svg", path=str(out))


async def _open_destination(pilot, app: TinyStenoApp, dest: str) -> None:
    """Open a Menu destination and let it finish reading its stand-in data."""
    app.open_destination(dest)
    await pilot.pause()
    await app.workers.wait_for_complete()
    await pilot.pause()


async def _capture_management(out: Path) -> None:
    """The Menu, then Settings, Personas and Recordings opened from it."""
    with _demo_app() as (app, _clock, _recorder, _gate):
        async with app.run_test(size=SIZE) as pilot:
            await pilot.pause()
            await pilot.press("m")
            await pilot.pause()
            app.save_screenshot("tui-menu.svg", path=str(out))
            await pilot.press("escape")
            await pilot.pause()

            await _open_destination(pilot, app, "personas")
            app.save_screenshot("tui-personas.svg", path=str(out))
            await pilot.press("escape")
            await pilot.pause()

            await _open_destination(pilot, app, "recordings")
            app.save_screenshot("tui-recordings.svg", path=str(out))
            await pilot.press("escape")
            await pilot.pause()

    with _demo_app() as (app, _clock, _recorder, _gate):
        async with app.run_test(size=SETTINGS_SIZE) as pilot:
            await pilot.pause()
            await _open_destination(pilot, app, "settings")
            app.save_screenshot("tui-settings.svg", path=str(out))


def main() -> None:
    """Write the screenshots into the output directory."""
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/images").resolve()
    out.mkdir(parents=True, exist_ok=True)
    asyncio.run(_capture_journey(out))
    asyncio.run(_capture_warning(out))
    asyncio.run(_capture_small_home(out))
    asyncio.run(_capture_management(out))
    for path in sorted(out.glob("tui-*.svg")):
        print(path)


if __name__ == "__main__":
    main()
