"""UI-free transcribe -> summarize -> export pipeline with per-stage reporting."""

import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Iterator, Optional, Protocol

from tinysteno.personas import Persona
from tinysteno.transcriber import Transcriber

logger = logging.getLogger(__name__)


class Stage(str, Enum):
    """Pipeline stages, in execution order."""

    TRANSCRIBE = "transcribe"
    SUMMARIZE = "summarize"
    METADATA = "metadata"
    EXPORT = "export"


class PipelineReporter(Protocol):
    """Receives stage lifecycle events from run_pipeline."""

    def stage_started(self, stage: Stage) -> None:
        """A stage began running."""

    def stage_progress(self, stage: Stage, fraction: float) -> None:
        """A running stage reports completion as a fraction in [0, 1]."""

    def stage_finished(self, stage: Stage) -> None:
        """A stage completed successfully."""

    def stage_skipped(self, stage: Stage) -> None:
        """A stage does not apply to this run."""


class NullReporter:
    """Reporter that ignores every event."""

    def stage_started(self, stage: Stage) -> None:
        """Ignore."""

    def stage_progress(self, stage: Stage, fraction: float) -> None:
        """Ignore."""

    def stage_finished(self, stage: Stage) -> None:
        """Ignore."""

    def stage_skipped(self, stage: Stage) -> None:
        """Ignore."""


class PipelineError(RuntimeError):
    """A pipeline stage failed; the original exception is the __cause__."""

    def __init__(self, stage: Stage, message: str = ""):
        super().__init__(message or f"{stage.value} failed")
        self.stage = stage


@dataclass(frozen=True)
class PipelineResult:
    """Outcome of a pipeline run."""

    wav_path: str
    note_path: Optional[str]
    title: Optional[str]
    duration_seconds: float
    no_speech: bool


def _format_duration(duration_seconds: float) -> str:
    """Format duration in seconds to HH:MM:SS, clamped at 99:59:59."""
    total = int(duration_seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h > 99:
        return "99:59:59"
    return f"{h:02d}:{m:02d}:{s:02d}"


def _extract_summary_text(persona: Persona, data: dict) -> Optional[str]:
    """Extract the best text from summarized data for title/tag generation."""
    first_string_value = next(
        (data.get(field, "") for field, defn in persona.schema.items()
         if defn["type"] == "string"),
        None,
    )
    if not first_string_value:
        all_items: list[str] = []
        for field, defn in persona.schema.items():
            if defn["type"] == "list":
                all_items.extend(data.get(field, []))
        return ". ".join(all_items) if all_items else None
    return first_string_value


@contextmanager
def _stage(reporter: PipelineReporter, stage: Stage) -> Iterator[None]:
    """Report a stage as started/finished; re-raise failures as PipelineError."""
    reporter.stage_started(stage)
    try:
        yield
    except Exception as exc:
        raise PipelineError(stage, str(exc)) from exc
    reporter.stage_finished(stage)


def run_pipeline(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    wav_path: str,
    name: Optional[str],
    config: dict,
    persona: Persona,
    timestamp: datetime,
    *,
    transcriber: Transcriber,
    orchestrator_cls: type,
    exporter_cls: type,
    reporter: Optional[PipelineReporter] = None,
) -> PipelineResult:
    """Transcribe, summarize, tag and export a recording, reporting each stage."""
    reporter = reporter or NullReporter()

    with _stage(reporter, Stage.TRANSCRIBE):
        result = transcriber.transcribe(
            wav_path,
            diarize=config.get("diarization", False),
            on_progress=lambda fraction: reporter.stage_progress(Stage.TRANSCRIBE, fraction),
        )
    duration_seconds = result.get("duration_seconds", 0.0)
    logger.debug("Detected language: %s", result.get("detected_language"))
    logger.debug("Duration: %.0fs", duration_seconds)

    transcript = result["diarised_text"] or result["text"]
    if not transcript.strip():
        for stage in (Stage.SUMMARIZE, Stage.METADATA, Stage.EXPORT):
            reporter.stage_skipped(stage)
        return PipelineResult(wav_path, None, None, duration_seconds, no_speech=True)

    data: dict = {}
    orchestrator = None
    if config.get("api_key"):
        with _stage(reporter, Stage.SUMMARIZE):
            orchestrator = orchestrator_cls(
                api_key=config["api_key"],
                base_url=config["base_url"],
                model=config["model"],
            )
            data = orchestrator.summarize(transcript, persona)
    else:
        reporter.stage_skipped(Stage.SUMMARIZE)

    # First string field for title/tag generation; list-only personas (e.g. 1on1)
    # fall back to joining all list values.
    summary_text = _extract_summary_text(persona, data)
    generated_title = None
    generated_tags: list = []
    wants_title = config.get("auto_title")
    wants_tags = config.get("auto_tags")
    if (wants_title or wants_tags) and orchestrator and summary_text:
        with _stage(reporter, Stage.METADATA):
            with ThreadPoolExecutor(max_workers=2) as executor:
                title_future = (
                    executor.submit(orchestrator.generate_title, summary_text)
                    if wants_title else None
                )
                tags_future = (
                    executor.submit(orchestrator.generate_tags, summary_text)
                    if wants_tags else None
                )
            if title_future is not None:
                generated_title = title_future.result()
            if tags_future is not None:
                generated_tags = tags_future.result()
    else:
        reporter.stage_skipped(Stage.METADATA)

    title = name or generated_title or Path(wav_path).stem

    with _stage(reporter, Stage.EXPORT):
        metadata = {
            "title": title,
            "date": timestamp.strftime("%Y-%m-%d %H:%M"),
            "duration": _format_duration(duration_seconds),
            "transcript": transcript,
            "detected_language": result.get("detected_language", ""),
            "generated_tags": generated_tags,
        }
        exporter = exporter_cls(
            vault_path=config["obsidian_vault"],
            output_folder=config.get("output_folder", "meetings"),
        )
        note_path = str(exporter.export(data, persona, metadata))

    return PipelineResult(wav_path, note_path, title, duration_seconds, no_speech=False)
