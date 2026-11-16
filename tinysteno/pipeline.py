"""UI-free transcribe -> summarize -> export pipeline with per-stage reporting."""

import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Iterator, Optional, Protocol

from tinysteno.orchestrator import PromptContext
from tinysteno.personas import Persona
from tinysteno.transcriber import Transcriber
from tinysteno.transcript import render_transcript

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
    warnings: tuple[str, ...] = ()


def _format_duration(duration_seconds: float) -> str:
    """Format duration in seconds to HH:MM:SS, clamped at 99:59:59."""
    total = int(duration_seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h > 99:
        return "99:59:59"
    return f"{h:02d}:{m:02d}:{s:02d}"


def _string_list(value: object) -> list[str]:
    """The strings in a list or tuple; anything else (a missing or mock attribute) is none."""
    if isinstance(value, (list, tuple)):
        return [item for item in value if isinstance(item, str)]
    return []


def _merge_warnings(*groups: list[str]) -> tuple[str, ...]:
    """Concatenate warning groups in order, keeping each distinct text once."""
    return tuple(dict.fromkeys(text for group in groups for text in group))


def _extract_summary_text(persona: Persona, data: dict) -> Optional[str]:
    """Extract the best text from summarized data for title/tag generation."""
    candidates = [
        value.strip()
        for field, defn in persona.schema.items()
        if defn["type"] == "string"
        for value in [data.get(field, "")]
        if isinstance(value, str)
    ]
    # Longest wins (a one-word field is a poor title source); max keeps the earliest on ties.
    first_string_value = max(candidates, key=len, default=None)
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


def run_pipeline(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals,too-many-statements
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
    notes: str = "",
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

    transcriber_warnings = _string_list(result.get("warnings"))
    segments = result.get("segments")
    diarization = config.get("diarization", False)
    timestamps = config.get("transcript_timestamps", True)
    if isinstance(segments, list):
        transcript = render_transcript(segments, labels=diarization, timestamps=timestamps)
        has_labels = bool(diarization) and any(seg.speaker for seg in segments)
        has_timestamps = bool(timestamps)
    else:
        transcript = result["diarised_text"] or result["text"]
        has_labels = bool(result["diarised_text"])
        has_timestamps = False
    if not transcript.strip():
        for stage in (Stage.SUMMARIZE, Stage.METADATA, Stage.EXPORT):
            reporter.stage_skipped(stage)
        return PipelineResult(
            wav_path, None, None, duration_seconds, no_speech=True,
            warnings=_merge_warnings(transcriber_warnings),
        )

    data: dict = {}
    orchestrator = None
    if config.get("api_key"):
        with _stage(reporter, Stage.SUMMARIZE):
            extra = {}
            if "llm_context_tokens" in config:
                extra["context_tokens"] = config["llm_context_tokens"]
            orchestrator = orchestrator_cls(
                api_key=config["api_key"],
                base_url=config["base_url"],
                model=config["model"],
                **extra,
            )
            data = orchestrator.summarize(
                transcript,
                persona,
                PromptContext(
                    meeting_date=timestamp,
                    has_speaker_labels=has_labels,
                    has_timestamps=has_timestamps,
                    notes=notes,
                ),
            )
    else:
        reporter.stage_skipped(Stage.SUMMARIZE)

    # Longest string field for title/tag generation; list-only personas (e.g. 1on1)
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
    warnings = _merge_warnings(
        transcriber_warnings, _string_list(getattr(orchestrator, "warnings", [])))

    with _stage(reporter, Stage.EXPORT):
        metadata = {
            "title": title,
            "date": timestamp.strftime("%Y-%m-%d %H:%M"),
            "duration": _format_duration(duration_seconds),
            "transcript": transcript,
            "detected_language": result.get("detected_language", ""),
            "generated_tags": generated_tags,
            "processing_warnings": list(warnings),
        }
        exporter = exporter_cls(
            vault_path=config["obsidian_vault"],
            output_folder=config.get("output_folder", "meetings"),
        )
        note_path = str(exporter.export(data, persona, metadata))

    return PipelineResult(
        wav_path, note_path, title, duration_seconds, no_speech=False, warnings=warnings)
