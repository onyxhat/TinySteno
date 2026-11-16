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
from tinysteno.transcript import load_transcript, render_transcript, save_transcript

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
    reused_transcript: bool = False


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


def summary_digest(persona: Persona, data: dict, limit: int = 4000) -> Optional[str]:
    """Every non-empty string field, then every list item, one per line, cut to `limit`.

    Fields and items come in schema order; None when there is nothing to digest.
    """
    lines: list[str] = []
    for kind in ("string", "list"):
        for field, defn in persona.schema.items():
            if defn["type"] != kind:
                continue
            value = data.get(field)
            for item in [value] if kind == "string" else value or []:
                if isinstance(item, str) and item.strip():
                    lines.append(item.strip())
    return "\n".join(lines)[:limit] or None


@contextmanager
def _stage(reporter: PipelineReporter, stage: Stage) -> Iterator[None]:
    """Report a stage as started/finished; re-raise failures as PipelineError."""
    reporter.stage_started(stage)
    try:
        yield
    except Exception as exc:
        raise PipelineError(stage, str(exc)) from exc
    reporter.stage_finished(stage)


def transcript_fingerprint(wav_path: str, config: dict) -> dict:
    """What a saved transcript depends on: the recording's size and the transcription settings."""
    backend = config.get("transcription_backend", "local")
    prompt = config.get("whisper_prompt", "")
    split = bool(config.get("speaker_split", False)) and bool(config.get("diarization", False))
    fingerprint = {
        "size": Path(wav_path).stat().st_size,
        "backend": backend,
        "model": (config.get("whisper_api_model", "whisper-1") if backend == "api"
                  else config.get("whisper_model", "small")),
        "prompt": prompt if isinstance(prompt, str) else "",
        "vad": bool(config.get("whisper_vad", True)),
        "hallucination_filter": bool(config.get("hallucination_filter", True)),
        "bleed_filter": bool(config.get("bleed_filter", True)),
        "speaker_split": split,
        "max_speakers": config.get("max_speakers") if split else None,
    }
    if split and backend == "api" and config.get("whisper_api_diarize"):
        # Present only when on, so existing saved fingerprints stay valid.
        fingerprint["whisper_api_diarize"] = True
    return fingerprint


def _transcript_result(
    wav_path: str, config: dict, transcriber: Transcriber, reporter: PipelineReporter,
    retranscribe: bool,
) -> tuple[dict, bool]:
    """The saved transcript when one is usable, else a fresh transcription (which is saved).

    The flag says whether a saved transcript was used.
    """
    try:
        fingerprint = transcript_fingerprint(wav_path, config)
    except OSError:
        fingerprint = None  # no recording on disk: nothing to reuse, nothing to save
    if fingerprint is not None and not retranscribe:
        result = load_transcript(wav_path, fingerprint)
        if result is not None:
            logger.info("Reusing the saved transcript for %s", wav_path)
            reporter.stage_progress(Stage.TRANSCRIBE, 1.0)
            return result, True
    result = transcriber.transcribe(
        wav_path,
        diarize=config.get("diarization", False),
        on_progress=lambda fraction: reporter.stage_progress(Stage.TRANSCRIBE, fraction),
    )
    if fingerprint is not None:
        save_transcript(wav_path, result, fingerprint)
    return result, False


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
    retranscribe: bool = False,
) -> PipelineResult:
    """Transcribe, summarize, tag and export a recording, reporting each stage.

    A transcript saved beside the recording is reused unless `retranscribe` is set.
    """
    reporter = reporter or NullReporter()

    with _stage(reporter, Stage.TRANSCRIBE):
        result, reused = _transcript_result(wav_path, config, transcriber, reporter, retranscribe)
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
        has_numbered = has_labels and any(seg.speaker.startswith("Speaker ") for seg in segments)
    else:
        transcript = result["diarised_text"] or result["text"]
        has_labels = bool(result["diarised_text"])
        has_timestamps = False
        has_numbered = False
    if not transcript.strip():
        for stage in (Stage.SUMMARIZE, Stage.METADATA, Stage.EXPORT):
            reporter.stage_skipped(stage)
        return PipelineResult(
            wav_path, None, None, duration_seconds, no_speech=True,
            warnings=_merge_warnings(transcriber_warnings), reused_transcript=reused,
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
                    has_numbered_speakers=has_numbered,
                    has_timestamps=has_timestamps,
                    notes=notes,
                ),
            )
    else:
        reporter.stage_skipped(Stage.SUMMARIZE)

    summary_text = summary_digest(persona, data)
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
        wav_path, note_path, title, duration_seconds, no_speech=False, warnings=warnings,
        reused_transcript=reused)
