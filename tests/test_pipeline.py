"""Tests for tinysteno.pipeline."""
# pylint: disable=unused-argument
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tinysteno.personas import Persona
from tinysteno.pipeline import PipelineError, PipelineResult, Stage, run_pipeline


def _persona() -> Persona:
    return Persona(
        slug="default",
        name="Default",
        description="desc",
        schema={"summary": {"type": "string", "description": "summary"}},
        system_prompt="You are a test assistant.",
        template="{{ title }}",
        template_path=Path("/fake/template.md"),
        tags=[],
    )


class RecordingReporter:
    """Reporter that records (event, stage) tuples."""

    def __init__(self):
        self.events = []

    def stage_started(self, stage):
        self.events.append(("started", stage))

    def stage_progress(self, stage, fraction):
        self.events.append(("progress", stage))

    def stage_finished(self, stage):
        self.events.append(("finished", stage))

    def stage_skipped(self, stage):
        self.events.append(("skipped", stage))


class FakeTranscriber:
    """Transcriber returning canned text after one progress callback."""

    def __init__(self, text):
        self.text = text

    def transcribe(self, path, diarize=False, on_progress=None):
        if on_progress:
            on_progress(0.5)
        return {
            "text": self.text,
            "diarised_text": "",
            "duration_seconds": 61.0,
            "detected_language": "en",
        }


def _orchestrator_cls():
    cls = MagicMock()
    cls.return_value.summarize.return_value = {"summary": "We met."}
    cls.return_value.generate_title.return_value = "Sync"
    cls.return_value.generate_tags.return_value = ["a"]
    return cls


def _exporter_cls():
    cls = MagicMock()
    cls.return_value.export.return_value = "/vault/meetings/Sync.md"
    return cls


def _run(  # pylint: disable=too-many-arguments
        tmp_path, *, name=None, text="hello there", config_overrides=None,
         reporter=None, orchestrator_cls=None, exporter_cls=None):
    config = {
        "api_key": "k", "base_url": "u", "model": "m", "auto_title": True,
        "auto_tags": True, "obsidian_vault": str(tmp_path),
        "output_folder": "meetings", "diarization": False,
    }
    config.update(config_overrides or {})
    return run_pipeline(
        "x.wav", name, config, _persona(), datetime(2024, 1, 1),
        transcriber=FakeTranscriber(text),
        orchestrator_cls=orchestrator_cls or _orchestrator_cls(),
        exporter_cls=exporter_cls or _exporter_cls(),
        reporter=reporter,
    )


def test_happy_path_reports_every_stage_in_order(tmp_path):
    reporter = RecordingReporter()
    result = _run(tmp_path, reporter=reporter)
    assert result == PipelineResult(
        wav_path="x.wav", note_path="/vault/meetings/Sync.md", title="Sync",
        duration_seconds=61.0, no_speech=False,
    )
    assert reporter.events == [
        ("started", Stage.TRANSCRIBE), ("progress", Stage.TRANSCRIBE),
        ("finished", Stage.TRANSCRIBE),
        ("started", Stage.SUMMARIZE), ("finished", Stage.SUMMARIZE),
        ("started", Stage.METADATA), ("finished", Stage.METADATA),
        ("started", Stage.EXPORT), ("finished", Stage.EXPORT),
    ]


def test_explicit_name_wins_over_generated_title(tmp_path):
    assert _run(tmp_path, name="Board call").title == "Board call"


def test_no_speech_skips_everything_after_transcription(tmp_path):
    reporter = RecordingReporter()
    exporter_cls = MagicMock()
    result = _run(tmp_path, text="   ", reporter=reporter, exporter_cls=exporter_cls)
    assert result.no_speech and result.note_path is None
    exporter_cls.assert_not_called()
    assert reporter.events[-3:] == [
        ("skipped", Stage.SUMMARIZE), ("skipped", Stage.METADATA), ("skipped", Stage.EXPORT),
    ]


def test_no_api_key_skips_llm_stages_and_titles_from_filename(tmp_path):
    reporter = RecordingReporter()
    result = _run(tmp_path, config_overrides={"api_key": ""}, reporter=reporter)
    assert result.title == "x"
    assert ("skipped", Stage.SUMMARIZE) in reporter.events
    assert ("skipped", Stage.METADATA) in reporter.events


def test_stage_failure_becomes_pipeline_error_with_stage(tmp_path):
    orchestrator_cls = MagicMock()
    orchestrator_cls.return_value.summarize.side_effect = ConnectionError("endpoint down")
    with pytest.raises(PipelineError) as info:
        _run(tmp_path, orchestrator_cls=orchestrator_cls)
    assert info.value.stage is Stage.SUMMARIZE
    assert isinstance(info.value.__cause__, ConnectionError)


def test_export_template_error_is_pipeline_error_at_export(tmp_path):
    exporter_cls = MagicMock()
    exporter_cls.return_value.export.side_effect = RuntimeError("Template render error")
    with pytest.raises(PipelineError) as info:
        _run(tmp_path, exporter_cls=exporter_cls)
    assert info.value.stage is Stage.EXPORT
