"""Tests for tinysteno.pipeline."""
# pylint: disable=unused-argument
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tinysteno.personas import Persona
from tinysteno.pipeline import (
    PipelineError, PipelineResult, Stage, _extract_summary_text, run_pipeline)
from tinysteno.transcript import Segment


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

    def __init__(self, text, warnings=None, segments=None):
        self.text = text
        self.warnings = warnings
        self.segments = segments

    def transcribe(self, path, diarize=False, on_progress=None):
        if on_progress:
            on_progress(0.5)
        return {
            "text": self.text,
            "diarised_text": "",
            "duration_seconds": 61.0,
            "detected_language": "en",
            **({"warnings": self.warnings} if self.warnings is not None else {}),
            **({"segments": self.segments} if self.segments is not None else {}),
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
         reporter=None, orchestrator_cls=None, exporter_cls=None, warnings=None,
         segments=None):
    config = {
        "api_key": "k", "base_url": "u", "model": "m", "auto_title": True,
        "auto_tags": True, "obsidian_vault": str(tmp_path),
        "output_folder": "meetings", "diarization": False,
    }
    config.update(config_overrides or {})
    return run_pipeline(
        "x.wav", name, config, _persona(), datetime(2024, 1, 1),
        transcriber=FakeTranscriber(text, warnings, segments),
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


TRANSCRIBER_WARNING = "System audio was present but no speech was recognised."
ORCHESTRATOR_WARNING = (
    "1 of 3 transcript sections could not be summarized; the summary is incomplete."
)


def test_warnings_from_transcriber_and_orchestrator_reach_result_and_note(tmp_path):
    orchestrator_cls = _orchestrator_cls()
    orchestrator_cls.return_value.warnings = [ORCHESTRATOR_WARNING]
    exporter_cls = _exporter_cls()
    result = _run(tmp_path, warnings=[TRANSCRIBER_WARNING], orchestrator_cls=orchestrator_cls,
                  exporter_cls=exporter_cls)
    assert result.warnings == (TRANSCRIBER_WARNING, ORCHESTRATOR_WARNING)
    metadata = exporter_cls.return_value.export.call_args.args[2]
    assert metadata["processing_warnings"] == [TRANSCRIBER_WARNING, ORCHESTRATOR_WARNING]


def test_duplicate_warnings_are_kept_once(tmp_path):
    orchestrator_cls = _orchestrator_cls()
    orchestrator_cls.return_value.warnings = [TRANSCRIBER_WARNING, ORCHESTRATOR_WARNING]
    result = _run(tmp_path, warnings=[TRANSCRIBER_WARNING], orchestrator_cls=orchestrator_cls)
    assert result.warnings == (TRANSCRIBER_WARNING, ORCHESTRATOR_WARNING)


def test_no_warnings_gives_empty_tuple_and_empty_metadata_list(tmp_path):
    exporter_cls = _exporter_cls()
    result = _run(tmp_path, exporter_cls=exporter_cls)  # mock .warnings is not a list
    assert not result.warnings
    assert exporter_cls.return_value.export.call_args.args[2]["processing_warnings"] == []


def test_no_speech_result_carries_transcriber_warnings(tmp_path):
    result = _run(tmp_path, text="  ", warnings=[TRANSCRIBER_WARNING])
    assert result.no_speech and result.warnings == (TRANSCRIBER_WARNING,)


def test_summary_text_is_the_longest_string_field():
    persona = _persona()
    persona.schema = {"ceremony_type": {"type": "string", "description": "x"},
                      "overview": {"type": "string", "description": "x"}}
    data = {"ceremony_type": "Review", "overview": "We reviewed the sprint."}
    assert _extract_summary_text(persona, data) == "We reviewed the sprint."


def test_summary_text_ties_go_to_the_earlier_field():
    persona = _persona()
    persona.schema = {"a": {"type": "string", "description": "x"},
                      "b": {"type": "string", "description": "x"}}
    assert _extract_summary_text(persona, {"a": " one ", "b": "two"}) == "one"


def _segment_run(tmp_path, segments, config_overrides=None):
    orchestrator_cls, exporter_cls = _orchestrator_cls(), _exporter_cls()
    result = _run(
        tmp_path, segments=segments, config_overrides=config_overrides,
        orchestrator_cls=orchestrator_cls, exporter_cls=exporter_cls)
    summarize = orchestrator_cls.return_value.summarize
    export = exporter_cls.return_value.export
    return result, summarize.call_args.args[0], export.call_args.args[2]["transcript"]


def test_pipeline_renders_transcript_from_segments(tmp_path):
    segs = [Segment(3.0, 4.0, "hello there", "You")]
    _, summarized, exported = _segment_run(tmp_path, segs, {"diarization": True})
    assert summarized == "[00:00:03] [You] hello there"
    assert exported == "[00:00:03] [You] hello there"


def test_pipeline_omits_labels_when_diarization_is_off(tmp_path):
    segs = [Segment(3.0, 4.0, "hello there", "You")]
    _, summarized, exported = _segment_run(tmp_path, segs)
    assert summarized == exported == "[00:00:03] hello there"


def test_pipeline_can_turn_timestamps_off(tmp_path):
    segs = [Segment(3.0, 4.0, "hello", "You"), Segment(5.0, 6.0, "again", "Others")]
    _, summarized, _ = _segment_run(
        tmp_path, segs, {"diarization": True, "transcript_timestamps": False})
    assert summarized == "[You] hello\n[Others] again"


def test_pipeline_empty_segments_is_no_speech(tmp_path):
    orchestrator_cls = _orchestrator_cls()
    result = _run(tmp_path, text="stale", segments=[], orchestrator_cls=orchestrator_cls)
    assert result.no_speech is True
    orchestrator_cls.return_value.summarize.assert_not_called()


def test_pipeline_non_list_segments_use_the_legacy_text(tmp_path):
    orchestrator_cls = _orchestrator_cls()
    _run(tmp_path, text="legacy words", segments="oops", orchestrator_cls=orchestrator_cls)
    assert orchestrator_cls.return_value.summarize.call_args.args[0] == "legacy words"
