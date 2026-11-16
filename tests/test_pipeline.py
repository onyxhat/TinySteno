"""Tests for tinysteno.pipeline."""
# pylint: disable=unused-argument
import json
import os
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tinysteno.personas import Persona
from tinysteno.pipeline import (
    PipelineError, PipelineResult, Stage, run_pipeline, summary_digest, transcript_fingerprint)
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
         segments=None, wav_path="x.wav", transcriber=None, **kwargs):
    config = {
        "api_key": "k", "base_url": "u", "model": "m", "auto_title": True,
        "auto_tags": True, "obsidian_vault": str(tmp_path),
        "output_folder": "meetings", "diarization": False,
    }
    config.update(config_overrides or {})
    return run_pipeline(
        wav_path, name, config, _persona(), datetime(2024, 1, 1),
        transcriber=transcriber or FakeTranscriber(text, warnings, segments),
        orchestrator_cls=orchestrator_cls or _orchestrator_cls(),
        exporter_cls=exporter_cls or _exporter_cls(),
        reporter=reporter,
        **kwargs,
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


PERSONA_SCHEMA = {
    "ceremony_type": {"type": "string", "description": "x"},
    "overview": {"type": "string", "description": "x"},
    "action_items": {"type": "list", "description": "x"},
}


def test_digest_covers_strings_then_lists_and_is_bounded():
    persona = _persona()
    persona.schema = PERSONA_SCHEMA
    data = {"overview": "We met.", "ceremony_type": "Review", "action_items": ["Ship it (Priya)"]}
    assert summary_digest(persona, data) == "Review\nWe met.\nShip it (Priya)"
    assert len(summary_digest(persona, {"overview": "x" * 9000})) == 4000


def test_digest_skips_empty_values_and_is_none_when_nothing_is_left():
    persona = _persona()
    persona.schema = PERSONA_SCHEMA
    assert summary_digest(persona, {"overview": "  ", "action_items": ["", " a "]}) == "a"
    assert summary_digest(persona, {}) is None


class CountingTranscriber(FakeTranscriber):
    """Fake transcriber that counts calls and returns real segments."""

    def __init__(self):
        super().__init__("hi there", segments=[Segment(0.0, 1.0, "hi there", "You")])
        self.calls = 0

    def transcribe(self, path, diarize=False, on_progress=None):
        self.calls += 1
        return super().transcribe(path, diarize, on_progress)


def _real_wav(tmp_path):
    wav = tmp_path / "Meeting-1.wav"
    wav.write_bytes(b"RIFF")
    return str(wav)


def test_second_run_reuses_the_saved_transcript(tmp_path):
    transcriber, wav = CountingTranscriber(), _real_wav(tmp_path)
    _run(tmp_path, transcriber=transcriber, wav_path=wav)
    result = _run(tmp_path, transcriber=transcriber, wav_path=wav)
    assert transcriber.calls == 1
    assert result.duration_seconds == 61.0 and not result.no_speech


def test_retranscribe_ignores_the_saved_transcript(tmp_path):
    transcriber, wav = CountingTranscriber(), _real_wav(tmp_path)
    _run(tmp_path, transcriber=transcriber, wav_path=wav)
    _run(tmp_path, transcriber=transcriber, wav_path=wav, retranscribe=True)
    assert transcriber.calls == 2


def test_reused_transcript_still_reports_the_transcribe_stage(tmp_path):
    transcriber, wav = CountingTranscriber(), _real_wav(tmp_path)
    _run(tmp_path, transcriber=transcriber, wav_path=wav)
    reporter = RecordingReporter()
    _run(tmp_path, transcriber=transcriber, wav_path=wav, reporter=reporter)
    assert reporter.events[:3] == [
        ("started", Stage.TRANSCRIBE), ("progress", Stage.TRANSCRIBE),
        ("finished", Stage.TRANSCRIBE)]


def test_reused_transcript_keeps_its_warnings(tmp_path):
    transcriber, wav = CountingTranscriber(), _real_wav(tmp_path)
    transcriber.warnings = [TRANSCRIBER_WARNING]
    _run(tmp_path, transcriber=transcriber, wav_path=wav)
    assert _run(tmp_path, transcriber=transcriber, wav_path=wav).warnings == (
        TRANSCRIBER_WARNING,)


def test_a_silent_recording_is_not_transcribed_again(tmp_path):
    transcriber, wav = CountingTranscriber(), _real_wav(tmp_path)
    transcriber.segments = []
    transcriber.text = ""
    assert _run(tmp_path, transcriber=transcriber, wav_path=wav).no_speech
    assert _run(tmp_path, transcriber=transcriber, wav_path=wav).no_speech
    assert transcriber.calls == 1


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


def _context_for(tmp_path, segments, config_overrides=None, **kwargs):
    orchestrator_cls = _orchestrator_cls()
    _run(tmp_path, segments=segments, config_overrides=config_overrides,
         orchestrator_cls=orchestrator_cls, **kwargs)
    return orchestrator_cls.return_value.summarize.call_args.args[2]


def test_pipeline_passes_meeting_date_labels_timestamps_and_notes(tmp_path):
    segs = [Segment(3.0, 4.0, "hello there", "You")]
    ctx = _context_for(tmp_path, segs, {"diarization": True}, notes="  my notes \n")
    assert ctx.meeting_date == datetime(2024, 1, 1)
    assert ctx.has_speaker_labels is True
    assert ctx.has_timestamps is True
    assert ctx.notes == "  my notes \n"


def test_pipeline_context_has_no_labels_when_diarization_is_off_or_speakers_are_empty(tmp_path):
    segs = [Segment(3.0, 4.0, "hello there", "You")]
    assert _context_for(tmp_path, segs).has_speaker_labels is False
    bare = [Segment(3.0, 4.0, "hello there")]
    assert _context_for(tmp_path, bare, {"diarization": True}).has_speaker_labels is False


def test_pipeline_context_timestamps_follow_the_config_key(tmp_path):
    segs = [Segment(3.0, 4.0, "hello there", "You")]
    ctx = _context_for(tmp_path, segs, {"transcript_timestamps": False})
    assert ctx.has_timestamps is False


def test_pipeline_context_for_legacy_text_has_no_timestamps(tmp_path):
    ctx = _context_for(tmp_path, None)
    assert ctx.has_timestamps is False and ctx.has_speaker_labels is False


def test_context_window_is_passed_to_the_orchestrator_only_when_configured(tmp_path):
    orchestrator_cls = _orchestrator_cls()
    _run(tmp_path, orchestrator_cls=orchestrator_cls)
    assert "context_tokens" not in orchestrator_cls.call_args.kwargs
    orchestrator_cls = _orchestrator_cls()
    _run(tmp_path, orchestrator_cls=orchestrator_cls,
         config_overrides={"llm_context_tokens": 4096})
    assert orchestrator_cls.call_args.kwargs["context_tokens"] == 4096


def test_reused_flag_is_false_then_true(tmp_path):
    transcriber, wav = CountingTranscriber(), _real_wav(tmp_path)
    assert not _run(tmp_path, transcriber=transcriber, wav_path=wav).reused_transcript
    assert _run(tmp_path, transcriber=transcriber, wav_path=wav).reused_transcript


@pytest.mark.parametrize("override", [
    {"whisper_prompt": "Acme"}, {"hallucination_filter": False}, {"bleed_filter": False},
    {"whisper_vad": False}, {"whisper_model": "large"}, {"transcription_backend": "api"}])
def test_changed_settings_transcribe_again(tmp_path, override):
    transcriber, wav = CountingTranscriber(), _real_wav(tmp_path)
    _run(tmp_path, transcriber=transcriber, wav_path=wav)
    _run(tmp_path, transcriber=transcriber, wav_path=wav, config_overrides=override)
    assert transcriber.calls == 2


def test_a_different_recording_with_an_old_timestamp_transcribes_again(tmp_path):
    transcriber, wav = CountingTranscriber(), _real_wav(tmp_path)
    _run(tmp_path, transcriber=transcriber, wav_path=wav)
    Path(wav).write_bytes(b"RIFF-different-recording")
    os.utime(wav, (1, 1))
    _run(tmp_path, transcriber=transcriber, wav_path=wav)
    assert transcriber.calls == 2


def test_fingerprint_keys_follow_the_backend(tmp_path):
    wav = _real_wav(tmp_path)
    config = {"whisper_model": "small", "whisper_api_model": "w-1", "whisper_prompt": 5}
    local = transcript_fingerprint(wav, config)
    assert local == {
        "size": 4, "backend": "local", "model": "small", "prompt": "", "vad": True,
        "hallucination_filter": True, "bleed_filter": True, "speaker_split": False,
        "max_speakers": None}
    assert transcript_fingerprint(wav, {**config, "transcription_backend": "api"})["model"] == "w-1"


def test_fake_recording_path_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _run(tmp_path / "vault", transcriber=CountingTranscriber())
    assert not [p for p in tmp_path.iterdir() if p.name != "vault"]


def _numbered_context(tmp_path, speaker, config_overrides):
    segs = [Segment(3.0, 4.0, "hello there", speaker)]
    orchestrator_cls = _orchestrator_cls()
    _run(tmp_path, segments=segs, config_overrides=config_overrides,
         orchestrator_cls=orchestrator_cls)
    return orchestrator_cls.return_value.summarize.call_args.args[2]


def test_pipeline_flags_numbered_speakers_for_the_prompt(tmp_path):
    ctx = _numbered_context(tmp_path, "Speaker 2", {"diarization": True})
    assert ctx.has_speaker_labels and ctx.has_numbered_speakers


def test_pipeline_does_not_flag_numbered_speakers_without_them_or_labels(tmp_path):
    assert not _numbered_context(tmp_path, "Others", {"diarization": True}).has_numbered_speakers
    assert not _numbered_context(tmp_path, "Speaker 2", {}).has_numbered_speakers


def test_fingerprint_changes_with_the_split_settings(tmp_path):
    wav = _real_wav(tmp_path)
    base = transcript_fingerprint(wav, {})
    assert base["speaker_split"] is False and base["max_speakers"] is None
    assert transcript_fingerprint(wav, {"speaker_split": True, "diarization": True}) != base


def test_fingerprint_records_the_effective_split_only(tmp_path):
    wav = _real_wav(tmp_path)
    off = transcript_fingerprint(wav, {"speaker_split": True, "diarization": False})
    assert off["speaker_split"] is False and off["max_speakers"] is None
    assert off == transcript_fingerprint(wav, {})
    assert off == transcript_fingerprint(wav, {"max_speakers": 2})
    assert off == transcript_fingerprint(wav, {"speaker_split": True, "max_speakers": 2})
    on = transcript_fingerprint(
        wav, {"speaker_split": True, "diarization": True, "max_speakers": 2})
    assert on["speaker_split"] is True and on["max_speakers"] == 2
    assert on != transcript_fingerprint(wav, {"speaker_split": True, "diarization": True})


def test_enabling_labels_does_not_reuse_an_unsplit_transcript(tmp_path):
    transcriber, wav = CountingTranscriber(), _real_wav(tmp_path)
    _run(tmp_path, transcriber=transcriber, wav_path=wav,
         config_overrides={"speaker_split": True, "diarization": False})
    _run(tmp_path, transcriber=transcriber, wav_path=wav,
         config_overrides={"speaker_split": True, "diarization": True})
    assert transcriber.calls == 2


def test_speakers_hint_with_the_split_off_reuses_the_transcript(tmp_path):
    transcriber, wav = CountingTranscriber(), _real_wav(tmp_path)
    _run(tmp_path, transcriber=transcriber, wav_path=wav)
    _run(tmp_path, transcriber=transcriber, wav_path=wav, config_overrides={"max_speakers": 3})
    assert transcriber.calls == 1


def test_fingerprint_tracks_server_diarization_only_when_effective(tmp_path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"x")
    on = {"speaker_split": True, "diarization": True, "transcription_backend": "api"}
    assert (transcript_fingerprint(str(wav), {**on, "whisper_api_diarize": True})
            != transcript_fingerprint(str(wav), on))
    for off in ({"transcription_backend": "local"}, {"speaker_split": False}):
        assert (transcript_fingerprint(str(wav), {**on, **off, "whisper_api_diarize": True})
                == transcript_fingerprint(str(wav), {**on, **off}))


class SplitTranscriber(CountingTranscriber):
    """Counting transcriber whose result says whether the speaker split failed."""

    def __init__(self, split_failed):
        super().__init__()
        self.split_failed = split_failed

    def transcribe(self, path, diarize=False, on_progress=None):
        result = super().transcribe(path, diarize, on_progress)
        return {**result, "split_failed": self.split_failed,
                "warnings": ["Speaker separation failed; channel labels were used."]
                if self.split_failed else []}


SPLIT_ON = {"speaker_split": True, "diarization": True}


def test_a_failed_split_is_not_saved_so_the_next_run_tries_again(tmp_path):
    transcriber, wav = SplitTranscriber(split_failed=True), _real_wav(tmp_path)
    _run(tmp_path, transcriber=transcriber, wav_path=wav, config_overrides=SPLIT_ON)
    result = _run(tmp_path, transcriber=transcriber, wav_path=wav, config_overrides=SPLIT_ON)
    assert transcriber.calls == 2 and not result.reused_transcript
    assert not Path(wav + ".transcript.json").exists()


def test_a_quiet_split_skip_is_saved_and_reused(tmp_path):
    transcriber, wav = SplitTranscriber(split_failed=False), _real_wav(tmp_path)
    _run(tmp_path, transcriber=transcriber, wav_path=wav, config_overrides=SPLIT_ON)
    result = _run(tmp_path, transcriber=transcriber, wav_path=wav, config_overrides=SPLIT_ON)
    assert transcriber.calls == 1 and result.reused_transcript


def test_warnings_are_made_single_line_and_bounded(tmp_path):
    native = "Speaker separation could not run:\n  native error\r\n\tat frame 1  "
    result = _run(tmp_path, warnings=[native, "x" * 400, "y" * 300])
    assert result.warnings == (
        "Speaker separation could not run: native error at frame 1",
        "x" * 299 + "…",
        "y" * 300,
    )


def test_prompt_names_the_recorder_only_when_a_line_is_labelled_you(tmp_path):
    with_you = [Segment(1.0, 2.0, "hi", "You"), Segment(3.0, 4.0, "yo", "Speaker 2")]
    ctx = _context_for(tmp_path, with_you, {"diarization": True})
    assert ctx.has_you_label and ctx.has_numbered_speakers
    in_room = [Segment(1.0, 2.0, "hi", "Speaker 1"), Segment(3.0, 4.0, "yo", "Speaker 2")]
    ctx = _context_for(tmp_path, in_room, {"diarization": True})
    assert not ctx.has_you_label and ctx.has_numbered_speakers and ctx.has_speaker_labels
    assert not _context_for(tmp_path, with_you).has_you_label  # labels not shown


def test_prompt_for_a_split_mono_recording_does_not_call_anyone_the_recorder(tmp_path):
    from tinysteno.orchestrator import Orchestrator
    in_room = [Segment(1.0, 2.0, "hi", "Speaker 1"), Segment(3.0, 4.0, "yo", "Speaker 2")]
    ctx = _context_for(tmp_path, in_room, {"diarization": True})
    block = Orchestrator._context_block(ctx, None)  # pylint: disable=protected-access
    assert "[You]" not in block and "recorded this meeting" in block
    assert "Speaker 1, Speaker 2" in block


def test_prompt_for_legacy_text_names_the_recorder_only_when_you_appears(tmp_path):
    class Legacy(FakeTranscriber):
        """A result in the old shape: labelled text and no segment list."""

        def __init__(self, diarised):
            super().__init__("a b")
            self.diarised = diarised

        def transcribe(self, path, diarize=False, on_progress=None):
            return {**super().transcribe(path, diarize, on_progress),
                    "diarised_text": self.diarised}

    assert _context_for(tmp_path, None, transcriber=Legacy("[You] a\n[Others] b")).has_you_label
    assert not _context_for(tmp_path, None, transcriber=Legacy("[Others] a b")).has_you_label


def test_fingerprint_tracks_the_transcription_server_on_the_api_backend_only(tmp_path):
    wav = _real_wav(tmp_path)
    api = {"transcription_backend": "api", "base_url": "http://llm/v1"}
    assert len(transcript_fingerprint(wav, api)["server_id"]) == 64
    assert (transcript_fingerprint(wav, api)
            == transcript_fingerprint(wav, {**api, "whisper_base_url": "http://llm/v1"}))
    assert (transcript_fingerprint(wav, {**api, "whisper_base_url": "http://a/v1"})
            != transcript_fingerprint(wav, {**api, "whisper_base_url": "http://b/v1"}))
    assert (transcript_fingerprint(wav, {**api, "whisper_base_url": "http://a/v1"})
            == transcript_fingerprint(wav, {**api, "whisper_base_url": "http://a/v1"}))
    local = {"whisper_base_url": "http://a/v1"}
    assert "server_id" not in transcript_fingerprint(wav, local)
    assert transcript_fingerprint(wav, local) == transcript_fingerprint(wav, {})


def test_fingerprint_and_saved_transcript_never_hold_the_server_url(tmp_path):
    transcriber, wav = CountingTranscriber(), _real_wav(tmp_path)
    url = "https://alice:secret@whisper.example.com/v1?token=tok123"
    config = {"transcription_backend": "api", "whisper_base_url": url}
    fingerprint = json.dumps(transcript_fingerprint(wav, config))
    _run(tmp_path, transcriber=transcriber, wav_path=wav, config_overrides=config)
    saved = Path(wav + ".transcript.json").read_text(encoding="utf-8")
    for text in (fingerprint, saved):
        for leak in (url, "whisper.example.com", "alice", "secret", "tok123", "https://"):
            assert leak not in text
    _run(tmp_path, transcriber=transcriber, wav_path=wav, config_overrides=config)
    assert transcriber.calls == 1  # the digest still matches on the next run



def test_pipeline_error_carries_the_warnings_gathered_so_far(tmp_path):
    orchestrator_cls = _orchestrator_cls()
    orchestrator_cls.return_value.warnings = [ORCHESTRATOR_WARNING]
    orchestrator_cls.return_value.summarize.side_effect = ConnectionError("down")
    with pytest.raises(PipelineError) as info:
        _run(tmp_path, warnings=[TRANSCRIBER_WARNING, "two\nlines"],
             orchestrator_cls=orchestrator_cls)
    assert info.value.warnings == (TRANSCRIBER_WARNING, "two lines", ORCHESTRATOR_WARNING)
    exporter_cls = _exporter_cls()
    exporter_cls.return_value.export.side_effect = RuntimeError("bad template")
    with pytest.raises(PipelineError) as info:
        _run(tmp_path, warnings=[TRANSCRIBER_WARNING], exporter_cls=exporter_cls)
    assert info.value.stage is Stage.EXPORT and info.value.warnings == (TRANSCRIBER_WARNING,)
    default = PipelineError(Stage.TRANSCRIBE).warnings
    assert isinstance(default, tuple) and not default
