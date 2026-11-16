"""Tests for transcript segments and the shared assembly path."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from tinysteno.sherpa_diarizer import DiarizerUnavailable
from tinysteno.speaker_split import Turn
from tinysteno.transcript import (
    OTHERS,
    YOU,
    Segment,
    TranscriptOptions,
    assemble,
    interleave,
    is_silent,
    load_transcript,
    labelled_text,
    plain_text,
    render_transcript,
    save_transcript,
    sidecar_path,
)

LOUD = np.full(32000, 0.1, dtype=np.float32)
SILENT = np.zeros(32000, dtype=np.float32)
LOUD_BOTH = {YOU: LOUD, OTHERS: LOUD}
SILENT_BOTH = {YOU: SILENT, OTHERS: SILENT}


def S(start, text, speaker=""):  # pylint: disable=invalid-name
    return Segment(start, start + 1.0, text, speaker)


def test_interleave_orders_by_start_and_puts_you_first_on_ties():
    segs = [S(2.0, "b", OTHERS), S(1.0, "a", OTHERS), S(1.0, "c", YOU)]
    assert [s.text for s in interleave(segs)] == ["c", "a", "b"]


def test_interleave_ties_keep_input_order_for_same_speaker():
    segs = [S(1.0, "x", YOU), S(1.0, "y", YOU), S(1.0, "z", "Alice")]
    assert [s.text for s in interleave(segs)] == ["x", "y", "z"]


def test_plain_text_joins_in_given_order():
    assert plain_text([S(1.0, "b"), S(0.0, "a")]) == "b a"
    assert plain_text([]) == ""


def test_labelled_text_formats_lines_or_none():
    assert labelled_text([S(0.0, "hi", YOU), S(1.0, "yo", OTHERS)]) == "[You] hi\n[Others] yo"
    assert labelled_text([]) is None
    assert labelled_text([S(0.0, "hi")]) is None


def test_assemble_stereo_with_labels():
    result = assemble({YOU: [S(0.0, "hi")], OTHERS: [S(1.0, "hello")]}, LOUD_BOTH,
                      diarize=True, language="en")
    assert result["text"] == "hi hello"
    assert result["diarised_text"] == "[You] hi\n[Others] hello"
    assert result["duration_seconds"] == 2.0 and result["warnings"] == []
    assert result["detected_language"] == "en"
    assert [s.speaker for s in result["segments"]] == [YOU, OTHERS]


def test_assemble_without_diarize_still_includes_both_channels():
    result = assemble({YOU: [S(0.0, "hi")], OTHERS: [S(1.0, "hello")]}, LOUD_BOTH,
                      diarize=False, language="en")
    assert result["text"] == "hi hello" and result["diarised_text"] is None


def test_assemble_nothing_recognised_is_empty_not_labels():
    result = assemble({YOU: [], OTHERS: []}, SILENT_BOTH, diarize=True, language="en")
    assert result["text"] == "" and result["diarised_text"] is None and result["warnings"] == []


def test_assemble_warns_when_a_loud_channel_produced_no_text():
    result = assemble({YOU: [S(0.0, "hi")], OTHERS: []}, LOUD_BOTH, diarize=True, language="en")
    assert result["warnings"] == ["System audio was present but no speech was recognised."]


def test_assemble_warns_for_microphone():
    result = assemble({YOU: [], OTHERS: [S(0.0, "hi")]}, LOUD_BOTH, diarize=True, language="en")
    assert result["warnings"] == ["Microphone audio was present but no speech was recognised."]


def test_assemble_mono_has_no_labels_even_with_diarize():
    result = assemble({"": [S(0.0, "hi")]}, {"": LOUD}, diarize=True, language="en")
    assert result["text"] == "hi" and result["diarised_text"] is None
    assert result["warnings"] == []


def test_assemble_keeps_an_existing_speaker():
    result = assemble({YOU: [S(0.0, "hi", "Alice")]}, {YOU: LOUD}, diarize=True, language="en")
    assert result["segments"][0].speaker == "Alice"


@pytest.mark.parametrize("level, silent", [(0.0, True), (0.0005, True), (0.002, False)])
def test_is_silent(level, silent):
    assert is_silent(np.full(1000, level, dtype=np.float32)) is silent


def test_is_silent_empty_array():
    assert is_silent(np.array([], dtype=np.float32)) is True


def test_assemble_applies_filters_by_default_and_not_when_disabled():
    def seg(start, text):
        return Segment(start, start + 2.0, text)

    channels = {YOU: [seg(1.1, "we should move the launch to friday")],
                OTHERS: [seg(1.0, "We should move the launch to Friday.")]}
    audio = {YOU: np.full(64000, 0.01, dtype=np.float32),
             OTHERS: np.full(64000, 0.2, dtype=np.float32)}
    assert assemble(channels, audio, diarize=True, language="en")["diarised_text"] \
        == "[Others] We should move the launch to Friday."
    off = TranscriptOptions(drop_hallucinations=False, suppress_bleed=False)
    assert len(assemble(channels, audio, diarize=True, language="en", options=off)["segments"]) == 2


def test_assemble_drops_credits_per_channel_but_still_warns_for_loud_empty_channel():
    channels = {YOU: [Segment(0.0, 1.0, "Thanks for watching!")], OTHERS: []}
    audio = {YOU: LOUD, OTHERS: LOUD}
    result = assemble(channels, audio, diarize=True, language="en")
    assert result["segments"] == []
    assert result["warnings"] == ["System audio was present but no speech was recognised."]


def _bleed_audio(mic_level, system_level):
    return {YOU: np.full(64000, mic_level, dtype=np.float32),
            OTHERS: np.full(64000, system_level, dtype=np.float32)}


def test_assemble_measures_bleed_by_channel_not_by_speaker_label():
    text = "we should move the launch to friday"
    channels = {YOU: [Segment(1.1, 3.0, text, "Alice")],
                OTHERS: [Segment(1.0, 3.0, text)]}
    result = assemble(channels, _bleed_audio(0.01, 0.2), diarize=True, language="en")
    assert [(s.speaker, s.text) for s in result["segments"]] == [(OTHERS, text)]


def test_assemble_drops_a_relabelled_system_copy_when_the_microphone_is_louder():
    text = "we should move the launch to friday"
    channels = {YOU: [Segment(1.1, 3.0, text)],
                OTHERS: [Segment(1.0, 3.0, text, "Speaker 2")]}
    result = assemble(channels, _bleed_audio(0.2, 0.01), diarize=True, language="en")
    assert [(s.speaker, s.text) for s in result["segments"]] == [(YOU, text)]


_RENDER_SEGS = [Segment(3.0, 4.0, "hi", "You"), Segment(3725.0, 3726.0, "bye", "Others")]


def test_render_with_labels_and_timestamps():
    assert render_transcript(_RENDER_SEGS, labels=True, timestamps=True) \
        == "[00:00:03] [You] hi\n[01:02:05] [Others] bye"


def test_render_timestamps_only():
    assert render_transcript(_RENDER_SEGS, labels=False, timestamps=True) \
        == "[00:00:03] hi\n[01:02:05] bye"


def test_render_labels_only():
    assert render_transcript(_RENDER_SEGS, labels=True, timestamps=False) \
        == labelled_text(_RENDER_SEGS)


def test_render_plain():
    assert render_transcript(_RENDER_SEGS, labels=False, timestamps=False) == "hi bye"


def test_render_mono_segment_has_no_speaker_prefix():
    assert render_transcript([Segment(3.9, 5.0, "hi")], labels=True, timestamps=True) \
        == "[00:00:03] hi"


def test_render_hours_are_not_truncated():
    assert render_transcript([Segment(360000.0, 360001.0, "x")], labels=False, timestamps=True) \
        == "[100:00:00] x"


RESULT = {
    "text": "hello there",
    "diarised_text": "[You] hello\n[Others] there",
    "duration_seconds": 12.5,
    "detected_language": "en",
    "segments": [Segment(0.0, 1.0, "hello", YOU), Segment(1.0, 2.0, "there", OTHERS)],
    "warnings": ["System audio was present but no speech was recognised."],
}


def _wav(tmp_path):
    wav = tmp_path / "Meeting-1.wav"
    wav.write_bytes(b"RIFF")
    return wav


FP = {"size": 4, "prompt": "", "bleed_filter": True}


def _sidecar_json(wav):
    return json.loads(sidecar_path(str(wav)).read_text(encoding="utf-8"))


def _write_sidecar(wav, **changes):
    data = {**_sidecar_json(wav), **changes}
    sidecar_path(str(wav)).write_text(json.dumps(data), encoding="utf-8")


def test_sidecar_name_keeps_the_audio_extension(tmp_path):
    assert str(sidecar_path("m/Meeting-1.wav")).endswith("Meeting-1.wav.transcript.json")
    assert sidecar_path(str(tmp_path / "a.wav")) != sidecar_path(str(tmp_path / "a.mp3"))


def test_save_then_load_round_trips(tmp_path):
    wav = _wav(tmp_path)
    save_transcript(str(wav), RESULT, FP)
    loaded = load_transcript(str(wav), FP)
    assert loaded["segments"] == RESULT["segments"]
    assert loaded["warnings"] == RESULT["warnings"]
    assert loaded["duration_seconds"] == 12.5 and loaded["detected_language"] == "en"
    assert loaded["text"] == "hello there"
    assert loaded["diarised_text"] == "[You] hello\n[Others] there"
    assert _sidecar_json(wav)["version"] == 2
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "Meeting-1.wav", "Meeting-1.wav.transcript.json"]


@pytest.mark.parametrize("other", [
    {**FP, "size": 5}, {**FP, "prompt": "Acme"}, {**FP, "bleed_filter": False}, {}])
def test_different_fingerprint_is_a_miss(tmp_path, other):
    wav = _wav(tmp_path)
    save_transcript(str(wav), RESULT, FP)
    assert load_transcript(str(wav), other) is None


def test_sidecar_older_than_the_wav_is_ignored(tmp_path):
    wav = _wav(tmp_path)
    save_transcript(str(wav), RESULT, FP)
    os.utime(sidecar_path(str(wav)), (1, 1))
    assert load_transcript(str(wav), FP) is None


def test_corrupt_sidecar_is_ignored(tmp_path):
    wav = _wav(tmp_path)
    sidecar_path(str(wav)).write_text("{not json", encoding="utf-8")
    assert load_transcript(str(wav), FP) is None


@pytest.mark.parametrize("version", [1, 3, None])
def test_other_versions_are_ignored(tmp_path, version):
    wav = _wav(tmp_path)
    save_transcript(str(wav), RESULT, FP)
    _write_sidecar(wav, version=version)
    assert load_transcript(str(wav), FP) is None


GOOD = {"start": 0.0, "end": 1.0, "text": "hi", "speaker": "You"}


@pytest.mark.parametrize("changes", [
    {"segments": [{**GOOD, "text": None}]},
    {"segments": [{**GOOD, "speaker": None}]},
    {"segments": [{**GOOD, "start": "0"}]},
    {"segments": [{**GOOD, "start": True}]},
    {"segments": [{**GOOD, "start": -1.0}]},
    {"segments": [{**GOOD, "start": 2.0, "end": 1.0}]},
    {"segments": [{**GOOD, "end": float("inf")}]},
    {"segments": [{**GOOD, "end": float("nan")}]},
    {"segments": ["x"]},
    {"segments": "abc"},
    {"warnings": "abc"},
    {"warnings": [1]},
    {"detected_language": None},
    {"duration_seconds": -1},
    {"duration_seconds": "3"},
    {"duration_seconds": float("inf")},
])
def test_wrong_shapes_are_a_miss(tmp_path, changes):
    wav = _wav(tmp_path)
    save_transcript(str(wav), RESULT, FP)
    _write_sidecar(wav, **changes)
    assert load_transcript(str(wav), FP) is None


def test_non_object_top_level_is_a_miss(tmp_path):
    wav = _wav(tmp_path)
    sidecar_path(str(wav)).write_text("[]", encoding="utf-8")
    assert load_transcript(str(wav), FP) is None


def test_missing_wav_or_sidecar_loads_and_saves_nothing(tmp_path):
    assert load_transcript(str(tmp_path / "none.wav"), FP) is None
    save_transcript(str(tmp_path / "none.wav"), RESULT, FP)
    assert not list(tmp_path.iterdir())
    assert load_transcript(str(_wav(tmp_path)), FP) is None


def test_result_without_segments_is_not_saved(tmp_path):
    wav = _wav(tmp_path)
    save_transcript(str(wav), {"text": "x"}, FP)
    assert not sidecar_path(str(wav)).exists()


def test_save_failure_is_ignored_and_leaves_no_temp_file(tmp_path, monkeypatch):
    wav = _wav(tmp_path)

    def boom(*_args):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    save_transcript(str(wav), RESULT, FP)
    assert [p.name for p in tmp_path.iterdir()] == ["Meeting-1.wav"]


def test_unserialisable_value_logs_a_warning_and_leaves_no_temp_file(tmp_path, caplog):
    wav = _wav(tmp_path)
    save_transcript(str(wav), {**RESULT, "warnings": [object()]}, FP)
    assert "Could not save the transcript" in caplog.text
    assert [p.name for p in tmp_path.iterdir()] == ["Meeting-1.wav"]


# --- speaker split -------------------------------------------------------------------------

LONG = np.full(16000 * 120, 0.1, dtype=np.float32)
SILENT_LONG = np.zeros(16000 * 120, dtype=np.float32)
LONG_BOTH = {YOU: LONG, OTHERS: LONG}
SPLIT_CHANNELS = {YOU: [S(5, "mine")], OTHERS: [S(1, "first voice"), S(40, "second voice")]}
ON = TranscriptOptions(speaker_split=True)
TWO_VOICES = [Turn(0, 30, 0), Turn(30, 90, 1)]
UNAVAILABLE = 'Speaker separation needs the optional extra: pip install "tinysteno[diarize]"'


def _must_not_be_called(_label, _audio):
    raise AssertionError("the diarizer must not run")


def _speakers(result):
    return [seg.speaker for seg in result["segments"]]


def test_split_relabels_the_remote_channel_only():
    calls = []

    def turns_for(label, audio):
        calls.append((label, len(audio)))
        return TWO_VOICES

    result = assemble(SPLIT_CHANNELS, LONG_BOTH, diarize=True, language="en", options=ON,
                      turns_for=turns_for)
    assert _speakers(result) == ["Speaker 2", "You", "Others"]  # 2nd voice spoke most
    assert calls == [(OTHERS, len(LONG))]
    assert result["warnings"] == []


def test_off_by_default_never_calls_the_diarizer():
    result = assemble(SPLIT_CHANNELS, LONG_BOTH, diarize=True, language="en",
                      turns_for=_must_not_be_called)
    assert set(_speakers(result)) == {YOU, OTHERS}


def test_split_is_skipped_when_labels_are_not_shown():
    result = assemble(SPLIT_CHANNELS, LONG_BOTH, diarize=False, language="en", options=ON,
                      turns_for=_must_not_be_called)
    assert set(_speakers(result)) == {YOU, OTHERS}


def test_in_person_meeting_splits_the_microphone():
    channels = {YOU: [S(1, "host"), S(70, "guest")], OTHERS: []}
    seen = []

    def turns_for(label, _audio):
        seen.append(label)
        return [Turn(0, 60, 0), Turn(60, 90, 1)]

    result = assemble(channels, {YOU: LONG, OTHERS: SILENT_LONG}, diarize=True, language="en",
                      options=ON, turns_for=turns_for)
    # The most-heard voice in the room is not necessarily the person recording.
    assert _speakers(result) == ["Speaker 1", "Speaker 2"] and seen == [YOU]


def test_mono_file_is_split_into_numbered_speakers_never_you():
    seen = []

    def turns_for(label, _audio):
        seen.append(label)
        return [Turn(0, 60, 0), Turn(60, 90, 1)]

    result = assemble({"": [S(1, "host"), S(70, "guest")]}, {"": LONG}, diarize=True,
                      language="en", options=ON, turns_for=turns_for)
    assert _speakers(result) == ["Speaker 1", "Speaker 2"] and seen == [YOU]


def test_mono_file_keeps_empty_speakers_when_off_or_failed():
    off = assemble({"": [S(1, "a")]}, {"": LONG}, diarize=True, language="en")
    assert _speakers(off) == [""]

    def boom(_label, _audio):
        raise RuntimeError("x")

    failed = assemble({"": [S(1, "a")]}, {"": LONG}, diarize=True, language="en",
                      options=ON, turns_for=boom)
    assert _speakers(failed) == [""]


def test_microphone_of_a_call_with_remote_speech_is_never_split():
    channels = {YOU: [S(1, "a"), S(50, "b")], OTHERS: [S(2, "c")]}
    labels = []

    def turns_for(label, _audio):
        labels.append(label)
        return [Turn(0, 60, 0), Turn(60, 90, 1)]

    result = assemble(channels, LONG_BOTH, diarize=True, language="en", options=ON,
                      turns_for=turns_for)
    assert labels == [OTHERS]
    assert [s.speaker for s in result["segments"] if s.text in ("a", "b")] == [YOU, YOU]


def test_missing_extra_keeps_labels_and_says_what_to_install():
    def unavailable(_label, _audio):
        raise DiarizerUnavailable(UNAVAILABLE)

    result = assemble(SPLIT_CHANNELS, LONG_BOTH, diarize=True, language="en", options=ON,
                      turns_for=unavailable)
    assert set(_speakers(result)) == {YOU, OTHERS}
    assert result["warnings"] == [UNAVAILABLE]


def test_default_diarizer_reports_the_missing_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "sherpa_onnx", None)  # makes `import sherpa_onnx` fail
    result = assemble(SPLIT_CHANNELS, LONG_BOTH, diarize=True, language="en", options=ON)
    assert set(_speakers(result)) == {YOU, OTHERS}
    assert len(result["warnings"]) == 1 and "tinysteno[diarize]" in result["warnings"][0]


def test_default_turns_come_from_the_diarizer_with_progress(monkeypatch):
    seen = []

    def fake_diarize(audio, *, on_progress=None, **_kwargs):
        assert len(audio) == len(LONG)
        on_progress(0.5)
        return TWO_VOICES

    monkeypatch.setattr("tinysteno.sherpa_diarizer.diarize", fake_diarize)
    result = assemble(SPLIT_CHANNELS, LONG_BOTH, diarize=True, language="en", options=ON,
                      on_progress=seen.append)
    assert _speakers(result) == ["Speaker 2", "You", "Others"] and seen == [0.5]


def test_any_other_failure_keeps_labels_with_a_generic_warning(caplog):
    def boom(_label, _audio):
        raise ValueError("model exploded")

    with caplog.at_level("WARNING"):
        result = assemble(SPLIT_CHANNELS, LONG_BOTH, diarize=True, language="en", options=ON,
                          turns_for=boom)
    assert set(_speakers(result)) == {YOU, OTHERS}
    assert result["warnings"] == ["Speaker separation failed; channel labels were used."]
    assert any(r.exc_info and "model exploded" in str(r.exc_info[1]) for r in caplog.records)


def test_policy_collapse_is_silent():
    result = assemble(SPLIT_CHANNELS, LONG_BOTH, diarize=True, language="en", options=ON,
                      turns_for=lambda _label, _audio: [Turn(0, 90, 0)])
    assert set(_speakers(result)) == {YOU, OTHERS} and result["warnings"] == []


def test_short_audio_is_not_split():
    short = np.full(16000 * 20, 0.1, dtype=np.float32)
    result = assemble(SPLIT_CHANNELS, {YOU: short, OTHERS: short}, diarize=True, language="en",
                      options=ON, turns_for=_must_not_be_called)
    assert set(_speakers(result)) == {YOU, OTHERS} and result["warnings"] == []


def test_a_cap_of_one_speaker_skips_the_split():
    result = assemble(SPLIT_CHANNELS, LONG_BOTH, diarize=True, language="en",
                      options=TranscriptOptions(speaker_split=True, max_speakers=1),
                      turns_for=_must_not_be_called)
    assert set(_speakers(result)) == {YOU, OTHERS}


def test_max_speakers_caps_the_split():
    turns = [Turn(0, 30, 0), Turn(30, 60, 1), Turn(60, 85, 2), Turn(85, 120, 0)]
    channels = {OTHERS: [S(1, "a"), S(35, "b"), S(65, "c")]}
    result = assemble(channels, {OTHERS: LONG}, diarize=True, language="en",
                      options=TranscriptOptions(speaker_split=True, max_speakers=2),
                      turns_for=lambda _label, _audio: turns)
    assert _speakers(result) == ["Others", "Speaker 2", "Speaker 2"]


def test_split_runs_after_the_filters():
    # Others holds only a hallucination; the filter empties it, so the split must go to You.
    channels = {YOU: [S(1, "real words"), S(70, "more words")],
                OTHERS: [S(2, "Thanks for watching!")]}
    seen = []

    def turns_for(label, _audio):
        seen.append(label)
        return [Turn(0, 60, 0), Turn(60, 90, 1)]

    result = assemble(channels, LONG_BOTH, diarize=True, language="en", options=ON,
                      turns_for=turns_for)
    assert seen == [YOU] and _speakers(result) == ["Speaker 1", "Speaker 2"]


def test_saved_numbered_speakers_load_unchanged(tmp_path):
    wav = _wav(tmp_path)
    result = {**RESULT, "segments": [Segment(0.0, 1.0, "hi", "Speaker 2")]}
    save_transcript(str(wav), result, FP)
    assert load_transcript(str(wav), FP)["segments"][0].speaker == "Speaker 2"


def test_importing_the_pipeline_does_not_import_sherpa():
    check = (
        "import sys, tinysteno.main, tinysteno.transcript, tinysteno.transcriber, "
        "tinysteno.api_transcriber; "
        "assert 'sherpa_onnx' not in sys.modules and "
        "'tinysteno.sherpa_diarizer' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", check], check=True)


def test_a_failure_while_relabelling_keeps_the_segments(monkeypatch):
    def boom(*_args):
        raise ValueError("bad turns")

    monkeypatch.setattr("tinysteno.transcript.label_segments", boom)
    result = assemble({"": [S(1, "a"), S(70, "b")]}, {"": LONG}, diarize=True, language="en",
                      options=ON, turns_for=lambda _label, _audio: TWO_VOICES)
    assert _speakers(result) == ["", ""]
    assert result["warnings"] == ["Speaker separation failed; channel labels were used."]


def test_the_exception_type_lives_in_the_policy_module():
    from tinysteno import sherpa_diarizer, speaker_split
    assert sherpa_diarizer.DiarizerUnavailable is speaker_split.DiarizerUnavailable


def test_lazy_import_holds_after_a_split_runs():
    check = (
        "import sys, numpy as np\n"
        "from tinysteno.speaker_split import Turn\n"
        "from tinysteno.transcript import assemble, Segment, TranscriptOptions\n"
        "a = np.full(16000 * 120, 0.1, dtype=np.float32)\n"
        "r = assemble({'Others': [Segment(1, 2, 'x'), Segment(70, 71, 'y')]}, {'Others': a},\n"
        "             diarize=True, language='en', options=TranscriptOptions(speaker_split=True),\n"
        "             turns_for=lambda l, x: [Turn(0, 60, 0), Turn(60, 90, 1)])\n"
        "assert [s.speaker for s in r['segments']] == ['Others', 'Speaker 2']\n"
        "assert 'sherpa_onnx' not in sys.modules\n"
        "assert 'tinysteno.sherpa_diarizer' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", check], check=True)


def test_a_failed_split_is_flagged_on_both_failure_paths():
    def unavailable(_label, _audio):
        raise DiarizerUnavailable(UNAVAILABLE)

    def boom(_label, _audio):
        raise ValueError("model exploded")

    for turns_for in (unavailable, boom):
        result = assemble(SPLIT_CHANNELS, LONG_BOTH, diarize=True, language="en", options=ON,
                          turns_for=turns_for)
        assert result["split_failed"] is True


def test_a_quiet_skip_or_a_split_is_not_a_failure():
    short = np.full(16000 * 20, 0.1, dtype=np.float32)
    runs = [
        assemble(SPLIT_CHANNELS, LONG_BOTH, diarize=True, language="en", options=ON,
                 turns_for=lambda _label, _audio: TWO_VOICES),
        assemble(SPLIT_CHANNELS, LONG_BOTH, diarize=True, language="en", options=ON,
                 turns_for=lambda _label, _audio: [Turn(0, 90, 0)]),
        assemble(SPLIT_CHANNELS, {YOU: short, OTHERS: short}, diarize=True, language="en",
                 options=ON, turns_for=_must_not_be_called),
        assemble(SPLIT_CHANNELS, LONG_BOTH, diarize=True, language="en"),
    ]
    assert [run["split_failed"] for run in runs] == [False] * 4
