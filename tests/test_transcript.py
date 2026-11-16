"""Tests for transcript segments and the shared assembly path."""
import numpy as np
import pytest

from tinysteno.transcript import (
    OTHERS,
    YOU,
    Segment,
    TranscriptOptions,
    assemble,
    interleave,
    is_silent,
    labelled_text,
    plain_text,
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
