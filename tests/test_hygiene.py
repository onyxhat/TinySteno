"""Tests for removing lines Whisper invents in silence or noise."""
import logging

from tinysteno.hygiene import drop_hallucinations
from tinysteno.transcript import Segment


def seg(start, text):
    return Segment(start, start + 1.5, text, "Others")


def test_caption_credits_are_removed():
    kept = drop_hallucinations([seg(0, "Let's begin."),
                                seg(4, "Subtitles by the Amara.org community"),
                                seg(9, "Thanks for watching!")])
    assert [s.text for s in kept] == ["Let's begin."]


def test_each_caption_credit_pattern_matches():
    texts = ["Captions provided by X", "Transcribed by someone", "Translated by Y",
             "Please subscribe", "see amara.org", "Thank you for watching"]
    assert not drop_hallucinations([seg(i * 3, t) for i, t in enumerate(texts)])


def test_a_stuck_run_collapses_to_one():
    kept = drop_hallucinations([seg(i, "I'm sorry.") for i in range(6)]
                               + [seg(7, "Right, so the budget.")])
    assert [s.text for s in kept] == ["I'm sorry.", "Right, so the budget."]


def test_four_identical_in_a_row_are_left_alone():
    talk = [seg(i, "I'm sorry.") for i in range(4)]
    assert drop_hallucinations(talk) == talk


def test_short_repeats_stranded_in_silence_are_removed():
    kept = drop_hallucinations([seg(0, "Thank you."), seg(60, "Thank you."),
                                seg(120, "Thank you."), seg(200, "Okay, let's start."),
                                seg(202, "First item.")])
    assert [s.text for s in kept] == ["Okay, let's start.", "First item."]


def test_short_replies_inside_a_conversation_are_kept():
    talk = [seg(0, "Shall we ship it?"), seg(2, "Yeah."), seg(4, "Any objections?"),
            seg(6, "Yeah."), seg(8, "Then it's decided."), seg(10, "Yeah.")]
    assert drop_hallucinations(talk) == talk


def test_two_stranded_occurrences_are_left_alone():
    talk = [seg(0, "Thank you."), seg(60, "Thank you."), seg(120, "Fine.")]
    assert drop_hallucinations(talk) == talk


def test_long_stranded_repeats_are_left_alone():
    text = "one two three four five six"
    talk = [seg(0, text), seg(60, text), seg(120, text)]
    assert drop_hallucinations(talk) == talk


def test_empty_input():
    assert not drop_hallucinations([])


def test_input_is_not_mutated_and_a_new_list_is_returned():
    talk = [seg(0, "Please subscribe"), seg(5, "Hello")]
    before = list(talk)
    kept = drop_hallucinations(talk)
    assert talk == before
    assert kept is not talk


def test_removals_are_logged_with_reason(caplog):
    with caplog.at_level(logging.INFO, logger="tinysteno.hygiene"):
        drop_hallucinations([seg(3, "Please subscribe")])
    assert "caption credit" in caplog.text
    assert "Others" in caplog.text
    assert "Please subscribe" in caplog.text
