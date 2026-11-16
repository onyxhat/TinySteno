"""Tests for removing lines Whisper invents in silence or noise."""
import logging

from tinysteno.hygiene import drop_hallucinations, suppress_bleed
from tinysteno.transcript import Segment


def seg(start, text):
    return Segment(start, start + 1.5, text, "Others")


def level_by_text(quiet_texts, loud=0.2, quiet=0.004):
    """An rms stub: quiet over the given texts, loud elsewhere, keyed by segment start."""
    def make(segments):
        quiet_starts = {s.start for s in segments if s.text in quiet_texts}
        return lambda start, end: quiet if start in quiet_starts else loud
    return make


def test_caption_credits_are_removed():
    kept = drop_hallucinations([seg(0, "Let's begin."),
                                seg(4, "Subtitles by the Amara.org community"),
                                seg(9, "Thanks for watching!")])
    assert [s.text for s in kept] == ["Let's begin."]


def test_each_caption_credit_form_is_removed():
    texts = ["Captions provided by X", "Transcribed by someone", "Translated by Y",
             "Please subscribe to my channel", "see amara.org", "Thank you for watching.",
             "Thanks so much for watching! Please subscribe.",
             "Thank you very much for watching and please subscribe"]
    assert not drop_hallucinations([seg(i * 3, t) for i, t in enumerate(texts)])


def test_real_sentences_that_mention_credit_words_are_kept():
    talk = [seg(0, "Thanks for watching the demo, any questions?"),
            seg(3, "It was transcribed by hand last year."),
            seg(6, "Please subscribe the team to the alerts channel before Friday, thanks."),
            seg(9, "Subtitles by hand are slow because we check every single line twice.")]
    assert drop_hallucinations(talk) == talk


def test_a_stuck_run_collapses_to_one():
    kept = drop_hallucinations([seg(i, "I'm sorry.") for i in range(6)]
                               + [seg(7, "Right, so the budget.")])
    assert [s.text for s in kept] == ["I'm sorry.", "Right, so the budget."]


def test_four_identical_in_a_row_are_left_alone():
    talk = [seg(i, "I'm sorry.") for i in range(4)]
    assert drop_hallucinations(talk) == talk


THANKS = "Thank you."


def stranded_thanks():
    return [seg(0, THANKS), seg(60, THANKS), seg(120, THANKS), seg(200, "Okay, let's start."),
            seg(202, "First item."),
            seg(204, "Second item."), seg(206, "Third item.")]


def test_quiet_short_repeats_stranded_in_silence_are_removed():
    talk = stranded_thanks()
    kept = drop_hallucinations(talk, level_by_text({THANKS})(talk))
    assert [s.text for s in kept] == ["Okay, let's start.", "First item.", "Second item.",
                                  "Third item."]


def test_stranded_repeats_are_kept_without_audio_evidence():
    talk = stranded_thanks()
    assert drop_hallucinations(talk) == talk


def test_stranded_repeats_are_kept_when_the_median_level_is_zero():
    talk = stranded_thanks()
    assert drop_hallucinations(talk, lambda start, end: 0.0) == talk


def test_loud_stranded_answers_from_a_listener_are_kept():
    talk = [seg(0, "Yes."), seg(60, "Yes."), seg(120, "Yes.")]
    assert drop_hallucinations(talk, lambda start, end: 0.2) == talk


def test_a_gap_of_exactly_ten_seconds_counts_as_stranded():
    talk = [Segment(0.0, 1.0, THANKS, "Others"), Segment(11.0, 12.0, THANKS, "Others"),
            Segment(22.0, 23.0, THANKS, "Others"), Segment(33.0, 34.0, "Real talk.", "Others"),
            Segment(35.0, 36.0, "More talk.", "Others"),
            Segment(37.0, 38.0, "Even more.", "Others")]
    kept = drop_hallucinations(talk, level_by_text({THANKS})(talk))
    assert [s.text for s in kept] == ["Real talk.", "More talk.", "Even more."]


def test_only_the_stranded_quiet_occurrences_are_removed():
    talk = [seg(0, THANKS), seg(60, THANKS), seg(120, "Right."), seg(122, THANKS),
            seg(124, "Next point."), seg(126, "Done.")]
    quiet = {s.start for s in talk if s.text == THANKS}
    kept = drop_hallucinations(talk, lambda start, end: 0.004 if start in quiet else 0.2)
    assert [(s.start, s.text) for s in kept] == [(120, "Right."), (122, THANKS),
                                                 (124, "Next point."), (126, "Done.")]


def test_first_and_last_segments_can_be_stranded():
    talk = [seg(0, THANKS), seg(50, "Real talk here."), seg(52, "More real talk."),
            seg(54, "And more."), seg(100, THANKS), seg(150, THANKS)]
    kept = drop_hallucinations(talk, level_by_text({THANKS})(talk))
    assert [s.text for s in kept] == ["Real talk here.", "More real talk.", "And more."]


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


def bseg(start, text, speaker):
    return Segment(start, start + 2.0, text, speaker)


def levels(mic_level, system_level):
    """Keyword-style helper: the (mic_rms, system_rms) pair for suppress_bleed."""
    return (lambda _start, _end: mic_level), (lambda _start, _end: system_level)


def test_bleed_drops_the_quieter_copy(caplog):
    mic = [bseg(10.2, "so the release slips to the twenty first", "You")]
    system = [bseg(10.0, "So the release slips to the twenty-first.", "Others")]
    with caplog.at_level(logging.INFO, logger="tinysteno.hygiene"):
        kept_mic, kept_system = suppress_bleed(mic, system, *levels(0.02, 0.3))
    assert kept_mic == [] and kept_system == system
    assert "speaker bleed" in caplog.text and "speaker=You" in caplog.text


def test_bleed_keeps_the_louder_microphone_when_it_is_the_source():
    mic = [bseg(10.2, "so the release slips to the twenty first", "You")]
    system = [bseg(10.0, "So the release slips to the twenty-first.", "Others")]
    kept_mic, kept_system = suppress_bleed(mic, system, *levels(0.3, 0.02))
    assert kept_mic == mic and kept_system == []


def test_bleed_tie_drops_the_microphone_copy():
    mic = [bseg(10.0, "so the release slips to the twenty first", "You")]
    system = [bseg(10.0, "So the release slips to the twenty-first.", "Others")]
    kept_mic, kept_system = suppress_bleed(mic, system, *levels(0.1, 0.1))
    assert kept_mic == [] and kept_system == system


def test_different_sentences_close_in_time_are_both_kept():
    mic = [bseg(10.0, "I will send the budget spreadsheet tomorrow", "You")]
    system = [bseg(10.5, "The customer asked about the renewal date", "Others")]
    assert suppress_bleed(mic, system, *levels(0.02, 0.3)) == (mic, system)


def test_same_words_more_than_three_seconds_apart_are_both_kept():
    mic = [bseg(10.0, "so the release slips to the twenty first", "You")]
    system = [bseg(13.5, "So the release slips to the twenty-first.", "Others")]
    assert suppress_bleed(mic, system, *levels(0.02, 0.3)) == (mic, system)


def test_short_echoed_agreement_is_never_bleed():
    mic, system = [bseg(5.0, "Sounds good.", "You")], [bseg(5.4, "Sounds good.", "Others")]
    assert suppress_bleed(mic, system, *levels(0.1, 0.1)) == (mic, system)


def test_each_segment_pairs_at_most_once():
    text = "so the release slips to the twenty first"
    mic = [bseg(10.0, text, "You")]
    system = [bseg(10.1, text, "Others"), bseg(10.2, text, "Others")]
    kept_mic, kept_system = suppress_bleed(mic, system, *levels(0.02, 0.3))
    assert kept_mic == [] and kept_system == system


def test_bleed_does_not_mutate_inputs():
    mic = [bseg(10.2, "so the release slips to the twenty first", "You")]
    system = [bseg(10.0, "So the release slips to the twenty-first.", "Others")]
    suppress_bleed(mic, system, *levels(0.02, 0.3))
    assert len(mic) == 1 and len(system) == 1
