"""Policy for trusting a diarizer's speaker split."""
import pytest

from tinysteno.speaker_split import Turn, apply_policy, cluster_threshold, label_segments
from tinysteno.transcript import Segment


@pytest.mark.parametrize("seconds, expected",
                         [(60, 0.6), (900, 0.6), (2250, 0.65), (3600, 0.7), (9000, 0.7)])
def test_cluster_threshold_ramps(seconds, expected):
    assert cluster_threshold(seconds) == pytest.approx(expected)


def test_one_to_one_call_keeps_channel_labels():
    assert apply_policy([Turn(0, 50, 0), Turn(52, 110, 0)]) is None


def test_a_phantom_cluster_does_not_become_a_speaker():
    turns = [Turn(0, 120, 0), Turn(121, 121.6, 1), Turn(125, 240, 0)]
    assert apply_policy(turns) is None


def test_near_total_dominance_collapses():
    assert apply_policy([Turn(0, 950, 0), Turn(950, 1000, 1)]) is None


def test_two_real_speakers_are_kept_and_ranked_by_speech():
    turns = [Turn(0, 30, 4), Turn(30, 110, 7), Turn(110, 140, 4)]
    assert apply_policy(turns) == [Turn(0, 30, 1), Turn(30, 110, 0), Turn(110, 140, 1)]


def test_speaker_cap_folds_the_smallest_into_its_neighbour():
    turns = [Turn(0, 60, 0), Turn(60, 100, 1), Turn(100, 112, 2), Turn(112, 170, 1)]
    out = apply_policy(turns, max_speakers=2)
    assert {t.speaker for t in out} == {0, 1} and len(out) == 4
    assert [t.speaker for t in out] == [1, 0, 0, 0]  # the tie goes to the earlier kept turn


def test_cap_of_one_means_no_split():
    assert apply_policy([Turn(0, 60, 0), Turn(60, 120, 1)], max_speakers=1) is None


def test_empty_turns():
    assert apply_policy([]) is None


def test_label_segments_by_overlap_and_leaves_other_channel_alone():
    segs = [Segment(1, 5, "a", "Others"), Segment(31, 40, "b", "Others"),
            Segment(6, 8, "c", "You")]
    turns = [Turn(0, 30, 1), Turn(30, 100, 0)]
    assert [s.speaker for s in label_segments(segs, turns, "Others")] == [
        "Speaker 2", "Others", "You"]


def test_segment_outside_every_turn_goes_to_the_nearest_or_dominant():
    turns = [Turn(0, 10, 0), Turn(20, 22, 1)]  # turn midpoints 5.0 and 21.0
    segs = [Segment(22.5, 23, "near", "Others"), Segment(60, 61, "far", "Others"),
            Segment(23, 24.5, "too far", "Others")]
    out = label_segments(segs, turns, "Others")
    assert [s.speaker for s in out] == ["Speaker 2", "Others", "Others"]


def test_label_segments_does_not_mutate_input():
    segs = [Segment(1, 5, "a", "Others")]
    label_segments(segs, [Turn(0, 30, 1)], "Others")
    assert segs == [Segment(1, 5, "a", "Others")]


def test_label_segments_can_number_the_dominant_voice_too():
    segs = [Segment(0, 1, "a", "You"), Segment(5, 6, "b", "You"), Segment(5, 6, "c", "Others")]
    turns = [Turn(0, 2, 0), Turn(4, 7, 1)]
    out = label_segments(segs, turns, "You", first_label="Speaker 1")
    assert [s.speaker for s in out] == ["Speaker 1", "Speaker 2", "Others"]
