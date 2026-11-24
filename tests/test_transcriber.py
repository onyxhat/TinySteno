"""Tests for WhisperTranscriber."""
# pylint: disable=protected-access
from unittest.mock import MagicMock, patch

import numpy as np
import soundfile as sf

_LEFT_LEVEL = 0.1
_RIGHT_LEVEL = 0.2
_SECONDS = 2


def _segments(spec):
    """Turn a string or [(start, text)] into fake faster-whisper segments."""
    if isinstance(spec, str):
        spec = [(0.0, spec)]
    return [MagicMock(start=start, end=start + 0.5, text=f" {text} ") for start, text in spec]


def _transcribe(
    tmp_path,
    *,
    left=None,
    right=None,
    mono=None,
    left_silent=False,
    right_silent=False,
    right_level=_RIGHT_LEVEL,
    diarize=False,
    on_progress=None,
    return_model=False,
):  # pylint: disable=too-many-arguments,too-many-locals
    """Write a WAV, run WhisperTranscriber on it with a fake model, return the result."""
    import tinysteno.transcriber as mod

    frames = 16000 * _SECONDS
    if mono is not None:
        data = np.full(frames, _LEFT_LEVEL, dtype=np.float32)
        by_level = {_LEFT_LEVEL: mono}
    else:
        data = np.stack(
            [
                np.full(frames, 0.0 if left_silent else _LEFT_LEVEL, dtype=np.float32),
                np.full(frames, 0.0 if right_silent else right_level, dtype=np.float32),
            ],
            axis=1,
        )
        by_level = {_LEFT_LEVEL: left or [], right_level: right or []}
    wav_path = tmp_path / "t.wav"
    sf.write(str(wav_path), data, 16000, subtype="FLOAT")

    def fake_transcribe(audio, **_kwargs):
        level = float(np.mean(audio))
        for key, spec in by_level.items():
            if np.isclose(level, key):
                return iter(_segments(spec)), MagicMock(language="en")
        raise AssertionError("a silent channel must not be decoded")

    mod._MODEL_CACHE.clear()
    with patch("tinysteno.transcriber.WhisperModel") as mock_wm:
        model = MagicMock()
        model.transcribe.side_effect = fake_transcribe
        mock_wm.return_value = model
        result = mod.WhisperTranscriber().transcribe(
            str(wav_path), diarize=diarize, on_progress=on_progress
        )
    return (result, model) if return_model else result


def _make_transcriber(model_size="tiny", **options):
    with patch("tinysteno.transcriber.WhisperModel") as mock_wm:
        mock_wm.return_value = MagicMock()
        from tinysteno.transcriber import WhisperTranscriber

        t = WhisperTranscriber(model_size=model_size, **options)
    return t


def _decode_kwargs(**options):
    """Run one decode on a fake model; return the keyword arguments it was given."""
    t = _make_transcriber(**options)
    t._model.transcribe.return_value = (iter([]), MagicMock(language="en"))
    t._decode(np.zeros(16000, dtype=np.float32), None)
    return t._model.transcribe.call_args.kwargs


def test_decode_uses_vad_and_prompt_by_default_settings():
    kwargs = _decode_kwargs(vad=True, prompt="Kubernetes, Priya")
    assert kwargs["vad_filter"] is True and kwargs["initial_prompt"] == "Kubernetes, Priya"


def test_decode_uses_the_configured_beam_size():
    assert _decode_kwargs()["beam_size"] == 5
    assert _decode_kwargs(beam_size=1)["beam_size"] == 1


def test_decode_without_prompt_passes_none():
    kwargs = _decode_kwargs(vad=False, prompt="")
    assert kwargs["initial_prompt"] is None and kwargs["vad_filter"] is False


def test_decode_defaults_enable_vad_and_no_prompt():
    kwargs = _decode_kwargs()
    assert kwargs["vad_filter"] is True and kwargs["initial_prompt"] is None


def test_resample_uses_scipy_not_np_interp():
    """convert_to_16khz_array should resample without calling np.interp."""
    from tinysteno.transcriber import convert_to_16khz_array

    data_44k = np.random.rand(44100).astype(np.float32)  # 1 second at 44.1kHz

    with patch("tinysteno.transcriber.np.interp") as mock_interp:
        result = convert_to_16khz_array(data_44k, sr=44100)
        mock_interp.assert_not_called()

    assert result.shape[0] == 16000
    assert result.dtype == np.float32


def test_resample_passthrough_at_16khz():
    """No resampling performed when input is already 16kHz."""
    from tinysteno.transcriber import convert_to_16khz_array

    data = np.ones(16000, dtype=np.float32)
    result = convert_to_16khz_array(data, sr=16000)
    np.testing.assert_array_equal(result, data)


def test_transcribe_writes_no_temp_files(tmp_path):
    """transcribe() should not write any .wav temp files."""
    from tinysteno.transcriber import WhisperTranscriber

    audio = np.zeros(16000, dtype=np.float32)
    wav_path = tmp_path / "test.wav"
    sf.write(str(wav_path), audio, 44100)

    with patch("tinysteno.transcriber.WhisperModel") as mock_wm:
        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter([]), MagicMock(language="en"))
        mock_wm.return_value = mock_model
        t = WhisperTranscriber()
        t.transcribe(str(wav_path))

    assert not list(tmp_path.glob("*.16khz.wav")), "Temp 16khz file should not exist"
    assert not list(tmp_path.glob("*_left.wav")), "Left channel temp file should not exist"
    assert not list(tmp_path.glob("*_right.wav")), "Right channel temp file should not exist"


def test_whisper_model_cached_across_instances():
    """Two WhisperTranscriber instances with same model_size share one WhisperModel."""
    import tinysteno.transcriber as mod

    mod._MODEL_CACHE.clear()

    with patch("tinysteno.transcriber.WhisperModel") as mock_wm:
        mock_wm.return_value = MagicMock()
        t1 = mod.WhisperTranscriber(model_size="tiny")
        t2 = mod.WhisperTranscriber(model_size="tiny")

    assert mock_wm.call_count == 1
    assert t1._model is t2._model


def test_whisper_transcriber_explicit_device():
    """WhisperTranscriber should accept explicit device and compute_type."""
    import tinysteno.transcriber as mod

    mod._MODEL_CACHE.clear()

    with patch("tinysteno.transcriber.WhisperModel") as mock_wm:
        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter([]), MagicMock(language="en"))
        mock_wm.return_value = mock_model
        from tinysteno.transcriber import WhisperTranscriber

        t = WhisperTranscriber(model_size="tiny", device="cpu", compute_type="int8")

    assert t.device == "cpu"
    assert t.compute_type == "int8"


def test_whisper_transcriber_different_devices_separate_cache():
    """Different device/compute_type values should create separate model instances."""
    import tinysteno.transcriber as mod

    mod._MODEL_CACHE.clear()

    with patch("tinysteno.transcriber.WhisperModel") as mock_wm:
        mock_wm.return_value = MagicMock()
        _t1 = mod.WhisperTranscriber(model_size="tiny", device="cpu", compute_type="int8")
        _t2 = mod.WhisperTranscriber(model_size="tiny", device="cuda", compute_type="float16")

    # Two different cache keys → two WhisperModel constructor calls
    assert mock_wm.call_count == 2


def test_detect_device_returns_tuple():
    """_detect_device should return a tuple of two strings."""
    from tinysteno.transcriber import _detect_device

    device, compute_type = _detect_device()
    assert isinstance(device, str)
    assert isinstance(compute_type, str)
    assert device in ("cpu", "cuda", "auto")


def test_stereo_without_diarize_transcribes_both_channels(tmp_path):
    result = _transcribe(tmp_path, left="hello from me", right="hello from them", diarize=False)
    assert result["text"] == "hello from me hello from them"
    assert result["diarised_text"] is None


def test_stereo_with_diarize_interleaves_by_time_in_two_passes(tmp_path):
    result, model = _transcribe(
        tmp_path, left=[(5.0, "second")], right=[(1.0, "first")], diarize=True, return_model=True
    )
    assert result["diarised_text"] == "[Others] first\n[You] second"
    assert model.transcribe.call_count == 2


def test_silent_channel_is_not_decoded(tmp_path):
    result, model = _transcribe(
        tmp_path, left=[(0.0, "only me")], right_silent=True, diarize=True, return_model=True
    )
    assert model.transcribe.call_count == 1 and result["warnings"] == []


def test_quiet_but_not_silent_channel_is_still_decoded(tmp_path):
    _result, model = _transcribe(
        tmp_path,
        left=[(0.0, "me")],
        right=[(1.0, "them")],
        right_level=0.004,
        diarize=True,
        return_model=True,
    )
    assert model.transcribe.call_count == 2


def test_both_channels_silent_gives_no_speech(tmp_path):
    result = _transcribe(tmp_path, left_silent=True, right_silent=True, diarize=True)
    assert result["text"] == "" and result["diarised_text"] is None


def test_mono_with_diarize_is_plain(tmp_path):
    result = _transcribe(tmp_path, mono=[(0.0, "just me")], diarize=True)
    assert result["text"] == "just me" and result["diarised_text"] is None


def test_progress_is_monotonic_across_both_passes(tmp_path):
    seen = []
    _transcribe(
        tmp_path,
        left=[(0.0, "a"), (1.0, "b")],
        right=[(0.5, "c")],
        diarize=True,
        on_progress=seen.append,
    )
    assert seen == sorted(seen) and seen[-1] == 1.0 and any(0.5 <= p < 1.0 for p in seen)
    assert all(0.0 <= p <= 1.0 for p in seen)


def test_single_pass_progress_spans_zero_to_one(tmp_path):
    seen = []
    _transcribe(tmp_path, left=[(0.0, "a"), (1.5, "b")], right_silent=True, on_progress=seen.append)
    assert seen == sorted(seen) and seen[-1] == 1.0 and any(0.5 < p < 1.0 for p in seen)


def test_decode_returns_stripped_segments_and_language():
    import tinysteno.transcriber as mod

    mod._MODEL_CACHE.clear()
    with patch("tinysteno.transcriber.WhisperModel") as mock_wm:
        model = MagicMock()
        model.transcribe.return_value = (
            iter(_segments([(0.5, "hello")]) + _segments([(1.0, "   ")])),
            MagicMock(language="fr"),
        )
        mock_wm.return_value = model
        transcriber = mod.WhisperTranscriber()

    segments, language = transcriber._decode(np.zeros(16000, dtype=np.float32))
    assert [(s.start, s.end, s.text) for s in segments] == [(0.5, 1.0, "hello")]
    assert language == "fr"
    assert isinstance(model.transcribe.call_args[0][0], np.ndarray)


def test_split_channels_mono_and_stereo():
    from tinysteno.transcriber import split_channels

    mono = split_channels(np.ones(16000, dtype=np.float32), 16000)
    assert list(mono) == [""]
    stereo = split_channels(np.ones((16000, 2), dtype=np.float32), 16000)
    assert list(stereo) == ["You", "Others"]


def test_detect_device_uses_cuda_when_a_device_is_present():
    from tinysteno.transcriber import _detect_device
    with patch("ctranslate2.get_cuda_device_count", return_value=1):
        assert _detect_device() == ("cuda", "float16")


def test_detect_device_is_not_cuda_without_devices():
    from tinysteno.transcriber import _detect_device
    with patch("ctranslate2.get_cuda_device_count", return_value=0):
        assert _detect_device()[0] != "cuda"


def test_detect_device_survives_a_failing_cuda_probe():
    from tinysteno.transcriber import _detect_device
    with patch("ctranslate2.get_cuda_device_count", side_effect=RuntimeError("no driver")):
        assert _detect_device()[0] != "cuda"


def _split_progress(tmp_path):
    import tinysteno.transcriber as mod
    from tinysteno.transcript import TranscriptOptions

    data = np.full((16000 * _SECONDS, 2), _LEFT_LEVEL, dtype=np.float32)
    wav_path = tmp_path / "t.wav"
    sf.write(str(wav_path), data, 16000, subtype="FLOAT")
    seen = []

    def fake_assemble(*_args, on_progress=None, **_kwargs):
        on_progress(0.5)
        return {}

    mod._MODEL_CACHE.clear()
    with patch("tinysteno.transcriber.WhisperModel") as mock_wm, \
         patch("tinysteno.transcriber.assemble", side_effect=fake_assemble):
        mock_wm.return_value.transcribe.side_effect = (
            lambda *_a, **_k: (iter(_segments("a")), MagicMock(language="en")))
        mod.WhisperTranscriber(options=TranscriptOptions(speaker_split=True)).transcribe(
            str(wav_path), diarize=True, on_progress=seen.append)
    return seen


def test_split_progress_decodes_in_the_first_70_percent_then_the_diarizer_fills_the_rest(
        tmp_path):
    seen = _split_progress(tmp_path)
    assert seen == sorted(seen) and seen[-1] == 1.0
    assert 0.7 in seen and 0.85 in seen
    assert max(v for v in seen[: seen.index(0.7) + 1]) == 0.7


def test_split_switched_on_without_labels_leaves_progress_as_it_is(tmp_path):
    import tinysteno.transcriber as mod
    from tinysteno.transcript import TranscriptOptions

    def run(options):
        seen = []
        mod._MODEL_CACHE.clear()
        data = np.full((16000 * _SECONDS, 2), _LEFT_LEVEL, dtype=np.float32)
        wav_path = tmp_path / "p.wav"
        sf.write(str(wav_path), data, 16000, subtype="FLOAT")
        with patch("tinysteno.transcriber.WhisperModel") as mock_wm:
            mock_wm.return_value.transcribe.side_effect = (
                lambda *_a, **_k: (iter(_segments("a")), MagicMock(language="en")))
            mod.WhisperTranscriber(options=options).transcribe(
                str(wav_path), diarize=False, on_progress=seen.append)
        return seen

    assert run(TranscriptOptions(speaker_split=True)) == run(None)


# --- splitting segments at pauses between words ---

# The words of one segment faster-whisper returned for a real recording's microphone channel
# (VAD joined three stretches of speech): "Okay," at 25.4 s, the question two minutes later.
_SPANNING = (25.4, 182.1, "Okay, so I guess where does my team come in? Let me just", [
    (25.4, 25.9, " Okay,"), (166.0, 166.2, " so"), (166.2, 166.3, " I"),
    (166.3, 166.4, " guess"), (166.4, 166.9, " where"), (166.9, 167.5, " does"),
    (167.5, 167.8, " my"), (167.8, 168.0, " team"), (168.0, 168.2, " come"),
    (168.2, 168.6, " in?"), (181.2, 181.7, " Let"), (181.7, 181.8, " me"),
    (181.8, 182.1, " just"),
])


def test_a_segment_is_split_at_each_long_pause_and_timed_from_its_words():
    from tinysteno.transcriber import split_at_word_gaps
    from tinysteno.transcript import Segment

    assert split_at_word_gaps(*_SPANNING) == [
        Segment(start=25.4, end=25.9, text="Okay,"),
        Segment(start=166.0, end=168.6, text="so I guess where does my team come in?"),
        Segment(start=181.2, end=182.1, text="Let me just"),
    ]


def test_words_without_a_long_pause_stay_one_segment_timed_from_the_words():
    from tinysteno.transcriber import split_at_word_gaps
    from tinysteno.transcript import Segment

    words = [(12.0, 12.4, " We"), (13.9, 14.2, " met.")]  # a 1.5 s pause is not a split
    assert split_at_word_gaps(10.0, 15.0, "We met.", words) == [
        Segment(start=12.0, end=14.2, text="We met.")]


def test_a_segment_without_words_is_kept_as_given():
    from tinysteno.transcriber import split_at_word_gaps
    from tinysteno.transcript import Segment

    assert split_at_word_gaps(1.0, 2.0, " hi ", []) == [Segment(start=1.0, end=2.0, text="hi")]


def test_decode_asks_for_word_timestamps_and_splits_at_pauses():
    t = _make_transcriber()
    start, end, text, words = _SPANNING
    segment = MagicMock(start=start, end=end, text=f" {text}",
                        words=[MagicMock(start=s, end=e, word=w) for s, e, w in words])
    t._model.transcribe.return_value = (iter([segment]), MagicMock(language="en"))
    segments, _ = t._decode(np.zeros(16000, dtype=np.float32), None)
    assert t._model.transcribe.call_args.kwargs["word_timestamps"] is True
    assert [(s.start, s.text) for s in segments] == [
        (25.4, "Okay,"), (166.0, "so I guess where does my team come in?"), (181.2, "Let me just")]
