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


def _make_transcriber(model_size="tiny"):
    with patch("tinysteno.transcriber.WhisperModel") as mock_wm:
        mock_wm.return_value = MagicMock()
        from tinysteno.transcriber import WhisperTranscriber

        t = WhisperTranscriber(model_size=model_size)
    return t


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
