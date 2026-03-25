"""Tests for WhisperTranscriber."""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch


def _make_transcriber(model_size="tiny"):
    with patch("tinysteno.transcriber.WhisperModel") as mock_wm:
        mock_wm.return_value = MagicMock()
        from tinysteno.transcriber import WhisperTranscriber
        t = WhisperTranscriber(model_size=model_size)
    return t


def test_resample_uses_scipy_not_np_interp():
    """_convert_to_16khz_array should resample without calling np.interp."""
    import tinysteno.transcriber as mod

    t = _make_transcriber()
    data_44k = np.random.rand(44100).astype(np.float32)  # 1 second at 44.1kHz

    with patch("tinysteno.transcriber.np.interp") as mock_interp:
        result = t._convert_to_16khz_array(data_44k, sr=44100)
        mock_interp.assert_not_called()

    assert result.shape[0] == 16000
    assert result.dtype == np.float32


def test_resample_passthrough_at_16khz():
    """No resampling performed when input is already 16kHz."""
    t = _make_transcriber()
    data = np.ones(16000, dtype=np.float32)
    result = t._convert_to_16khz_array(data, sr=16000)
    np.testing.assert_array_equal(result, data)


def test_transcribe_writes_no_temp_files(tmp_path):
    """transcribe() should not write any .wav temp files."""
    import soundfile as sf
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


def test_run_whisper_accepts_array():
    """_run_whisper should accept numpy array, not require a file path."""
    import tinysteno.transcriber as mod
    from tinysteno.transcriber import WhisperTranscriber

    mod._MODEL_CACHE.clear()

    with patch("tinysteno.transcriber.WhisperModel") as mock_wm:
        mock_model = MagicMock()
        seg = MagicMock()
        seg.text = " hello"
        mock_model.transcribe.return_value = (iter([seg]), MagicMock(language="en"))
        mock_wm.return_value = mock_model
        t = WhisperTranscriber()

    audio = np.zeros(16000, dtype=np.float32)
    text, lang = t._run_whisper(audio)
    assert text == " hello"
    assert lang == "en"
    call_args = mock_model.transcribe.call_args
    assert isinstance(call_args[0][0], np.ndarray)


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


def test_transcribe_calls_progress_callback(tmp_path):
    """on_progress callback should be called with values between 0.0 and 1.0."""
    import soundfile as sf
    from unittest.mock import patch, MagicMock
    import tinysteno.transcriber as mod

    audio = np.zeros(32000, dtype=np.float32)  # 2 seconds at 16kHz
    wav_path = tmp_path / "t.wav"
    sf.write(str(wav_path), audio, 16000)

    seg1 = MagicMock()
    seg1.text = " hello"
    seg1.start = 0.5

    seg2 = MagicMock()
    seg2.text = " world"
    seg2.start = 1.5

    progress_values = []

    mod._MODEL_CACHE.clear()
    with patch("tinysteno.transcriber.WhisperModel") as mock_wm:
        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter([seg1, seg2]), MagicMock(language="en"))
        mock_wm.return_value = mock_model
        t = mod.WhisperTranscriber()

    t.transcribe(str(wav_path), on_progress=lambda r: progress_values.append(r))

    assert len(progress_values) >= 1
    assert all(0.0 <= v <= 1.0 for v in progress_values)
