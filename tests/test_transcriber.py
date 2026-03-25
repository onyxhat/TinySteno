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
