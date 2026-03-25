"""Tests for AudioRecorder streaming behavior."""
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from tinysteno.recorder import AudioRecorder


def _make_recorder(tmp_path) -> AudioRecorder:
    return AudioRecorder(sample_rate=44100, channels=1, recordings_dir=tmp_path)


def test_buffer_not_used_during_recording(tmp_path):
    """After start(), _buffer should remain empty (data goes to file)."""
    rec = _make_recorder(tmp_path)
    with patch("tinysteno.recorder.sd") as mock_sd:
        mock_sd.query_devices.return_value = {"index": 0, "max_input_channels": 1}
        mock_sd.InputStream.return_value.__enter__ = lambda s: s
        mock_sd.InputStream.return_value.__exit__ = MagicMock(return_value=False)
        mock_sd.InputStream.return_value.start = MagicMock()
        rec.start()
    assert rec._buffer == [], "Buffer should be empty; data streams to file"


def test_stop_produces_valid_wav(tmp_path):
    """stop() should write a readable WAV file with correct properties."""
    rec = _make_recorder(tmp_path)
    fake_chunk = np.zeros((1024, 1), dtype=np.float32)

    with patch("tinysteno.recorder.sd") as mock_sd:
        mock_sd.query_devices.return_value = {"index": 0, "max_input_channels": 1}
        mock_sd.InputStream.return_value.start = MagicMock()
        mock_sd.InputStream.return_value.stop = MagicMock()
        mock_sd.InputStream.return_value.close = MagicMock()
        rec.start()
        for _ in range(10):
            rec._write_mic_frame(fake_chunk)
        rec.stop()

    assert rec.output_path is not None
    assert rec.output_path.exists()
    with wave.open(str(rec.output_path), "rb") as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getframerate() == 44100
