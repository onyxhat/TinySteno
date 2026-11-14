"""Tests for AudioRecorder streaming behavior."""
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tinysteno.recorder import AudioRecorder

# The real method, captured before the autouse fixture below replaces it.
_REAL_START_MACOS_LOOPBACK = AudioRecorder._start_macos_loopback  # pylint: disable=protected-access


@pytest.fixture(autouse=True)
def _no_real_system_capture():
    """Never start real ScreenCaptureKit capture (macOS) from these tests."""
    with patch.object(AudioRecorder, "_start_macos_loopback", return_value=False):
        yield


def _stub_input_device(mock_sd) -> None:
    mock_sd.query_devices.return_value = {"index": 0, "max_input_channels": 1}
    mock_sd.InputStream.return_value.start = MagicMock()
    mock_sd.InputStream.return_value.stop = MagicMock()
    mock_sd.InputStream.return_value.close = MagicMock()


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
        assert not rec._buffer, "Buffer should be empty; data streams to file"  # pylint: disable=protected-access
        rec.discard()


def test_stop_produces_valid_wav(tmp_path):
    """stop() should write a readable WAV file with correct properties."""
    rec = _make_recorder(tmp_path)
    fake_chunk = np.zeros((1024, 1), dtype=np.float32)

    with patch("tinysteno.recorder.sd") as mock_sd:
        _stub_input_device(mock_sd)
        rec.start()
        for _ in range(10):
            rec._write_mic_frame(fake_chunk)  # pylint: disable=protected-access
        rec.stop()

    assert rec.output_path is not None
    assert rec.output_path.exists()
    with wave.open(str(rec.output_path), "rb") as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getframerate() == 44100


def test_mic_level_tracks_rms_of_latest_mic_frame(tmp_path):
    rec = _make_recorder(tmp_path)
    with patch("tinysteno.recorder.sd") as mock_sd:
        _stub_input_device(mock_sd)
        rec.start()
        assert rec.mic_level == 0.0
        rec._write_mic_frame(np.full((1024, 1), 0.5, dtype=np.float32))  # pylint: disable=protected-access
        assert rec.mic_level == pytest.approx(0.5, abs=1e-3)
        rec._write_mic_frame(np.full((1024, 1), 4.0, dtype=np.float32))  # pylint: disable=protected-access
        assert rec.mic_level == 1.0
        rec.stop()
    assert rec.mic_level == 0.0


@pytest.mark.parametrize("feed", ["_loopback_callback", "_write_loopback_frame"])
def test_system_level_tracks_both_loopback_paths_independently_of_mic(tmp_path, feed):
    rec = _make_recorder(tmp_path)
    frame = np.full((1024, 1), 0.25, dtype=np.float32)
    with patch("tinysteno.recorder.sd") as mock_sd:
        _stub_input_device(mock_sd)
        rec.start()
        assert rec.system_level == 0.0
        if feed == "_loopback_callback":
            rec._loopback_callback(frame, 1024, None, None)  # pylint: disable=protected-access
        else:
            rec._write_loopback_frame(frame)  # pylint: disable=protected-access
        assert rec.system_level == pytest.approx(0.25, abs=1e-3)
        assert rec.mic_level == 0.0
        rec.stop()
    assert rec.system_level == 0.0


def test_level_handles_multichannel_sequence_and_empty_frames(tmp_path):
    rec = _make_recorder(tmp_path)
    with patch("tinysteno.recorder.sd") as mock_sd:
        _stub_input_device(mock_sd)
        rec.start()
        rec._write_mic_frame(np.full((8, 2), 0.5, dtype=np.float32))  # pylint: disable=protected-access
        assert rec.mic_level == pytest.approx(0.5, abs=1e-3)
        rec._write_loopback_frame([0.2] * 16)  # pylint: disable=protected-access
        assert rec.system_level == pytest.approx(0.2, abs=1e-3)
        rec._write_mic_frame(np.zeros((0, 1), dtype=np.float32))  # pylint: disable=protected-access
        assert rec.mic_level == 0.0
        rec.stop()


def test_discard_writes_no_wav_and_removes_temp_files(tmp_path):
    rec = _make_recorder(tmp_path)
    with patch("tinysteno.recorder.sd") as mock_sd:
        _stub_input_device(mock_sd)
        wav_path = rec.start()
        rec._write_mic_frame(np.full((1024, 1), 0.5, dtype=np.float32))  # pylint: disable=protected-access
        temp_files = [rec._mic_raw_path, rec._loopback_raw_path]  # pylint: disable=protected-access
        rec.discard()
    assert not Path(wav_path).exists()
    assert not any(p.exists() for p in temp_files)
    assert rec.mic_level == 0.0
    assert rec.stop() is False


def test_discard_when_not_recording_is_a_no_op(tmp_path):
    _make_recorder(tmp_path).discard()


def test_failed_wav_write_keeps_the_raw_files_and_names_them(tmp_path):
    rec = _make_recorder(tmp_path)
    with patch("tinysteno.recorder.sd") as mock_sd:
        _stub_input_device(mock_sd)
        rec.start()
        rec._write_mic_frame(np.full((1024, 1), 0.5, dtype=np.float32))  # pylint: disable=protected-access
        raw_files = [rec._mic_raw_path, rec._loopback_raw_path]  # pylint: disable=protected-access
        with patch("tinysteno.recorder.wave.open", side_effect=OSError("disk full")):
            with pytest.raises(RuntimeError) as exc:
                rec.stop()
    try:
        assert all(p.exists() for p in raw_files)
        assert rec.raw_paths == tuple(raw_files)
        message = str(exc.value)
        assert "disk full" in message
        assert all(str(p) in message for p in raw_files)
        assert rec.mic_level == 0.0 and rec.system_level == 0.0
    finally:
        for path in raw_files:
            path.unlink(missing_ok=True)


def test_successful_stop_deletes_the_raw_files(tmp_path):
    rec = _make_recorder(tmp_path)
    with patch("tinysteno.recorder.sd") as mock_sd:
        _stub_input_device(mock_sd)
        rec.start()
        rec._write_mic_frame(np.full((1024, 1), 0.5, dtype=np.float32))  # pylint: disable=protected-access
        raw_files = [rec._mic_raw_path, rec._loopback_raw_path]  # pylint: disable=protected-access
        assert rec.stop() is True
    assert not any(p.exists() for p in raw_files)
    assert not rec.raw_paths


def test_stop_with_no_audio_written_returns_false_and_resets_levels(tmp_path):
    rec = _make_recorder(tmp_path)
    with patch("tinysteno.recorder.sd") as mock_sd:
        _stub_input_device(mock_sd)
        rec.start()
        rec._mic_level, rec._system_level = 0.4, 0.2  # pylint: disable=protected-access
        raw_files = [rec._mic_raw_path, rec._loopback_raw_path]  # pylint: disable=protected-access
        assert rec.stop() is False
    assert rec.mic_level == 0.0 and rec.system_level == 0.0
    assert not any(p.exists() for p in raw_files)
    assert rec.stop() is False  # already stopped


def test_failed_stop_resets_levels_even_before_any_audio_is_read(tmp_path):
    rec = _make_recorder(tmp_path)
    with patch("tinysteno.recorder.sd") as mock_sd:
        _stub_input_device(mock_sd)
        rec.start()
        rec._write_mic_frame(np.full((1024, 1), 0.5, dtype=np.float32))  # pylint: disable=protected-access
        raw_files = [rec._mic_raw_path, rec._loopback_raw_path]  # pylint: disable=protected-access
        mock_sd.InputStream.return_value.stop.side_effect = OSError("device gone")
        with pytest.raises(RuntimeError, match="device gone"):
            rec.stop()
    try:
        assert rec.mic_level == 0.0 and rec.system_level == 0.0
        assert rec.raw_paths == tuple(raw_files)
    finally:
        for path in raw_files:
            path.unlink(missing_ok=True)


@pytest.mark.parametrize("captured", [True, False])
def test_has_loopback_reflects_whether_system_audio_started(tmp_path, captured):
    rec = _make_recorder(tmp_path)
    with patch("tinysteno.recorder.sd") as mock_sd, \
         patch("tinysteno.recorder.platform.system", return_value="Darwin"), \
         patch.object(AudioRecorder, "_start_macos_loopback", return_value=captured):
        _stub_input_device(mock_sd)
        assert rec.has_loopback is False
        rec.start()
        assert rec.has_loopback is captured
        rec.discard()


def test_macos_loopback_failure_reason_is_kept(tmp_path, capsys):
    class Broken:  # pylint: disable=too-few-public-methods
        """Stands in for MacOSLoopback; start() fails like a missing permission."""
        def __init__(self, **_kwargs):
            pass

        def start(self):
            """Fail to start."""
            raise RuntimeError("Screen Recording permission denied")

    rec = _make_recorder(tmp_path)
    with patch.dict("sys.modules", {"tinysteno._macos_loopback": MagicMock(MacOSLoopback=Broken)}):
        assert _REAL_START_MACOS_LOOPBACK(rec) is False
    assert rec.loopback_error == "Screen Recording permission denied"
    assert "Screen Recording permission denied" in capsys.readouterr().out
