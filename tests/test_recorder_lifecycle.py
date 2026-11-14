"""AudioRecorder lifecycle: every way start() and stop() can go leaves it in a known state."""
# pylint: disable=redefined-outer-name,protected-access,unused-argument
import tempfile
import wave
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tinysteno.recorder import AudioRecorder

CHUNK = np.full((1024, 1), 0.25, dtype=np.float32)


@pytest.fixture(autouse=True)
def raw_dir(tmp_path, monkeypatch):
    """Raw audio temp files land in a per-test directory instead of the system one."""
    directory = tmp_path / "raw"
    directory.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(directory))
    return directory


@pytest.fixture
def sd():
    """A stand-in sounddevice with one mono input device; no real audio, no real capture."""
    with patch("tinysteno.recorder.sd") as mock_sd, \
         patch.object(AudioRecorder, "_start_macos_loopback", return_value=False):
        mock_sd.PortAudioError = type("PortAudioError", (Exception,), {})
        mock_sd.query_devices.return_value = {"index": 0, "max_input_channels": 1}
        mock_sd.InputStream.side_effect = lambda **_kwargs: MagicMock()
        yield mock_sd


@pytest.fixture
def rec(tmp_path):
    return AudioRecorder(sample_rate=44100, channels=1, recordings_dir=tmp_path / "out")


def _frames(path) -> tuple[int, int]:
    with wave.open(str(path), "rb") as wav:
        return wav.getnframes(), wav.getnchannels()


def test_start_failure_before_any_stream_leaves_the_recorder_idle(rec, sd, raw_dir):
    sd.query_devices.return_value = {"index": 0, "max_input_channels": 0}
    with pytest.raises(RuntimeError, match="No audio input devices found"):
        rec.start()
    assert rec.stop() is False          # nothing was started: nothing to save, nothing to raise
    assert not list(raw_dir.iterdir())


def test_start_failure_after_the_mic_opened_stops_the_mic_and_cleans_up(rec, sd, raw_dir):
    streams = []

    def open_stream(**_kwargs):
        streams.append(MagicMock())
        return streams[-1]

    sd.InputStream.side_effect = open_stream
    with patch("tinysteno.recorder.platform.system", return_value="Linux"), \
         patch.object(AudioRecorder, "_find_loopback_device", side_effect=OSError("no pulse")):
        with pytest.raises(RuntimeError, match="no pulse"):
            rec.start()
    assert len(streams) == 1
    streams[0].stop.assert_called_once()     # the microphone is no longer being captured
    streams[0].close.assert_called_once()
    assert not list(raw_dir.iterdir())       # and no raw audio is left behind
    assert rec.stop() is False


def test_unexpected_error_opening_the_mic_cleans_up(rec, sd, raw_dir):
    sd.InputStream.side_effect = ValueError("bad sample rate")
    with pytest.raises(RuntimeError, match="bad sample rate"):
        rec.start()
    assert not list(raw_dir.iterdir())
    assert rec.stop() is False


def test_starting_twice_is_refused_and_the_first_recording_survives(rec, sd):
    rec.start()
    rec._write_mic_frame(CHUNK)
    with pytest.raises(RuntimeError, match="already recording"):
        rec.start()
    assert rec.stop() is True
    assert _frames(rec.output_path) == (1024, 1)


def test_a_failed_start_can_be_followed_by_a_successful_one(rec, sd):
    sd.InputStream.side_effect = ValueError("device busy")
    with pytest.raises(RuntimeError):
        rec.start()
    sd.InputStream.side_effect = lambda **_kwargs: MagicMock()
    rec.start()
    rec._write_mic_frame(CHUNK)
    assert rec.stop() is True


def test_system_audio_is_kept_when_the_mic_captured_nothing(rec, sd):
    rec.start()
    rec._has_loopback = True
    for _ in range(3):
        rec._write_loopback_frame(CHUNK)
    assert rec.stop() is True
    assert _frames(rec.output_path) == (3 * 1024, 2)


def test_the_wav_is_not_cut_short_when_system_audio_ends_early(rec, sd):
    rec.start()
    rec._has_loopback = True
    for _ in range(10):
        rec._write_mic_frame(CHUNK)
    for _ in range(2):                       # system capture died two chunks in
        rec._write_loopback_frame(CHUNK)
    assert rec.stop() is True
    assert _frames(rec.output_path) == (10 * 1024, 2)


def test_stop_after_a_failed_save_does_not_claim_an_empty_recording(rec, sd):
    rec.start()
    rec._write_mic_frame(CHUNK)
    with patch("tinysteno.recorder.wave.open", side_effect=OSError("disk full")):
        with pytest.raises(RuntimeError, match="disk full"):
            rec.stop()
    kept = rec.raw_paths
    assert kept and all(p.exists() for p in kept)
    with pytest.raises(RuntimeError, match="already failed"):
        rec.stop()                           # not False: False means "no audio was captured"
    assert rec.raw_paths == kept             # and the kept audio is still there
    rec.discard()                            # an explicit discard is what removes it
    assert not any(p.exists() for p in kept)
