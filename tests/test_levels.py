"""The recorded channel levels: the recorder's gains file and how transcription uses it."""
# pylint: disable=protected-access
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from tinysteno.levels import levels_path, load_levels, save_levels
from tinysteno.recorder import AudioRecorder, _to_int16
from tinysteno.transcript import OTHERS, YOU, Segment, assemble

SENTENCE = "we should move the launch to friday"
MIC_MARK, SYSTEM_MARK = 0.9, 0.4  # normalised levels in the WAV, also used to tell channels apart


# --- the recorder's side -------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _no_real_system_capture(tmp_path, monkeypatch):
    """Never start real ScreenCaptureKit capture; raw temp files go to the test directory."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(raw_dir))
    with patch.object(AudioRecorder, "_start_macos_loopback", return_value=False):
        yield


def test_to_int16_reports_the_gain_applied_to_each_channel():
    audio = np.stack([np.full(100, 0.5, dtype=np.float32),
                      np.full(100, 0.001, dtype=np.float32)], axis=1)
    out, gains = _to_int16(audio)
    assert out[0, 0] == 32767 and out[0, 1] == int(0.001 * 32767)
    assert gains == [pytest.approx(2.0), 1.0]  # below the floor: not scaled


def _record(tmp_path, mic_level, system_level=None):
    rec = AudioRecorder(sample_rate=16000, channels=1, recordings_dir=tmp_path / "rec")
    with patch("tinysteno.recorder.sd") as mock_sd:
        mock_sd.query_devices.return_value = {"index": 0, "max_input_channels": 1}
        rec.start()
        rec._write_mic_frame(np.full((1024, 1), mic_level, dtype=np.float32))
        if system_level is not None:
            rec._has_loopback = True
            rec._write_loopback_frame(np.full(1024, system_level, dtype=np.float32))
        assert rec.stop() is True
    return rec.output_path


def test_stop_writes_the_gains_beside_the_wav(tmp_path):
    wav = _record(tmp_path, 0.25, system_level=0.5)
    path = Path(str(wav) + ".levels.json")
    assert levels_path(wav) == path
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data == {"version": 1, "gains": [pytest.approx(4.0), pytest.approx(2.0)]}
    mono = _record(tmp_path / "mono", 0.5)
    assert load_levels(str(mono), 1) == [pytest.approx(2.0)]


def test_a_failure_writing_the_levels_file_does_not_fail_the_save(tmp_path, caplog):
    with patch("tinysteno.levels.os.replace", side_effect=OSError("read-only")):
        wav = _record(tmp_path, 0.25)
    assert wav.exists() and sf.info(str(wav)).frames == 1024
    assert not levels_path(wav).exists() and "read-only" in caplog.text
    assert [p.name for p in wav.parent.iterdir()] == [wav.name]  # no temp file left


# --- reading it back -----------------------------------------------------------------------

def _wav(tmp_path):
    path = tmp_path / "Meeting-1.wav"
    path.write_bytes(b"RIFF")
    return str(path)


def test_save_then_load_round_trips(tmp_path):
    wav = _wav(tmp_path)
    save_levels(wav, [2.0, 250.0])
    assert load_levels(wav, 2) == [2.0, 250.0]


@pytest.mark.parametrize("text", [
    '{"version": 2, "gains": [1.0, 1.0]}',
    '{"version": 1, "gains": [1.0]}',
    '{"version": 1, "gains": [1.0, 0.0]}',
    '{"version": 1, "gains": [1.0, -2.0]}',
    '{"version": 1, "gains": [1.0, NaN]}',
    '{"version": 1, "gains": [1.0, Infinity]}',
    '{"version": 1, "gains": [1.0, "2"]}',
    '{"version": 1, "gains": [1.0, true]}',
    '{"version": 1, "gains": "1.0"}',
    '{"version": 1}',
    '[1.0, 1.0]',
    '{"version": 1, "gains": [1.0, 1.0]',
])
def test_an_invalid_levels_file_is_ignored(tmp_path, text):
    wav = _wav(tmp_path)
    levels_path(wav).write_text(text, encoding="utf-8")
    assert load_levels(wav, 2) is None


def test_a_missing_or_older_levels_file_is_ignored(tmp_path):
    wav = _wav(tmp_path)
    assert load_levels(wav, 2) is None
    save_levels(wav, [1.0, 1.0])
    os.utime(levels_path(wav), (1, 1))
    assert load_levels(wav, 2) is None


# --- transcription with the recorded levels ------------------------------------------------

def _stereo_wav(tmp_path, mic, system, gains=None):
    frames = 16000 * 4
    data = np.stack([np.full(frames, mic, dtype=np.float32),
                     np.full(frames, system, dtype=np.float32)], axis=1)
    path = tmp_path / "call.wav"
    sf.write(str(path), data, 16000, subtype="FLOAT")
    if gains is not None:
        save_levels(str(path), gains)
    return str(path)


def _local(path, mic_segments, system_segments):
    import tinysteno.transcriber as mod

    def fake_transcribe(audio, **_kwargs):
        spec = system_segments if np.isclose(np.mean(audio), SYSTEM_MARK) else mic_segments
        segs = [MagicMock(start=start, end=start + 2.0, text=text) for start, text in spec]
        return iter(segs), MagicMock(language="en")

    mod._MODEL_CACHE.clear()
    with patch("tinysteno.transcriber.WhisperModel") as mock_wm:
        mock_wm.return_value.transcribe.side_effect = fake_transcribe
        result = mod.WhisperTranscriber().transcribe(path, diarize=True)
        return result, mock_wm.return_value.transcribe.call_count


# Listen-only call on speakers: system 0.2 normalised to 0.4 (gain 2), the microphone holds
# only the bleed, 47x quieter (0.2 / 47), normalised to 0.9 (gain 0.9 * 47 / 0.2).
LISTEN_ONLY_GAINS = [MIC_MARK * 47 / 0.2, SYSTEM_MARK / 0.2]


def test_listen_only_bleed_keeps_the_system_line_when_the_levels_are_known(tmp_path):
    path = _stereo_wav(tmp_path, MIC_MARK, SYSTEM_MARK, LISTEN_ONLY_GAINS)
    result, _ = _local(path, [(1.1, SENTENCE)], [(1.0, SENTENCE)])
    assert [(s.speaker, s.text) for s in result["segments"]] == [(OTHERS, SENTENCE)]


def test_without_the_levels_file_the_normalised_levels_decide_as_before(tmp_path):
    path = _stereo_wav(tmp_path, MIC_MARK, SYSTEM_MARK)
    result, _ = _local(path, [(1.1, SENTENCE)], [(1.0, SENTENCE)])
    assert [(s.speaker, s.text) for s in result["segments"]] == [(YOU, SENTENCE)]


QUIET_MIC = 0.125  # normalised; 0.0005 as recorded with a gain of 250, under SILENCE_RMS


def test_a_quiet_live_microphone_is_not_decoded_when_the_levels_are_known(tmp_path):
    path = _stereo_wav(tmp_path, QUIET_MIC, SYSTEM_MARK, [250.0, 2.0])
    result, calls = _local(path, [], [(1.0, "hello from them")])
    assert calls == 1 and result["warnings"] == []


@pytest.mark.parametrize("gains", [None, [250.0], [250.0, 0.0]])
def test_missing_or_invalid_levels_decode_the_quiet_microphone_as_before(tmp_path, gains):
    path = _stereo_wav(tmp_path, QUIET_MIC, SYSTEM_MARK)
    if gains is not None:
        levels_path(path).write_text(json.dumps({"version": 1, "gains": gains}), "utf-8")
    result, calls = _local(path, [], [(1.0, "hello from them")])
    assert calls == 2
    assert result["warnings"] == ["Microphone audio was present but no speech was recognised."]


def test_the_api_backend_uses_the_recorded_levels_too(tmp_path):
    from tinysteno.api_transcriber import ApiTranscriber

    path = _stereo_wav(tmp_path, QUIET_MIC, SYSTEM_MARK, [250.0, 2.0])
    reply = SimpleNamespace(text="them", language="en", segments=[
        SimpleNamespace(start=1.0, end=2.0, text="hello from them")])
    with patch("tinysteno.api_transcriber.OpenAI") as mock_class:
        client = mock_class.return_value
        client.audio.transcriptions.create.return_value = reply
        result = ApiTranscriber(api_key="k").transcribe(path, diarize=True)
    assert client.audio.transcriptions.create.call_count == 1
    assert result["warnings"] == []


def test_assemble_measures_bleed_and_empty_channels_at_the_recorded_level():
    audio = {YOU: np.full(64000, MIC_MARK, dtype=np.float32),
             OTHERS: np.full(64000, SYSTEM_MARK, dtype=np.float32)}
    channels = {YOU: [Segment(1.1, 3.0, SENTENCE)], OTHERS: [Segment(1.0, 3.0, SENTENCE)]}
    gains = dict(zip((YOU, OTHERS), LISTEN_ONLY_GAINS))
    result = assemble(channels, audio, diarize=True, language="en", gains=gains)
    assert [s.speaker for s in result["segments"]] == [OTHERS]
    quiet = {YOU: np.full(64000, QUIET_MIC, dtype=np.float32), OTHERS: audio[OTHERS]}
    empty_mic = {YOU: [], OTHERS: [Segment(1.0, 3.0, SENTENCE)]}
    assert assemble(empty_mic, quiet, diarize=True, language="en",
                    gains={YOU: 250.0, OTHERS: 2.0})["warnings"] == []
    assert assemble(empty_mic, quiet, diarize=True, language="en")["warnings"] == [
        "Microphone audio was present but no speech was recognised."]


def test_one_non_finite_sample_does_not_zero_a_channel():
    column = np.array([0.25, np.inf, -np.inf, np.nan, 0.5], dtype=np.float32)
    out, gains = _to_int16(column.reshape(-1, 1))
    assert out[:, 0].tolist() == [int(0.25 * 32767), 32767, -32767, 0, int(0.5 * 32767)]
    assert gains == [1.0]
