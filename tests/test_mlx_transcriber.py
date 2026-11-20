"""Tests for MlxWhisperTranscriber: mlx_whisper is never installed in CI, so it is faked."""
# pylint: disable=protected-access
import importlib
import sys
import types

import numpy as np
import pytest
import soundfile as sf

from tinysteno.transcript import Segment

_LEVEL = 0.1


@pytest.fixture(name="fake_mlx")
def fake_mlx_fixture(monkeypatch):
    """Install a fake mlx_whisper module; return the list of (audio, kwargs) calls it saw."""
    calls = []

    def transcribe(audio, **kwargs):
        calls.append((audio, kwargs))
        return {
            "text": " hi ",
            "segments": [
                {"start": 0.0, "end": 0.5, "text": " hi "},
                {"start": 0.5, "end": 0.9, "text": "   "},
            ],
            "language": "en",
        }

    monkeypatch.setitem(sys.modules, "mlx_whisper",
                        types.SimpleNamespace(transcribe=transcribe))
    return calls


def _make(model_size="small", **kwargs):
    from tinysteno.mlx_transcriber import MlxWhisperTranscriber
    return MlxWhisperTranscriber(model_size, **kwargs)


def test_decode_adapts_mlx_result_to_segments(fake_mlx):
    from tinysteno.mlx_transcriber import _MLX_MODELS
    segments, language = _make()._decode(np.zeros(16000, dtype=np.float32))
    assert segments == [Segment(start=0.0, end=0.5, text="hi")]
    assert language == "en"
    _, kwargs = fake_mlx[0]
    assert kwargs["path_or_hf_repo"] == _MLX_MODELS["small"]
    assert kwargs["initial_prompt"] is None


def test_decode_reports_progress_complete(fake_mlx):
    seen = []
    _make()._decode(np.zeros(16000, dtype=np.float32), seen.append)
    assert fake_mlx and seen[-1] == 1.0


@pytest.mark.parametrize("size", ["tiny", "base", "small", "medium"])
def test_known_sizes_map_to_mlx_community_repos(size):
    assert _make(size).model == f"mlx-community/whisper-{size}-mlx"


def test_large_v3_maps_to_its_repo():
    assert _make("large-v3").model == "mlx-community/whisper-large-v3-mlx"


@pytest.mark.parametrize("name", ["mlx-community/whisper-large-v3-turbo", "my-model"])
def test_unknown_model_names_pass_through(name):
    assert _make(name).model == name


def test_prompt_forwarded(fake_mlx):
    _make(prompt="Acme")._decode(np.zeros(16000, dtype=np.float32))
    assert fake_mlx[0][1]["initial_prompt"] == "Acme"


def test_dual_channel_skips_silent_channel(fake_mlx, tmp_path):
    frames = 16000 * 2
    data = np.stack([np.full(frames, _LEVEL, dtype=np.float32),
                     np.zeros(frames, dtype=np.float32)], axis=1)
    wav = tmp_path / "t.wav"
    sf.write(str(wav), data, 16000, subtype="FLOAT")
    result = _make().transcribe(str(wav))
    assert len(fake_mlx) == 1
    assert np.isclose(float(np.mean(fake_mlx[0][0])), _LEVEL)
    assert "hi" in result["text"]
    assert isinstance(result["warnings"], list)


def test_mlx_available_true_with_module(fake_mlx):
    from tinysteno.mlx_transcriber import mlx_available
    assert fake_mlx == [] and mlx_available() is True


def test_mlx_available_false_without_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "mlx_whisper", None)  # import raises ImportError
    from tinysteno.mlx_transcriber import mlx_available
    assert mlx_available() is False


def test_module_imports_without_mlx_whisper(monkeypatch):
    monkeypatch.setitem(sys.modules, "mlx_whisper", None)
    monkeypatch.delitem(sys.modules, "tinysteno.mlx_transcriber", raising=False)
    module = importlib.import_module("tinysteno.mlx_transcriber")
    assert module.MlxWhisperTranscriber("tiny").model == "mlx-community/whisper-tiny-mlx"
