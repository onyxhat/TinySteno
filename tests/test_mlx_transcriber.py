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
    segments, language = _make()._decode(np.zeros(16000, dtype=np.float32))
    assert segments == [Segment(start=0.0, end=0.5, text="hi")]
    assert language == "en"
    _, kwargs = fake_mlx[0]
    assert kwargs["path_or_hf_repo"] == "mlx-community/whisper-small-mlx"
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


@pytest.mark.parametrize("name", ["mlx-community/whisper-large-v3-turbo", "me/my-model"])
def test_repo_ids_pass_through(name):
    assert _make(name).model == name


@pytest.mark.parametrize("name, repo", [
    ("large", "mlx-community/whisper-large-v3-mlx"),
    ("turbo", "mlx-community/whisper-large-v3-turbo"),
    ("large-v3-turbo", "mlx-community/whisper-large-v3-turbo"),
    ("large-v2", "mlx-community/whisper-large-v2-mlx"),
    ("small.en", "mlx-community/whisper-small.en-mlx"),
    ("distil-large-v3", "mlx-community/distil-whisper-large-v3"),
    ("distil-medium.en", "mlx-community/distil-whisper-medium.en"),
])
def test_faster_whisper_names_map_to_mlx_repos(name, repo):
    assert _make(name).model == repo


@pytest.mark.parametrize("name, local", [
    ("small", "small"),
    ("mlx-community/whisper-small-mlx", "small"),
    ("mlx-community/whisper-small.en-mlx", "small.en"),
    ("mlx-community/whisper-large-v3-turbo", "large-v3-turbo"),
    ("mlx-community/whisper-turbo", "large-v3-turbo"),
    ("mlx-community/distil-whisper-large-v3", "distil-large-v3"),
    ("Systran/faster-whisper-small", "Systran/faster-whisper-small"),
    ("mlx-community/some-other-model", None),
])
def test_local_model_name_for_the_fallback(name, local):
    from tinysteno.mlx_transcriber import local_model_name
    assert local_model_name(name) == local


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


def test_mlx_whisper_installs_automatically_on_apple_silicon_only():
    import tomllib
    from pathlib import Path
    project = tomllib.loads(
        (Path(__file__).parent.parent / "pyproject.toml").read_text())["project"]
    deps = [d for d in project["dependencies"] if d.startswith("mlx-whisper")]
    assert len(deps) == 1
    assert "sys_platform == 'darwin'" in deps[0] and "platform_machine == 'arm64'" in deps[0]
    assert "mlx" not in project.get("optional-dependencies", {})


def test_decode_asks_for_word_timestamps_and_splits_at_pauses(monkeypatch):
    calls = []

    def transcribe(_audio, **kwargs):
        calls.append(kwargs)
        return {"language": "en", "segments": [{
            "start": 151.0, "end": 182.4, "text": " OK, so where does my team come in? Let me",
            "words": [
                {"word": " OK,", "start": 166.0, "end": 166.3},
                {"word": " so", "start": 166.3, "end": 166.5},
                {"word": " where", "start": 166.5, "end": 166.9},
                {"word": " does", "start": 166.9, "end": 167.5},
                {"word": " my", "start": 167.5, "end": 167.8},
                {"word": " team", "start": 167.8, "end": 168.0},
                {"word": " come", "start": 168.0, "end": 168.2},
                {"word": " in?", "start": 168.2, "end": 168.6},
                {"word": " Let", "start": 181.2, "end": 181.7},
                {"word": " me", "start": 181.7, "end": 182.4},
            ]}]}

    monkeypatch.setitem(sys.modules, "mlx_whisper", types.SimpleNamespace(transcribe=transcribe))
    segments, _ = _make()._decode(np.zeros(16000, dtype=np.float32))
    assert calls[0]["word_timestamps"] is True
    assert segments == [
        Segment(start=166.0, end=168.6, text="OK, so where does my team come in?"),
        Segment(start=181.2, end=182.4, text="Let me"),
    ]
