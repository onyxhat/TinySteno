"""cmd_tui builds TuiDeps whose settings, recorder and pipeline follow the live config."""
import argparse
from unittest.mock import patch

import pytest
import yaml

from tinysteno.config import config_path, load_config
from tinysteno.main import cmd_tui


class FakeRecorder:  # pylint: disable=too-few-public-methods
    """Records its constructor arguments; never touches audio hardware."""
    made: list[dict] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        FakeRecorder.made.append(kwargs)


@pytest.fixture(autouse=True)
def _stub_recorder(monkeypatch):
    """No test here may construct a real AudioRecorder."""
    monkeypatch.setattr("tinysteno.main.AudioRecorder", FakeRecorder)


def _launch(args=None, config=None, first_run=False):
    """Run cmd_tui with run_tui patched; return (deps, live config)."""
    FakeRecorder.made = []
    captured = {}

    def fake_run_tui(_cfg, deps, **_kwargs):
        captured["deps"] = deps
        return None, 0

    args = args or argparse.Namespace()
    if config is None:
        config = load_config()
    with patch("tinysteno.tui.run_tui", side_effect=fake_run_tui):
        cmd_tui(args, config, first_run=first_run)
    return captured["deps"], config


def _stored():
    return yaml.safe_load(config_path().read_text())


def test_save_updates_the_live_config_and_the_file():
    deps, config = _launch()
    deps.save_settings({"model": "new"})
    assert config["model"] == "new"
    assert _stored()["model"] == "new"
    assert deps.load_settings()["model"] == "new"


def test_save_does_not_persist_cli_overrides():
    args = argparse.Namespace(backend="api", whisper_api_key="sk-secret")
    deps, config = _launch(args)
    deps.save_settings({"model": "new"})
    assert _stored()["transcription_backend"] == "local"
    assert "sk-secret" not in config_path().read_text()
    assert config["transcription_backend"] == "api"
    assert config["whisper_api_key"] == "sk-secret"


def test_flag_wins_for_the_session_but_the_file_gets_the_new_value():
    deps, config = _launch(argparse.Namespace(backend="api"))
    deps.save_settings({"transcription_backend": "local"})
    assert _stored()["transcription_backend"] == "local"
    assert config["transcription_backend"] == "api"


def test_save_moves_the_recordings_dir(tmp_path):
    deps, _config = _launch()
    target = tmp_path / "rec"
    deps.save_settings({"recordings_path": str(target)})
    assert deps.recordings_dir == target
    deps.make_recorder()
    assert FakeRecorder.made[-1]["recordings_dir"] == target


def test_recorder_uses_saved_audio_settings():
    deps, _config = _launch()
    deps.save_settings({"channels": 2, "sample_rate": 16000})
    deps.make_recorder()
    assert FakeRecorder.made[-1]["channels"] == 2
    assert FakeRecorder.made[-1]["sample_rate"] == 16000


def test_invalid_save_changes_nothing_in_memory():
    deps, config = _launch()
    before = dict(config)
    file_before = config_path().read_bytes()
    with pytest.raises(ValueError):
        deps.save_settings({"channels": 5})
    assert config == before
    assert config_path().read_bytes() == file_before


def test_first_run_flag_follows_the_config_file():
    deps, _config = _launch(first_run=True)
    assert deps.first_run is True
    deps, _config = _launch()
    assert deps.first_run is False


def test_main_passes_first_run():
    from tinysteno.main import main
    seen = []
    for _ in range(2):
        with patch("tinysteno.main.cmd_tui",
                   side_effect=lambda a, c, first_run=False: seen.append(first_run)), \
             patch("sys.argv", ["tinysteno", "tui"]):
            main()
    assert seen == [True, False]


def test_run_checks_and_delete_follow_the_live_state(tmp_path):
    deps, config = _launch()
    rec = tmp_path / "rec"
    deps.save_settings({"recordings_path": str(rec)})
    rec.mkdir()
    wav = rec / "a.wav"
    wav.write_bytes(b"")
    assert deps.delete_recording(wav) == [wav]
    assert not wav.exists()
    with patch("tinysteno.main.run_checks", return_value=[]) as checks:
        deps.run_checks()
    checks.assert_called_once_with(config)


def test_seed_personas_forces_a_restore():
    deps, _config = _launch()
    with patch("tinysteno.main.seed_builtin_personas",
               return_value={"copied": ["x"], "skipped": []}) as seed:
        assert deps.seed_personas() == {"copied": ["x"], "skipped": []}
    seed.assert_called_once_with(force=True)


def test_pipeline_forwards_retranscribe_and_reads_the_live_config():
    deps, _config = _launch()
    with patch("tinysteno.main.run_pipeline") as run, \
         patch("tinysteno.main.create_transcriber"):
        deps.run_pipeline("a.wav", None, object(), None, None, retranscribe=True)
    assert run.call_args.kwargs["retranscribe"] is True
