"""Tests for tinysteno.config: merge-saves that keep unknown keys, atomic writes."""
import argparse
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from tinysteno import main
from tinysteno.config import DEFAULT_CONFIG, config_path, update_config


@pytest.fixture(name="home")
def home_fixture(tmp_path, monkeypatch):
    """Redirect HOME, Path.home and the import-time personas dir to a temp directory."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr("tinysteno.personas._USER_DIR", tmp_path / ".tinysteno" / "personas")
    return tmp_path


def write_yaml(home, data):
    path = home / ".tinysteno" / "config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(data))


def raise_oserror(*_args, **_kwargs):
    raise OSError("disk full")


def test_update_keeps_keys_it_was_not_given(home):
    write_yaml(home, {"model": "a", "whisper_prompt": "Acme", "custom_thing": 7})
    stored = update_config({"model": "b"})
    assert stored == {"model": "b", "whisper_prompt": "Acme", "custom_thing": 7}
    assert yaml.safe_load(config_path().read_text()) == stored


def test_none_removes_a_key(home):
    write_yaml(home, {"recordings_path": "/x", "model": "a"})
    assert "recordings_path" not in update_config({"recordings_path": None})


def test_invalid_change_leaves_the_file_untouched(home):
    write_yaml(home, {"channels": 1})
    before = config_path().read_bytes()
    with pytest.raises(ValueError):
        update_config({"channels": 5})
    assert config_path().read_bytes() == before


@pytest.mark.usefixtures("home")
def test_malformed_file_is_reported_and_untouched():
    config_path().parent.mkdir(parents=True)
    config_path().write_text("- just\n- a list\n")
    before = config_path().read_bytes()
    with pytest.raises(ValueError):
        update_config({"model": "b"})
    assert config_path().read_bytes() == before


@pytest.mark.usefixtures("home")
def test_invalid_yaml_is_a_value_error_and_untouched():
    config_path().parent.mkdir(parents=True)
    config_path().write_text("model: [unclosed\n")
    before = config_path().read_bytes()
    with pytest.raises(ValueError):
        update_config({"model": "b"})
    assert config_path().read_bytes() == before


def test_failed_write_leaves_the_old_file(home, monkeypatch):
    write_yaml(home, {"model": "a"})
    before = config_path().read_bytes()
    monkeypatch.setattr(os, "replace", raise_oserror)
    with pytest.raises(OSError):
        update_config({"model": "b"})
    assert config_path().read_bytes() == before
    assert not list(config_path().parent.glob("*.tmp*"))


@pytest.mark.usefixtures("home")
def test_new_file_is_private_and_existing_mode_is_kept():
    update_config({"model": "a"})
    assert stat.S_IMODE(config_path().stat().st_mode) == 0o600
    config_path().chmod(0o644)
    update_config({"model": "b"})
    assert stat.S_IMODE(config_path().stat().st_mode) == 0o644


def test_setup_wizard_keeps_keys_it_does_not_ask_about(home, monkeypatch):
    write_yaml(home, {**DEFAULT_CONFIG, "whisper_prompt": "Acme", "tui_theme": "nord",
                      "transcription_backend": "api"})
    monkeypatch.setattr("builtins.input", lambda *_: "")  # Enter on every prompt
    main.cmd_setup(argparse.Namespace(reset_personas=False))
    stored = yaml.safe_load(config_path().read_text())
    assert stored["whisper_prompt"] == "Acme"
    assert stored["tui_theme"] == "nord"
    assert stored["transcription_backend"] == "api"


def test_importing_config_does_not_import_the_cli_or_the_diarizer():
    code = (
        "import sys, tinysteno.config;"
        "bad = [m for m in ('tinysteno.main', 'sherpa_onnx', 'argparse', 'rich')"
        " if m in sys.modules and m != 'argparse'];"
        "sys.exit(1 if bad else 0)"
    )
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0
