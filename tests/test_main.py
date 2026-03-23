"""Tests for tinysteno.main module."""
import argparse
import sys
from pathlib import Path
from unittest.mock import patch

import yaml

from tinysteno.main import cmd_setup, load_config


def _write_config(path: Path) -> None:
    """Write a minimal valid config file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "obsidian_vault": str(path.parent),
        "api_key": "ollama",
        "base_url": "http://localhost:11434/v1",
        "model": "llama3.2:3b",
        "whisper_model": "small",
        "diarization": False,
        "auto_title": True,
        "output_folder": "meetings",
        "sample_rate": 44100,
        "channels": 1,
        "persona": "default",
    }
    path.write_text(yaml.dump(config))


def test_load_config_seeds_personas_when_dir_absent(tmp_path):
    _write_config(tmp_path / ".tinysteno" / "config.yaml")
    # personas dir does NOT exist

    with patch.object(Path, "home", return_value=tmp_path):
        with patch("tinysteno.main.seed_builtin_personas") as mock_seed:
            load_config()

    mock_seed.assert_called_once_with()


def test_load_config_does_not_seed_when_dir_exists(tmp_path):
    _write_config(tmp_path / ".tinysteno" / "config.yaml")
    (tmp_path / ".tinysteno" / "personas").mkdir(parents=True)  # dir exists

    with patch.object(Path, "home", return_value=tmp_path):
        with patch("tinysteno.main.seed_builtin_personas") as mock_seed:
            load_config()

    mock_seed.assert_not_called()


def test_cmd_setup_reset_personas_forces_seed_without_wizard(monkeypatch):
    args = argparse.Namespace(reset_personas=True)
    calls = []
    monkeypatch.setattr("builtins.input", lambda _: calls.append("input") or "")

    with patch("tinysteno.main.seed_builtin_personas") as mock_seed:
        mock_seed.return_value = {"copied": ["default"], "skipped": []}
        cmd_setup(args)

    mock_seed.assert_called_once_with(force=True)
    assert not calls, "input() must not be called when --reset-personas is set"


def test_cmd_setup_normal_seeds_interactively_after_config_write(monkeypatch):
    args = argparse.Namespace(reset_personas=False)
    monkeypatch.setattr("builtins.input", lambda _: "")  # accept all defaults

    with patch("tinysteno.main.seed_builtin_personas") as mock_seed:
        mock_seed.return_value = {"copied": ["default", "rca"], "skipped": []}
        with patch("tinysteno.main._write_config"):
            with patch("tinysteno.main.list_personas", return_value=["default"]):
                cmd_setup(args)

    mock_seed.assert_called_once_with(interactive=True)


def test_setup_argparser_has_reset_personas_flag():
    """Verify the real main() parser wires --reset-personas correctly."""
    from tinysteno.main import main

    with patch("tinysteno.main.seed_builtin_personas") as mock_seed:
        mock_seed.return_value = {"copied": [], "skipped": []}
        with patch.object(sys, "argv", ["tinysteno", "setup", "--reset-personas"]):
            main()

    mock_seed.assert_called_once_with(force=True)
