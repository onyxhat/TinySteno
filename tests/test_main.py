"""Tests for tinysteno.main module."""
import argparse
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import yaml

from tinysteno.main import cmd_setup, load_config, _process_audio
from tinysteno.personas import Persona


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


def _make_list_only_persona() -> Persona:
    return Persona(
        slug="1on1",
        name="1-on-1",
        description="desc",
        schema={
            "goals":   {"type": "list", "description": "goals"},
            "actions": {"type": "list", "description": "actions"},
        },
        system_prompt="You are a test assistant.",
        template="{{ title }}",
        template_path=Path("/fake/template.md"),
    )


def test_auto_tags_called_for_list_only_persona(tmp_path):
    """auto_tags must fire even when the persona has no string fields."""
    persona = _make_list_only_persona()
    config = {
        "api_key": "ollama",
        "base_url": "http://localhost",
        "model": "test",
        "whisper_model": "small",
        "diarization": False,
        "auto_title": False,
        "auto_tags": True,
        "obsidian_vault": str(tmp_path),
        "output_folder": "meetings",
    }

    mock_transcribe_result = {
        "text": "hello",
        "diarised_text": "",
        "detected_language": "en",
        "duration_seconds": 60.0,
    }
    summarize_data = {"goals": ["ship feature"], "actions": ["write tests"]}

    with patch("tinysteno.main.WhisperTranscriber") as mock_transcriber_cls, \
         patch("tinysteno.main.Orchestrator") as mock_orchestrator_cls, \
         patch("tinysteno.main.ObsidianExporter") as mock_exporter_cls:

        mock_transcriber_cls.return_value.transcribe.return_value = mock_transcribe_result
        mock_orch = mock_orchestrator_cls.return_value
        mock_orch.summarize.return_value = summarize_data
        mock_orch.generate_tags.return_value = ["management", "goals"]
        mock_exporter_cls.return_value.export.return_value = tmp_path / "note.md"

        _process_audio(
            wav_path=str(tmp_path / "audio.wav"),
            name=None,
            config=config,
            logger=MagicMock(),
            persona=persona,
            timestamp=datetime(2024, 1, 1),
        )

    mock_orch.generate_tags.assert_called_once()
    call_arg = mock_orch.generate_tags.call_args[0][0]
    assert "ship feature" in call_arg
    assert "write tests" in call_arg


def test_setup_argparser_has_reset_personas_flag():
    """Verify the real main() parser wires --reset-personas correctly."""
    from tinysteno.main import main

    with patch("tinysteno.main.seed_builtin_personas") as mock_seed:
        mock_seed.return_value = {"copied": [], "skipped": []}
        with patch.object(sys, "argv", ["tinysteno", "setup", "--reset-personas"]):
            main()

    mock_seed.assert_called_once_with(force=True)


def test_title_and_tags_generated_in_parallel(tmp_path):
    """generate_title and generate_tags should start concurrently."""
    import time

    persona = Persona(
        slug="default",
        name="Default",
        description="desc",
        schema={"summary": {"type": "string", "description": "summary"}},
        system_prompt="You are a test assistant.",
        template="{{ title }}",
        template_path=Path("/fake/template.md"),
    )
    config = {
        "api_key": "ollama",
        "base_url": "http://localhost",
        "model": "test",
        "whisper_model": "small",
        "diarization": False,
        "auto_title": True,
        "auto_tags": True,
        "obsidian_vault": str(tmp_path),
        "output_folder": "meetings",
    }

    call_start_times: dict = {}

    def fake_title(_text):
        call_start_times["title"] = time.monotonic()
        time.sleep(0.05)
        return "Test Title"

    def fake_tags(_text):
        call_start_times["tags"] = time.monotonic()
        time.sleep(0.05)
        return ["test"]

    mock_transcribe_result = {
        "text": "hello world",
        "diarised_text": "",
        "detected_language": "en",
        "duration_seconds": 10.0,
    }

    with patch("tinysteno.main.WhisperTranscriber") as mock_tc, \
         patch("tinysteno.main.Orchestrator") as mock_oc, \
         patch("tinysteno.main.ObsidianExporter") as mock_ec:

        mock_tc.return_value.transcribe.return_value = mock_transcribe_result
        mock_orch = mock_oc.return_value
        mock_orch.summarize.return_value = {"summary": "hello world"}
        mock_orch.generate_title.side_effect = fake_title
        mock_orch.generate_tags.side_effect = fake_tags
        mock_ec.return_value.export.return_value = tmp_path / "note.md"

        _process_audio(
            wav_path=str(tmp_path / "audio.wav"),
            name=None,
            config=config,
            logger=MagicMock(),
            persona=persona,
            timestamp=datetime(2024, 1, 1),
        )

    assert "title" in call_start_times and "tags" in call_start_times
    overlap = abs(call_start_times["title"] - call_start_times["tags"])
    assert overlap < 0.02, f"title and tags should start concurrently, gap={overlap:.3f}s"
