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
from tinysteno.config import (
    DEFAULT_CONFIG, config_path, load_config, read_stored, update_config,
)


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
        "bad = [m for m in ('tinysteno.main', 'tinysteno.sherpa_diarizer', 'sherpa_onnx', 'rich')"
        " if m in sys.modules];"
        "sys.exit(1 if bad else 0)"
    )
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0


# --- write safety -----------------------------------------------------------------------

@pytest.mark.usefixtures("home")
def test_read_only_file_is_refused_before_any_temp_file():
    config_path().parent.mkdir(parents=True)
    config_path().write_text("model: a\n")
    config_path().chmod(0o444)
    with pytest.raises(PermissionError):
        update_config({"model": "b"})
    assert config_path().read_text() == "model: a\n"
    assert not list(config_path().parent.glob(".*tmp*"))


@pytest.mark.usefixtures("home")
def test_symlinked_config_is_written_through_and_stays_a_link(tmp_path):
    target = tmp_path / "elsewhere" / "real.yaml"
    target.parent.mkdir()
    target.write_text("model: a\n")
    config_path().parent.mkdir(parents=True)
    config_path().symlink_to(target)
    update_config({"model": "b"})
    assert config_path().is_symlink()
    assert yaml.safe_load(target.read_text()) == {"model": "b"}


@pytest.mark.usefixtures("home")
def test_dangling_symlink_creates_its_target(tmp_path):
    target = tmp_path / "elsewhere" / "real.yaml"
    target.parent.mkdir()
    config_path().parent.mkdir(parents=True)
    config_path().symlink_to(target)
    update_config({"model": "b"})
    assert config_path().is_symlink()
    assert yaml.safe_load(target.read_text()) == {"model": "b"}


@pytest.mark.parametrize("text", ["", "# only a comment\n"])
@pytest.mark.usefixtures("home")
def test_empty_or_comment_only_file_reads_as_empty(text):
    config_path().parent.mkdir(parents=True)
    config_path().write_text(text)
    assert read_stored() == {}
    assert update_config({"model": "b"}) == {"model": "b"}


@pytest.mark.usefixtures("home")
def test_load_config_still_rejects_an_empty_file():
    config_path().parent.mkdir(parents=True)
    config_path().write_text("")
    with pytest.raises(ValueError, match="malformed"):
        load_config()


# --- setup repairs invalid files --------------------------------------------------------

def run_setup(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *_: "")
    main.cmd_setup(argparse.Namespace(reset_personas=False))


def test_setup_works_on_an_empty_file(home, monkeypatch):
    write_yaml(home, None)
    config_path().write_text("")
    run_setup(monkeypatch)
    assert yaml.safe_load(config_path().read_text())["persona"] == "default"


@pytest.mark.parametrize("text", ["- a\n- list\n", "model: [unclosed\n"])
@pytest.mark.usefixtures("home")
def test_setup_refuses_an_unreadable_file_without_prompting(monkeypatch, capsys, text):
    config_path().parent.mkdir(parents=True)
    config_path().write_text(text)

    def no_prompt(*_):
        raise AssertionError("must not prompt")
    monkeypatch.setattr("builtins.input", no_prompt)
    with pytest.raises(SystemExit) as info:
        main.cmd_setup(argparse.Namespace(reset_personas=False))
    out = capsys.readouterr().out
    assert info.value.code == 1
    assert "Cannot read" in out and "tinysteno config --edit" in out
    assert config_path().read_text() == text


def test_setup_removes_an_invalid_key_it_does_not_ask_about(home, monkeypatch, capsys):
    write_yaml(home, {"max_speakers": 0, "whisper_api_timeout": -1, "llm_context_tokens": "8k",
                      "tui_theme": "nord"})
    run_setup(monkeypatch)
    stored = yaml.safe_load(config_path().read_text())
    for key in ("max_speakers", "whisper_api_timeout", "llm_context_tokens"):
        assert key not in stored
    assert stored["tui_theme"] == "nord"
    out = capsys.readouterr().out
    assert "Removing invalid setting max_speakers (was 0)" in out


def test_setup_repairs_an_invalid_asked_key_with_enter(home, monkeypatch, capsys):
    write_yaml(home, {"channels": 3})
    run_setup(monkeypatch)
    assert yaml.safe_load(config_path().read_text())["channels"] == 1
    assert "Current value of channels is invalid" in capsys.readouterr().out


def test_setup_never_prints_secret_values(home, monkeypatch, capsys):
    write_yaml(home, {"whisper_api_key_timeout": -1})
    monkeypatch.setattr(main, "invalid_settings",
                        lambda _stored: {"whisper_api_key_timeout": "bad"})
    run_setup(monkeypatch)
    out = capsys.readouterr().out
    assert "(hidden)" in out and "-1" not in out


def test_setup_still_exits_1_when_the_save_fails(home, monkeypatch, capsys):
    write_yaml(home, {"model": "a"})
    config_path().chmod(0o444)
    with pytest.raises(SystemExit) as info:
        run_setup(monkeypatch)
    assert info.value.code == 1
    assert "Could not save config" in capsys.readouterr().out


# --- tinysteno config -------------------------------------------------------------------

def test_config_edit_opens_the_editor_on_an_invalid_file(home, monkeypatch):
    write_yaml(home, [1, 2])
    called = []
    monkeypatch.setattr(main.subprocess, "call", called.append)
    monkeypatch.setenv("EDITOR", "ed")
    main.cmd_config(argparse.Namespace(edit=True))
    assert called == [["ed", str(config_path())]]


def test_config_show_of_an_invalid_file_explains_and_exits_1(home, capsys):
    write_yaml(home, [1, 2])
    with pytest.raises(SystemExit) as info:
        main.cmd_config(argparse.Namespace(edit=False))
    assert info.value.code == 1
    assert "--edit" in capsys.readouterr().err
