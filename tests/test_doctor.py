"""Tests for the setup checks (tinysteno.doctor) and `tinysteno test`."""
import argparse
import subprocess
import sys

import pytest
import yaml

from tinysteno import main
from tinysteno.config import DEFAULT_CONFIG, config_path
from tinysteno.doctor import Check, run_checks
from tinysteno.personas import seed_builtin_personas


def by_name(checks):
    return {c.name: c for c in checks}


def make_config(tmp_path, **over):
    vault = tmp_path / "vault"
    vault.mkdir(exist_ok=True)
    return {**DEFAULT_CONFIG, "obsidian_vault": str(vault), **over}


def test_checks_report_a_missing_vault(tmp_path):
    config = make_config(tmp_path, obsidian_vault=str(tmp_path / "nope"))
    check = by_name(run_checks(config))["Vault folder"]
    assert not check.ok
    assert str(tmp_path / "nope") in check.detail


def test_existing_vault_is_ok(tmp_path):
    assert by_name(run_checks(make_config(tmp_path)))["Vault folder"].ok


def test_no_config_runs_only_module_and_file_checks():
    names = [c.name for c in run_checks(None)]
    assert names == ["Recorder module", "Transcriber module", "sounddevice", "faster-whisper",
                     "openai", "pyyaml", "rich", "Config file"]


def test_missing_config_file_names_the_path():
    check = by_name(run_checks(None))["Config file"]
    assert not check.ok
    assert str(config_path()) in check.detail


def test_invalid_config_file_detail_is_the_reason():
    config_path().parent.mkdir(parents=True)
    config_path().write_text("channels: 7\n")
    check = by_name(run_checks(None))["Config file"]
    assert not check.ok
    assert "channels" in check.detail


def test_broken_default_persona_is_reported(tmp_path):
    seed_builtin_personas()
    check = by_name(run_checks(make_config(tmp_path, persona="does-not-exist")))
    assert not check["Default persona"].ok
    assert "does-not-exist" in check["Default persona"].detail


def test_default_persona_ok_when_seeded(tmp_path):
    seed_builtin_personas()
    assert by_name(run_checks(make_config(tmp_path)))["Default persona"].ok


def test_recordings_folder_ok_when_it_can_be_created(tmp_path):
    config = make_config(tmp_path, recordings_path=str(tmp_path / "new" / "deep"))
    check = by_name(run_checks(config))["Recordings folder"]
    assert check.ok
    assert str(tmp_path / "new" / "deep") in check.detail


def test_recordings_folder_fails_under_a_file(tmp_path):
    (tmp_path / "afile").write_text("x")
    config = make_config(tmp_path, recordings_path=str(tmp_path / "afile" / "sub"))
    assert not by_name(run_checks(config))["Recordings folder"].ok


def test_speaker_check_only_when_split_is_on_and_never_imports(tmp_path):
    sys.modules.pop("sherpa_onnx", None)
    assert "Speaker separation" not in by_name(run_checks(make_config(tmp_path)))
    checks = by_name(run_checks(make_config(tmp_path, speaker_split=True)))
    assert checks["Speaker separation"].required is False
    assert "sherpa_onnx" not in sys.modules


def test_a_check_never_raises(tmp_path, monkeypatch):
    def boom(_slug):
        raise RuntimeError("multi\nline")
    monkeypatch.setattr("tinysteno.doctor.load_persona", boom)
    check = by_name(run_checks(make_config(tmp_path)))["Default persona"]
    assert check == Check("Default persona", False, "multi line")


@pytest.mark.usefixtures("isolated_home")
def test_cmd_test_exit_code_and_output(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("tinysteno.doctor.importlib.import_module", lambda name: object())
    assert main.cmd_test(argparse.Namespace()) == 1
    out = capsys.readouterr().out
    assert "✓ Recorder module" in out
    assert "✗ Config: " in out and "not found" in out
    assert "✗ Found 1 issue(s)" in out

    seed_builtin_personas()
    vault = tmp_path / "vault"
    vault.mkdir()
    config_path().parent.mkdir(parents=True, exist_ok=True)
    config_path().write_text(yaml.dump({**DEFAULT_CONFIG, "obsidian_vault": str(vault),
                                        "speaker_split": True}))
    monkeypatch.setattr("tinysteno.doctor.importlib.util.find_spec", lambda name: None)
    assert main.cmd_test(argparse.Namespace()) == 0
    out = capsys.readouterr().out
    assert "✓ Config file exists" in out
    assert "! Speaker separation:" in out
    assert "✓ All checks passed!" in out


@pytest.mark.usefixtures("isolated_home")
def test_cmd_test_prints_legacy_failure_wording(capsys, monkeypatch):
    def fake_import(name):
        if name == "sounddevice":
            raise ImportError("no")
        return object()
    monkeypatch.setattr("tinysteno.doctor.importlib.import_module", fake_import)
    assert main.cmd_test(argparse.Namespace()) == 1
    assert "✗ sounddevice (pip install sounddevice)" in capsys.readouterr().out


def test_importing_doctor_and_recordings_stays_light():
    code = (
        "import sys, tinysteno.doctor, tinysteno.recordings;"
        "bad = [m for m in ('tinysteno.main', 'tinysteno.sherpa_diarizer', 'sherpa_onnx',"
        " 'textual', 'rich') if m in sys.modules];"
        "sys.exit(1 if bad else 0)"
    )
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0
