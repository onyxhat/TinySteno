"""Tests for the setup checks (tinysteno.doctor) and `tinysteno test`."""
import argparse
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

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


def test_cmd_test_exit_code_and_output(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("tinysteno.doctor._import", lambda name: MagicMock())
    assert main.cmd_test(argparse.Namespace()) == 1
    out = capsys.readouterr().out
    assert "✓ Recorder module" in out
    assert f"✗ Config: {config_path()} not found\n" in out
    assert "✗ Found 1 issue(s)" in out

    seed_builtin_personas()
    vault = tmp_path / "vault"
    vault.mkdir()
    config_path().parent.mkdir(parents=True, exist_ok=True)
    config_path().write_text(yaml.dump({**DEFAULT_CONFIG, "obsidian_vault": str(vault),
                                        "speaker_split": True}))
    monkeypatch.setattr("tinysteno.doctor._find_spec", lambda name: None)
    assert main.cmd_test(argparse.Namespace()) == 0
    out = capsys.readouterr().out
    assert "✓ Config file exists" in out
    assert "! Speaker separation:" in out
    assert "✓ All checks passed!" in out


def test_cmd_test_prints_legacy_failure_wording(capsys, monkeypatch):
    def fake_import(name):
        if name == "sounddevice":
            raise ImportError("no")
        return MagicMock()
    monkeypatch.setattr("tinysteno.doctor._import", fake_import)
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


def test_cmd_test_prints_the_legacy_recorder_failure_line(capsys, monkeypatch):
    def fake_import(name):
        if name == "tinysteno.recorder":
            raise ImportError("boom")
        return MagicMock()
    monkeypatch.setattr("tinysteno.doctor._import", fake_import)
    main.cmd_test(argparse.Namespace())
    assert "✗ Recorder: boom\n" in capsys.readouterr().out


def test_recorder_and_transcriber_checks_need_their_classes(monkeypatch):
    monkeypatch.setattr("tinysteno.doctor._import", lambda name: object())
    checks = by_name(run_checks(None))
    assert not checks["Recorder module"].ok
    assert not checks["Transcriber module"].ok


def test_cmd_test_prints_brackets_literally(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("tinysteno.doctor._find_spec", lambda name: None)
    config_path().parent.mkdir(parents=True)
    config_path().write_text(yaml.dump({**DEFAULT_CONFIG, "speaker_split": True,
                                        "obsidian_vault": str(tmp_path / "Notes [work]")}))
    main.cmd_test(argparse.Namespace())
    out = capsys.readouterr().out
    assert "! Speaker separation: pip install 'tinysteno[diarize]'\n" in out
    assert f"{tmp_path / 'Notes [work]'} is not an existing directory\n" in out


def tree(root):
    return sorted((str(p.relative_to(root)), p.stat().st_size if p.is_file() else -1)
                  for p in root.rglob("*"))


def test_cmd_test_writes_nothing_and_reports_missing_personas(tmp_path, capsys):
    home = Path.home()
    config_path().parent.mkdir(parents=True)
    config_path().write_text(yaml.dump({"obsidian_vault": str(tmp_path)}))
    before = tree(home)
    assert main.cmd_test(argparse.Namespace()) == 1
    out = capsys.readouterr().out
    assert tree(home) == before
    assert "✗ Default persona:" in out


def test_cmd_test_with_an_invalid_config_cannot_pass(capsys):
    config_path().parent.mkdir(parents=True)
    config_path().write_text("channels: 7\n")
    assert main.cmd_test(argparse.Namespace()) == 1
    assert "✗ Config:" in capsys.readouterr().out


def test_empty_and_non_string_vault_fail_with_a_clear_detail(tmp_path):
    for value in ("", "   ", None, 5):
        check = by_name(run_checks(make_config(tmp_path, obsidian_vault=value)))["Vault folder"]
        assert not check.ok
        assert check.detail == "No vault folder is set"


def test_vault_tilde_is_expanded_by_the_check():
    (Path.home() / "vault").mkdir()
    config = {**DEFAULT_CONFIG, "obsidian_vault": "~/vault"}
    assert by_name(run_checks(config))["Vault folder"].ok


def test_broken_symlink_in_the_recordings_path_fails(tmp_path):
    (tmp_path / "link").symlink_to(tmp_path / "gone")
    for path in (tmp_path / "link", tmp_path / "link" / "sub"):
        config = make_config(tmp_path, recordings_path=str(path))
        check = by_name(run_checks(config))["Recordings folder"]
        assert not check.ok
        assert str(tmp_path / "link") in check.detail


def test_recordings_tilde_path_is_expanded():
    config = {**DEFAULT_CONFIG, "recordings_path": "~/rec"}
    check = by_name(run_checks(config))["Recordings folder"]
    assert check.ok
    assert str(Path.home() / "rec") in check.detail
