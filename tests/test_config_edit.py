"""`tinysteno config --edit`: the editor always gets a file it can save."""
import argparse
import stat
from pathlib import Path

import pytest
import yaml

from tinysteno import main
from tinysteno.config import DEFAULT_CONFIG, config_dir, config_path
from tinysteno.personas import BUILTIN_ORDER


def _editor_stub(tmp_path: Path, monkeypatch) -> Path:
    """Point EDITOR at a script that records the path it was given; return the record."""
    record = tmp_path / "editor-calls.txt"
    stub = tmp_path / "editor-stub.sh"
    stub.write_text(f'#!/bin/sh\nprintf "%s\\n" "$1" >> "{record}"\n')
    stub.chmod(0o755)
    monkeypatch.delenv("VISUAL", raising=False)
    monkeypatch.setenv("EDITOR", str(stub))
    return record


def test_edit_on_a_fresh_home_creates_the_default_config_first(tmp_path, monkeypatch,
                                                                isolated_home):
    record = _editor_stub(tmp_path, monkeypatch)
    assert not config_dir().exists()
    main.cmd_config(argparse.Namespace(edit=True))
    assert config_dir().is_dir() and config_path().is_file()
    assert stat.S_IMODE(config_path().stat().st_mode) == 0o600
    expected = dict(DEFAULT_CONFIG, obsidian_vault=str(isolated_home / "Obsidian" / "Vault"))
    assert yaml.safe_load(config_path().read_text(encoding="utf-8")) == expected
    personas = config_dir() / "personas"
    assert sorted(p.name for p in personas.iterdir()) == sorted(BUILTIN_ORDER)
    assert record.read_text().splitlines() == [str(config_path())]


def test_edit_leaves_an_invalid_file_byte_for_byte_and_still_opens_it(tmp_path, monkeypatch):
    record = _editor_stub(tmp_path, monkeypatch)
    config_path().parent.mkdir(parents=True)
    broken = b"model: [unclosed\napi_key: sk-x\n\xff"
    config_path().write_bytes(broken)
    main.cmd_config(argparse.Namespace(edit=True))
    assert config_path().read_bytes() == broken
    assert record.read_text().splitlines() == [str(config_path())]


def test_edit_keeps_an_existing_valid_file(tmp_path, monkeypatch):
    record = _editor_stub(tmp_path, monkeypatch)
    config_path().parent.mkdir(parents=True)
    config_path().write_text("model: mine\n", encoding="utf-8")
    main.cmd_config(argparse.Namespace(edit=True))
    assert config_path().read_text(encoding="utf-8") == "model: mine\n"
    assert record.read_text().splitlines() == [str(config_path())]


def test_edit_reports_a_config_that_cannot_be_created(tmp_path, monkeypatch, capsys):
    record = _editor_stub(tmp_path, monkeypatch)

    def refuse():
        raise PermissionError("read-only home")
    monkeypatch.setattr(main, "ensure_config_file", refuse)
    with pytest.raises(SystemExit) as exit_info:
        main.cmd_config(argparse.Namespace(edit=True))
    assert exit_info.value.code == 1
    assert capsys.readouterr().err == f"Cannot create {config_path()}: read-only home\n"
    assert not record.exists()
