"""Odd config files never give tracebacks; saves keep the file's key order and its text."""
import sys

import pytest
import yaml

from tinysteno import main
from tinysteno.config import (
    DEFAULT_CONFIG, config_path, is_positive_number, is_timeout_minutes, load_config,
    read_stored, update_config, validate_config,
)

HUGE = 10 ** 400          # an int YAML reads exactly, too large for a float
SECRET = b"api_key: sk-latin1-SECRET-\xe9\xff\n"


def _write(data: bytes) -> None:
    config_path().parent.mkdir(parents=True, exist_ok=True)
    (config_path().parent / "personas").mkdir(exist_ok=True)   # load_config: no seeding
    config_path().write_bytes(data)


def _chain(exc: BaseException) -> list[BaseException]:
    seen: list[BaseException] = []
    while exc is not None and exc not in seen:
        seen.append(exc)
        exc = exc.__cause__ or exc.__context__
    return seen


# ── numbers too large for a float ────────────────────────────────────────────────────

@pytest.mark.parametrize("check", [is_timeout_minutes, is_positive_number])
@pytest.mark.parametrize("value", [HUGE, -HUGE], ids=["huge", "minus-huge"])
def test_number_checks_refuse_an_int_too_large_for_a_float(check, value):
    assert check(value) is False


@pytest.mark.parametrize("key", ["session_timeout_minutes", "whisper_api_timeout"])
def test_validation_reports_a_huge_number_as_invalid(key):
    with pytest.raises(ValueError):
        validate_config({key: HUGE})


def test_a_stored_huge_number_is_a_value_error_not_a_traceback():
    _write(f"session_timeout_minutes: {HUGE}\n".encode())
    with pytest.raises(ValueError):
        load_config()


# ── a file that is not UTF-8 ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("loader", [read_stored, load_config])
def test_a_non_utf8_file_names_the_path_and_keeps_no_file_bytes(loader):
    _write(b"model: a\n" + SECRET)
    with pytest.raises(ValueError) as caught:
        loader()
    assert str(caught.value) == f"Config file is not valid UTF-8: {config_path()}"
    assert caught.value.__cause__ is None and caught.value.__suppress_context__ is True
    for exc in _chain(caught.value):
        assert not isinstance(exc, UnicodeDecodeError)
        assert b"SECRET" not in repr(vars(exc)).encode() + repr(exc.args).encode()


def test_main_reports_a_non_utf8_file_without_a_traceback(monkeypatch, capsys):
    _write(SECRET)
    monkeypatch.setattr(sys, "argv", ["tinysteno", "process", "x.wav"])
    with pytest.raises(SystemExit) as exit_info:
        main.main()          # fails at load_config, before any audio or app is touched
    assert exit_info.value.code == 1
    err = capsys.readouterr().err
    assert err.splitlines()[0] == f"Error: Config file is not valid UTF-8: {config_path()}"
    assert "SECRET" not in err and "Traceback" not in err


# ── a file that cannot be read at all ────────────────────────────────────────────────

@pytest.mark.parametrize("argv", [
    ["tinysteno", "process", "x.wav"],
    ["tinysteno", "record"],
    ["tinysteno", "tui"],
])
def test_main_reports_an_unreadable_config_without_a_traceback(argv, monkeypatch, capsys):
    _write(b"model: a\n")
    config_path().chmod(0o000)
    monkeypatch.setattr(sys, "argv", argv)
    try:
        with pytest.raises(SystemExit) as exit_info:
            main.main()      # fails at load_config, before any audio or app is touched
    finally:
        config_path().chmod(0o600)
    assert exit_info.value.code == 1
    out, err = capsys.readouterr()
    assert not out
    assert err.splitlines() == [f"Error: Cannot read {config_path()}: Permission denied"]


# ── saving keeps the file's order and its text ───────────────────────────────────────

def test_a_save_keeps_the_key_order_of_the_file():
    _write(b"model: a\napi_key: b\nbase_url: c\ntui_theme: nord\n")
    update_config({"api_key": "z", "base_url": "y"})
    assert config_path().read_text(encoding="utf-8") == (
        "model: a\napi_key: z\nbase_url: y\ntui_theme: nord\n")


def test_a_new_key_is_added_at_the_end():
    _write(b"model: a\napi_key: b\n")
    update_config({"whisper_prompt": "Acme"})
    assert list(yaml.safe_load(config_path().read_text(encoding="utf-8"))) == [
        "model", "api_key", "whisper_prompt"]


def test_non_ascii_text_is_written_readably_and_round_trips():
    _write(b"model: a\n")
    update_config({"whisper_prompt": "Zoë, Łukasz"})
    text = config_path().read_bytes().decode("utf-8")
    assert "whisper_prompt: Zoë, Łukasz\n" in text
    assert read_stored()["whisper_prompt"] == "Zoë, Łukasz"


def test_the_first_run_file_follows_the_default_order(isolated_home):
    load_config()
    stored = yaml.safe_load(config_path().read_text(encoding="utf-8"))
    assert list(stored) == list(DEFAULT_CONFIG)
    assert stored["obsidian_vault"] == str(isolated_home / "Obsidian" / "Vault")
