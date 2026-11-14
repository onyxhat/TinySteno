"""Tests for tinysteno.main module."""
import argparse
import logging
import os
import signal
import sys
import threading
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from tinysteno.main import (
    cmd_record, cmd_setup, load_config, main, _process_audio, _validate_config,
)
from tinysteno.main import cmd_tui as run_cmd_tui
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
        tags=[],
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
        tags=[],
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


@pytest.mark.parametrize("bad", ["two hours", -5, True, None, float("inf"), float("nan")])
def test_validate_config_rejects_bad_session_timeout(bad):
    with pytest.raises(ValueError, match="session_timeout_minutes"):
        _validate_config({"session_timeout_minutes": bad})


@pytest.mark.parametrize("good", [0, 120, 0.5])
def test_validate_config_accepts_session_timeout(good):
    _validate_config({"session_timeout_minutes": good})


def _record_args(**overrides):
    base = {"persona": None, "name": None, "timeout": None}
    return argparse.Namespace(**{**base, **overrides})


def test_cmd_record_stops_and_processes_when_timeout_elapses(tmp_path, capsys):
    config = {"session_timeout_minutes": 0.001, "obsidian_vault": str(tmp_path)}
    with patch("tinysteno.main.AudioRecorder") as rec_cls, \
         patch("tinysteno.main.load_persona", return_value=_make_list_only_persona()), \
         patch("tinysteno.main._process_audio") as process:
        rec_cls.return_value.start.return_value = "/tmp/x.wav"
        cmd_record(_record_args(), config)  # returns without KeyboardInterrupt
    rec_cls.return_value.stop.assert_called_once()
    process.assert_called_once()
    assert "timeout" in capsys.readouterr().out.lower()


def _timer_limit(tmp_path, *, persona_minutes, cli_minutes):
    """Run cmd_record with a 120-minute global config; return the limit given to SessionTimer."""
    config = {"session_timeout_minutes": 120, "obsidian_vault": str(tmp_path)}
    persona = _make_list_only_persona()
    persona.session_timeout_minutes = persona_minutes
    with patch("tinysteno.main.AudioRecorder") as rec_cls, \
         patch("tinysteno.main.load_persona", return_value=persona), \
         patch("tinysteno.main._process_audio"), \
         patch("tinysteno.main.SessionTimer") as timer_cls:
        rec_cls.return_value.start.return_value = "/tmp/x.wav"
        timer_cls.return_value.expired = True
        cmd_record(_record_args(timeout=cli_minutes), config)
    return timer_cls.call_args.args[0]


def test_global_timeout_applies_when_persona_sets_none(tmp_path):
    assert _timer_limit(tmp_path, persona_minutes=None, cli_minutes=None) == 7200


def test_persona_timeout_overrides_global(tmp_path):
    assert _timer_limit(tmp_path, persona_minutes=30, cli_minutes=None) == 1800


def test_persona_can_disable_the_timeout(tmp_path):
    assert _timer_limit(tmp_path, persona_minutes=0, cli_minutes=None) is None


def test_timeout_flag_overrides_persona_and_global(tmp_path):
    assert _timer_limit(tmp_path, persona_minutes=30, cli_minutes=5.0) == 300


@pytest.mark.parametrize("command", ["record", "tui"])
@pytest.mark.parametrize("minutes", ["-1", "inf", "nan", "-inf", "Infinity"])
def test_timeout_flag_rejects_negative_and_non_finite_minutes(command, minutes, capsys):
    # The commands are stubbed: if parsing ever accepted the value, nothing real may start.
    with patch.object(sys, "argv", ["tinysteno", command, f"--timeout={minutes}"]), \
         patch("tinysteno.main.load_config", return_value={}), \
         patch("tinysteno.main.cmd_record") as record, \
         patch("tinysteno.main.cmd_tui") as tui:
        with pytest.raises(SystemExit):
            main()
    assert not record.called and not tui.called
    assert "minutes must be >= 0" in capsys.readouterr().err


def _setup_timeout(tmp_path, monkeypatch, typed: str, existing: dict | None = None):
    """Run the setup wizard answering ``typed`` to the timeout prompt; return the saved value."""
    if existing is not None:
        (tmp_path / ".tinysteno").mkdir()
        (tmp_path / ".tinysteno" / "config.yaml").write_text(yaml.dump(existing))

    def answer(_console, label, default, _hint=""):
        return typed if label == "Session timeout (minutes)" else default

    monkeypatch.setattr("builtins.input", lambda _prompt: "")
    with patch.object(Path, "home", return_value=tmp_path), \
         patch("tinysteno.main._prompt", side_effect=answer), \
         patch("tinysteno.main.seed_builtin_personas",
               return_value={"copied": [], "skipped": []}), \
         patch("tinysteno.main.list_personas", return_value=["default"]), \
         patch("tinysteno.main._write_config") as write:
        cmd_setup(argparse.Namespace(reset_personas=False))
    return write.call_args.args[1]["session_timeout_minutes"]


@pytest.mark.parametrize("typed, saved", [
    ("45", 45), ("22.5", 22.5), ("0", 0),
    ("two hours", 75), ("-5", 75), ("inf", 75), ("nan", 75),
])
def test_setup_timeout_prompt_accepts_numbers_and_falls_back_on_garbage(
    tmp_path, monkeypatch, typed, saved
):
    value = _setup_timeout(tmp_path, monkeypatch, typed)
    assert value == saved and type(value) is type(saved)


@pytest.mark.parametrize("existing", [float("nan"), float("inf"), "soon", 30])
def test_setup_timeout_prompt_keeps_a_valid_existing_value_else_the_default(
    tmp_path, monkeypatch, existing
):
    value = _setup_timeout(tmp_path, monkeypatch, "", {"session_timeout_minutes": existing})
    assert value == (30 if existing == 30 else 75)


def _run_process_audio(tmp_path, *, orchestrator_setup=None, exporter_setup=None):
    """Call _process_audio with mocked backends, letting callers set side effects."""
    config = {
        "api_key": "ollama", "base_url": "http://localhost", "model": "test",
        "whisper_model": "small", "diarization": False, "auto_title": False,
        "auto_tags": False, "obsidian_vault": str(tmp_path), "output_folder": "meetings",
    }
    transcript = {
        "text": "hello", "diarised_text": "", "detected_language": "en",
        "duration_seconds": 5.0,
    }
    persona = Persona(
        slug="default", name="Default", description="desc",
        schema={"summary": {"type": "string", "description": "summary"}},
        system_prompt="p", template="{{ title }}",
        template_path=Path("/fake/template.md"), tags=[],
    )
    with patch("tinysteno.main.WhisperTranscriber") as transcriber_cls, \
         patch("tinysteno.main.Orchestrator") as orchestrator_cls, \
         patch("tinysteno.main.ObsidianExporter") as exporter_cls:
        transcriber_cls.return_value.transcribe.return_value = transcript
        orchestrator_cls.return_value.summarize.return_value = {"summary": "We met."}
        if orchestrator_setup:
            orchestrator_setup(orchestrator_cls.return_value)
        if exporter_setup:
            exporter_setup(exporter_cls.return_value)
        _process_audio(str(tmp_path / "a.wav"), None, config, MagicMock(), persona,
                       datetime(2024, 1, 1))


def test_process_audio_reports_template_error_and_returns(tmp_path, capsys):
    def setup(exporter):
        exporter.export.side_effect = RuntimeError("Template render error")
    _run_process_audio(tmp_path, exporter_setup=setup)
    assert "Error rendering note: Template render error" in capsys.readouterr().out


def test_process_audio_propagates_export_os_error(tmp_path):
    def setup(exporter):
        exporter.export.side_effect = OSError("disk full")
    with pytest.raises(OSError):
        _run_process_audio(tmp_path, exporter_setup=setup)


def test_process_audio_propagates_summarize_error(tmp_path):
    def setup(orchestrator):
        orchestrator.summarize.side_effect = ConnectionError("endpoint down")
    with pytest.raises(ConnectionError):
        _run_process_audio(tmp_path, orchestrator_setup=setup)


def test_no_subcommand_launches_tui_on_a_tty(tmp_path):
    with patch.object(sys, "argv", ["tinysteno"]), \
         patch("tinysteno.main._is_interactive", return_value=True), \
         patch("tinysteno.main.load_config", return_value={"obsidian_vault": str(tmp_path)}), \
         patch("tinysteno.main.cmd_tui") as cmd_tui:
        main()
    cmd_tui.assert_called_once()


def test_no_subcommand_prints_help_when_not_a_tty(capsys):
    with patch.object(sys, "argv", ["tinysteno"]), \
         patch("tinysteno.main._is_interactive", return_value=False), \
         patch("tinysteno.main.cmd_tui") as cmd_tui:
        main()
    cmd_tui.assert_not_called()
    assert "usage:" in capsys.readouterr().out


def test_tui_subcommand_passes_now_flag(tmp_path):
    with patch.object(sys, "argv", ["tinysteno", "tui", "--now", "--timeout", "30"]), \
         patch("tinysteno.main.load_config", return_value={"obsidian_vault": str(tmp_path)}), \
         patch("tinysteno.main.cmd_tui") as cmd_tui:
        main()
    args = cmd_tui.call_args.args[0]
    assert args.now is True and args.timeout == 30.0


def test_cmd_tui_failed_save_goes_to_stderr_and_exits_1(tmp_path, capsys):
    message = "Could not save the recording: disk full. Raw audio kept at x.f32"
    with patch("tinysteno.tui.run_tui", return_value=(message, 1)), \
         patch.object(Path, "home", return_value=tmp_path):
        with pytest.raises(SystemExit) as exc:
            run_cmd_tui(argparse.Namespace(), {"obsidian_vault": str(tmp_path)})
    assert exc.value.code == 1
    out, err = capsys.readouterr()
    assert message in err and message not in out
    assert err.endswith(f"Log: {tmp_path / '.tinysteno' / 'tui.log'}\n")


def test_cmd_tui_saved_recording_goes_to_stdout(tmp_path, capsys):
    with patch("tinysteno.tui.run_tui", return_value=("Saved recording: x", 0)), \
         patch.object(Path, "home", return_value=tmp_path):
        run_cmd_tui(argparse.Namespace(), {"obsidian_vault": str(tmp_path)})
    out, err = capsys.readouterr()
    assert out == "Saved recording: x\n" and not err


def _terminal_stream_handlers() -> list:
    return [
        h for h in logging.getLogger().handlers
        if isinstance(h, logging.StreamHandler)
        and getattr(h, "stream", None) in (sys.stderr, sys.stdout, sys.__stderr__, sys.__stdout__)
    ]


def test_cmd_tui_logs_to_a_file_not_the_terminal(tmp_path, capsys):
    root = logging.getLogger()
    root.addHandler(logging.StreamHandler(sys.stderr))  # as setup_logging() leaves it
    before = root.handlers[:]
    seen = {}

    def fake_run_tui(*_args, **_kwargs):
        seen["terminal_handlers"] = _terminal_stream_handlers()
        seen["level"] = root.level
        logging.getLogger("tinysteno.orchestrator").warning("LLM endpoint unreachable")
        return None, 0

    try:
        with patch("tinysteno.tui.run_tui", side_effect=fake_run_tui), \
             patch.object(Path, "home", return_value=tmp_path):
            run_cmd_tui(argparse.Namespace(verbose=True), {"obsidian_vault": str(tmp_path)})
        log_path = tmp_path / ".tinysteno" / "tui.log"
        assert seen["terminal_handlers"] == [] and seen["level"] == logging.DEBUG
        assert "WARNING: LLM endpoint unreachable" in log_path.read_text(encoding="utf-8")
        out, err = capsys.readouterr()
        assert out == f"Log: {log_path}\n" and not err
        assert root.handlers == before          # restored for anything that runs after
    finally:
        for handler in root.handlers[:]:
            if handler not in before:
                root.removeHandler(handler)
        root.removeHandler(before[-1])


def test_cmd_tui_without_warnings_or_errors_does_not_mention_the_log(tmp_path, capsys):
    with patch("tinysteno.tui.run_tui", return_value=(None, 0)), \
         patch.object(Path, "home", return_value=tmp_path):
        run_cmd_tui(argparse.Namespace(), {"obsidian_vault": str(tmp_path)})
    assert capsys.readouterr() == ("", "")


def _run_record(tmp_path, recorder_setup, **patches):
    """cmd_record on a mocked recorder; returns (recorder mock, _process_audio mock)."""
    config = {"session_timeout_minutes": 0.001, "obsidian_vault": str(tmp_path)}
    with patch("tinysteno.main.AudioRecorder") as rec_cls, \
         patch("tinysteno.main.load_persona", return_value=_make_list_only_persona()), \
         patch("tinysteno.main._process_audio") as process:
        rec_cls.return_value.start.return_value = "/tmp/x.wav"
        recorder_setup(rec_cls.return_value)
        for target, value in patches.items():
            setattr(rec_cls.return_value, target, value)
        cmd_record(_record_args(), config)
    return rec_cls.return_value, process


def test_cmd_record_does_not_process_when_no_audio_was_captured(tmp_path, capsys):
    def no_audio(recorder):
        recorder.stop.return_value = False

    _recorder, process = _run_record(tmp_path, no_audio)
    process.assert_not_called()
    assert "No audio was captured" in capsys.readouterr().out


@pytest.mark.skipif(not hasattr(signal, "SIGTERM"), reason="needs POSIX signals")
def test_cmd_record_saves_and_does_not_process_when_killed(tmp_path, capsys):
    delivered = []
    config = {"session_timeout_minutes": 5, "obsidian_vault": str(tmp_path)}
    previous = signal.signal(signal.SIGTERM, lambda *_: delivered.append("after"))
    killer = threading.Timer(0.3, os.kill, (os.getpid(), signal.SIGTERM))
    try:
        with patch("tinysteno.main.AudioRecorder") as rec_cls, \
             patch("tinysteno.main.load_persona", return_value=_make_list_only_persona()), \
             patch("tinysteno.main._process_audio") as process:
            rec_cls.return_value.start.return_value = "/tmp/x.wav"
            rec_cls.return_value.stop.side_effect = lambda: delivered.append("saved") or True
            killer.start()
            cmd_record(_record_args(), config)
    finally:
        killer.cancel()
        signal.signal(signal.SIGTERM, previous)
    assert delivered == ["saved", "after"]       # the signal takes effect only after the save
    process.assert_not_called()
    out = capsys.readouterr().out
    assert "Saved recording: /tmp/x.wav" in out and "tinysteno process /tmp/x.wav" in out
