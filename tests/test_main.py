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
    cmd_process, cmd_record, cmd_setup, create_transcriber, load_config, main,
    _apply_cli_overrides, _process_audio, _validate_config,
)
from tinysteno.main import cmd_tui as run_cmd_tui
from tinysteno.personas import Persona
from tinysteno.transcript import Segment, TranscriptOptions


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


def test_create_transcriber_reads_vad_and_prompt_from_config():
    with patch("tinysteno.main.WhisperTranscriber") as cls:
        create_transcriber({"whisper_vad": False, "whisper_prompt": "Acme"})
    assert cls.call_args.kwargs["vad"] is False and cls.call_args.kwargs["prompt"] == "Acme"


def test_create_transcriber_defaults_and_ignores_non_string_prompt():
    with patch("tinysteno.main.WhisperTranscriber") as cls:
        create_transcriber({})
        assert cls.call_args.kwargs["vad"] is True and cls.call_args.kwargs["prompt"] == ""
        create_transcriber({"whisper_prompt": None})
        assert cls.call_args.kwargs["prompt"] == ""
        create_transcriber({"whisper_prompt": 42})
        assert cls.call_args.kwargs["prompt"] == ""


def test_create_transcriber_passes_prompt_to_api_backend():
    with patch("tinysteno.main.ApiTranscriber") as cls:
        create_transcriber({"transcription_backend": "api", "api_key": "k",
                            "whisper_prompt": "Acme"})
    assert cls.call_args.kwargs["prompt"] == "Acme"


def test_create_transcriber_maps_filter_switches_for_both_backends():
    with patch("tinysteno.main.WhisperTranscriber") as cls:
        create_transcriber({})
        assert cls.call_args.kwargs["options"] == TranscriptOptions(True, True)
        create_transcriber({"hallucination_filter": False, "bleed_filter": False})
        assert cls.call_args.kwargs["options"] == TranscriptOptions(False, False)
    with patch("tinysteno.main.ApiTranscriber") as cls:
        create_transcriber({"transcription_backend": "api", "api_key": "k",
                            "bleed_filter": False})
    assert cls.call_args.kwargs["options"] == TranscriptOptions(True, False)


def test_default_config_includes_filter_switches(tmp_path):
    with patch.object(Path, "home", return_value=tmp_path):
        with patch("tinysteno.config.seed_builtin_personas"):
            config = load_config()
    assert config["hallucination_filter"] is True and config["bleed_filter"] is True


def test_default_config_includes_vad_and_prompt(tmp_path):
    with patch.object(Path, "home", return_value=tmp_path):
        with patch("tinysteno.config.seed_builtin_personas"):
            config = load_config()
    assert config["whisper_vad"] is True and config["whisper_prompt"] == ""


def test_load_config_seeds_personas_when_dir_absent(tmp_path):
    _write_config(tmp_path / ".tinysteno" / "config.yaml")
    # personas dir does NOT exist

    with patch.object(Path, "home", return_value=tmp_path):
        with patch("tinysteno.config.seed_builtin_personas") as mock_seed:
            load_config()

    mock_seed.assert_called_once_with()


def test_load_config_does_not_seed_when_dir_exists(tmp_path):
    _write_config(tmp_path / ".tinysteno" / "config.yaml")
    (tmp_path / ".tinysteno" / "personas").mkdir(parents=True)  # dir exists

    with patch.object(Path, "home", return_value=tmp_path):
        with patch("tinysteno.config.seed_builtin_personas") as mock_seed:
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
        with patch("tinysteno.main.update_config"):
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
    base = {"persona": None, "name": None, "timeout": None, "notes": None}
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
         patch("tinysteno.main.update_config") as write:
        cmd_setup(argparse.Namespace(reset_personas=False))
    return write.call_args.args[0]["session_timeout_minutes"]


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


def _run_process_audio(tmp_path, *, orchestrator_setup=None, exporter_setup=None,
                       transcript_extra=None):
    """Call _process_audio with mocked backends, letting callers set side effects."""
    config = {
        "api_key": "ollama", "base_url": "http://localhost", "model": "test",
        "whisper_model": "small", "diarization": False, "auto_title": False,
        "auto_tags": False, "obsidian_vault": str(tmp_path), "output_folder": "meetings",
    }
    transcript = {
        "text": "hello", "diarised_text": "", "detected_language": "en",
        "duration_seconds": 5.0, **(transcript_extra or {}),
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


def test_process_audio_prints_warnings_before_the_meeting_line(tmp_path, capsys):
    def setup(orchestrator):
        orchestrator.warnings = ["summary incomplete"]
        orchestrator.summarize.return_value = {"summary": "We met."}
    _run_process_audio(tmp_path, orchestrator_setup=setup,
                       transcript_extra={"warnings": ["one channel silent"]})
    out = capsys.readouterr().out
    assert out.index("Warning: one channel silent\n") < out.index("Warning: summary incomplete\n")
    assert out.index("Warning: summary incomplete") < out.index("Meeting:")


def test_process_audio_says_when_it_reused_the_saved_transcript(tmp_path, capsys):
    (tmp_path / "a.wav").write_bytes(b"RIFF")
    extra = {"segments": [Segment(0.0, 1.0, "hello", "You")]}
    for _ in range(2):
        _run_process_audio(tmp_path, transcript_extra=extra)
    out = capsys.readouterr().out
    assert out.count("Using the saved transcript for this recording") == 1
    assert "(--retranscribe to transcribe again)." in out
    assert out.index("Using the saved") < out.rindex("Meeting:")


@pytest.mark.parametrize("flag, expected", [(["--retranscribe"], True), ([], False)])
def test_process_passes_retranscribe_through(tmp_path, flag, expected):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x")
    with patch("tinysteno.main._process_audio") as process, \
         patch("tinysteno.main.load_persona"), patch("tinysteno.main.setup_logging"), \
         patch("tinysteno.main.load_config", return_value={}), \
         patch.object(sys, "argv", ["tinysteno", "process", str(audio), *flag]):
        main()
    assert process.call_args.kwargs["retranscribe"] is expected


def test_process_audio_prints_warnings_before_the_no_speech_message(tmp_path, capsys):
    _run_process_audio(tmp_path, transcript_extra={
        "text": " ", "warnings": ["both channels silent"]})
    out = capsys.readouterr().out
    assert out.index("Warning: both channels silent") < out.index("No speech detected")


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


def _run_setup_capturing_prompts(tmp_path, monkeypatch, diarization: bool, existing=None):
    """Run the wizard; return (diarization hint, channels default/hint, saved channels)."""
    if existing is not None:
        (tmp_path / ".tinysteno").mkdir()
        (tmp_path / ".tinysteno" / "config.yaml").write_text(yaml.dump(existing))
    seen = {}

    def answer(_console, label, default, hint=""):
        if label == "Channels":
            seen["channels"] = (default, hint)
        return default

    def answer_bool(_console, label, default, hint=""):
        if label.startswith("Enable speaker diarization"):
            seen["diarization_hint"] = hint
            return diarization
        return default

    monkeypatch.setattr("builtins.input", lambda _prompt: "")
    with patch.object(Path, "home", return_value=tmp_path), \
         patch("tinysteno.main._prompt", side_effect=answer), \
         patch("tinysteno.main._prompt_bool", side_effect=answer_bool), \
         patch("tinysteno.main.seed_builtin_personas",
               return_value={"copied": [], "skipped": []}), \
         patch("tinysteno.main.list_personas", return_value=["default"]), \
         patch("tinysteno.main.update_config") as write:
        cmd_setup(argparse.Namespace(reset_personas=False))
    return seen, write.call_args.args[0]["channels"]


def test_setup_diarization_hint_does_not_ask_for_two_channels(tmp_path, monkeypatch):
    seen, _ = _run_setup_capturing_prompts(tmp_path, monkeypatch, diarization=False)
    assert "channels = 2" not in seen["diarization_hint"]
    assert "recording system audio" in seen["diarization_hint"]


def test_setup_diarization_yes_keeps_the_channels_default_at_one(tmp_path, monkeypatch):
    seen, saved = _run_setup_capturing_prompts(tmp_path, monkeypatch, diarization=True)
    assert seen["channels"][0] == "1" and saved == 1
    assert "1 is right for almost every microphone" in seen["channels"][1]


def test_setup_still_offers_an_existing_channels_value(tmp_path, monkeypatch):
    seen, saved = _run_setup_capturing_prompts(
        tmp_path, monkeypatch, diarization=True, existing={"channels": 2}
    )
    assert seen["channels"][0] == "2" and saved == 2


def test_missing_notes_file_fails_before_recording(tmp_path, capsys):
    with patch("tinysteno.main.AudioRecorder") as rec_cls, patch("tinysteno.main.load_persona"), \
         pytest.raises(SystemExit) as exit_info:
        cmd_record(
            _record_args(notes=str(tmp_path / "nope.txt")), {"obsidian_vault": str(tmp_path)})
    assert exit_info.value.code == 1 and "cannot read notes file" in capsys.readouterr().err
    rec_cls.assert_not_called()
    rec_cls.return_value.start.assert_not_called()


def test_notes_file_is_read_and_passed_to_processing(tmp_path):
    notes = tmp_path / "n.txt"
    notes.write_text("Budget is 40k \u2013 agreed", encoding="utf-8")
    config = {"session_timeout_minutes": 0.001, "obsidian_vault": str(tmp_path)}
    with patch("tinysteno.main.AudioRecorder") as rec_cls, \
         patch("tinysteno.main.load_persona", return_value=_make_list_only_persona()), \
         patch("tinysteno.main._process_audio") as process:
        rec_cls.return_value.start.return_value = "/tmp/x.wav"
        cmd_record(_record_args(notes=str(notes)), config)
    assert process.call_args.kwargs["notes"] == "Budget is 40k \u2013 agreed"


def test_process_reads_notes_before_processing(tmp_path, capsys):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x")
    args = argparse.Namespace(
        audio=str(audio), persona=None, name=None, notes=str(tmp_path / "nope.txt"))
    with patch("tinysteno.main._process_audio") as process, \
         patch("tinysteno.main.load_persona"), pytest.raises(SystemExit) as exit_info:
        cmd_process(args, {})
    assert exit_info.value.code == 1
    assert "Error: cannot read notes file" in capsys.readouterr().err
    process.assert_not_called()


def test_tui_reads_notes_before_the_app_starts(tmp_path, capsys):
    args = argparse.Namespace(notes=str(tmp_path / "nope.txt"))
    with patch("tinysteno.tui.run_tui") as run_tui, pytest.raises(SystemExit) as exit_info:
        run_cmd_tui(args, {})
    assert exit_info.value.code == 1
    assert "cannot read notes file" in capsys.readouterr().err
    run_tui.assert_not_called()


@pytest.mark.parametrize("bad", ["8192", 8192.5, True, None])
def test_validate_config_rejects_bad_llm_context_tokens(bad):
    with pytest.raises(ValueError, match="llm_context_tokens"):
        _validate_config({"llm_context_tokens": bad})


def test_validate_config_accepts_llm_context_tokens():
    _validate_config({"llm_context_tokens": 4096})


@pytest.mark.parametrize("command", ["record", "tui", "process"])
@pytest.mark.parametrize("value", ["0", "-2", "two", "1.5"])
def test_speakers_flag_rejects_zero_and_non_integers(command, value, capsys):
    argv = ["tinysteno", command, f"--speakers={value}"]
    if command == "process":
        argv.insert(2, "x.wav")
    with patch.object(sys, "argv", argv), \
         patch("tinysteno.main.load_config", return_value={}), \
         patch("tinysteno.main.cmd_record") as record, \
         patch("tinysteno.main.cmd_tui") as tui, \
         patch("tinysteno.main.cmd_process") as process:
        with pytest.raises(SystemExit):
            main()
    assert "must be a whole number of at least 1" in capsys.readouterr().err
    record.assert_not_called()
    tui.assert_not_called()
    process.assert_not_called()


def test_speakers_flag_sets_max_speakers_for_the_run_only():
    config = {}
    _apply_cli_overrides(config, argparse.Namespace(speakers=3))
    assert config == {"max_speakers": 3}
    untouched = {}
    _apply_cli_overrides(untouched, argparse.Namespace(speakers=None))
    assert not untouched


def test_speaker_split_is_off_in_the_default_config(tmp_path):
    with patch.object(Path, "home", return_value=tmp_path):
        with patch("tinysteno.config.seed_builtin_personas"):
            config = load_config()
    assert config["speaker_split"] is False and "max_speakers" not in config


@pytest.mark.parametrize("bad", [0, -1, 1.5, "2", True, None])
def test_validate_config_rejects_bad_max_speakers(bad):
    with pytest.raises(ValueError, match="max_speakers"):
        _validate_config({"max_speakers": bad})


def test_validate_config_accepts_max_speakers():
    _validate_config({"max_speakers": 4})


def test_create_transcriber_passes_split_options():
    with patch("tinysteno.main.WhisperTranscriber") as cls:
        create_transcriber({"speaker_split": True, "max_speakers": 3})
        assert cls.call_args.kwargs["options"].speaker_split is True
        assert cls.call_args.kwargs["options"].max_speakers == 3
        create_transcriber({})
        assert cls.call_args.kwargs["options"].speaker_split is False
        assert cls.call_args.kwargs["options"].max_speakers is None
    with patch("tinysteno.main.ApiTranscriber") as cls:
        create_transcriber({"transcription_backend": "api", "api_key": "k",
                            "speaker_split": True})
        assert cls.call_args.kwargs["options"].speaker_split is True


@pytest.mark.parametrize("bad", [0, -1, "600", True, float("inf"), float("nan"), None])
def test_validate_config_rejects_bad_api_timeout(bad):
    with pytest.raises(ValueError, match="whisper_api_timeout"):
        _validate_config({"whisper_api_timeout": bad})


def test_validate_config_accepts_api_timeout():
    _validate_config({"whisper_api_timeout": 0.5})


def test_process_audio_prints_warnings_gathered_before_a_failed_stage(tmp_path, capsys):
    def setup(exporter):
        exporter.export.side_effect = RuntimeError("Template render error")
    _run_process_audio(tmp_path, exporter_setup=setup,
                       transcript_extra={"warnings": ["one channel silent"]})
    out = capsys.readouterr().out
    assert out.index("Warning: one channel silent") < out.index("Error rendering note")


def test_process_audio_prints_warnings_before_raising_a_failed_stage(tmp_path, capsys):
    def setup(orchestrator):
        orchestrator.summarize.side_effect = ConnectionError("endpoint down")
    with pytest.raises(ConnectionError):
        _run_process_audio(tmp_path, orchestrator_setup=setup,
                           transcript_extra={"warnings": ["one channel silent"]})
    assert "Warning: one channel silent\n" in capsys.readouterr().out


def test_create_transcriber_passes_beam_size():
    with patch("tinysteno.main.WhisperTranscriber") as cls:
        create_transcriber({"whisper_beam_size": 1})
        assert cls.call_args.kwargs["beam_size"] == 1
        create_transcriber({})
        assert cls.call_args.kwargs["beam_size"] == 5
        create_transcriber({"whisper_beam_size": 0})
        assert cls.call_args.kwargs["beam_size"] == 5


def test_create_transcriber_mlx_backend_when_available(monkeypatch):
    import types
    from tinysteno.mlx_transcriber import MlxWhisperTranscriber
    monkeypatch.setitem(sys.modules, "mlx_whisper", types.SimpleNamespace())
    result = create_transcriber({"transcription_backend": "mlx", "whisper_model": "base",
                                 "whisper_prompt": "Acme"})
    assert isinstance(result, MlxWhisperTranscriber)
    assert result.model == "mlx-community/whisper-base-mlx" and result.prompt == "Acme"


def test_create_transcriber_mlx_falls_back_to_local(monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "mlx_whisper", None)
    with patch("tinysteno.main.WhisperTranscriber") as cls, caplog.at_level(logging.WARNING):
        result = create_transcriber({"transcription_backend": "mlx", "whisper_beam_size": 1})
    assert result is cls.return_value and cls.call_args.kwargs["beam_size"] == 1
    assert [r for r in caplog.records if "mlx" in r.getMessage()]


@pytest.mark.parametrize("command", [["record"], ["tui"], ["process", "x.wav"]])
def test_backend_flag_accepts_mlx(command):
    argv = ["tinysteno", *command, "--backend", "mlx"]
    with patch.object(sys, "argv", argv), patch("tinysteno.main.load_config", return_value={}), \
            patch("tinysteno.main._config_path") as path, \
            patch("tinysteno.main.cmd_record") as rec, patch("tinysteno.main.cmd_tui") as tui, \
            patch("tinysteno.main.cmd_process") as proc:
        path.return_value.exists.return_value = True
        main()
    called = next(m for m in (rec, tui, proc) if m.called)
    assert called.call_args.args[0].backend == "mlx"


@pytest.mark.parametrize("model, expected", [
    ("mlx-community/whisper-large-v3-turbo", "large-v3-turbo"),
    ("mlx-community/some-other-model", "small"),
    ("medium", "medium"),
])
def test_mlx_fallback_translates_the_model_name(monkeypatch, caplog, model, expected):
    monkeypatch.setitem(sys.modules, "mlx_whisper", None)
    with patch("tinysteno.main.WhisperTranscriber") as cls, caplog.at_level(logging.WARNING):
        create_transcriber({"transcription_backend": "mlx", "whisper_model": model})
    assert cls.call_args.kwargs["model_size"] == expected
