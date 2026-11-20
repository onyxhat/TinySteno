"""CLI entry point for TinySteno."""

import argparse
import logging
import math
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from tinysteno.recorder import AudioRecorder
from tinysteno.api_transcriber import ApiTranscriber
from tinysteno.transcriber import Transcriber, WhisperTranscriber
from tinysteno.transcript import TranscriptOptions
from tinysteno.orchestrator import Orchestrator
from tinysteno.obsidian import ObsidianExporter
from tinysteno.session import (
    DEFAULT_SESSION_TIMEOUT_MINUTES, SessionTimer, resolve_timeout_seconds, saved_message,
)
from tinysteno.pipeline import PipelineError, Stage, run_pipeline
from tinysteno.doctor import run_checks
from tinysteno.recordings import delete_recording
from tinysteno.log_file import log_to_file
from tinysteno.signals import deferred_exit_signals
from tinysteno.config import (
    config_dir as _config_dir, config_path as _config_path,
    DEFAULT_CONFIG, ensure_config_file, read_stored, invalid_settings, load_config,
    recordings_dir as _recordings_dir,
    validate_config as _validate_config, update_config,
    is_timeout_minutes as _is_timeout_minutes, is_positive_number as _is_positive_number,
    is_speaker_count as _is_speaker_count,
)
from tinysteno.personas import (
    load_persona, list_personas, seed_builtin_personas,
    PersonaNotFoundError, PersonaInvalidError, Persona,
)


def setup_logging(verbose: bool = False):
    """Configure logging level."""
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def _speaker_count(value: str) -> int:
    """argparse type: a whole number of at least 1."""
    try:
        count = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a whole number of at least 1") from None
    if count < 1:
        raise argparse.ArgumentTypeError("must be a whole number of at least 1")
    return count


def _non_negative_minutes(value: str) -> float:
    """argparse type: minutes as a finite number >= 0."""
    try:
        minutes = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid minutes: {value!r}") from None
    if not math.isfinite(minutes) or minutes < 0:
        raise argparse.ArgumentTypeError(f"minutes must be >= 0, got {value!r}")
    return minutes


def create_transcriber(config: dict) -> Transcriber:
    """Factory: build the appropriate transcriber backend from config."""
    backend = config.get("transcription_backend", "local")
    prompt = config.get("whisper_prompt", "")
    if not isinstance(prompt, str):
        prompt = ""
    options = TranscriptOptions(
        drop_hallucinations=bool(config.get("hallucination_filter", True)),
        suppress_bleed=bool(config.get("bleed_filter", True)),
        speaker_split=bool(config.get("speaker_split", False)),
        max_speakers=(config["max_speakers"]
                      if _is_speaker_count(config.get("max_speakers")) else None),
    )
    if backend == "api":
        return ApiTranscriber(
            api_key=config.get("whisper_api_key") or config["api_key"],
            base_url=config.get("whisper_base_url") or config.get("base_url"),
            model=config.get("whisper_api_model", "whisper-1"),
            prompt=prompt,
            options=options,
            server_diarize=bool(config.get("whisper_api_diarize", False)),
            timeout=(float(config["whisper_api_timeout"])
                     if _is_positive_number(config.get("whisper_api_timeout")) else 600.0),
        )
    model_size = config.get("whisper_model", "small")
    if backend == "mlx":
        from tinysteno.mlx_transcriber import (
            MlxWhisperTranscriber, local_model_name, mlx_available)
        if mlx_available():
            return MlxWhisperTranscriber(model_size, prompt=prompt, options=options)
        local = local_model_name(model_size) or "small"
        logging.getLogger(__name__).warning(
            "The mlx backend is unavailable on this machine (it needs Apple Silicon and "
            "`uv sync --extra mlx` or `pip install 'tinysteno[mlx]'`); using the local "
            "faster-whisper backend with model %s.", local)
        model_size = local
    return WhisperTranscriber(
        model_size=model_size,
        device=config.get("whisper_device", "auto"),
        compute_type=config.get("whisper_compute_type", "auto"),
        beam_size=(config["whisper_beam_size"]
                   if _is_speaker_count(config.get("whisper_beam_size")) else 5),
        vad=config.get("whisper_vad", True),
        prompt=prompt,
        options=options,
    )


_CLI_TO_CONFIG = {
    "backend": "transcription_backend",
    "whisper_device": "whisper_device",
    "whisper_compute_type": "whisper_compute_type",
    "whisper_api_key": "whisper_api_key",
    "whisper_api_model": "whisper_api_model",
    "speakers": "max_speakers",
}


def _cli_overridden(args: argparse.Namespace) -> frozenset[str]:
    """The config keys these CLI flags set."""
    return frozenset(
        key for flag, key in _CLI_TO_CONFIG.items() if getattr(args, flag, None) is not None
    )


def _apply_cli_overrides(config: dict, args: argparse.Namespace) -> None:
    """Merge CLI flags into config in-place (CLI overrides take precedence)."""
    for cli_key, config_key in _CLI_TO_CONFIG.items():
        val = getattr(args, cli_key, None)
        if val is not None:
            config[config_key] = val


class _RichReporter:
    """Drives a transient Rich progress display from pipeline stage events."""

    _LABELS = {
        Stage.TRANSCRIBE: "Transcribing...",
        Stage.SUMMARIZE: "Summarizing...",
        Stage.METADATA: "Generating title and tags...",
        Stage.EXPORT: "Exporting...",
    }

    def __init__(self, progress):
        self._progress = progress
        self._tasks: dict = {}

    def stage_started(self, stage: Stage) -> None:
        """Add a task; transcription gets a real bar, other stages a spinner."""
        total = 1.0 if stage is Stage.TRANSCRIBE else None
        self._tasks[stage] = self._progress.add_task(self._LABELS[stage], total=total)

    def stage_progress(self, stage: Stage, fraction: float) -> None:
        """Advance the stage's bar."""
        if stage in self._tasks:
            self._progress.update(self._tasks[stage], completed=fraction)

    def stage_finished(self, stage: Stage) -> None:
        """Remove the stage's task."""
        self._progress.remove_task(self._tasks.pop(stage))

    def stage_skipped(self, stage: Stage) -> None:
        """Nothing to show for a skipped stage."""


def _read_notes(args) -> str:
    """The text of the `--notes` file, or "" without one; exit 1 if it cannot be read."""
    path = getattr(args, "notes", None)
    if not path:
        return ""
    try:
        return Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeError) as err:
        reason = err.strerror if isinstance(err, OSError) and err.strerror else err
        print(f"Error: cannot read notes file {path}: {reason}", file=sys.stderr)
        sys.exit(1)


def _process_audio(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    wav_path: str,
    name: Optional[str],
    config: dict,
    logger: logging.Logger,
    persona: Persona,
    timestamp: datetime,
    notes: str = "",
    retranscribe: bool = False,
) -> None:
    """Shared pipeline: transcribe → summarize → export, with Rich progress."""
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        transient=True,
    ) as progress:
        try:
            result = run_pipeline(
                wav_path, name, config, persona, timestamp,
                transcriber=create_transcriber(config),
                orchestrator_cls=Orchestrator,
                exporter_cls=ObsidianExporter,
                reporter=_RichReporter(progress),
                notes=notes,
                retranscribe=retranscribe,
            )
        except PipelineError as err:
            for text in err.warnings:
                print(f"Warning: {text}")
            cause = err.__cause__
            if err.stage is Stage.EXPORT and isinstance(cause, RuntimeError):
                print(f"Error rendering note: {cause}")
                return
            raise cause from None  # pylint: disable=raising-non-exception

    logger.debug(f"Duration: {result.duration_seconds:.0f}s")
    if result.reused_transcript:
        print("Using the saved transcript for this recording "
              "(--retranscribe to transcribe again).")
    for text in result.warnings:
        print(f"Warning: {text}")
    if result.no_speech:
        print("No speech detected, skipping export.")
        return
    print(f"Meeting: {result.note_path}")


def _capture(recorder, timer: SessionTimer, name: Optional[str], minutes: float):
    """Record until the timer runs out, Ctrl+C, or a kill signal; then stop and save.

    Returns (wav path, whether a WAV was written, whether a signal ended it). A closed
    terminal or a kill saves like Ctrl+C but the caller must not start processing: the
    signal is delivered again as soon as this returns.
    """
    import time

    with deferred_exit_signals() as exit_signals:
        wav_path = recorder.start(name)
        timer.start()
        logging.getLogger(__name__).debug("Recording to: %s", wav_path)
        try:
            while not timer.expired and not exit_signals:
                time.sleep(0.1)
            if not exit_signals:
                print(f"Session timeout reached ({minutes:g} min).")
        except KeyboardInterrupt:
            pass
        finally:
            saved = recorder.stop()
        if exit_signals and saved:
            print(saved_message(wav_path), flush=True)
        return wav_path, saved, bool(exit_signals)


def cmd_record(args, config):  # pylint: disable=too-many-locals
    """Record audio and process into meeting notes."""
    logger = logging.getLogger(__name__)

    slug = args.persona or config.get("persona", "default")
    try:
        persona = load_persona(slug)
    except PersonaNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except PersonaInvalidError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    notes = _read_notes(args)
    name = args.name or None
    timestamp = datetime.now()  # capture at start of recording

    _apply_cli_overrides(config, args)

    recorder = AudioRecorder(
        sample_rate=config.get("sample_rate", 44100),
        channels=config.get("channels", 1),
        recordings_dir=_recordings_dir(config),
    )

    limit = resolve_timeout_seconds(
        config, persona.session_timeout_minutes, getattr(args, "timeout", None)
    )
    timer = SessionTimer(limit)
    minutes = limit / 60 if limit is not None else 0

    if limit is None:
        print("Recording started...\nPress Ctrl+C to stop.")
    else:
        print(
            "Recording started...\n"
            f"Press Ctrl+C to stop. Auto-stop after {minutes:g} min (--timeout 0 to disable)."
        )
    wav_path, saved, killed = _capture(recorder, timer, name, minutes)
    if killed:
        return
    print("Recording stopped.")
    if not saved:
        print("No audio was captured.")
        return
    _process_audio(wav_path, name, config, logger, persona, timestamp, notes=notes)


def _is_interactive() -> bool:
    """True when both stdin and stdout are terminals."""
    return sys.stdin.isatty() and sys.stdout.isatty()


def cmd_tui(args, config, *, first_run: bool = False):
    """Launch the Textual interface."""
    from tinysteno.tui import run_tui
    from tinysteno.tui.deps import SettingsNotApplied, TuiDeps

    notes = _read_notes(args)
    _apply_cli_overrides(config, args)

    def make_recorder():
        # Reads the live config at call time, so saved settings apply to the next recording.
        return AudioRecorder(
            sample_rate=config.get("sample_rate", 44100),
            channels=config.get("channels", 1),
            recordings_dir=_recordings_dir(config),
        )

    def pipeline(  # pylint: disable=too-many-arguments
        wav_path, name, persona, timestamp, reporter, *, retranscribe=False
    ):
        snapshot = dict(config)  # one consistent view for the whole run, whatever is saved
        return run_pipeline(
            wav_path, name, snapshot, persona, timestamp,
            transcriber=create_transcriber(snapshot),
            orchestrator_cls=Orchestrator,
            exporter_cls=ObsidianExporter,
            reporter=reporter,
            notes=notes,
            retranscribe=retranscribe,
        )

    def save_settings(changes: dict) -> None:
        update_config(changes)  # ValueError / OSError: nothing written, nothing changed
        try:
            fresh = load_config()
            _apply_cli_overrides(fresh, args)  # a CLI flag wins for this session, never the file
            new_dir = _recordings_dir(fresh)
        except Exception as err:
            raise SettingsNotApplied(str(err)) from err
        # In place, never empty: overwrite first, then drop the keys that went away.
        config.update(fresh)
        for key in [key for key in config if key not in fresh]:
            del config[key]
        deps.recordings_dir = new_dir

    deps = TuiDeps(
        make_recorder=make_recorder,
        run_pipeline=pipeline,
        list_personas=list_personas,
        load_persona=load_persona,
        recordings_dir=_recordings_dir(config),
        load_settings=read_stored,
        save_settings=save_settings,
        run_checks=lambda: run_checks(dict(config)),
        seed_personas=lambda: seed_builtin_personas(force=True),
        delete_recording=lambda wav: delete_recording(deps.recordings_dir, wav),
        first_run=first_run,
        overridden_keys=_cli_overridden(args),
    )
    log_path = _config_dir() / "tui.log"
    with log_to_file(log_path, getattr(args, "verbose", False)) as log:
        message, code = run_tui(
            config, deps,
            autostart=getattr(args, "now", False),
            name=getattr(args, "name", None),
            persona=getattr(args, "persona", None),
            timeout_override=getattr(args, "timeout", None),
        )
    if log is not None and (code or log.warned):
        message = f"{message}\nLog: {log_path}" if message else f"Log: {log_path}"
    if message:
        print(message, file=sys.stderr if code else sys.stdout)
    if code:
        sys.exit(code)


def cmd_process(args, config):
    """Process an existing audio file."""
    audio_file = Path(args.audio)
    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_file}")

    slug = args.persona or config.get("persona", "default")
    try:
        persona = load_persona(slug)
    except PersonaNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except PersonaInvalidError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    notes = _read_notes(args)
    timestamp = datetime.fromtimestamp(audio_file.stat().st_mtime)

    _apply_cli_overrides(config, args)

    logger = logging.getLogger(__name__)
    _process_audio(
        str(audio_file), args.name or None, config, logger, persona, timestamp, notes=notes,
        retranscribe=getattr(args, "retranscribe", False))


def cmd_list(args):
    """List processed meetings."""
    vault_path = Path(args.vault)
    meetings_dir = vault_path / "meetings"

    if not meetings_dir.exists():
        print("No meetings found.")
        return

    meetings = list(meetings_dir.glob("*.md"))
    for meeting in sorted(meetings, reverse=True):
        print(meeting.name)


def _check_line(check) -> str:
    """One printed line for a check; the checks `test` always had keep their original wording."""
    if check.ok:
        return "✓ Config file exists" if check.name == "Config file" else f"✓ {check.name}"
    legacy = {"Recorder module": "Recorder", "Transcriber module": "Transcriber",
              "Config file": "Config"}
    if check.name in legacy:
        return f"✗ {legacy[check.name]}: {check.detail}"
    if check.name in ("sounddevice", "faster-whisper", "openai", "pyyaml", "rich"):
        return f"✗ {check.name} ({check.detail})"
    return f"{'✗' if check.required else '!'} {check.name}: {check.detail}"


def cmd_test(_args):
    """Verify setup before first use."""
    from rich.console import Console

    console = Console()
    console.print("\n[TinySteno Setup Test]")

    config = None  # nothing is created or seeded here; the Config file check reports why
    if _config_path().exists():
        try:
            stored = read_stored()
            _validate_config(stored)
            config = {**DEFAULT_CONFIG, **stored}
        except (ValueError, OSError):
            config = None

    issues = 0
    for check in run_checks(config):
        console.print(_check_line(check), markup=False, highlight=False, soft_wrap=True)
        if not check.ok and check.required:
            issues += 1

    if not issues:
        console.print("\n✓ All checks passed!")
        return 0
    console.print(f"\n✗ Found {issues} issue(s)")
    return 1


def cmd_config(args):
    """Show or edit configuration."""
    import yaml

    config_path = _config_path()

    if args.edit:
        try:
            ensure_config_file()  # the editor needs a folder to save into; never reads the file
        except OSError as exc:
            print(f"Cannot create {config_path}: {exc}", file=sys.stderr)
            sys.exit(1)
        editor = (
            os.environ.get("VISUAL")
            or os.environ.get("EDITOR")
            or "nano"
        )
        editor_cmd = shlex.split(editor) + [str(config_path)]
        subprocess.call(editor_cmd)
        return
    try:
        config = load_config()
    except (ValueError, OSError, yaml.YAMLError) as exc:
        print(f"Cannot read {config_path}: {exc}", file=sys.stderr)
        print("Fix the file with: tinysteno config --edit", file=sys.stderr)
        sys.exit(1)
    print(yaml.dump(config, indent=2))


def _prompt(console, label: str, default: str, hint: str = "") -> str:
    """Prompt the user for a value, returning the default on empty input."""
    hint_str = f" [dim]{hint}[/dim]" if hint else ""
    console.print(f"  [bold]{label}[/bold]{hint_str}")
    console.print(f"  [dim]default: {default}[/dim]")
    value = input("  > ").strip()
    return value if value else default


def _prompt_bool(console, label: str, default: bool, hint: str = "") -> bool:
    """Prompt the user for a yes/no value."""
    hint_str = f" [dim]{hint}[/dim]" if hint else ""
    default_str = "Y/n" if default else "y/N"
    console.print(f"  [bold]{label}[/bold]{hint_str}")
    value = input(f"  [{default_str}] > ").strip().lower()
    if not value:
        return default
    return value in ("y", "yes")


def _print_seed_summary(console, result: dict) -> None:
    """Print persona seeding summary using rich."""
    user_dir = _config_dir() / "personas"
    if result["copied"]:
        console.print(f"[green]✓[/green] Personas seeded in [bold]{user_dir}[/bold]")
        console.print(f"  Copied: {', '.join(result['copied'])}")
    if result["skipped"]:
        console.print(f"  Skipped (kept yours): {', '.join(result['skipped'])}")


_ASKED_KEYS = frozenset({
    "obsidian_vault", "output_folder", "recordings_path", "api_key", "base_url", "model",
    "auto_title", "auto_tags", "whisper_model", "diarization", "channels", "sample_rate",
    "persona", "session_timeout_minutes",
})


def _shown(key: str, value) -> str:
    """Repr of a value for messages; secrets are hidden."""
    lowered = key.lower()
    return "(hidden)" if "key" in lowered or "token" in lowered else repr(value)


def _report_invalid_settings(existing: dict) -> dict:
    """Report invalid stored settings. Asked keys are dropped from `existing` (so the prompt
    default falls back to the built-in); unasked keys are returned as removals (key -> None)."""
    removals = {}
    for key, reason in invalid_settings(existing).items():
        if key in _ASKED_KEYS:
            print(f"Current value of {key} is invalid: {reason}")
            del existing[key]
        else:
            print(f"Removing invalid setting {key} (was {_shown(key, existing[key])}): {reason}")
            removals[key] = None
    return removals


def cmd_setup(_args):  # pylint: disable=too-many-statements,too-many-locals  # large interactive CLI setup function
    """Interactively create or update ~/.tinysteno/config.yaml."""
    from rich.console import Console
    from rich.rule import Rule

    console = Console()

    # --reset-personas: re-seed built-ins without running the config wizard
    if getattr(_args, "reset_personas", False):
        result = seed_builtin_personas(force=True)
        _print_seed_summary(console, result)
        return

    cfg_path = _config_path()

    # Read the stored file once, before asking anything.
    try:
        existing = read_stored()
    except (ValueError, OSError) as exc:
        print(f"Cannot read {cfg_path}: {exc}")
        print("Fix or remove the file, or run: tinysteno config --edit")
        sys.exit(1)
    action = "Updating" if cfg_path.exists() else "Creating"
    removals = _report_invalid_settings(existing)

    def get(key, fallback):
        return existing.get(key, fallback)

    console.print()
    console.print(Rule("[bold]TinySteno Setup[/bold]"))
    console.print()
    console.print(f"{action} [bold]{cfg_path}[/bold]")
    console.print("Press Enter to keep the current value.\n")

    # --- Obsidian ---
    console.print(Rule("[dim]Obsidian Vault[/dim]"))
    obsidian_vault = _prompt(
        console,
        "Vault path",
        get("obsidian_vault", str(Path("~/Obsidian/Vault").expanduser())),
        "absolute path to your Obsidian vault",
    )

    output_folder = _prompt(
        console,
        "Output folder",
        get("output_folder", "meetings"),
        "subfolder inside vault where notes are saved",
    )

    # --- LLM ---
    console.print()
    console.print(Rule("[dim]LLM / Summarization[/dim]"))
    console.print(
        "  [dim]Ollama (local): base_url = http://localhost:11434/v1, api_key = ollama[/dim]"
    )
    console.print(
        "  [dim]OpenAI (cloud): base_url = https://api.openai.com/v1, api_key = sk-...[/dim]"
    )
    console.print()

    base_url = _prompt(
        console,
        "API base URL",
        get("base_url", "http://localhost:11434/v1"),
    )
    api_key = _prompt(
        console,
        "API key",
        get("api_key", "ollama"),
        "use 'ollama' for local Ollama; enter your key for OpenAI/Groq",
    )
    model = _prompt(
        console,
        "Model name",
        get("model", "llama3.2:3b"),
        "e.g. llama3.2:3b, gpt-4o-mini",
    )
    auto_title = _prompt_bool(
        console,
        "Auto-generate titles from content?",
        get("auto_title", True),
    )
    auto_tags = _prompt_bool(
        console,
        "Auto-generate tags from content?",
        get("auto_tags", True),
    )

    # --- Whisper ---
    console.print()
    console.print(Rule("[dim]Transcription (Whisper)[/dim]"))
    console.print(
        "  [dim]Model sizes (speed ↔ accuracy): tiny · base · small · medium · large[/dim]"
    )
    console.print()

    whisper_model = _prompt(
        console,
        "Whisper model size",
        get("whisper_model", "small"),
    )

    diarization = _prompt_bool(
        console,
        "Enable speaker diarization ([You]/[Others] labels)?",
        get("diarization", False),
        "labels come from recording system audio; no microphone setting is needed",
    )

    # --- Audio ---
    console.print()
    console.print(Rule("[dim]Audio Recording[/dim]"))

    computed_recordings = str(Path(obsidian_vault) / output_folder / "audio")
    recordings_path = _prompt(
        console,
        "Recordings path",
        get("recordings_path", ""),
        f"leave blank to use default: {computed_recordings}",
    )

    channels_input = _prompt(
        console,
        "Channels",
        str(get("channels", 1)),
        "microphone input channels (1 is right for almost every microphone)",
    )
    try:
        channels = int(channels_input)
    except ValueError:
        channels = 1

    sample_rate_input = _prompt(console, "Sample rate", str(get("sample_rate", 44100)), "Hz")
    try:
        sample_rate = int(sample_rate_input)
    except ValueError:
        sample_rate = 44100

    timeout_default = get("session_timeout_minutes", DEFAULT_SESSION_TIMEOUT_MINUTES)
    if not _is_timeout_minutes(timeout_default):
        timeout_default = DEFAULT_SESSION_TIMEOUT_MINUTES
    timeout_input = _prompt(
        console,
        "Session timeout (minutes)",
        str(timeout_default),
        "auto-stop a recording after this long; 0 disables",
    )
    try:
        session_timeout_minutes = _non_negative_minutes(timeout_input)
    except argparse.ArgumentTypeError:
        session_timeout_minutes = timeout_default
    if session_timeout_minutes == int(session_timeout_minutes):
        session_timeout_minutes = int(session_timeout_minutes)

    # --- Persona ---
    console.print()
    console.print(Rule("[dim]Persona[/dim]"))
    available_personas = list_personas()
    console.print(f"  [dim]Available personas: {', '.join(available_personas)}[/dim]")
    console.print()

    persona_slug = _prompt(
        console,
        "Default persona",
        get("persona", "default"),
        "slug of the persona to use for processing",
    )
    if persona_slug not in available_personas:
        console.print(
            f"  [yellow]Warning: '{persona_slug}' is not a known persona."
            " It will be saved but may fail at runtime.[/yellow]"
        )

    # --- Write ---
    try:
        update_config({
            **removals,
            "obsidian_vault": obsidian_vault,
            "output_folder": output_folder,
            "recordings_path": recordings_path or None,
            "api_key": api_key,
            "base_url": base_url,
            "model": model,
            "auto_title": auto_title,
            "auto_tags": auto_tags,
            "whisper_model": whisper_model,
            "diarization": diarization,
            "channels": channels,
            "sample_rate": sample_rate,
            "persona": persona_slug,
            "session_timeout_minutes": session_timeout_minutes,
        })
    except (ValueError, OSError) as exc:
        console.print(f"[red]✗ Could not save config: {exc}[/red]")
        sys.exit(1)

    # Seed built-in personas, prompting on conflicts
    seed_result = seed_builtin_personas(interactive=True)
    _print_seed_summary(console, seed_result)

    console.print()
    console.print(Rule())
    console.print(f"[green]✓[/green] Config written to [bold]{cfg_path}[/bold]")
    console.print()
    console.print("Run [bold]tinysteno setup[/bold] again any time to update settings.")
    console.print("Run [bold]tinysteno test[/bold] to verify your setup.")
    console.print()


def main():  # pylint: disable=too-many-statements  # argparse setup + dispatch
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="TinySteno - Meeting recorder")

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    record_parser = subparsers.add_parser("record", help="Record a meeting")
    record_parser.add_argument("--name", help="Meeting name")
    record_parser.add_argument("--persona", help="Persona slug to use for this recording")
    record_parser.add_argument(
        "--timeout", type=_non_negative_minutes, metavar="MINUTES",
        help="Auto-stop after MINUTES (0 disables; default from config)",
    )
    record_parser.add_argument(
        "--notes", metavar="PATH", help="Text file of your own notes for the summary (UTF-8)")
    record_parser.add_argument("--verbose", action="store_true", help="Verbose output")
    record_parser.add_argument(
        "--backend", choices=["local", "mlx", "api"], help="Transcription backend")
    record_parser.add_argument("--whisper-device", help="Whisper device override (cpu, cuda, auto)")
    record_parser.add_argument(
        "--whisper-compute-type",
        help="Whisper compute type (int8, float16, auto)",
    )
    record_parser.add_argument("--whisper-api-key", help="API key for API backend")
    record_parser.add_argument("--whisper-api-model", help="Model name for API backend")
    record_parser.add_argument(
        "--speakers", type=_speaker_count, metavar="N",
        help="Most voices to expect when splitting the remote side by speaker (this run only)",
    )

    tui_parser = subparsers.add_parser("tui", help="Interactive terminal interface")
    tui_parser.add_argument("--now", action="store_true", help="Start recording immediately")
    tui_parser.add_argument("--name", help="Meeting name")
    tui_parser.add_argument("--persona", help="Persona slug to use for this recording")
    tui_parser.add_argument(
        "--timeout", type=_non_negative_minutes, metavar="MINUTES",
        help="Auto-stop after MINUTES (0 disables; default from config)",
    )
    tui_parser.add_argument(
        "--notes", metavar="PATH", help="Text file of your own notes for the summary (UTF-8)")
    tui_parser.add_argument("--verbose", action="store_true", help="Verbose output")
    tui_parser.add_argument(
        "--backend", choices=["local", "mlx", "api"], help="Transcription backend")
    tui_parser.add_argument("--whisper-device", help="Whisper device override (cpu, cuda, auto)")
    tui_parser.add_argument(
        "--whisper-compute-type",
        help="Whisper compute type (int8, float16, auto)",
    )
    tui_parser.add_argument("--whisper-api-key", help="API key for API backend")
    tui_parser.add_argument("--whisper-api-model", help="Model name for API backend")
    tui_parser.add_argument(
        "--speakers", type=_speaker_count, metavar="N",
        help="Most voices to expect when splitting the remote side by speaker (this run only)",
    )

    process_parser = subparsers.add_parser("process", help="Process existing audio")
    process_parser.add_argument("audio", help="Audio file path")
    process_parser.add_argument("--name", help="Meeting name")
    process_parser.add_argument("--persona", help="Persona slug to use for this audio file")
    process_parser.add_argument(
        "--notes", metavar="PATH", help="Text file of your own notes for the summary (UTF-8)")
    process_parser.add_argument(
        "--retranscribe", action="store_true",
        help="Transcribe again even if a saved transcript sits beside the recording")
    process_parser.add_argument("--verbose", action="store_true", help="Verbose output")
    process_parser.add_argument(
        "--backend", choices=["local", "mlx", "api"], help="Transcription backend")
    process_parser.add_argument("--whisper-device", help="Whisper device (cpu, cuda, auto)")
    process_parser.add_argument(
        "--whisper-compute-type",
        help="Whisper compute type (int8, float16, auto)",
    )
    process_parser.add_argument("--whisper-api-key", help="API key for API backend")
    process_parser.add_argument("--whisper-api-model", help="Model name for API backend")
    process_parser.add_argument(
        "--speakers", type=_speaker_count, metavar="N",
        help="Most voices to expect when splitting the remote side by speaker (this run only)",
    )

    list_parser = subparsers.add_parser("list", help="List meetings")
    list_parser.add_argument("--vault", default="", help="Vault path")

    test_parser = subparsers.add_parser("test", help="Verify setup")
    test_parser.add_argument("--verbose", action="store_true", help="Verbose output")

    config_parser = subparsers.add_parser("config", help="Show/edit config")
    config_parser.add_argument("--edit", action="store_true", help="Edit config")

    setup_parser = subparsers.add_parser("setup", help="Create or update config interactively")
    setup_parser.add_argument(
        "--reset-personas",
        action="store_true",
        default=False,
        help="Re-copy all built-in personas to ~/.tinysteno/personas/ (overwrites existing)",
    )

    args = parser.parse_args()

    if args.command == "setup":
        cmd_setup(args)
        return

    if args.command == "test":
        setup_logging(args.verbose)
        sys.exit(cmd_test(args))

    if args.command == "config":
        cmd_config(args)
        return

    if args.command == "list":
        cmd_list(args)
        return

    if not args.command:
        if not _is_interactive():
            parser.print_help()
            return
        args = parser.parse_args(["tui"])

    setup_logging(getattr(args, "verbose", False))
    first_run = not _config_path().exists()
    try:
        config = load_config()
    except ValueError as exc:  # its message names the file, never the file's content
        print(f"Error: {exc}", file=sys.stderr)
        print("Fix the file, or run: tinysteno config --edit", file=sys.stderr)
        sys.exit(1)
    except OSError as exc:
        print(f"Error: Cannot read {_config_path()}: {exc.strerror or exc}", file=sys.stderr)
        sys.exit(1)

    if args.command == "tui":
        cmd_tui(args, config, first_run=first_run)
    elif args.command == "record":
        cmd_record(args, config)
    elif args.command == "process":
        cmd_process(args, config)


if __name__ == "__main__":
    main()
