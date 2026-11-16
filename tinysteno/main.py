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
from tinysteno.log_file import log_to_file
from tinysteno.signals import deferred_exit_signals
from tinysteno.personas import (
    load_persona, list_personas, seed_builtin_personas,
    PersonaNotFoundError, PersonaInvalidError, Persona,
)


def setup_logging(verbose: bool = False):
    """Configure logging level."""
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def load_config() -> dict:
    """Load configuration from ~/.tinysteno/config.yaml."""
    import yaml

    config_path = Path.home() / ".tinysteno" / "config.yaml"

    if not config_path.exists():
        default_config = {
            "obsidian_vault": str(Path.home() / "Obsidian" / "Vault"),
            "api_key": "ollama",
            "base_url": "http://localhost:11434/v1",
            "model": "llama3.2:3b",
            "whisper_model": "small",
            "transcription_backend": "local",
            "whisper_device": "auto",
            "whisper_compute_type": "auto",
            "whisper_api_key": "",
            "whisper_base_url": "",
            "whisper_api_model": "whisper-1",
            "whisper_vad": True,
            "whisper_prompt": "",
            "hallucination_filter": True,
            "bleed_filter": True,
            "diarization": False,
            "transcript_timestamps": True,
            "auto_title": True,
            "auto_tags": True,
            "output_folder": "meetings",
            "sample_rate": 44100,
            "channels": 1,
            "persona": "default",
            "session_timeout_minutes": DEFAULT_SESSION_TIMEOUT_MINUTES,
        }
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.dump(default_config, default_flow_style=False))

    # Seed built-in personas on first run (silent; only when dir absent)
    personas_dir = config_path.parent / "personas"
    if not personas_dir.exists():
        seed_builtin_personas()

    config = yaml.safe_load(config_path.read_text())
    if not isinstance(config, dict):
        raise ValueError(f"Config file is malformed: {config_path}")
    _validate_config(config)
    for key in ("obsidian_vault", "recordings_path"):
        if config.get(key):
            config[key] = str(Path(config[key]).expanduser())
    return config


def _recordings_dir(config: dict) -> Path:
    """Return the effective recordings directory from config."""
    if config.get("recordings_path"):
        return Path(config["recordings_path"]).expanduser()
    vault = config.get("obsidian_vault", str(Path.home() / "Obsidian" / "Vault"))
    folder = config.get("output_folder", "meetings")
    return Path(vault) / folder / "audio"


def _validate_config(config: dict) -> None:
    """Raise ValueError if required config fields have wrong types."""
    int_fields = {"sample_rate": int, "channels": int}
    for field, expected_type in int_fields.items():
        if field in config and not isinstance(config[field], expected_type):
            raise ValueError(
                f"Config field '{field}' must be {expected_type.__name__}, "
                f"got {type(config[field]).__name__}"
            )
    if "channels" in config and config["channels"] not in (1, 2):
        raise ValueError("Config field 'channels' must be 1 or 2")
    if "session_timeout_minutes" in config:
        if not _is_timeout_minutes(config["session_timeout_minutes"]):
            raise ValueError(
                "Config field 'session_timeout_minutes' must be a number >= 0 (0 disables)"
            )


def _is_timeout_minutes(value) -> bool:
    """True for a finite int/float >= 0 that is not a bool."""
    return (
        isinstance(value, (int, float)) and not isinstance(value, bool)
        and math.isfinite(value) and value >= 0
    )


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
    )
    if backend == "api":
        return ApiTranscriber(
            api_key=config.get("whisper_api_key") or config["api_key"],
            base_url=config.get("whisper_base_url") or config.get("base_url"),
            model=config.get("whisper_api_model", "whisper-1"),
            prompt=prompt,
            options=options,
        )
    return WhisperTranscriber(
        model_size=config.get("whisper_model", "small"),
        device=config.get("whisper_device", "auto"),
        compute_type=config.get("whisper_compute_type", "auto"),
        vad=config.get("whisper_vad", True),
        prompt=prompt,
        options=options,
    )


def _apply_cli_overrides(config: dict, args: argparse.Namespace) -> None:
    """Merge CLI flags into config in-place (CLI overrides take precedence)."""
    cli_to_config = {
        "backend": "transcription_backend",
        "whisper_device": "whisper_device",
        "whisper_compute_type": "whisper_compute_type",
        "whisper_api_key": "whisper_api_key",
        "whisper_api_model": "whisper_api_model",
    }
    for cli_key, config_key in cli_to_config.items():
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


def _process_audio(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    wav_path: str,
    name: Optional[str],
    config: dict,
    logger: logging.Logger,
    persona: Persona,
    timestamp: datetime,
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
            )
        except PipelineError as err:
            cause = err.__cause__
            if err.stage is Stage.EXPORT and isinstance(cause, RuntimeError):
                print(f"Error rendering note: {cause}")
                return
            raise cause from None  # pylint: disable=raising-non-exception

    logger.debug(f"Duration: {result.duration_seconds:.0f}s")
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


def cmd_record(args, config):
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
    _process_audio(wav_path, name, config, logger, persona, timestamp)


def _is_interactive() -> bool:
    """True when both stdin and stdout are terminals."""
    return sys.stdin.isatty() and sys.stdout.isatty()


def cmd_tui(args, config):
    """Launch the Textual interface."""
    from tinysteno.tui import run_tui
    from tinysteno.tui.deps import TuiDeps

    _apply_cli_overrides(config, args)

    def make_recorder():
        return AudioRecorder(
            sample_rate=config.get("sample_rate", 44100),
            channels=config.get("channels", 1),
            recordings_dir=_recordings_dir(config),
        )

    deps = TuiDeps(
        make_recorder=make_recorder,
        run_pipeline=lambda wav_path, name, persona, timestamp, reporter: run_pipeline(
            wav_path, name, config, persona, timestamp,
            transcriber=create_transcriber(config),
            orchestrator_cls=Orchestrator,
            exporter_cls=ObsidianExporter,
            reporter=reporter,
        ),
        list_personas=list_personas,
        load_persona=load_persona,
        recordings_dir=_recordings_dir(config),
    )
    log_path = Path.home() / ".tinysteno" / "tui.log"
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

    timestamp = datetime.fromtimestamp(audio_file.stat().st_mtime)

    _apply_cli_overrides(config, args)

    logger = logging.getLogger(__name__)
    _process_audio(str(audio_file), args.name or None, config, logger, persona, timestamp)


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


def cmd_test(_args):  # pylint: disable=too-many-statements  # large interactive CLI setup function
    """Verify setup before first use."""
    from rich.console import Console

    console = Console()
    issues = []

    console.print("\n[TinySteno Setup Test]")

    try:
        from tinysteno.recorder import AudioRecorder as _  # noqa: F401  # pylint: disable=unused-import,reimported,redefined-outer-name  # import-existence check

        console.print("✓ Recorder module")
    except ImportError as e:
        issues.append(f"Recorder: {e}")
        console.print(f"✗ Recorder: {e}")

    try:
        from tinysteno.transcriber import WhisperTranscriber as _  # noqa: F401  # pylint: disable=unused-import,reimported,redefined-outer-name  # import-existence check

        console.print("✓ Transcriber module")
    except ImportError as e:
        issues.append(f"Transcriber: {e}")
        console.print(f"✗ Transcriber: {e}")

    try:
        import sounddevice as _  # noqa: F401  # pylint: disable=unused-import  # import-existence check

        console.print("✓ sounddevice")
    except ImportError:
        issues.append("sounddevice")
        console.print("✗ sounddevice (pip install sounddevice)")

    try:
        import faster_whisper as _  # noqa: F401  # pylint: disable=unused-import  # import-existence check

        console.print("✓ faster-whisper")
    except ImportError:
        issues.append("faster-whisper")
        console.print("✗ faster-whisper (pip install faster-whisper)")

    try:
        import openai as _  # noqa: F401  # pylint: disable=unused-import  # import-existence check

        console.print("✓ openai")
    except ImportError:
        issues.append("openai")
        console.print("✗ openai (pip install openai)")

    try:
        import yaml as _  # noqa: F401  # pylint: disable=unused-import  # import-existence check

        console.print("✓ pyyaml")
    except ImportError:
        issues.append("pyyaml")
        console.print("✗ pyyaml (pip install pyyaml)")

    try:
        import rich as _  # noqa: F401  # pylint: disable=unused-import  # import-existence check

        console.print("✓ rich")
    except ImportError:
        issues.append("rich")
        console.print("✗ rich (pip install rich)")

    config_path = Path.home() / ".tinysteno" / "config.yaml"
    if config_path.exists():
        console.print("✓ Config file exists")
    else:
        issues.append("config missing")
        console.print(f"✗ Config: {config_path} not found")

    if not issues:
        console.print("\n✓ All checks passed!")
        return 0
    console.print(f"\n✗ Found {len(issues)} issue(s)")
    return 1


def cmd_config(args):
    """Show or edit configuration."""
    import yaml

    config_path = Path.home() / ".tinysteno" / "config.yaml"
    config = load_config()

    if args.edit:
        editor = (
            os.environ.get("VISUAL")
            or os.environ.get("EDITOR")
            or "nano"
        )
        editor_cmd = shlex.split(editor) + [str(config_path)]
        subprocess.call(editor_cmd)
    else:
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
    user_dir = Path.home() / ".tinysteno" / "personas"
    if result["copied"]:
        console.print(f"[green]✓[/green] Personas seeded in [bold]{user_dir}[/bold]")
        console.print(f"  Copied: {', '.join(result['copied'])}")
    if result["skipped"]:
        console.print(f"  Skipped (kept yours): {', '.join(result['skipped'])}")


def _write_config(config_path: Path, config: dict) -> None:
    """Persist config dict to YAML file."""
    import yaml  # pylint: disable=import-outside-toplevel
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.dump(config, default_flow_style=False), encoding="utf-8")


def cmd_setup(_args):  # pylint: disable=too-many-statements,too-many-locals  # large interactive CLI setup function
    """Interactively create or update ~/.tinysteno/config.yaml."""
    import yaml
    from rich.console import Console
    from rich.rule import Rule

    console = Console()

    # --reset-personas: re-seed built-ins without running the config wizard
    if getattr(_args, "reset_personas", False):
        result = seed_builtin_personas(force=True)
        _print_seed_summary(console, result)
        return

    config_path = Path.home() / ".tinysteno" / "config.yaml"

    # Load existing values as defaults, or fall back to built-in defaults.
    if config_path.exists():
        existing = yaml.safe_load(config_path.read_text()) or {}
        action = "Updating"
    else:
        existing = {}
        action = "Creating"

    def get(key, fallback):
        return existing.get(key, fallback)

    console.print()
    console.print(Rule("[bold]TinySteno Setup[/bold]"))
    console.print()
    console.print(f"{action} [bold]{config_path}[/bold]")
    console.print("Press Enter to keep the current value.\n")

    # --- Obsidian ---
    console.print(Rule("[dim]Obsidian Vault[/dim]"))
    obsidian_vault = _prompt(
        console,
        "Vault path",
        get("obsidian_vault", str(Path.home() / "Obsidian" / "Vault")),
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
    config = {
        "obsidian_vault": obsidian_vault,
        "output_folder": output_folder,
        **( {"recordings_path": recordings_path} if recordings_path else {} ),
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
    }

    _write_config(config_path, config)

    # Seed built-in personas, prompting on conflicts
    seed_result = seed_builtin_personas(interactive=True)
    _print_seed_summary(console, seed_result)

    console.print()
    console.print(Rule())
    console.print(f"[green]✓[/green] Config written to [bold]{config_path}[/bold]")
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
    record_parser.add_argument("--verbose", action="store_true", help="Verbose output")
    record_parser.add_argument("--backend", choices=["local", "api"], help="Transcription backend")
    record_parser.add_argument("--whisper-device", help="Whisper device override (cpu, cuda, auto)")
    record_parser.add_argument(
        "--whisper-compute-type",
        help="Whisper compute type (int8, float16, auto)",
    )
    record_parser.add_argument("--whisper-api-key", help="API key for API backend")
    record_parser.add_argument("--whisper-api-model", help="Model name for API backend")

    tui_parser = subparsers.add_parser("tui", help="Interactive terminal interface")
    tui_parser.add_argument("--now", action="store_true", help="Start recording immediately")
    tui_parser.add_argument("--name", help="Meeting name")
    tui_parser.add_argument("--persona", help="Persona slug to use for this recording")
    tui_parser.add_argument(
        "--timeout", type=_non_negative_minutes, metavar="MINUTES",
        help="Auto-stop after MINUTES (0 disables; default from config)",
    )
    tui_parser.add_argument("--verbose", action="store_true", help="Verbose output")
    tui_parser.add_argument("--backend", choices=["local", "api"], help="Transcription backend")
    tui_parser.add_argument("--whisper-device", help="Whisper device override (cpu, cuda, auto)")
    tui_parser.add_argument(
        "--whisper-compute-type",
        help="Whisper compute type (int8, float16, auto)",
    )
    tui_parser.add_argument("--whisper-api-key", help="API key for API backend")
    tui_parser.add_argument("--whisper-api-model", help="Model name for API backend")

    process_parser = subparsers.add_parser("process", help="Process existing audio")
    process_parser.add_argument("audio", help="Audio file path")
    process_parser.add_argument("--name", help="Meeting name")
    process_parser.add_argument("--persona", help="Persona slug to use for this audio file")
    process_parser.add_argument("--verbose", action="store_true", help="Verbose output")
    process_parser.add_argument("--backend", choices=["local", "api"], help="Transcription backend")
    process_parser.add_argument("--whisper-device", help="Whisper device (cpu, cuda, auto)")
    process_parser.add_argument(
        "--whisper-compute-type",
        help="Whisper compute type (int8, float16, auto)",
    )
    process_parser.add_argument("--whisper-api-key", help="API key for API backend")
    process_parser.add_argument("--whisper-api-model", help="Model name for API backend")

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
    config = load_config()

    if args.command == "tui":
        cmd_tui(args, config)
    elif args.command == "record":
        cmd_record(args, config)
    elif args.command == "process":
        cmd_process(args, config)


if __name__ == "__main__":
    main()
