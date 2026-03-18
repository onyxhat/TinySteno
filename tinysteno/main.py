"""CLI entry point for TinySteno."""

import argparse
import logging
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from tinysteno.recorder import AudioRecorder
from tinysteno.transcriber import WhisperTranscriber
from tinysteno.summarizer import Summarizer
from tinysteno.obsidian import ObsidianExporter


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
            "diarization": False,
            "auto_title": True,
            "tags": ["meeting"],
            "output_folder": "meetings",
            "sample_rate": 44100,
            "channels": 1,
        }
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.dump(default_config, default_flow_style=False))

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


def _process_audio(
    wav_path: str,
    name: Optional[str],
    config: dict,
    logger: logging.Logger,
) -> None:
    """Shared pipeline: transcribe → summarize → export."""
    print("Transcribing...")
    transcriber = WhisperTranscriber(
        model_size=config.get("whisper_model", "small")
    )
    result = transcriber.transcribe(wav_path, diarize=config.get("diarization", False))
    logger.debug(f"Detected language: {result['detected_language']}")
    logger.debug(f"Duration: {result['duration_seconds']:.0f}s")

    transcript = result["diarised_text"] or result["text"]
    if not transcript.strip():
        print("No speech detected, skipping export.")
        return

    summary = None
    summarizer = None
    if config.get("api_key"):
        print("Summarizing...")
        summarizer = Summarizer(
            api_key=config["api_key"],
            base_url=config["base_url"],
            model=config["model"],
        )
        summary = summarizer.summarize(transcript)

    title = name or "Meeting"
    if config.get("auto_title") and summary and summarizer:
        print("Generating title...")
        title = summarizer.generate_title(summary.overview)

    timestamp = datetime.now()

    exporter = ObsidianExporter(
        vault_path=config["obsidian_vault"],
        output_folder=config.get("output_folder", "meetings"),
        tags=config.get("tags", ["meeting"]),
    )

    meeting_path = exporter.export_meeting(
        data=summary.to_dict() if summary else {},
        title=title,
        timestamp=timestamp,
        transcript=transcript,
    )

    print(f"Meeting: {meeting_path}")


def cmd_record(args, config):
    """Record audio and process into meeting notes."""
    logger = logging.getLogger(__name__)

    name = args.name or None
    recorder = AudioRecorder(
        sample_rate=config.get("sample_rate", 44100),
        channels=config.get("channels", 1),
        recordings_dir=_recordings_dir(config),
    )

    print("Recording started... Press Ctrl+C to stop.")
    wav_path = recorder.start(name)
    logger.debug(f"Recording to: {wav_path}")

    try:
        import time

        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        recorder.stop()

    print("Recording stopped.")
    _process_audio(wav_path, name, config, logger)


def cmd_process(args, config):
    """Process an existing audio file."""
    audio_file = Path(args.audio)
    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_file}")

    logger = logging.getLogger(__name__)
    _process_audio(str(audio_file), args.name or None, config, logger)


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


def cmd_test(args):
    """Verify setup before first use."""
    from rich.console import Console

    console = Console()
    issues = []

    console.print("\n[TinySteno Setup Test]")

    try:
        from tinysteno.recorder import AudioRecorder

        console.print("✓ Recorder module")
    except ImportError as e:
        issues.append(f"Recorder: {e}")
        console.print(f"✗ Recorder: {e}")

    try:
        from tinysteno.transcriber import WhisperTranscriber

        console.print("✓ Transcriber module")
    except ImportError as e:
        issues.append(f"Transcriber: {e}")
        console.print(f"✗ Transcriber: {e}")

    try:
        import sounddevice

        console.print("✓ sounddevice")
    except ImportError:
        issues.append("sounddevice")
        console.print("✗ sounddevice (pip install sounddevice)")

    try:
        import faster_whisper

        console.print("✓ faster-whisper")
    except ImportError:
        issues.append("faster-whisper")
        console.print("✗ faster-whisper (pip install faster-whisper)")

    try:
        import openai

        console.print("✓ openai")
    except ImportError:
        issues.append("openai")
        console.print("✗ openai (pip install openai)")

    try:
        import yaml

        console.print("✓ pyyaml")
    except ImportError:
        issues.append("pyyaml")
        console.print("✗ pyyaml (pip install pyyaml)")

    try:
        import rich

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
    else:
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


def cmd_setup(args):
    """Interactively create or update ~/.tinysteno/config.yaml."""
    import yaml
    from rich.console import Console
    from rich.rule import Rule

    console = Console()
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
    console.print(Rule(f"[bold]TinySteno Setup[/bold]"))
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

    tags_default = ", ".join(get("tags", ["meeting"]))
    tags_input = _prompt(
        console,
        "Default tags",
        tags_default,
        "comma-separated list of tags",
    )
    tags = [t.strip() for t in tags_input.split(",") if t.strip()]

    # --- LLM ---
    console.print()
    console.print(Rule("[dim]LLM / Summarization[/dim]"))
    console.print("  [dim]Ollama (local): base_url = http://localhost:11434/v1, api_key = ollama[/dim]")
    console.print("  [dim]OpenAI (cloud): base_url = https://api.openai.com/v1, api_key = sk-...[/dim]")
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

    # --- Whisper ---
    console.print()
    console.print(Rule("[dim]Transcription (Whisper)[/dim]"))
    console.print("  [dim]Model sizes (speed ↔ accuracy): tiny · base · small · medium · large[/dim]")
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
        "requires stereo recording (channels = 2)",
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

    channels_fallback = "2" if diarization else "1"
    channels_input = _prompt(
        console,
        "Channels",
        str(get("channels", channels_fallback)),
        "1 = mono, 2 = stereo (required for diarization)",
    )
    try:
        channels = int(channels_input)
    except ValueError:
        channels = 2 if diarization else 1

    sample_rate_input = _prompt(console, "Sample rate", str(get("sample_rate", 44100)), "Hz")
    try:
        sample_rate = int(sample_rate_input)
    except ValueError:
        sample_rate = 44100

    # --- Write ---
    config = {
        "obsidian_vault": obsidian_vault,
        "output_folder": output_folder,
        **( {"recordings_path": recordings_path} if recordings_path else {} ),
        "tags": tags,
        "api_key": api_key,
        "base_url": base_url,
        "model": model,
        "auto_title": auto_title,
        "whisper_model": whisper_model,
        "diarization": diarization,
        "channels": channels,
        "sample_rate": sample_rate,
    }

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.dump(config, default_flow_style=False))

    console.print()
    console.print(Rule())
    console.print(f"[green]✓[/green] Config written to [bold]{config_path}[/bold]")
    console.print()
    console.print("Run [bold]tinysteno setup[/bold] again any time to update settings.")
    console.print("Run [bold]tinysteno test[/bold] to verify your setup.")
    console.print()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="TinySteno - Meeting recorder")

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    record_parser = subparsers.add_parser("record", help="Record a meeting")
    record_parser.add_argument("--name", help="Meeting name")
    record_parser.add_argument("--verbose", action="store_true", help="Verbose output")

    process_parser = subparsers.add_parser("process", help="Process existing audio")
    process_parser.add_argument("audio", help="Audio file path")
    process_parser.add_argument("--name", help="Meeting name")
    process_parser.add_argument("--verbose", action="store_true", help="Verbose output")

    list_parser = subparsers.add_parser("list", help="List meetings")
    list_parser.add_argument("--vault", default="", help="Vault path")

    test_parser = subparsers.add_parser("test", help="Verify setup")
    test_parser.add_argument("--verbose", action="store_true", help="Verbose output")

    config_parser = subparsers.add_parser("config", help="Show/edit config")
    config_parser.add_argument("--edit", action="store_true", help="Edit config")

    subparsers.add_parser("setup", help="Create or update config interactively")

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
        parser.print_help()
        return

    setup_logging(getattr(args, "verbose", False))
    config = load_config()

    if args.command == "record":
        cmd_record(args, config)
    elif args.command == "process":
        cmd_process(args, config)


if __name__ == "__main__":
    main()
