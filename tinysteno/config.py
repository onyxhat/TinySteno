"""Configuration for TinySteno: locating, loading, validating and saving config.yaml.

This module is UI-free: it must not import the CLI, argparse, rich or the TUI.
"""
import math
import os
import stat
import tempfile
from pathlib import Path

from tinysteno.personas import seed_builtin_personas
from tinysteno.session import DEFAULT_SESSION_TIMEOUT_MINUTES

DEFAULT_CONFIG: dict = {
    "obsidian_vault": "~/Obsidian/Vault",
    "api_key": "ollama",
    "base_url": "http://localhost:11434/v1",
    "model": "llama3.2:3b",
    "llm_context_tokens": 8192,
    "whisper_model": "small",
    "transcription_backend": "local",
    "whisper_device": "auto",
    "whisper_compute_type": "auto",
    "whisper_api_key": "",
    "whisper_base_url": "",
    "whisper_api_model": "whisper-1",
    "whisper_api_diarize": False,
    "whisper_api_timeout": 600,
    "whisper_vad": True,
    "whisper_prompt": "",
    "hallucination_filter": True,
    "bleed_filter": True,
    "diarization": False,
    "speaker_split": False,
    "transcript_timestamps": True,
    "auto_title": True,
    "auto_tags": True,
    "output_folder": "meetings",
    "sample_rate": 44100,
    "channels": 1,
    "persona": "default",
    "session_timeout_minutes": DEFAULT_SESSION_TIMEOUT_MINUTES,
}


def config_dir() -> Path:
    """Return ~/.tinysteno, resolved against the current home on every call."""
    return Path.home() / ".tinysteno"


def config_path() -> Path:
    """Return ~/.tinysteno/config.yaml, resolved against the current home on every call."""
    return config_dir() / "config.yaml"


def is_timeout_minutes(value) -> bool:
    """True for a finite int/float >= 0 that is not a bool."""
    return (
        isinstance(value, (int, float)) and not isinstance(value, bool)
        and math.isfinite(value) and value >= 0
    )


def is_positive_number(value) -> bool:
    """True for a finite int/float > 0 that is not a bool."""
    return (
        isinstance(value, (int, float)) and not isinstance(value, bool)
        and math.isfinite(value) and value > 0
    )


def is_speaker_count(value) -> bool:
    """True for an int >= 1 that is not a bool."""
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def validate_config(config: dict) -> None:
    """Raise ValueError if required config fields have wrong types."""
    int_fields = {"sample_rate": int, "channels": int}
    for field, expected_type in int_fields.items():
        if field in config and not isinstance(config[field], expected_type):
            raise ValueError(
                f"Config field '{field}' must be {expected_type.__name__}, "
                f"got {type(config[field]).__name__}"
            )
    if "llm_context_tokens" in config and (
        isinstance(config["llm_context_tokens"], bool)
        or not isinstance(config["llm_context_tokens"], int)
    ):
        raise ValueError(
            "Config field 'llm_context_tokens' must be int, "
            f"got {type(config['llm_context_tokens']).__name__}"
        )
    if "channels" in config and config["channels"] not in (1, 2):
        raise ValueError("Config field 'channels' must be 1 or 2")
    if "max_speakers" in config and not is_speaker_count(config["max_speakers"]):
        raise ValueError("Config field 'max_speakers' must be a whole number >= 1")
    if "whisper_api_timeout" in config and not is_positive_number(config["whisper_api_timeout"]):
        raise ValueError("Config field 'whisper_api_timeout' must be a positive number of seconds")
    if "session_timeout_minutes" in config:
        if not is_timeout_minutes(config["session_timeout_minutes"]):
            raise ValueError(
                "Config field 'session_timeout_minutes' must be a number >= 0 (0 disables)"
            )


def recordings_dir(config: dict) -> Path:
    """Return the effective recordings directory from config."""
    if config.get("recordings_path"):
        return Path(config["recordings_path"]).expanduser()
    vault = Path(config.get("obsidian_vault", DEFAULT_CONFIG["obsidian_vault"])).expanduser()
    folder = config.get("output_folder", "meetings")
    return vault / folder / "audio"


def write_config(path: Path, config: dict) -> None:
    """Atomically persist a config dict as YAML.

    Writes a temp file next to the target, then os.replace()s it into place. A new file gets
    mode 0o600 (it may hold API keys); an existing file keeps its mode. On failure the target
    is untouched and no temp file is left behind.
    """
    import yaml
    path = Path(os.path.realpath(path))  # write through a symlink, never replace it
    if path.exists() and not os.access(path, os.W_OK):
        raise PermissionError(f"Config file is read-only: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        mode = stat.S_IMODE(path.stat().st_mode)
    except FileNotFoundError:
        mode = 0o600
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(yaml.dump(config, default_flow_style=False))
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp_name, mode)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _parse_yaml(text: str, path: Path):
    """yaml.safe_load, raising ValueError that names where the file is broken, never what it
    says there: PyYAML's own message quotes the line, which may hold an API key."""
    import yaml
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        where = "unreadable"
        if mark is not None:
            where = f"line {mark.line + 1}, column {mark.column + 1}"
        message = f"Config file is not valid YAML ({where}): {path}"
    # Raised outside the except block, and from None: the original error is neither
    # chained nor kept as context, so no traceback can show its text.
    raise ValueError(message) from None


def read_stored() -> dict:
    """Return the config file's own contents, unexpanded; {} when the file is missing.

    An empty or comments-only file counts as {}. Raises ValueError when the file is not
    valid YAML or not a mapping.
    """
    path = config_path()
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    stored = _parse_yaml(text, path)
    if stored is None:
        return {}
    if not isinstance(stored, dict):
        raise ValueError(f"Config file is malformed: {path}")
    return stored


def invalid_settings(stored: dict) -> dict:
    """Return {key: reason} for each stored key that fails validation on its own."""
    problems = {}
    for key, value in stored.items():
        try:
            validate_config({key: value})
        except ValueError as exc:
            problems[key] = str(exc)
    return problems


def update_config(changes: dict) -> dict:
    """Merge `changes` into the stored config and save it; a value of None removes the key.

    Unknown keys already in the file are kept. Raises ValueError (malformed file or invalid
    result) or OSError (cannot write); in both cases the file is left unchanged.
    """
    stored = read_stored()
    for key, value in changes.items():
        if value is None:
            stored.pop(key, None)
        else:
            stored[key] = value
    validate_config(stored)
    write_config(config_path(), stored)
    return stored


def load_config() -> dict:
    """Load configuration from ~/.tinysteno/config.yaml, creating defaults on first run.

    Raises ValueError when the file is not valid YAML, not a mapping or holds invalid values.
    """
    path = config_path()

    if not path.exists():
        default_config = dict(DEFAULT_CONFIG)
        default_config["obsidian_vault"] = str(Path.home() / "Obsidian" / "Vault")
        write_config(path, default_config)

    # Seed built-in personas on first run (silent; only when dir absent)
    personas_dir = path.parent / "personas"
    if not personas_dir.exists():
        seed_builtin_personas()

    config = _parse_yaml(path.read_text(), path)
    if not isinstance(config, dict):
        raise ValueError(f"Config file is malformed: {path}")
    validate_config(config)
    for key in ("obsidian_vault", "recordings_path"):
        if config.get(key):
            config[key] = str(Path(config[key]).expanduser())
    return config
