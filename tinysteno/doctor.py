"""Setup checks as data: what `tinysteno test` and the TUI System screen both show."""

import importlib
import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from tinysteno.config import (
    DEFAULT_CONFIG, config_path, read_stored, recordings_dir, validate_config,
)
from tinysteno.personas import load_persona

# Seams for tests: patch these, not importlib itself.
_import = importlib.import_module
_find_spec = importlib.util.find_spec


@dataclass(frozen=True)
class Check:
    """One setup check. `detail` is one line; the fix when the check failed."""

    name: str
    ok: bool
    detail: str = ""
    required: bool = True  # a failed optional check is a warning, not a failure


def _one_line(exc: BaseException) -> str:
    text = " ".join(str(exc).split())
    return text or type(exc).__name__


def _safely(name: str, func: Callable[[], Check], required: bool = True) -> Check:
    """Run a check; any exception becomes a failed Check carrying its text."""
    try:
        return func()
    except Exception as exc:
        return Check(name, False, _one_line(exc), required)


def _import_check(name: str, module: str, fix: str = "", attr: str = "") -> Check:
    try:
        loaded = _import(module)
        if attr:
            getattr(loaded, attr)
    except (ImportError, AttributeError) as exc:
        return Check(name, False, fix or _one_line(exc))
    return Check(name, True)


def _config_file() -> Check:
    path = config_path()
    if not path.exists():
        return Check("Config file", False, f"{path} not found")
    validate_config(read_stored())
    return Check("Config file", True)


def _vault(config: dict) -> Check:
    value = config.get("obsidian_vault")
    if not isinstance(value, str) or not value.strip():
        return Check("Vault folder", False, "No vault folder is set")
    vault = Path(value).expanduser()
    if vault.is_dir():
        return Check("Vault folder", True, str(vault))
    return Check("Vault folder", False, f"{vault} is not an existing directory")


def _recordings_folder(config: dict) -> Check:
    folder = recordings_dir(config)
    probe = folder
    while not probe.exists():
        if probe.is_symlink():
            return Check("Recordings folder", False, f"{probe} is a broken symlink")
        if probe == probe.parent:
            break
        probe = probe.parent
    if not probe.is_dir():
        return Check("Recordings folder", False, f"{folder}: {probe} is not a directory")
    if not os.access(probe, os.W_OK | os.X_OK):
        return Check("Recordings folder", False, f"{folder} is not writable ({probe})")
    return Check("Recordings folder", True, str(folder))


def _default_persona(config: dict) -> Check:
    slug = config.get("persona", DEFAULT_CONFIG["persona"])
    load_persona(slug)
    return Check("Default persona", True, str(slug))


def _speaker_separation() -> Check:
    if _find_spec("sherpa_onnx") is None:
        return Check("Speaker separation", False, "pip install 'tinysteno[diarize]'", False)
    return Check("Speaker separation", True, required=False)


def _mlx_backend() -> Check:
    if _find_spec("mlx_whisper") is None:
        return Check("MLX backend", False,
                     "needs an Apple Silicon Mac with a native arm64 Python; "
                     "until then the local backend is used", False)
    return Check("MLX backend", True, required=False)


def run_checks(config: dict | None) -> list[Check]:
    """Run every setup check. `config=None` means no usable config: only module and file checks."""
    checks = [
        _safely("Recorder module", lambda: _import_check("Recorder module", "tinysteno.recorder",
                                                         attr="AudioRecorder")),
        _safely("Transcriber module",
                lambda: _import_check("Transcriber module", "tinysteno.transcriber",
                                                            attr="WhisperTranscriber")),
        _safely("sounddevice",
                lambda: _import_check("sounddevice", "sounddevice", "pip install sounddevice")),
        _safely("faster-whisper",
                lambda: _import_check("faster-whisper", "faster_whisper",
                                      "pip install faster-whisper")),
        _safely("openai", lambda: _import_check("openai", "openai", "pip install openai")),
        _safely("pyyaml", lambda: _import_check("pyyaml", "yaml", "pip install pyyaml")),
        _safely("rich", lambda: _import_check("rich", "rich", "pip install rich")),
        _safely("Config file", _config_file),
    ]
    if config is not None:
        checks.append(_safely("Vault folder", lambda: _vault(config)))
        checks.append(_safely("Recordings folder", lambda: _recordings_folder(config)))
        checks.append(_safely("Default persona", lambda: _default_persona(config)))
        if config.get("transcription_backend") == "mlx":
            checks.append(_safely("MLX backend", _mlx_backend, required=False))
        if config.get("speaker_split"):
            checks.append(_safely("Speaker separation", _speaker_separation, required=False))
    return checks
