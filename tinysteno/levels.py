"""The gain the recorder applied to each channel, saved beside the WAV.

The recorder scales each channel to its own peak, so the WAV no longer says how loud the
channels were relative to each other. `Meeting-1.wav.levels.json` keeps the gains so level
decisions (silence, speaker bleed) can be made on the levels that were recorded.
"""
import json
import logging
import math
import os
import tempfile
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)

LEVELS_VERSION = 1


def levels_path(wav_path: str) -> Path:
    """Where a recording's gains live: `Meeting-1.wav.levels.json`."""
    return Path(str(wav_path) + ".levels.json")


def save_levels(wav_path: str, gains: Sequence[float]) -> None:
    """Write the per-channel gains beside the recording, atomically.

    Any failure is logged and ignored: it never fails saving the recording.
    """
    target = levels_path(wav_path)
    temp_name = None
    try:
        payload = {"version": LEVELS_VERSION, "gains": [float(gain) for gain in gains]}
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=target.parent, prefix=target.name + ".",
            suffix=".tmp", delete=False,
        ) as handle:
            temp_name = handle.name
            json.dump(payload, handle, allow_nan=False)
        os.replace(temp_name, target)
    except Exception as err:
        logger.warning("Could not save the recording levels to %s: %s", target, err)
    finally:
        if temp_name and os.path.exists(temp_name):
            try:
                os.unlink(temp_name)
            except OSError:
                pass


def _is_gain(value: object) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value > 0)


def load_levels(wav_path: str, channels: int) -> list[float] | None:
    """The gains for a recording of `channels` channels, or None when they cannot be trusted.

    None when the file is missing, older than the recording, unreadable, of another version,
    or not a list of finite positive numbers with one entry per channel.
    """
    path = levels_path(wav_path)
    try:
        if path.stat().st_mtime < Path(wav_path).stat().st_mtime:
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as err:
        logger.info("Ignoring recording levels %s: %s", path, err)
        return None
    return _parse_levels(data, channels)


def _parse_levels(data: object, channels: int) -> list[float] | None:
    """The gains a parsed levels file holds, or None unless it is exactly as written."""
    if not isinstance(data, dict) or data.get("version") != LEVELS_VERSION:
        return None
    gains = data.get("gains")
    if (not isinstance(gains, list) or len(gains) != channels
            or not all(_is_gain(gain) for gain in gains)):
        return None
    return [float(gain) for gain in gains]
