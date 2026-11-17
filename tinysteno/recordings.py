"""List recordings and delete one safely. No UI, no side effects beyond the file system."""

import os
import stat
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from tinysteno.levels import levels_path
from tinysteno.transcript import sidecar_path


@dataclass(frozen=True)
class RecordingInfo:
    """One WAV in the recordings directory."""

    path: Path
    size_bytes: int
    modified: datetime  # naive local time of the file's mtime
    has_transcript: bool  # <wav>.transcript.json exists


def _is_wav(path: Path) -> bool:
    return path.suffix.lower() == ".wav"


def list_recordings(directory: Path) -> list[RecordingInfo]:
    """WAV files directly in `directory`, newest first; [] when it is missing."""
    try:
        entries = list(directory.iterdir())
    except (FileNotFoundError, NotADirectoryError):
        return []
    found = []
    for entry in entries:
        if not _is_wav(entry):
            continue
        try:
            info = entry.stat()  # follows a symlinked WAV: it can still be processed
            if not entry.is_file():
                continue
            has_transcript = sidecar_path(entry).exists()
        except OSError:  # vanished between listing and stat
            continue
        found.append(RecordingInfo(
            path=entry,
            size_bytes=info.st_size,
            modified=datetime.fromtimestamp(info.st_mtime),
            has_transcript=has_transcript,
        ))
    found.sort(key=lambda info: (info.modified, info.path.name), reverse=True)
    return found


def _remove_by_name(directory: Path, names: list[str]) -> list[str]:
    """Unlink `names` (WAV first) inside one opened directory; return those removed.

    Every operation is relative to the descriptor, so swapping the folder for a symlink
    after the checks cannot redirect the deletion.
    """
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        mode = os.stat(names[0], dir_fd=fd, follow_symlinks=False).st_mode
        if stat.S_ISLNK(mode):
            raise ValueError(f"Refusing to delete a symlink: {names[0]}")
        if not stat.S_ISREG(mode):
            raise FileNotFoundError(f"No such recording: {names[0]}")
        removed = []
        for index, name in enumerate(names):
            if index:
                try:
                    mode = os.stat(name, dir_fd=fd, follow_symlinks=False).st_mode
                except FileNotFoundError:
                    continue
                if stat.S_ISDIR(mode):
                    continue
            os.unlink(name, dir_fd=fd)
            removed.append(name)
        return removed
    finally:
        os.close(fd)


def delete_recording(directory: Path, wav: Path) -> list[Path]:
    """Delete `wav` and its transcript and levels sidecars; return what was removed.

    Raises ValueError unless `wav` is a `.wav` that is not a symlink and lives directly in
    `directory`; FileNotFoundError when it is not an existing regular file. Sidecars are
    unlinked themselves, never followed. The deletion happens by name inside the opened
    directory where the platform supports it.
    """
    directory, wav = Path(directory), Path(wav)
    if not _is_wav(wav):
        raise ValueError(f"Not a .wav file: {wav}")
    if wav.is_symlink():
        raise ValueError(f"Refusing to delete a symlink: {wav}")
    if wav.parent.resolve() != directory.resolve():
        raise ValueError(f"{wav} is not directly inside {directory}")
    sidecars = (sidecar_path(wav), levels_path(wav))
    if os.unlink in os.supports_dir_fd and os.stat in os.supports_dir_fd:
        names = [wav.name] + [sidecar.name for sidecar in sidecars]
        return [wav.parent / name for name in _remove_by_name(directory.resolve(), names)]
    if not wav.is_file():
        raise FileNotFoundError(f"No such recording: {wav}")
    removed = []
    wav.unlink()
    removed.append(wav)
    for sidecar in sidecars:
        if sidecar.is_symlink() or sidecar.is_file():
            sidecar.unlink()
            removed.append(sidecar)
    return removed
