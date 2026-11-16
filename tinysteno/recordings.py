"""List recordings and delete one safely. No UI, no side effects beyond the file system."""

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
            stat = entry.stat()  # follows a symlinked WAV: it can still be processed
            if not entry.is_file():
                continue
            has_transcript = sidecar_path(entry).exists()
        except OSError:  # vanished between listing and stat
            continue
        found.append(RecordingInfo(
            path=entry,
            size_bytes=stat.st_size,
            modified=datetime.fromtimestamp(stat.st_mtime),
            has_transcript=has_transcript,
        ))
    found.sort(key=lambda info: (info.modified, info.path.name), reverse=True)
    return found


def delete_recording(directory: Path, wav: Path) -> list[Path]:
    """Delete `wav` and its transcript and levels sidecars; return what was removed.

    Raises ValueError unless `wav` is a `.wav` that is not a symlink and lives directly in
    `directory`; FileNotFoundError when it is not an existing regular file. Sidecars are
    unlinked themselves, never followed.
    """
    if not _is_wav(wav):
        raise ValueError(f"Not a .wav file: {wav}")
    if wav.is_symlink():
        raise ValueError(f"Refusing to delete a symlink: {wav}")
    if wav.parent.resolve() != directory.resolve():
        raise ValueError(f"{wav} is not directly inside {directory}")
    if not wav.is_file():
        raise FileNotFoundError(f"No such recording: {wav}")
    removed = []
    wav.unlink()
    removed.append(wav)
    for sidecar in (sidecar_path(wav), levels_path(wav)):
        if sidecar.is_symlink() or sidecar.is_file():
            sidecar.unlink()
            removed.append(sidecar)
    return removed
