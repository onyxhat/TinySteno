"""Shared steps for the Recordings screen tests."""
import os
from pathlib import Path

from textual.widgets import DataTable, Static

from tinysteno.tui.recordings import RecordingsScreen

CONFIG = {"persona": "1on1", "session_timeout_minutes": 120}
BASE_TIME = 1_700_000_000


def make_wav(directory: Path, name: str, index: int, sidecars: bool) -> Path:
    """One WAV whose mtime rises with ``index``."""
    wav = directory / name
    wav.write_bytes(b"RIFF" * 10)
    os.utime(wav, (BASE_TIME + index * 60, BASE_TIME + index * 60))
    if sidecars:
        Path(str(wav) + ".transcript.json").write_text("{}", encoding="utf-8")
        Path(str(wav) + ".levels.json").write_text("{}", encoding="utf-8")
    return wav


def make_wavs(directory: Path, names, sidecars=False) -> list[Path]:
    """Create WAVs whose mtime rises with their position in ``names``."""
    return [make_wav(directory, name, i, sidecars) for i, name in enumerate(names)]


async def settle(pilot, screen) -> None:
    """Wait until the screen has no read or delete in flight."""
    for _ in range(40):
        await pilot.pause()
        if screen._load is None and screen._delete is None:  # pylint: disable=protected-access
            await pilot.pause()
            return
    raise AssertionError("the screen never settled")


async def open_recordings(app, pilot) -> RecordingsScreen:
    """Open the Recordings screen from Home and wait for the table to load."""
    app.open_destination("recordings")
    screen = app.screen
    assert isinstance(screen, RecordingsScreen)
    await settle(pilot, screen)
    return screen


def table_of(screen) -> DataTable:
    """The recordings table."""
    return screen.query_one("#recordings", DataTable)


def listed(screen) -> list[str]:
    """The recording column, top to bottom."""
    table = table_of(screen)
    return [str(table.get_row_at(i)[0]) for i in range(table.row_count)]


def cursor_name(screen) -> str:
    """The file name under the cursor."""
    table = table_of(screen)
    return str(table.get_row_at(table.cursor_row)[0])


def move_to(screen, name) -> None:
    """Put the cursor on the row of a file name."""
    table = table_of(screen)
    table.move_cursor(row=listed(screen).index(name))


def note(screen) -> str:
    """The line above the table."""
    return str(screen.query_one("#recordings-note", Static).content)


async def confirm(pilot, screen) -> None:
    """Press the Delete button of the open dialog and let the delete finish."""
    await pilot.click("#confirm")
    await settle(pilot, screen)
