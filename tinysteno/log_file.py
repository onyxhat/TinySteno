"""Send logging to a file for the duration of a full-screen session.

Textual redirects only ``sys.stdout`` / ``sys.stderr``; a root ``StreamHandler`` created
before the app started still holds the real terminal and would paint over the screen.
"""

import contextlib
import logging
from pathlib import Path

LOG_FORMAT = "%(asctime)s %(levelname)s: %(message)s"


class _SessionLog(logging.FileHandler):
    """Appends to the log file and remembers whether anything WARNING or worse arrived."""

    def __init__(self, path: Path) -> None:
        super().__init__(path, mode="a", encoding="utf-8")
        self.warned = False

    def emit(self, record: logging.LogRecord) -> None:
        if record.levelno >= logging.WARNING:
            self.warned = True
        super().emit(record)


@contextlib.contextmanager
def log_to_file(path: Path, verbose: bool = False):
    """Replace the root logger's handlers with one appending to ``path``; restore them after.

    Yields the handler (its ``warned`` says whether WARNING-or-above records were logged), or
    None when the file could not be opened, in which case logging is discarded instead.
    """
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handler: logging.Handler | None = _SessionLog(path)
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
    except OSError:
        handler = None
    for existing in saved_handlers:
        root.removeHandler(existing)
    root.addHandler(handler or logging.NullHandler())
    root.setLevel(logging.DEBUG if verbose else logging.WARNING)
    try:
        yield handler
    finally:
        for current in root.handlers[:]:
            root.removeHandler(current)
        if handler is not None:
            handler.close()
        for existing in saved_handlers:
            root.addHandler(existing)
        root.setLevel(saved_level)
