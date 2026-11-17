"""Helper processes started while the TUI owns the terminal.

Textual replaces ``sys.stderr`` with a capture object whose ``fileno()`` is -1. Python's
multiprocessing resource tracker passes ``sys.stderr.fileno()`` to the helper process it
starts on first use, so a first use *inside* the app fails with
``ValueError: bad value(s) in fds_to_keep``. Loading a Whisper model does exactly that (the
model download takes a multiprocessing lock for its progress bars), which made every local
transcription from the TUI fail.
"""
import subprocess
import sys
import textwrap
from unittest.mock import patch

from tinysteno import tui
from tests.tui_fakes import make_deps

_SCRIPT = textwrap.dedent(
    """
    import asyncio, multiprocessing, sys, threading
    from textual.app import App
    import tinysteno.tui as tui

    if sys.argv[1] == "prepared":
        tui.start_process_helpers()

    async def main():
        result = {}
        def work():  # like the pipeline: a worker thread, while the app owns the terminal
            try:
                multiprocessing.RLock()
                result["outcome"] = "ok"
            except Exception as err:
                result["outcome"] = f"{type(err).__name__}: {err}"
        async with App().run_test():
            thread = threading.Thread(target=work)
            thread.start()
            thread.join()
        print(result["outcome"])

    asyncio.run(main())
    """
)


def _run(mode: str) -> str:
    done = subprocess.run(
        [sys.executable, "-c", _SCRIPT, mode], capture_output=True, text=True, timeout=120,
        check=False,
    )
    return done.stdout.strip().splitlines()[-1] if done.stdout.strip() else done.stderr


def test_a_lock_taken_inside_the_app_fails_without_preparation():
    """The cause, pinned: if this starts passing, the workaround can go."""
    assert _run("unprepared") == "ValueError: bad value(s) in fds_to_keep"


def test_a_lock_taken_inside_the_app_works_once_helpers_are_started():
    assert _run("prepared") == "ok"


def test_starting_helpers_never_raises():
    with patch("multiprocessing.resource_tracker.ensure_running", side_effect=OSError("no")):
        tui.start_process_helpers()


def test_run_tui_starts_the_helpers_before_the_app(tmp_path):
    order = []
    with patch.object(tui, "start_process_helpers", side_effect=lambda: order.append("helpers")), \
            patch.object(tui.TinyStenoApp, "run", side_effect=lambda: order.append("app")):
        tui.run_tui({}, make_deps(tmp_path))
    assert order == ["helpers", "app"]
