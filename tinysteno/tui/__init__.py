"""Textual user interface for TinySteno."""

import sys

from tinysteno.signals import deferred_exit_signals
from tinysteno.tui.app import TinyStenoApp
from tinysteno.tui.deps import TuiDeps


def start_process_helpers() -> None:
    """Start Python's multiprocessing resource tracker while stderr is still the terminal.

    The tracker is a helper process started on first use, and it is handed
    ``sys.stderr.fileno()``. Textual swaps ``sys.stderr`` for a capture object whose
    ``fileno()`` is -1, so a first use inside the app fails with ``ValueError: bad value(s)
    in fds_to_keep``. Loading a Whisper model is such a first use (its download takes a
    multiprocessing lock for progress bars). Once running, the tracker is reused.
    """
    try:
        from multiprocessing import resource_tracker
        resource_tracker.ensure_running()
    except Exception:  # never worth failing the launch for; transcription reports any error
        pass


def run_tui(  # pylint: disable=too-many-arguments
    config: dict,
    deps: TuiDeps,
    *,
    autostart: bool = False,
    name: str | None = None,
    persona: str | None = None,
    timeout_override: float | None = None,
) -> tuple[str | None, int]:
    """Run the app to completion; return its exit message (if any) and exit code.

    However the app ends (quit, closed terminal, crash), a recording that was started and
    never stopped or discarded is stopped here and the message says where it was saved.
    """
    app = TinyStenoApp(
        config, deps, autostart=autostart, name=name, persona=persona,
        timeout_override=timeout_override,
    )
    returned = False
    start_process_helpers()
    try:
        app.run()
        returned = True
    finally:
        # The app's own signal handling ended with the app: a kill arriving while the safety
        # net is still writing the WAV waits until the user has been told where it is.
        with deferred_exit_signals():
            message, code = app.final_result()
            if not returned and message:  # app.run() raised: say it before the traceback
                print(message, file=sys.stderr)
    return message, code
