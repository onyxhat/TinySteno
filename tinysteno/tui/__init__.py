"""Textual user interface for TinySteno."""

import sys

from tinysteno.signals import deferred_exit_signals
from tinysteno.tui.app import TinyStenoApp
from tinysteno.tui.deps import TuiDeps


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
