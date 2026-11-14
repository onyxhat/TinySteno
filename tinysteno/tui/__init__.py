"""Textual user interface for TinySteno."""

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
    """Run the app to completion; return its exit message (if any) and exit code."""
    app = TinyStenoApp(
        config, deps, autostart=autostart, name=name, persona=persona,
        timeout_override=timeout_override,
    )
    app.run()
    return app.exit_message, app.return_code or 0
