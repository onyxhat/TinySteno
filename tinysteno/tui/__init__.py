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
) -> str | None:
    """Run the app to completion and return its exit message, if any."""
    app = TinyStenoApp(
        config, deps, autostart=autostart, name=name, persona=persona,
        timeout_override=timeout_override,
    )
    app.run()
    return app.exit_message
