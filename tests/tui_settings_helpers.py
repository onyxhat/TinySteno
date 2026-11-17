"""Shared steps for the Settings screen tests."""
from tinysteno.tui.settings import SettingsScreen

CONFIG = {"persona": "1on1", "session_timeout_minutes": 120}


async def open_settings(app, pilot) -> SettingsScreen:
    """Open Settings from Home and return the screen."""
    app.open_destination("settings")
    await pilot.pause()
    assert isinstance(app.screen, SettingsScreen)
    return app.screen


def field(screen, key):
    """The widget of one setting."""
    return screen.query_one(f"#set-{key}")


def notes(app) -> list[tuple[str, str]]:
    """Every notification so far, as (message, severity)."""
    return [(n.message, n.severity) for n in app._notifications]  # pylint: disable=protected-access


def form_error(screen) -> str:
    """The text of the error line under the form."""
    return str(screen.query_one("#form-error").content)


async def save(pilot):
    """Press ctrl+s and let the screen react."""
    await pilot.press("ctrl+s")
    await pilot.pause()
