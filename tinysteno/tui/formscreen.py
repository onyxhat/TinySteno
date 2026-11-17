"""Shared base of the screens that edit settings: the Settings screen and the Setup guide."""

from collections.abc import Callable

from textual.screen import Screen
from textual.widgets import Static

from tinysteno.tui.confirm import ConfirmScreen
from tinysteno.tui.deps import SettingsNotApplied
from tinysteno.tui.forms import FormError, SettingsForm
from tinysteno.tui.leaving import LeaveOnceMixin


def readable_reason(err: Exception) -> str:
    """The error's text without quoted file content.

    A YAML error quotes the offending line of the file (indented four spaces, with a ``^``
    under it), and that line may hold an API key. config.py no longer passes that text on;
    this is a second layer for any other error that might.
    """
    lines = [line for line in str(err).splitlines() if not line.startswith("    ")]
    return "\n".join(line.rstrip(":") if line.startswith("  in ") else line for line in lines)


def unexpected(err: Exception) -> str:
    """One line for an error nobody planned for: its type, never its text (which could
    hold a value from the form)."""
    return f"Unexpected error: {type(err).__name__}"


class FormScreen(LeaveOnceMixin, Screen):
    """A screen with a ``SettingsForm`` and a ``#form-error`` line under it.

    Subclasses set ``self._form`` and decide what happens after a save. Stored values live
    only inside the form: no handler keeps them in a local variable where an unexpected
    error could print them.
    """

    def __init__(self) -> None:
        super().__init__()
        self._form: SettingsForm | None = None

    def _show_error(self, message: str) -> None:
        error = self.query_one("#form-error", Static)
        error.update(message)
        error.display = bool(message)

    def _can_act(self) -> bool:
        """Only the top screen that is not already closing handles Save and Back (queued
        presses arriving after the first save must not pop the screen below)."""
        return self._may_leave()

    def _close(self) -> None:
        self._leave_once()

    def has_unsaved_changes(self) -> bool:
        """True when leaving now would lose edits (the app asks before quitting)."""
        if self._form is None or self._leaving:
            return False
        try:
            return self._form.has_changes()
        except Exception:
            return True  # cannot tell: ask rather than lose edits

    def _leave_asking(self) -> None:
        """Leave; with unsaved changes, ask first."""
        if not self._can_act():
            return
        try:
            if not self.has_unsaved_changes():
                self._close()
                return
            self.app.push_screen(
                ConfirmScreen("Discard unsaved changes?", "Discard"), self._discard_answered
            )
        except Exception as err:
            self._show_error(unexpected(err))

    def _discard_answered(self, confirmed: bool | None) -> None:
        if confirmed and self._can_act():
            self._close()

    def _collect(self, form: SettingsForm, read: Callable[[], dict]) -> dict | None:
        """The values ``read`` returns, or None after showing why they are not valid."""
        try:
            return read()
        except FormError as err:
            self._show_error(err.message)
            form.focus_field(err.key)
            return None

    def _write(self, changes: dict) -> bool | None:
        """Save ``changes``. True: saved and applied. False: saved, but this session could
        not apply it (a warning is shown). None: not saved, the reason is on screen."""
        app = self.app
        try:
            app.deps.save_settings(changes)
        except SettingsNotApplied as err:
            self._saved(changes)
            app.notify(
                "Settings saved, but they could not be applied to this session: "
                f"{readable_reason(err)}. Restart TinySteno to use them.",
                severity="warning", markup=False,
            )
            return False
        except ValueError as err:
            self._show_error(readable_reason(err))
            return None
        except Exception as err:  # OSError, or anything else: keep the edits on screen
            self._show_error(f"Could not save settings: {readable_reason(err)}")
            return None
        self._saved(changes)
        if "tui_theme" in changes:
            app.use_theme(changes["tui_theme"])
        app.notify("Settings saved")
        return True

    def _saved(self, changes: dict) -> None:
        # Key names only: a value may be an API key.
        self.app.events.append(("settings_saved", tuple(sorted(changes))))
