"""Leaving a screen once, so a queued second activation never closes the screen below."""


class LeaveOnceMixin:
    """For a Textual ``Screen``: dismiss it only while it is the active screen and is not
    already leaving.

    ``Screen.dismiss()`` pops whatever screen is on top of the stack. A second Enter, Back or
    button press queued behind the first is handled after the first has already dismissed
    this screen, and would otherwise close the screen below it (Home, leaving a blank app).
    List this mixin before the Screen class.
    """

    # Not `_closing`: Textual's message pump uses that name, and the app freezes.
    _leaving = False

    def _may_leave(self) -> bool:
        """True while this screen is the active one and has not started leaving."""
        return not self._leaving and self.is_active

    def _leave_once(self, result: object = None) -> bool:
        """Dismiss with ``result`` if ``_may_leave()``; True when it did."""
        if not self._may_leave():
            return False
        self._leaving = True
        self.dismiss(result)
        return True
