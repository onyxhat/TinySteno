"""A yes/no question in a dialog, used before anything that discards or deletes."""

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.content import Content
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Static

from tinysteno.tui.leaving import LeaveOnceMixin


class ConfirmScreen(LeaveOnceMixin, ModalScreen[bool]):
    """Ask ``question``; dismiss with True when the user presses the confirm button.

    Focus starts on Cancel, so Enter on its own never confirms; escape cancels. Only the
    first answer counts: one queued behind it never closes the screen below.
    """

    BINDINGS = [Binding("escape", "cancel", "Cancel")]
    AUTO_FOCUS = "#cancel"

    def __init__(self, question: str, confirm_label: str) -> None:
        super().__init__()
        self.question = question
        self.confirm_label = confirm_label

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm-box"):
            # Questions name files and settings: shown literally, never as markup.
            yield Static(self.question, id="confirm-title", markup=False)
            with Horizontal(id="confirm-buttons"):
                yield Button("Cancel", id="cancel", variant="primary")
                yield Button(Content(self.confirm_label), id="confirm", variant="error")
        yield Footer()

    def action_cancel(self) -> None:
        """Answer no."""
        self._leave_once(False)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self._leave_once(event.button.id == "confirm")
