"""The Menu: one entry per management screen the app has registered."""

from collections.abc import Sequence

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Footer, Label, OptionList
from textual.widgets.option_list import Option

from tinysteno.tui.leaving import LeaveOnceMixin

# Every destination the Menu can show, in the order it shows them. The app lists only the
# ones it has registered (TinyStenoApp.destinations).
DESTINATION_ORDER: tuple[str, ...] = ("settings", "setup", "personas", "recordings", "checks")


class MenuScreen(LeaveOnceMixin, ModalScreen[str | None]):
    """Pick a destination; dismisses with its id, or None on escape (once: a second choice
    queued behind the first is ignored)."""

    BINDINGS = [Binding("escape", "close", "Close")]
    AUTO_FOCUS = "#menu"

    def __init__(self, entries: Sequence[tuple[str, str]]) -> None:
        """``entries`` are (destination id, label) pairs, shown in the given order."""
        super().__init__()
        self.entries = list(entries)

    def compose(self) -> ComposeResult:
        with Vertical(id="menu-box"):
            yield Label("Menu", id="menu-title")
            yield OptionList(
                *(Option(label, id=dest) for dest, label in self.entries), id="menu"
            )
        yield Footer()

    def action_close(self) -> None:
        """Close without choosing."""
        self._leave_once(None)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        self._leave_once(event.option.id)
