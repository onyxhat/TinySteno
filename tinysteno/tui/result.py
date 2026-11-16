"""Result screen: the exported note, where it was written, and what to do next."""

import re
import urllib.parse
from pathlib import Path

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.screen import Screen
from textual.widgets import Footer, Header, Markdown, Static

from tinysteno.pipeline import PipelineResult
from tinysteno.session import format_clock

FADE_SECONDS = 0.4
WARNING_SECONDS = 20
# A leading ``---`` line, any YAML, then a closing ``---`` line (LF or CRLF line endings).
_FRONTMATTER = re.compile(r"\A---\r?\n(?:.*?\r?\n)?---[ \t]*(?:\r?\n|\Z)", re.DOTALL)


def strip_frontmatter(text: str) -> str:
    """Remove a leading YAML front-matter block; anything else is returned unchanged."""
    match = _FRONTMATTER.match(text)
    return text[match.end():] if match else text


def obsidian_url(note_path: str) -> str:
    """The obsidian:// URL that opens a note by absolute path."""
    return "obsidian://open?path=" + urllib.parse.quote(note_path, safe="")


def _call_ui(app: App, callback, *args) -> None:
    """Run ``callback(*args)`` on the UI thread; silently drop it once the app is shutting down."""
    if not app.is_running:
        return
    try:
        app.call_from_thread(callback, *args)
    except RuntimeError:  # the app stopped between the check and the call
        pass


class ResultScreen(Screen):
    """Shows the finished note."""

    BINDINGS = [
        Binding("o", "open_obsidian", "Open in Obsidian"),
        Binding("n", "new", "New recording"),
    ]

    def __init__(self, result: PipelineResult) -> None:
        super().__init__()
        self.result = result
        self._fade = False

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll(id="note-scroll"):
            yield Markdown("", id="note")
        yield Static(self.result.note_path, id="note-path", markup=False)
        yield Footer()

    def on_mount(self) -> None:
        scroll = self.query_one("#note-scroll")
        scroll.border_title = self.result.title
        scroll.border_subtitle = format_clock(self.result.duration_seconds)
        # Fade in only for a live run; pilot tests wait on running animations.
        self._fade = self.app.deps.live_updates and self.app.animation_level != "none"
        if self._fade:
            self.styles.opacity = 0.0
        if self.result.reused_transcript:
            self.app.notify("Used the saved transcript for this recording.", markup=False)
        for text in self.result.warnings:
            self.app.notify(text, severity="warning", timeout=WARNING_SECONDS, markup=False)
        self._read_note(self.app, self.result.note_path)

    @work(thread=True)
    def _read_note(self, app: App, note_path: str) -> None:
        try:
            text = strip_frontmatter(Path(note_path).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError) as err:
            text = f"Note written to {note_path} (could not be displayed: {err})"
        _call_ui(app, self._show_note, text)

    async def _show_note(self, text: str) -> None:
        await self.query_one("#note", Markdown).update(text)
        if self._fade:
            self.styles.animate("opacity", 1.0, duration=FADE_SECONDS)

    def action_open_obsidian(self) -> None:
        """Open the note in Obsidian."""
        self.app.open_url(obsidian_url(self.result.note_path))

    def action_new(self) -> None:
        """Back to Home for another recording."""
        self.app.pop_screen()
