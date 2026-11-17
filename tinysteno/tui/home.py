"""Home screen: pick a name and persona, start recording, or reopen a recording."""

from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Input, Label, OptionList, Select, Static
from textual.widgets.option_list import Option

from tinysteno.personas import PersonaInvalidError, PersonaNotFoundError
from tinysteno.recordings import list_recordings
from tinysteno.tui.formscreen import readable_reason

NO_RECORDINGS = "No recordings yet"
NO_PERSONAS = "No personas found. Open the menu (m) and choose Personas."


def recent_recordings(directory: Path, limit: int = 8) -> list[Path]:
    """Newest-first WAV files in a directory (empty when it does not exist)."""
    try:
        found = list_recordings(directory)  # skips what cannot be read: a broken link, a folder
    except OSError:  # the folder itself cannot be read: Home stays usable with an empty list
        return []
    return [info.path for info in found[:limit]]


def initial_persona(config: dict, personas: list[str], preferred: str | None = None) -> object:
    """The persona a Select starts on: ``preferred``, then the configured default, then the
    first persona there is; ``Select.NULL`` when there are none."""
    candidates = [preferred, config.get("persona"), *personas]
    return next((slug for slug in candidates if slug in personas), Select.NULL)


RECENT_LABEL = "Recent recordings"
RECENT_LABEL_MORE = "Recent recordings (m → Recordings for all)"
RECENT_LIMIT = 8


def _recent_and_more(directory: Path) -> tuple[list[Path], bool]:
    """The newest recordings Home shows, and whether there are more than that."""
    found = recent_recordings(directory, RECENT_LIMIT + 1)
    return found[:RECENT_LIMIT], len(found) > RECENT_LIMIT


class HomeScreen(Screen):
    """Landing screen."""

    BINDINGS = [("r", "record", "Record"), ("m", "app.open_menu", "Menu")]

    def __init__(self, persona: str | None = None) -> None:
        super().__init__()
        self._persona_arg = persona
        self._personas: list[str] = []
        self._recent: list[Path] = []
        # The persona Home itself put in the Select (the configured default); once the user
        # picks another (or gave a valid --persona), a new default no longer replaces it.
        self._default_shown: str | None = None
        self._persona_picked = False
        self._personas_error: str | None = None   # why the personas could not be listed

    def compose(self) -> ComposeResult:
        app = self.app
        self._personas = personas = self._list_personas() or []
        # --persona first, then the configured default, then the first persona there is.
        initial = initial_persona(app.config, personas, self._persona_arg)
        self._persona_picked = self._persona_arg in personas
        self._default_shown = initial if isinstance(initial, str) else None
        self._recent, more = _recent_and_more(app.deps.recordings_dir)
        yield Header()
        with Vertical(id="panel"):
            with Horizontal(classes="field"):
                yield Label("Name", classes="field-label")
                yield Input(placeholder="Untitled meeting (optional)", id="name")
            with Horizontal(classes="field"):
                yield Label("Persona", classes="field-label")
                yield self._persona_select(personas, initial)
            yield Static("", id="timeout-hint")
            yield Button("● Record", id="record", variant="success", compact=True,
                         disabled=not personas)
            yield Label(RECENT_LABEL_MORE if more else RECENT_LABEL, id="recent-label")
            yield OptionList(*self._recent_options(), id="recent")
        yield Footer()

    @staticmethod
    def _persona_select(personas: list[str], initial: object) -> Select:
        return Select(
            [(slug, slug) for slug in personas], value=initial,
            allow_blank=not personas, prompt="No personas", id="persona",
        )

    def _recent_options(self) -> list[Option]:
        if self._recent:
            return [Option(p.name) for p in self._recent]
        return [Option(NO_RECORDINGS, disabled=True)]

    def on_mount(self) -> None:
        self.query_one("#record", Button).focus()
        self._update_hint()
        wanted = self._persona_arg
        # With --now the Recording screen reports the unknown persona itself.
        if wanted and self._personas and wanted not in self._personas and not self.app.autostart:
            self.app.notify(
                f"Unknown persona '{wanted}'; using '{self._slug()}' instead.",
                severity="error", markup=False,
            )

    async def on_screen_resume(self) -> None:
        """Back from another screen: show recordings made meanwhile (from the recordings
        folder, which a settings save may have changed), the personas as they are now, the
        new default persona unless the user picked one, and Record ready."""
        await self._follow_persona_list()
        recent, more = _recent_and_more(self.app.deps.recordings_dir)
        self.query_one("#recent-label", Label).update(RECENT_LABEL_MORE if more else RECENT_LABEL)
        if recent != self._recent:
            self._recent = recent
            listing = self.query_one("#recent", OptionList)
            listing.clear_options()
            listing.add_options(self._recent_options())
        self._follow_default_persona()
        self._update_hint()  # the time limit may have changed in Settings
        self.query_one("#record", Button).focus()

    async def _follow_persona_list(self) -> None:
        """Rebuild the persona Select when the personas on disk changed (restored, added or
        removed in the Personas screen): keep the current choice if it still exists, else
        --persona, else the configured default, else the first."""
        personas = self._list_personas()
        if personas is None or personas == self._personas:  # cannot tell: keep what is shown
            return
        current = self._slug_or_none()
        # The current choice stays unless the user never picked it and --persona (which did
        # not exist at startup) exists now.
        if current in personas and (self._persona_picked or self._persona_arg not in personas):
            initial = current
        else:
            # --persona (perhaps restored just now), then the configured default, then the first.
            initial = initial_persona(self.app.config, personas, self._persona_arg)
            self._persona_picked = self._persona_arg in personas
            self._default_shown = initial if isinstance(initial, str) else None
        self._personas = personas
        old = self.query_one("#persona", Select)
        parent = old.parent
        await old.remove()
        await parent.mount(self._persona_select(personas, initial))
        self.query_one("#record", Button).disabled = not personas

    def _list_personas(self) -> list[str] | None:
        """The personas there are; None (with the reason kept for the hint) when they cannot
        be listed, for example a personas folder that cannot be read."""
        try:
            personas = list(self.app.deps.list_personas())
        except Exception as err:
            self._personas_error = (
                f"Personas cannot be listed: {readable_reason(err) or type(err).__name__}")
            return None
        self._personas_error = None
        return personas

    def _slug_or_none(self) -> str | None:
        value = self.query_one("#persona", Select).value
        return value if isinstance(value, str) else None

    def _follow_default_persona(self) -> None:
        default = self.app.config.get("persona")
        if self._persona_picked or default not in self._personas:
            return
        select = self.query_one("#persona", Select)
        if select.value != default:
            self._default_shown = default
            select.value = default

    def _slug(self) -> str:
        value = self.query_one("#persona", Select).value
        return value if isinstance(value, str) else self.app.config.get("persona", "default")

    def _update_hint(self) -> None:
        hint = self.query_one("#timeout-hint", Static)
        if not self._personas:
            hint.add_class("error")
            hint.update(self._personas_error or NO_PERSONAS)
            return
        try:
            persona = self.app.deps.load_persona(self._slug())
        except (PersonaNotFoundError, PersonaInvalidError) as err:
            hint.add_class("error")
            hint.update(f"Persona error: {err}")
            return
        except Exception as err:
            hint.add_class("error")
            hint.update(f"Persona error: {type(err).__name__}")
            return
        hint.remove_class("error")
        limit = self.app.timeout_seconds_for(persona)
        hint.update("No time limit" if limit is None else f"Auto-stop after {limit / 60:g} min")

    def _start(self) -> None:
        if not self._personas:  # Record is disabled: nothing to record with
            self.app.notify(NO_PERSONAS, severity="error")
            return
        name = self.query_one("#name", Input).value.strip()
        self.app.start_recording(name or None, self._slug())

    def action_record(self) -> None:
        """Start recording with the current name and persona."""
        self._start()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "record":
            self._start()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "name":
            self._start()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "persona":
            if isinstance(event.value, str) and event.value != self._default_shown:
                self._persona_picked = True
            self._update_hint()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id != "recent" or not self._recent:
            return
        if not self._personas:  # nothing to process it with
            self.app.notify(NO_PERSONAS, severity="error")
            return
        self.app.process_file(str(self._recent[event.option_index]), self._slug())
