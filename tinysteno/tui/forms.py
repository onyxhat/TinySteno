"""A form for editing settings, shared by the Settings screen and the first-run guide.

The form starts from the config file's own values: a key the file lacks shows its
DEFAULT_CONFIG value. A stored value the form cannot represent is never lost or rewritten:
a choice outside the list is offered as an extra option, and a value of the wrong type is
shown as text. Leaving such a field untouched means it is not part of ``changes()``.
"""

from collections.abc import Collection, Sequence

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.widget import Widget
from textual.widgets import Collapsible, Input, Label, Select, Static, Switch

from tinysteno.config import DEFAULT_CONFIG
from tinysteno.settings import SECTIONS, Setting, from_text, to_text

OVERRIDE_NOTE = "(overridden by a command-line flag this session)"


class FormError(Exception):
    """A field holds an invalid value. ``message`` starts with the field's label and never
    contains the text of a secret."""

    def __init__(self, key: str, message: str) -> None:
        super().__init__(message)
        self.key = key
        self.message = message


def _same(a: object, b: object) -> bool:
    """Equal and of the same kind; 120 and 120.0 are the same number, 1 and True are not."""
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b
    return type(a) is type(b) and a == b


class SettingsForm(VerticalScroll):
    """One field per setting, grouped in a collapsible section per entry of SECTIONS.

    Field widgets have the id ``set-{key}``: ``Input`` for text, path and numbers,
    ``Input(password=True)`` for secrets, ``Switch`` for yes/no, ``Select`` for choices.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        settings: Sequence[Setting],
        values: dict,
        personas: Sequence[str],
        *,
        overridden: Collection[str] = frozenset(),
        id: str | None = None,  # pylint: disable=redefined-builtin
        classes: str | None = None,
    ) -> None:
        """``values`` are the config file's own values (not the session's effective config);
        ``personas`` are the slugs offered for the ``persona`` setting; ``overridden`` keys
        are marked as fixed by a command-line flag this session."""
        super().__init__(id=id, classes=classes)
        rank = {section: i for i, section in enumerate(SECTIONS)}
        # Display order: by section, then as listed in SETTINGS.
        self._order = sorted(settings, key=lambda s: rank.get(s.section, len(rank)))
        self._stored = dict(values)
        self._personas = list(personas)
        self._overridden = frozenset(overridden)
        self._initial = {
            s.key: values[s.key] if s.key in values else DEFAULT_CONFIG.get(s.key)
            for s in self._order
        }
        self._widgets: dict[str, Widget] = {}
        self._start: dict[str, object] = {}  # each widget's state as first shown

    # ── building ─────────────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        sections = list(dict.fromkeys(s.section for s in self._order))
        for section in sections:
            with Collapsible(title=section, collapsed=False, classes="settings-section"):
                for item in (s for s in self._order if s.section == section):
                    yield from self._compose_field(item)

    def _compose_field(self, item: Setting) -> ComposeResult:
        label = item.label
        if item.key in self._overridden:
            label = f"{label} {OVERRIDE_NOTE}"
        widget, state = self._make_widget(item)
        self._widgets[item.key] = widget
        self._start[item.key] = state
        with Vertical(classes="setting"):
            yield Label(label, classes="setting-label", markup=False)
            yield widget
            if item.help:
                yield Static(item.help, classes="setting-help", markup=False)

    def _make_widget(self, item: Setting) -> tuple[Widget, object]:
        value = self._initial[item.key]
        wid = f"set-{item.key}"
        if item.kind == "bool" and isinstance(value, bool):
            return Switch(value=value, id=wid), value
        if item.kind == "choice":
            return self._make_select(item, value)
        if item.kind == "bool" and value is not None:
            text = str(value)  # not a yes/no value: shown as stored so it can be corrected
        else:
            text = to_text(item, value)
        return Input(value=text, id=wid, password=item.kind == "secret"), text

    def _make_select(self, item: Setting, value: object) -> tuple[Widget, object]:
        choices = self._personas if item.key == "persona" else list(item.choices)
        current = to_text(item, value)
        options: list[tuple[Text, str]] = [(Text(choice), choice) for choice in choices]
        if current and current not in choices:
            extra = f"{current} (missing)" if item.key == "persona" else current
            options.append((Text(extra), current))
        selected = current if current else Select.NULL
        select: Select[str] = Select(
            options, value=selected, allow_blank=not current, id=f"set-{item.key}"
        )
        return select, selected

    # ── reading ──────────────────────────────────────────────────────────────

    def _state(self, key: str) -> object:
        widget = self._widgets[key]
        if isinstance(widget, (Input, Switch, Select)):
            return widget.value
        raise TypeError(f"Unexpected field widget for {key}")

    def _parse(self, item: Setting) -> object:
        state = self._state(item.key)
        if isinstance(state, bool):
            return state
        text = "" if state is Select.NULL else str(state)
        try:
            value = from_text(item, text)
        except ValueError as err:
            raise FormError(item.key, str(err)) from None
        if item.key == "persona" and value is not None and value not in self._personas:
            raise FormError(item.key, f"{item.label}: '{value}' no longer exists")
        return value

    def read(self) -> dict:
        """Every field's value: {key: value}, None for a blank optional field.

        Raises FormError for the first invalid field in display order.
        """
        return {item.key: self._parse(item) for item in self._order}

    def changes(self) -> dict:
        """Only the fields the user changed: {key: value}, None to remove a stored key.

        A field counts as changed when its parsed value differs from the value it showed
        at first. A blank optional field removes its key only when the file holds that key.
        Raises FormError for the first invalid changed field in display order.
        """
        result: dict = {}
        for item in self._order:
            key = item.key
            if self._state(key) == self._start[key]:
                continue
            value = self._parse(item)
            if _same(value, self._initial[key]):
                continue
            if value is None and key not in self._stored:
                continue
            result[key] = value
        return result

    def has_changes(self) -> bool:
        """True when saving would write something (an invalid edit counts as a change)."""
        try:
            return bool(self.changes())
        except FormError:
            return True

    def focus_field(self, key: str) -> None:
        """Move focus to a field, expanding its section first if it is collapsed."""
        widget = self._widgets[key]
        collapsed = [a for a in widget.ancestors if isinstance(a, Collapsible) and a.collapsed]
        if not collapsed:
            widget.focus()
            return
        for section in collapsed:
            section.collapsed = False
        self.call_after_refresh(widget.focus)
