"""A form for editing settings, shared by the Settings screen and the first-run guide.

The form starts from the config file's own values: a key the file lacks shows its
DEFAULT_CONFIG value. A stored value the form cannot represent is never lost or rewritten:
a choice outside the list is offered as an extra option, and a value of the wrong type is
shown as text. A field the user has not touched gives back exactly the value it started with.

Every widget is built in ``__init__``: nothing that runs later (compose, read, changes) holds
a stored value in a local variable where an unexpected error could show it in a traceback.
"""

from collections.abc import Collection, Sequence

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.widget import Widget
from textual.widgets import Collapsible, Input, Label, Select, Static, Switch

from tinysteno.config import DEFAULT_CONFIG
from tinysteno.settings import SECTIONS, Setting, accepts_any, from_text, to_text

OVERRIDE_NOTE = "(overridden by a command-line flag this session)"


class FormError(Exception):
    """A field holds an invalid value. ``message`` starts with the field's label and never
    contains the text of a field."""

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
    """One field per setting, grouped in a collapsible section per entry of SECTIONS
    (or, with ``flat=True``, listed in the given order without sections).

    Field widgets have the id ``set-{key}``: ``Input`` for text, path and numbers (and for
    a choice that accepts any value), ``Input(password=True)`` for secrets, ``Switch`` for
    yes/no, ``Select`` for choices.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        settings: Sequence[Setting],
        values: dict,
        personas: Sequence[str] | None,
        *,
        overridden: Collection[str] = frozenset(),
        flat: bool = False,
        id: str | None = None,  # pylint: disable=redefined-builtin
        classes: str | None = None,
    ) -> None:
        """``values`` are the config file's own values (not the session's effective config).
        ``personas`` are the slugs offered for the ``persona`` setting; None means they could
        not be listed, and the stored persona is then offered as it is. ``overridden`` keys
        are marked as fixed by a command-line flag this session."""
        super().__init__(id=id, classes=classes)
        self._flat = flat
        if flat:
            self._order = list(settings)
        else:
            rank = {section: i for i, section in enumerate(SECTIONS)}
            self._order = sorted(settings, key=lambda s: rank.get(s.section, len(rank)))
        self._stored_keys = frozenset(values)
        self._personas = None if personas is None else list(personas)
        self._overridden = frozenset(overridden)
        self._initial_values = {
            s.key: values[s.key] if s.key in values else DEFAULT_CONFIG.get(s.key)
            for s in self._order
        }
        self._field_widgets: dict[str, Widget] = {}
        self._start: dict[str, object] = {}  # each widget's state as first shown
        self._field_rows: dict[str, Vertical] = {}
        for item in self._order:
            self._build_field(item)

    # ── building ─────────────────────────────────────────────────────────────

    def _build_field(self, item: Setting) -> None:
        label = item.label
        if item.key in self._overridden:
            label = f"{label} {OVERRIDE_NOTE}"
        widget, state = self._make_widget(item)
        self._field_widgets[item.key] = widget
        self._start[item.key] = state
        parts: list[Widget] = [Label(label, classes="setting-label", markup=False), widget]
        if item.help:
            parts.append(Static(item.help, classes="setting-help", markup=False))
        self._field_rows[item.key] = Vertical(*parts, classes="setting")

    def _make_widget(self, item: Setting) -> tuple[Widget, object]:
        value = self._initial_values[item.key]
        wid = f"set-{item.key}"
        if item.kind == "bool" and isinstance(value, bool):
            return Switch(value=value, id=wid), value
        if item.kind == "choice" and not accepts_any(item):
            return self._make_select(item, value)
        if item.kind == "bool" and value is not None:
            text = str(value)  # not a yes/no value: shown as stored so it can be corrected
        else:
            text = to_text(item, value)
        return Input(value=text, id=wid, password=item.kind == "secret"), text

    def _make_select(self, item: Setting, value: object) -> tuple[Widget, object]:
        known = self._personas if item.key == "persona" else list(item.choices)
        choices = known or []
        current = to_text(item, value)
        options: list[tuple[Text, str]] = [(Text(choice), choice) for choice in choices]
        if current and current not in choices:
            missing = item.key == "persona" and known is not None
            options.append((Text(f"{current} (missing)" if missing else current), current))
        selected = current if current else Select.NULL
        select: Select[str] = Select(
            options, value=selected, allow_blank=not current, id=f"set-{item.key}"
        )
        return select, selected

    def compose(self) -> ComposeResult:
        if self._flat:
            yield from (self._field_rows[item.key] for item in self._order)
            return
        for section in dict.fromkeys(item.section for item in self._order):
            with Collapsible(title=section, collapsed=False, classes="settings-section"):
                for item in self._order:
                    if item.section == section:
                        yield self._field_rows[item.key]

    # ── reading ──────────────────────────────────────────────────────────────

    def _touched(self, key: str) -> bool:
        widget = self._field_widgets[key]
        if not isinstance(widget, (Input, Switch, Select)):
            raise TypeError(f"Unexpected field widget for {key}")
        return widget.value != self._start[key]

    def _parse(self, item: Setting) -> object:
        """The value of a touched field; any failure is a FormError naming only the label."""
        try:
            return self._parse_widget(item)
        except FormError:
            raise
        except ValueError as err:
            message = str(err)  # from_text: starts with the label, never quotes the text
        except Exception as err:
            message = f"{item.label}: could not be read ({type(err).__name__})"
        raise FormError(item.key, message) from None

    def _parse_widget(self, item: Setting) -> object:
        widget = self._field_widgets[item.key]
        if isinstance(widget, Switch):
            return widget.value
        state = widget.value if isinstance(widget, (Input, Select)) else ""
        value = from_text(item, "" if state is Select.NULL else str(state))
        if item.key == "persona" and value is not None:
            self._check_persona(item, value)
        return value

    def _check_persona(self, item: Setting, value: object) -> None:
        if self._personas and value not in self._personas:
            raise FormError(item.key, f"{item.label}: '{value}' no longer exists")

    def read(self) -> dict:
        """Every field's value: {key: value}, None for a blank optional field.

        A field the user has not touched gives back its starting value unchanged (the stored
        value, else the default); touched fields are parsed. A persona that no longer exists
        is refused while other personas exist. Raises FormError for the first invalid field
        in display order.
        """
        result: dict = {}
        for item in self._order:
            if self._touched(item.key):
                result[item.key] = self._parse(item)
            else:
                if item.key == "persona" and self._initial_values[item.key] is not None:
                    self._check_persona(item, self._initial_values[item.key])
                result[item.key] = self._initial_values[item.key]
        return result

    def changes(self) -> dict:
        """Only the fields the user changed: {key: value}, None to remove a stored key.

        A field counts as changed when its parsed value differs from the value it showed
        at first. A blank optional field removes its key only when the file holds that key.
        Raises FormError for the first invalid changed field in display order.
        """
        result: dict = {}
        for item in self._order:
            key = item.key
            if not self._touched(key):
                continue
            value = self._parse(item)
            if _same(value, self._initial_values[key]):
                continue
            if value is None and key not in self._stored_keys:
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
        widget = self._field_widgets[key]
        collapsed = [a for a in widget.ancestors if isinstance(a, Collapsible) and a.collapsed]
        if not collapsed:
            widget.focus()
            return
        for section in collapsed:
            section.collapsed = False
        self.call_after_refresh(widget.focus)
