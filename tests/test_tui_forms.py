"""SettingsForm on its own: what read() returns, open choices, the flat layout."""
import pytest
from textual.app import App
from textual.widgets import Collapsible, Input, Select

import tinysteno.tui.forms
from tinysteno.settings import SETTINGS, setting
from tinysteno.tui.forms import FormError, SettingsForm

ESSENTIAL = [s for s in SETTINGS if s.essential]


class FormApp(App):
    """Just the form, built in compose (its widgets need a running app)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.form_args = args, kwargs
        self.form: SettingsForm | None = None

    def compose(self):
        args, kwargs = self.form_args
        self.form = SettingsForm(*args, **kwargs)
        yield self.form


async def test_read_returns_untouched_stored_values_unchanged():
    app = FormApp(ESSENTIAL, {"model": " x ", "transcription_backend": "API"}, ["default"])
    async with app.run_test() as pilot:
        form = app.form
        values = form.read()
        assert values["model"] == " x " and values["transcription_backend"] == "API"
        assert values["base_url"] == "http://localhost:11434/v1"      # the default
        form.query_one("#set-model", Input).value = " y "
        await pilot.pause()
        assert form.read()["model"] == "y"


async def test_no_personas_accepts_the_untouched_persona():
    app = FormApp(ESSENTIAL, {}, [])
    async with app.run_test():
        form = app.form
        assert form.read()["persona"] == "default"
        assert not form.changes()


async def test_a_persona_that_no_longer_exists_is_refused_while_others_exist():
    app = FormApp(ESSENTIAL, {"persona": "gone"}, ["default"])
    async with app.run_test():
        form = app.form
        with pytest.raises(FormError) as caught:
            form.read()
        assert caught.value.key == "persona"
        assert not form.changes()


async def test_unknown_persona_list_offers_the_stored_value_plainly():
    app = FormApp(ESSENTIAL, {"persona": "rca"}, None)
    async with app.run_test():
        form = app.form
        select = form.query_one("#set-persona", Select)
        options = [(str(p), v) for p, v in select._options]  # pylint: disable=protected-access
        assert options == [("rca", "rca")]
        assert form.read()["persona"] == "rca"


async def test_flat_layout_has_no_sections_and_keeps_the_given_order():
    chosen = [setting("model"), setting("obsidian_vault"), setting("api_key")]
    app = FormApp(chosen, {}, ["default"], flat=True)
    async with app.run_test():
        form = app.form
        assert not form.query(Collapsible)
        ids = [w.id for w in form.query("Input, Select, Switch")]
        assert ids == ["set-model", "set-obsidian_vault", "set-api_key"]


async def test_open_choice_is_a_text_field():
    app = FormApp([setting("whisper_model")], {"whisper_model": "small"}, [])
    async with app.run_test() as pilot:
        form = app.form
        widget = form.query_one("#set-whisper_model")
        assert isinstance(widget, Input) and widget.value == "small"
        widget.value = "distil-large-v3"
        await pilot.pause()
        assert form.changes() == {"whisper_model": "distil-large-v3"}


async def test_an_unexpected_parse_failure_is_a_form_error_without_the_value(monkeypatch):
    def broken(item, text):
        raise RuntimeError(f"cannot parse {text}")
    app = FormApp([setting("api_key")], {"api_key": "old"}, [])
    async with app.run_test() as pilot:
        form = app.form
        form.query_one("#set-api_key", Input).value = "sk-typed-SECRET"
        await pilot.pause()
        monkeypatch.setattr(tinysteno.tui.forms, "from_text", broken)
        with pytest.raises(FormError) as caught:
            form.changes()
        assert caught.value.key == "api_key"
        assert caught.value.message == "LLM API key: could not be read (RuntimeError)"
        assert caught.value.__cause__ is None and caught.value.__suppress_context__
