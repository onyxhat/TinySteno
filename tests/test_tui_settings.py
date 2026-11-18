"""Settings screen: shows the stored file's values, saves only what changed, never leaks keys."""
import pytest
from textual.widgets import Collapsible, Input, Select, Switch

from tinysteno.config import DEFAULT_CONFIG, config_path
from tinysteno.settings import SECTIONS, SETTINGS, setting
from tinysteno.tui.app import TinyStenoApp
from tinysteno.tui.confirm import ConfirmScreen
from tinysteno.tui.deps import SettingsNotApplied
from tinysteno.tui.forms import FormError, SettingsForm
from tinysteno.tui.home import HomeScreen
from tinysteno.tui.settings import SettingsScreen
from tests.tui_fakes import make_deps
from tests.tui_settings_helpers import CONFIG, field, form_error, notes, open_settings, save

OVERRIDE_NOTE = "(overridden by a command-line flag this session)"


async def test_fields_show_stored_values_and_defaults(tmp_path):
    stored = {"model": "gpt-4o", "auto_title": False, "recordings_path": "~/Audio", "channels": 2}
    app = TinyStenoApp(CONFIG, make_deps(tmp_path, settings=stored))
    async with app.run_test() as pilot:
        screen = await open_settings(app, pilot)
        for item in SETTINGS:                  # one widget per setting
            assert field(screen, item.key) is not None
        assert field(screen, "model").value == "gpt-4o"
        assert field(screen, "base_url").value == DEFAULT_CONFIG["base_url"]
        assert field(screen, "obsidian_vault").value == "~/Obsidian/Vault"   # unexpanded
        assert field(screen, "recordings_path").value == "~/Audio"
        assert field(screen, "whisper_prompt").value == ""
        assert field(screen, "max_speakers").value == ""
        auto_title, vad = field(screen, "auto_title"), field(screen, "whisper_vad")
        assert isinstance(auto_title, Switch) and auto_title.value is False
        assert isinstance(vad, Switch) and vad.value is True
        assert isinstance(field(screen, "channels"), Select)
        assert field(screen, "channels").value == "2"
        assert field(screen, "transcription_backend").value == "local"
        assert field(screen, "persona").value == "default"
        titles = [str(c.title) for c in screen.query(Collapsible)]
        assert titles == list(SECTIONS)


async def test_form_read_returns_every_value(tmp_path):
    app = TinyStenoApp(CONFIG, make_deps(tmp_path, settings={"api_key": "k"}))
    async with app.run_test() as pilot:
        screen = await open_settings(app, pilot)
        values = screen.query_one(SettingsForm).read()
        assert set(values) == {s.key for s in SETTINGS}
        assert values["channels"] == 1 and values["whisper_vad"] is True
        assert values["recordings_path"] is None and values["api_key"] == "k"
        field(screen, "sample_rate").value = "fast"
        with pytest.raises(FormError) as caught:
            screen.query_one(SettingsForm).read()
        assert caught.value.key == "sample_rate"
        assert str(caught.value).startswith(setting("sample_rate").label)


async def test_form_read_refuses_a_persona_that_no_longer_exists(tmp_path):
    app = TinyStenoApp(CONFIG, make_deps(tmp_path, settings={"persona": "gone"}))
    async with app.run_test() as pilot:
        screen = await open_settings(app, pilot)
        with pytest.raises(FormError) as caught:
            screen.query_one(SettingsForm).read()
        assert caught.value.key == "persona"
        assert caught.value.message == "Persona: 'gone' no longer exists"


async def test_save_sends_only_changed_keys(tmp_path):
    deps = make_deps(tmp_path, settings={"model": "llama3.2:3b", "base_url": "http://x/v1"})
    app = TinyStenoApp(CONFIG, deps)
    async with app.run_test() as pilot:
        screen = await open_settings(app, pilot)
        field(screen, "model").value = "gpt-4o-mini"
        await save(pilot)
        assert deps.saved == [{"model": "gpt-4o-mini"}]
        assert isinstance(app.screen, HomeScreen)
        assert ("Settings saved", "information") in notes(app)
        assert ("settings_saved", ("model",)) in app.events


async def test_switch_and_choice_changes_are_typed(tmp_path):
    deps = make_deps(tmp_path)
    app = TinyStenoApp(CONFIG, deps)
    async with app.run_test() as pilot:
        screen = await open_settings(app, pilot)
        field(screen, "auto_tags").value = False
        field(screen, "channels").value = "2"
        field(screen, "session_timeout_minutes").value = "90"
        await save(pilot)
        assert deps.saved == [{"auto_tags": False, "channels": 2, "session_timeout_minutes": 90}]


async def test_thinking_is_a_summary_switch_that_saves_a_bool(tmp_path):
    deps = make_deps(tmp_path)
    app = TinyStenoApp(CONFIG, deps)
    async with app.run_test() as pilot:
        screen = await open_settings(app, pilot)
        thinking = field(screen, "enable_thinking")
        assert isinstance(thinking, Switch) and thinking.value is False
        section = next(a for a in thinking.ancestors if isinstance(a, Collapsible))
        assert str(section.title) == "Summary (LLM)"
        thinking.value = True
        await save(pilot)
        assert deps.saved == [{"enable_thinking": True}]


async def test_retyping_the_same_value_is_not_a_change(tmp_path):
    deps = make_deps(tmp_path, settings={"session_timeout_minutes": 120.0})
    app = TinyStenoApp(CONFIG, deps)
    async with app.run_test() as pilot:
        screen = await open_settings(app, pilot)
        field(screen, "session_timeout_minutes").value = " 120 "
        await pilot.press("escape")              # no unsaved change: leaves without asking
        await pilot.pause()
        assert isinstance(app.screen, HomeScreen)
        assert not deps.saved


async def test_nothing_changed_leaves_without_saving(tmp_path):
    deps = make_deps(tmp_path, settings={"model": "x"})
    app = TinyStenoApp(CONFIG, deps)
    async with app.run_test() as pilot:
        await open_settings(app, pilot)
        await save(pilot)
        assert not deps.saved
        assert isinstance(app.screen, HomeScreen)


async def test_blank_optional_field_removes_the_key(tmp_path):
    deps = make_deps(tmp_path, settings={"recordings_path": "/x"})
    app = TinyStenoApp(CONFIG, deps)
    async with app.run_test() as pilot:
        screen = await open_settings(app, pilot)
        field(screen, "recordings_path").value = "  "
        await save(pilot)
        assert deps.saved == [{"recordings_path": None}]


async def test_key_absent_from_the_file_is_not_written_by_an_unrelated_save(tmp_path):
    deps = make_deps(tmp_path, settings={})
    app = TinyStenoApp(CONFIG, deps)
    async with app.run_test() as pilot:
        screen = await open_settings(app, pilot)
        field(screen, "base_url").value = "http://llm:8080/v1"
        field(screen, "whisper_prompt").value = "  "     # blank: nothing stored to remove
        await save(pilot)
        assert deps.saved == [{"base_url": "http://llm:8080/v1"}]


async def test_stored_value_outside_the_choices_survives_an_unrelated_save(tmp_path):
    deps = make_deps(tmp_path, settings={"whisper_device": "mps", "whisper_model": "large-v2"})
    app = TinyStenoApp(CONFIG, deps)
    async with app.run_test() as pilot:
        screen = await open_settings(app, pilot)
        assert field(screen, "whisper_device").value == "mps"
        assert field(screen, "whisper_model").value == "large-v2"
        field(screen, "model").value = "m2"
        await save(pilot)
        assert deps.saved == [{"model": "m2"}]


async def test_stored_value_of_the_wrong_type_is_shown_as_text_and_kept(tmp_path):
    deps = make_deps(tmp_path, settings={"sample_rate": "fast", "auto_title": "maybe"})
    app = TinyStenoApp(CONFIG, deps)
    async with app.run_test() as pilot:
        screen = await open_settings(app, pilot)
        assert field(screen, "sample_rate").value == "fast"
        auto_title = field(screen, "auto_title")
        assert isinstance(auto_title, Input) and auto_title.value == "maybe"
        field(screen, "model").value = "m2"
        await save(pilot)
        assert deps.saved == [{"model": "m2"}]
        assert isinstance(app.screen, HomeScreen)


@pytest.mark.parametrize("key, text", [
    ("session_timeout_minutes", "soon"),
    ("session_timeout_minutes", "-5"),
    ("obsidian_vault", ""),
    ("auto_title", "maybe?"),
])
async def test_invalid_value_is_refused_and_focused(tmp_path, key, text):
    deps = make_deps(tmp_path, settings={"auto_title": "yes"})
    app = TinyStenoApp(CONFIG, deps)
    async with app.run_test() as pilot:
        screen = await open_settings(app, pilot)
        field(screen, key).value = text
        await save(pilot)
        assert not deps.saved
        assert isinstance(app.screen, SettingsScreen)
        assert form_error(screen).startswith(setting(key).label)
        assert app.focused.id == f"set-{key}"


async def test_invalid_field_in_a_collapsed_section_is_expanded_and_focused(tmp_path):
    deps = make_deps(tmp_path)
    app = TinyStenoApp(CONFIG, deps)
    async with app.run_test() as pilot:
        screen = await open_settings(app, pilot)
        rate = field(screen, "sample_rate")
        rate.value = "0"
        section = next(c for c in screen.query(Collapsible) if str(c.title) == "Recording")
        section.collapsed = True
        await pilot.pause()
        await save(pilot)
        await pilot.pause()
        assert not section.collapsed
        assert app.focused is rate
        assert "must be more than 0" in form_error(screen)


async def test_validation_error_from_save_is_shown_and_screen_stays(tmp_path):
    deps = make_deps(tmp_path, save_error=ValueError("LLM model: not allowed"))
    app = TinyStenoApp(CONFIG, deps)
    async with app.run_test() as pilot:
        screen = await open_settings(app, pilot)
        field(screen, "model").value = "m2"
        await save(pilot)
        assert app.screen is screen
        assert form_error(screen) == "LLM model: not allowed"
        assert field(screen, "model").value == "m2"


async def test_write_failure_is_shown(tmp_path):
    deps = make_deps(tmp_path, save_error=OSError("read-only"))
    app = TinyStenoApp(CONFIG, deps)
    async with app.run_test() as pilot:
        screen = await open_settings(app, pilot)
        field(screen, "model").value = "m2"
        await save(pilot)
        assert app.screen is screen
        assert form_error(screen) == "Could not save settings: read-only"


async def test_unexpected_save_failure_does_not_crash(tmp_path):
    deps = make_deps(tmp_path, save_error=RuntimeError("boom"))
    app = TinyStenoApp(CONFIG, deps)
    async with app.run_test() as pilot:
        screen = await open_settings(app, pilot)
        field(screen, "model").value = "m2"
        await save(pilot)
        assert app.screen is screen
        assert form_error(screen) == "Could not save settings: boom"


async def test_settings_not_applied_warns_and_leaves(tmp_path):
    deps = make_deps(tmp_path, save_error=SettingsNotApplied("bad persona"))
    app = TinyStenoApp(CONFIG, deps)
    async with app.run_test() as pilot:
        screen = await open_settings(app, pilot)
        field(screen, "model").value = "m2"
        await save(pilot)
        assert isinstance(app.screen, HomeScreen)
        message = ("Settings saved, but they could not be applied to this session: bad persona."
                   " Restart TinySteno to use them.")
        assert (message, "warning") in notes(app)


def _unreadable(reason):
    def load():
        raise ValueError(reason)
    return load


async def test_malformed_file_disables_save(tmp_path):
    deps = make_deps(tmp_path, load_settings=_unreadable("Config file is malformed: x"))
    app = TinyStenoApp(CONFIG, deps)
    async with app.run_test() as pilot:
        screen = await open_settings(app, pilot)
        assert not screen.query(SettingsForm)
        text = str(screen.query_one("#load-error").content)
        assert text == "Config file cannot be read: Config file is malformed: x"
        assert str(config_path()) in str(screen.query_one("#config-path").content)
        assert screen.query_one("#save").disabled
        await save(pilot)
        assert not deps.saved and app.screen is screen
        await pilot.press("escape")
        await pilot.pause()
        assert isinstance(app.screen, HomeScreen)


async def test_unreadable_file_is_reported_not_a_crash(tmp_path):
    def load():
        raise PermissionError("Permission denied: config.yaml")
    app = TinyStenoApp(CONFIG, make_deps(tmp_path, load_settings=load))
    async with app.run_test() as pilot:
        screen = await open_settings(app, pilot)
        text = str(screen.query_one("#load-error").content)
        assert text == "Config file cannot be read: Permission denied: config.yaml"
        assert screen.query_one("#save").disabled


async def test_yaml_error_does_not_show_the_offending_line(tmp_path):
    reason = ('Config file is not valid YAML: /c.yaml: while scanning a simple key\n'
              '  in "<unicode string>", line 2, column 1:\n'
              '    api_key sk-live-123\n'
              '    ^\n'
              "could not find expected ':'")
    deps = make_deps(tmp_path, load_settings=_unreadable(reason))
    app = TinyStenoApp(CONFIG, deps)
    async with app.run_test() as pilot:
        screen = await open_settings(app, pilot)
        text = str(screen.query_one("#load-error").content)
        assert "sk-live-123" not in text
        assert "line 2, column 1" in text and "could not find expected" in text


async def test_api_key_is_masked_and_never_in_notifications(tmp_path):
    old, new = "sk-old-secret-1", "sk-new-secret-2"
    deps = make_deps(tmp_path, settings={"api_key": old, "whisper_api_key": old})
    app = TinyStenoApp(CONFIG, deps)
    async with app.run_test() as pilot:
        screen = await open_settings(app, pilot)
        for key in ("api_key", "whisper_api_key"):
            assert field(screen, key).password is True
        field(screen, "api_key").value = new
        field(screen, "whisper_api_key").value = new
        await save(pilot)
        assert deps.saved == [{"api_key": new, "whisper_api_key": new}]
        seen = repr(app.events) + repr(notes(app))
        assert old not in seen and new not in seen
        assert ("settings_saved", ("api_key", "whisper_api_key")) in app.events


async def test_secret_is_not_in_the_error_when_its_save_fails(tmp_path):
    secret = "sk-new-secret-2"
    deps = make_deps(tmp_path, save_error=OSError("disk full"))
    app = TinyStenoApp(CONFIG, deps)
    async with app.run_test() as pilot:
        screen = await open_settings(app, pilot)
        field(screen, "api_key").value = secret
        await save(pilot)
        assert secret not in form_error(screen) + repr(notes(app)) + repr(app.events)


async def test_escape_with_changes_asks_and_cancel_keeps_them(tmp_path):
    deps = make_deps(tmp_path)
    app = TinyStenoApp(CONFIG, deps)
    async with app.run_test() as pilot:
        screen = await open_settings(app, pilot)
        field(screen, "model").value = "m2"
        await pilot.press("escape")
        await pilot.pause()
        assert isinstance(app.screen, ConfirmScreen)
        assert "Discard unsaved changes?" in str(app.screen.query_one("#confirm-title").content)
        await pilot.press("escape")             # cancel: stay with the edit
        await pilot.pause()
        assert app.screen is screen and field(screen, "model").value == "m2"
        await pilot.press("escape")
        await pilot.pause()
        app.screen.query_one("#confirm").focus()
        await pilot.press("enter")              # Discard
        await pilot.pause()
        assert isinstance(app.screen, HomeScreen)
        assert not deps.saved


async def test_escape_without_changes_leaves_at_once(tmp_path):
    app = TinyStenoApp(CONFIG, make_deps(tmp_path, settings={"model": "x"}))
    async with app.run_test() as pilot:
        await open_settings(app, pilot)
        await pilot.press("escape")
        await pilot.pause()
        assert isinstance(app.screen, HomeScreen)


async def test_missing_persona_is_shown_as_missing(tmp_path):
    deps = make_deps(tmp_path, settings={"persona": "gone"})
    app = TinyStenoApp(CONFIG, deps)
    async with app.run_test() as pilot:
        screen = await open_settings(app, pilot)
        select = field(screen, "persona")
        assert select.value == "gone"
        options = [(str(prompt), value) for prompt, value in select._options]  # pylint: disable=protected-access
        assert ("gone (missing)", "gone") in options
        assert ("default", "default") in options and ("1on1", "1on1") in options
        field(screen, "model").value = "m2"
        await save(pilot)
        assert deps.saved == [{"model": "m2"}]


async def test_persona_can_be_changed_to_an_existing_one(tmp_path):
    deps = make_deps(tmp_path, settings={"persona": "gone"})
    app = TinyStenoApp(CONFIG, deps)
    async with app.run_test() as pilot:
        screen = await open_settings(app, pilot)
        field(screen, "persona").value = "1on1"
        await save(pilot)
        assert deps.saved == [{"persona": "1on1"}]


async def test_overridden_key_is_marked(tmp_path):
    deps = make_deps(tmp_path, overridden_keys=frozenset({"transcription_backend"}))
    app = TinyStenoApp(CONFIG, deps)
    async with app.run_test() as pilot:
        screen = await open_settings(app, pilot)
        marked = [str(label.content) for label in screen.query(".setting-label")
                  if OVERRIDE_NOTE in str(label.content)]
        assert marked == [f"{setting('transcription_backend').label} {OVERRIDE_NOTE}"]


def _live_save(app_ref: list, deps, new_dir=None):
    """A save that, like the real one, updates the app's live config and recordings_dir."""
    def live_save(changes):
        deps.saved.append(dict(changes))
        app_ref[0].config.update(changes)
        if new_dir is not None:
            deps.recordings_dir = new_dir
    return live_save


async def test_home_uses_new_default_persona_after_save(tmp_path):
    deps = make_deps(tmp_path, settings={"persona": "1on1"})
    app = TinyStenoApp(CONFIG, deps)
    deps.save_settings = _live_save([app], deps)
    async with app.run_test() as pilot:
        assert app.screen.query_one("#persona").value == "1on1"
        screen = await open_settings(app, pilot)
        field(screen, "persona").value = "default"
        await save(pilot)
        await pilot.pause()
        assert deps.saved == [{"persona": "default"}]
        assert isinstance(app.screen, HomeScreen)
        assert app.screen.query_one("#persona").value == "default"


async def test_home_keeps_a_persona_the_user_picked(tmp_path):
    deps = make_deps(tmp_path, list_personas=lambda: ["default", "1on1", "rca"])
    app = TinyStenoApp({"persona": "default"}, deps)
    deps.save_settings = _live_save([app], deps)
    async with app.run_test() as pilot:
        app.screen.query_one("#persona").value = "1on1"     # the user's choice this session
        await pilot.pause()
        screen = await open_settings(app, pilot)
        field(screen, "persona").value = "rca"
        await save(pilot)
        await pilot.pause()
        assert app.config["persona"] == "rca"
        assert app.screen.query_one("#persona").value == "1on1"


async def test_home_lists_recordings_from_the_new_folder_and_applies_the_theme(tmp_path):
    new_dir = tmp_path / "audio"
    new_dir.mkdir()
    (new_dir / "Fresh.wav").write_bytes(b"")
    deps = make_deps(tmp_path, recordings_dir=tmp_path / "old")
    app = TinyStenoApp(CONFIG, deps)
    deps.save_settings = _live_save([app], deps, new_dir=new_dir)
    async with app.run_test() as pilot:
        screen = await open_settings(app, pilot)
        field(screen, "recordings_path").value = str(new_dir)
        field(screen, "tui_theme").value = "nord"
        await save(pilot)
        await pilot.pause()
        recent = app.screen.query_one("#recent")
        assert str(recent.get_option_at_index(0).prompt) == "Fresh.wav"
        assert app.theme == "nord"


async def test_settings_fit_80x24(tmp_path):
    deps = make_deps(tmp_path)
    app = TinyStenoApp(CONFIG, deps)
    async with app.run_test(size=(80, 24)) as pilot:
        screen = await open_settings(app, pilot)
        await pilot.pause()
        for selector in ("#save", "#back"):
            region = screen.query_one(selector).region
            assert region.height >= 1 and region.bottom <= 24 and region.right <= 80
        form = screen.query_one(SettingsForm)
        assert form.max_scroll_y > 0                 # the form scrolls; the buttons do not
        field(screen, "model").value = "m2"
        await pilot.click("#save")
        await pilot.pause()
        assert deps.saved == [{"model": "m2"}]


async def test_a_whisper_model_outside_the_list_can_be_typed(tmp_path):
    deps = make_deps(tmp_path)
    app = TinyStenoApp(CONFIG, deps)
    async with app.run_test() as pilot:
        screen = await open_settings(app, pilot)
        model = field(screen, "whisper_model")
        assert isinstance(model, Input)
        model.value = ""
        model.focus()
        await pilot.press(*"distil-large-v3")
        await save(pilot)
        assert deps.saved == [{"whisper_model": "distil-large-v3"}]
