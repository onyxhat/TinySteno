"""Result screen: front-matter stripping, Obsidian link, and the full two-keypress flow."""
from unittest.mock import patch

from tinysteno.pipeline import PipelineResult
from tinysteno.tui.app import TinyStenoApp
from tinysteno.tui.home import HomeScreen
from tinysteno.tui.result import ResultScreen, obsidian_url, strip_frontmatter
from tests.tui_fakes import NOTE_TEXT, FakeRecorder, make_deps

CONFIG = {"persona": "default", "session_timeout_minutes": 120}


def test_strip_frontmatter_removes_only_a_leading_block():
    assert strip_frontmatter("---\ntitle: A\ntags: [x]\n---\n# A\n\nBody\n") == "# A\n\nBody\n"
    assert strip_frontmatter("# A\n\n---\n\nrule above\n") == "# A\n\n---\n\nrule above\n"
    assert strip_frontmatter("---\nunterminated\n") == "---\nunterminated\n"


def test_strip_frontmatter_treats_crlf_like_lf():
    assert strip_frontmatter("---\r\ntitle: A\r\n---\r\n# A\r\n") == "# A\r\n"
    assert strip_frontmatter("---\r\nunterminated\r\n") == "---\r\nunterminated\r\n"
    assert strip_frontmatter("---\n---\nBody\n") == "Body\n"


def test_obsidian_url_percent_encodes_the_path():
    assert obsidian_url("/v/meetings/Q3 sync (2026-11-14).md") == \
        "obsidian://open?path=%2Fv%2Fmeetings%2FQ3%20sync%20%282026-11-14%29.md"


async def _settle(app, pilot):
    await app.workers.wait_for_complete()
    await pilot.pause()


async def _record_to_result(app, pilot):
    await pilot.press("enter")                 # 1: record
    await _settle(app, pilot)
    await pilot.press("enter")                 # 2: stop
    await _settle(app, pilot)
    await _settle(app, pilot)


async def test_full_flow_two_keypresses_from_launch_to_note(tmp_path):
    rec = FakeRecorder(tmp_path)
    app = TinyStenoApp(CONFIG, make_deps(tmp_path, make_recorder=lambda: rec))
    async with app.run_test() as pilot:
        await _record_to_result(app, pilot)
        assert isinstance(app.screen, ResultScreen)
        assert app.screen.query_one("#note").source == "# Sync\n\nWe met.\n"
        assert str(app.screen.query_one("#note-path").content) == str(tmp_path / "Sync.md")
        scroll = app.screen.query_one("#note-scroll")
        assert scroll.border_title == "Sync"
        assert scroll.border_subtitle == "01:01"


async def test_o_opens_obsidian_and_n_returns_home(tmp_path):
    rec = FakeRecorder(tmp_path)
    app = TinyStenoApp(CONFIG, make_deps(tmp_path, make_recorder=lambda: rec))
    async with app.run_test() as pilot:
        await _record_to_result(app, pilot)
        with patch.object(app, "open_url") as open_url:
            await pilot.press("o")
        assert open_url.call_args.args[0] == obsidian_url(str(tmp_path / "Sync.md"))
        await pilot.press("n")
        await pilot.pause()
        assert isinstance(app.screen, HomeScreen) and app.focused.id == "record"
        assert sum(isinstance(s, HomeScreen) for s in app.screen_stack) == 1
        recent = app.screen.query_one("#recent")
        assert str(recent.get_option_at_index(0).prompt) == "Meeting.wav"


async def test_unreadable_note_still_reports_its_path(tmp_path):
    missing = str(tmp_path / "missing.md")

    def fake(wav, name, persona, ts, reporter):  # pylint: disable=unused-argument
        return PipelineResult(wav_path=wav, note_path=missing, title="T",
                              duration_seconds=5.0, no_speech=False)

    wav = tmp_path / "A.wav"
    wav.write_bytes(b"")
    app = TinyStenoApp(CONFIG, make_deps(tmp_path, run_pipeline=fake))
    async with app.run_test() as pilot:
        app.process_file(str(wav), "default")
        await _settle(app, pilot)
        await _settle(app, pilot)
        assert isinstance(app.screen, ResultScreen)
        source = app.screen.query_one("#note").source
        assert "missing.md" in source and "could not be displayed" in source


async def test_note_path_with_brackets_is_not_read_as_markup(tmp_path):
    note = tmp_path / "Q[3] [bold]sync.md"
    note.write_text("# Hi\n", encoding="utf-8")
    result = PipelineResult(wav_path="x.wav", note_path=str(note), title="T",
                            duration_seconds=5.0, no_speech=False)
    app = TinyStenoApp(CONFIG, make_deps(tmp_path))
    async with app.run_test() as pilot:
        app.show_result(result)
        await _settle(app, pilot)
        assert str(app.screen.query_one("#note-path").content) == str(note)
        assert app.screen.query_one("#note").source == "# Hi\n"


async def test_each_warning_becomes_a_warning_notification(tmp_path):
    note = tmp_path / "Sync.md"
    note.write_text(NOTE_TEXT)
    result = PipelineResult(
        wav_path="a.wav", note_path=str(note), title="Sync", duration_seconds=61.0,
        no_speech=False, warnings=("first [bold]problem[/]", "second problem"),
    )
    app = TinyStenoApp(CONFIG, make_deps(tmp_path))
    async with app.run_test(notifications=True) as pilot:
        await app.push_screen(ResultScreen(result))
        await _settle(app, pilot)
        shown = [(n.message, n.severity) for n in app._notifications]  # pylint: disable=protected-access
    assert shown == [("first [bold]problem[/]", "warning"), ("second problem", "warning")]
