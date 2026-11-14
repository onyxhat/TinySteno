# Textual TUI + Session Timeout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give TinySteno a Textual TUI in which a meeting goes from launch to rendered note in two keypresses, and give every recording session a default timeout that the user can cancel or extend from the TUI.

**Architecture:** Three pieces of UI-free core land first: a `SessionTimer` (pure, clock-injected), a `run_pipeline()` extracted from `main._process_audio` that reports stage progress through a reporter protocol, and two small `AudioRecorder` additions (live per-stream levels, discard). A new `tinysteno/tui/` package then builds four screens (Home → Recording → Processing → Result) on those pieces, with every side-effecting dependency injected through a `TuiDeps` dataclass so screens are testable with Textual's `run_test()` pilot. The headless `record`/`process` commands keep working and share the same timer and pipeline.

**Tech Stack:** Python 3.12, uv, Textual (`textual>=8.2`, current release 8.2.8), Rich (existing), pytest + pytest-asyncio, pylint 4.0.5.

**Spec:** None exists. The "Design Decisions" section below is the spec for this plan; executors read it before any task.

## Global Constraints

- All Python runs through `uv run` (`uv run pytest`, `uv run pylint $(git ls-files '*.py')`). No `npm`.
- Pylint must pass on every tracked `.py` file, tests included; max line length 100. Do not "fix" `import-outside-toplevel` or `broad-exception-caught` (disabled on purpose).
- `textual` is imported lazily: only `tinysteno/tui/*` imports it at module top level; `tinysteno/main.py` imports `tinysteno.tui` inside `cmd_tui`.
- `tinysteno/tui/*` never imports `tinysteno.main`. Everything it needs arrives through `TuiDeps`.
- New files stay under 500 lines. `main.py` (736 lines) must not grow: Task 3 shrinks it.
- Existing tests in `tests/test_main.py` patch `tinysteno.main.WhisperTranscriber`, `tinysteno.main.Orchestrator` and `tinysteno.main.ObsidianExporter` and call `tinysteno.main._process_audio(wav_path, name, config, logger, persona, timestamp)`. That function, its signature and those patch points must keep working.
- Blocking calls never run on the Textual event loop: `recorder.start()` (up to 8 s on macOS), `recorder.stop()` (reads and mixes the whole recording) and the pipeline all run in `@work(thread=True)` workers; UI updates from them go through `self.app.call_from_thread(...)`.
- A recording is never lost to a UI path: timeout, quit, and pipeline failure all leave a WAV on disk and tell the user where it is. Only the explicit, confirmed Discard deletes audio.
- Do not edit `CHANGELOG.md` or the version (the bump workflow owns both).
- Commit messages follow the repo's conventional style (`feat:`, `refactor:`, `test:`, `docs:`).
- Widget, class and constant names in this plan were chosen against Textual 8.2. If the installed version names one differently, follow the installed version and keep this plan's ids (`#record`, `#elapsed`, …) unchanged. The same goes for how tests read a widget's text (`Static.renderable`, `Markdown.source`): keep the assertion's intent, use the attribute the installed version exposes.

## Design Decisions

### Session timeout

There is no recording-session timeout anywhere in this repo's git history (the only timeouts are the LLM 120 s, Whisper API 180 s and ScreenCaptureKit waits), so this plan defines it fresh:

| Decision | Value |
|---|---|
| Meaning | Maximum session length. When it elapses the recording **stops and is processed normally**; nothing is discarded. |
| Config key | `session_timeout_minutes` in `~/.tinysteno/config.yaml`; int or float, `>= 0`; `0` disables |
| Global default | `75` (also applied when the key is absent from an existing config) |
| Per-persona override | optional `session_timeout_minutes` key in a persona's `persona.yaml`; same rules (number `>= 0`, `0` = no limit for that persona); absent = use the global value |
| Precedence | `--timeout` flag, then the persona's value, then the global config value, then `75` |
| CLI override | `tinysteno record --timeout MINUTES`, `tinysteno tui --timeout MINUTES` |
| Warning window | last `60` seconds |
| Abort from TUI | `k` = keep recording (cancels the timeout for this session), available at any time while recording |
| Extend from TUI | `+` = add `15` minutes |
| Headless `record` | same timeout; Ctrl+C still stops early; prints the limit at start and a line when it fires |

### Entry points

- `tinysteno` with no subcommand: launches the TUI when stdin and stdout are TTYs; otherwise prints help (today's behaviour).
- `tinysteno tui [--now] [--persona SLUG] [--name NAME] [--timeout MINUTES]` plus the existing transcription override flags. `--now` skips Home and starts recording immediately.
- `record`, `process`, `list`, `config`, `setup`, `test` are unchanged apart from `record --timeout`.

### Screens and keys

Happy path: `tinysteno` → `Enter` (record) → meeting → `Enter` (stop) → note on screen. With `--now` it is one keypress.

```
HOME                                   RECORDING
┌ TinySteno ───────────────────────┐   ┌ ● REC  Weekly sync ───────────────┐
│ Name    [ optional            ]  │   │          00:42:17                 │  Digits
│ Persona [ default           ▾ ]  │   │  Mic    ▁▂▅▇▅▃▂▁▁▁▁▂▃▅▇▇▅▃▂▁▁▁▁▁  │  Sparkline
│        [  ● Record  ]            │   │  System ▁▁▁▁▁▂▅▇▇▅▂▁▁▁▁▁▁▂▃▅▇▅▃▂  │  Sparkline
│ Recent recordings                │   │  Auto-stop in 32:43    ━━━━━━──── │
│  Meeting-20261113-1502.wav       │   └───────────────────────────────────┘
└──────────────────────────────────┘    enter Stop  k Keep  + 15m  d Discard
 enter Record  q Quit

PROCESSING                             RESULT
 ✓ Saving audio                         (Markdown render of the exported note)
 ⠋ Transcribing  ━━━━━━━──── 61%
 · Summarizing                          o Open in Obsidian  n New  q Quit
 · Title & tags
 · Exporting
```

| Screen | Key | Action |
|---|---|---|
| Home | `enter` on Record button, or `r` anywhere but the Name input | start recording with the current name/persona |
| Home | `enter` on a recent recording | process that WAV with the selected persona |
| Recording | `enter`, `space`, `s` | stop and process |
| Recording | `k` | keep recording: cancel the timeout |
| Recording | `+` | extend the timeout by 15 minutes |
| Recording | `d` | discard: opens a confirm modal (`y` confirm, `n`/`escape` cancel) |
| Result | `o` | open the note in Obsidian (`obsidian://open?path=…`) |
| Result | `n` | back to Home |
| any | `q` | quit; while recording this stops and saves the WAV first (no processing) |

The Record button has initial focus. Persona defaults to the configured persona; Name is optional. Nothing on Home must be filled in. Under the persona selector a hint line shows the timeout that will apply to the selected persona (`Auto-stop after 75 min`, or `No time limit`), so a persona override is visible before recording starts.

### Visual effects

In this plan (all built-in Textual features, no extra dependencies):

1. `Digits` widget for the large elapsed clock.
2. Two rolling `Sparkline` meters, one per audio stream (`Mic` from `AudioRecorder.mic_level`, `System` from `AudioRecorder.system_level`), each showing the last 12 seconds and coloured with the `sparkline--min-color` / `sparkline--max-color` component classes. When system audio is not being captured the System row reads `System audio unavailable` instead.
3. Pulsing `● REC` indicator: a looping `styles.animate("opacity", …)`.
4. Timeout bar that changes state: `ProgressBar` counting down; in the warning window the panel takes a `-warning` class and CSS `transition: background 400ms, border 400ms` moves it to `$warning`, with a toast (`notify`) and `app.bell()`.
5. Stage checklist on Processing: rows move `pending → active → done` by CSS class with colour transitions; the active row shows a spinner; transcription shows a real `ProgressBar` from the `on_progress` fraction.
6. Result fades in (`styles.animate("opacity", 1.0, duration=0.4)`) and renders the note with the `Markdown` widget.
7. Themes: Textual's built-in themes (`tokyo-night`, `nord`, `gruvbox`, `catppuccin-mocha`, …), selectable from the command palette for free; optional config key `tui_theme` sets the starting one.
8. Border titles/subtitles carry state (`● REC`, capture mode), so the frame itself is the status line.

Stretch, deliberately not in this plan:

- Inline mode (`App.run(inline=True)`): the app renders under the shell prompt instead of fullscreen. Suits a tiny tool; not supported on Windows.
- Splash/text effects through the third-party `textualeffects` package (terminaltexteffects: decrypt, beams, rain).
- A real waveform via the Line API (`render_line` with half-block characters).
- SVG snapshot tests with `pytest-textual-snapshot`.

## Review Focus

1. **Quit while recording** (`q`, or the terminal closing): the user expects the audio kept. The recorder is stopped, the WAV written, and after the app exits the terminal shows `Saved recording: <path>` and `Process it with: tinysteno process <path>`. Test in Task 6.
2. **No input device / PortAudio failure at start**: `recorder.start()` raises `RuntimeError`. The user expects an error message and to land back on Home, not a traceback. Test in Task 6.
3. **Pipeline failure after a long meeting** (LLM endpoint down, template error, vault not writable): the user expects the WAV path shown and a retry key, not an app crash. Test in Task 7.
4. **Silence**: transcription returns no speech. The user expects "No speech detected" with the WAV path, and no empty note. Tests in Tasks 3 and 7.
5. **Bad timeout config** (`session_timeout_minutes: "two hours"`, `-5`, `true`, in the global config or in a persona's `persona.yaml`): the user expects a clear error naming the file's field before recording starts, not an auto-stop at a nonsense time. Tests in Task 2.

---

### Task 1: SessionTimer

**Files:**
- Create: `tinysteno/session.py`
- Test: `tests/test_session.py`

**Interfaces:**
- Produces:
  - `DEFAULT_SESSION_TIMEOUT_MINUTES = 75`, `TIMEOUT_WARNING_SECONDS = 60`, `EXTEND_SECONDS = 900`
  - `class SessionTimer(limit_seconds: float | None, clock: Callable[[], float] = time.monotonic)` with `start() -> None`, `cancel() -> None`, `extend(seconds: float = EXTEND_SECONDS) -> None`, and read-only properties `elapsed: float`, `limit_seconds: float | None`, `remaining: float | None` (None when there is no limit; never negative), `expired: bool`, `in_warning: bool`
  - `resolve_timeout_seconds(config: dict, persona_minutes: float | None = None, cli_minutes: float | None = None) -> float | None`: picks the first value that is not `None` from `cli_minutes`, `persona_minutes`, `config.get("session_timeout_minutes")`, `DEFAULT_SESSION_TIMEOUT_MINUTES`; returns it in seconds, or `None` when the picked value is `0`
  - `format_clock(seconds: float) -> str` (`"MM:SS"` under one hour, `"H:MM:SS"` from one hour)

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for the recording session timer."""
import pytest

from tinysteno.session import (
    DEFAULT_SESSION_TIMEOUT_MINUTES, SessionTimer, format_clock, resolve_timeout_seconds,
)


class FakeClock:
    """Manually advanced monotonic clock."""
    def __init__(self):
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _started(limit):
    clock = FakeClock()
    timer = SessionTimer(limit, clock=clock)
    timer.start()
    return timer, clock


def test_counts_down_and_expires():
    timer, clock = _started(600)
    clock.now += 599
    assert timer.elapsed == 599 and timer.remaining == 1 and not timer.expired
    clock.now += 5
    assert timer.remaining == 0 and timer.expired


def test_no_limit_never_expires():
    timer, clock = _started(None)
    clock.now += 10 ** 6
    assert timer.remaining is None and not timer.expired and not timer.in_warning


def test_warning_window_is_last_60_seconds():
    timer, clock = _started(600)
    clock.now += 539
    assert not timer.in_warning
    clock.now += 1
    assert timer.in_warning


def test_cancel_removes_limit_and_keeps_elapsed():
    timer, clock = _started(600)
    clock.now += 590
    timer.cancel()
    clock.now += 100
    assert timer.limit_seconds is None and not timer.expired and timer.elapsed == 690


def test_extend_adds_15_minutes_by_default():
    timer, clock = _started(600)
    clock.now += 590
    timer.extend()
    assert timer.remaining == 910 and not timer.in_warning


def test_extend_after_cancel_stays_unlimited():
    timer, _ = _started(600)
    timer.cancel()
    timer.extend()
    assert timer.limit_seconds is None


def test_before_start_nothing_has_elapsed():
    timer = SessionTimer(600, clock=FakeClock())
    assert timer.elapsed == 0 and timer.remaining == 600 and not timer.expired


def test_default_is_75_minutes():
    assert DEFAULT_SESSION_TIMEOUT_MINUTES == 75
    assert resolve_timeout_seconds({}) == 4500


@pytest.mark.parametrize("config, persona, cli, expected", [
    ({"session_timeout_minutes": 90}, None, None, 5400),     # global config
    ({"session_timeout_minutes": 0.5}, None, None, 30),
    ({"session_timeout_minutes": 0}, None, None, None),      # 0 disables
    ({"session_timeout_minutes": 90}, 30, None, 1800),       # persona overrides global
    ({}, 30, None, 1800),                                    # persona overrides default
    ({"session_timeout_minutes": 90}, 0, None, None),        # persona can disable
    ({"session_timeout_minutes": 0}, 45, None, 2700),        # persona can re-enable
    ({"session_timeout_minutes": 90}, 30, 10, 600),          # flag overrides persona
    ({"session_timeout_minutes": 90}, 30, 0, None),          # flag can disable
])
def test_resolve_timeout_precedence(config, persona, cli, expected):
    assert resolve_timeout_seconds(config, persona, cli) == expected


@pytest.mark.parametrize("seconds, text", [
    (0, "00:00"), (59.9, "00:59"), (3599, "59:59"), (3600, "1:00:00"), (4663, "1:17:43"),
])
def test_format_clock(seconds, text):
    assert format_clock(seconds) == text
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_session.py -v`
Expected: FAIL at import, `ModuleNotFoundError: No module named 'tinysteno.session'`

- [ ] **Step 3: Implement `tinysteno/session.py`** with the interface above. No threads, no sleeping: the timer only reads the injected clock.

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_session.py -v && uv run pylint tinysteno/session.py tests/test_session.py`
Expected: all tests PASS; pylint 10.00/10

- [ ] **Step 5: Commit**

```bash
git add tinysteno/session.py tests/test_session.py
git commit -m "feat: add SessionTimer for recording session timeouts"
```

---

### Task 2: Timeout config, persona override, `--timeout`, and headless `record`

**Files:**
- Modify: `tinysteno/main.py` (`load_config` default dict, `_validate_config`, `cmd_record` wait loop, `cmd_setup` Audio section and written dict, `record` argparser)
- Modify: `tinysteno/personas/__init__.py` (`Persona`, `_validate_dir`, `_load_from_dir`)
- Test: `tests/test_main.py`, `tests/test_personas.py`

**Interfaces:**
- Consumes: `SessionTimer`, `resolve_timeout_seconds`, `DEFAULT_SESSION_TIMEOUT_MINUTES` from Task 1
- Produces:
  - config key `session_timeout_minutes` (validated; default `75` written into new configs)
  - `Persona.session_timeout_minutes: float | None = None`: a new last dataclass field with a default, so every existing `Persona(...)` construction keeps working; loaded from the optional `session_timeout_minutes` key of `persona.yaml`
  - `_non_negative_minutes(value: str) -> float` in `main.py`: argparse `type=` that raises `argparse.ArgumentTypeError` for non-numbers and negatives (Task 5 reuses it for `tui --timeout`)
  - `args.timeout` is **not** merged into `config` (that would let a persona override the flag); it is passed to `resolve_timeout_seconds` as `cli_minutes`

- [ ] **Step 1: Write the failing tests** (append to `tests/test_main.py`; add `import pytest` and extend the existing `from tinysteno.main import …` line with `_validate_config, cmd_record, main`)

```python
@pytest.mark.parametrize("bad", ["two hours", -5, True, None])
def test_validate_config_rejects_bad_session_timeout(bad):
    with pytest.raises(ValueError, match="session_timeout_minutes"):
        _validate_config({"session_timeout_minutes": bad})


@pytest.mark.parametrize("good", [0, 120, 0.5])
def test_validate_config_accepts_session_timeout(good):
    _validate_config({"session_timeout_minutes": good})


def _record_args(**overrides):
    base = {"persona": None, "name": None, "timeout": None}
    return argparse.Namespace(**{**base, **overrides})


def test_cmd_record_stops_and_processes_when_timeout_elapses(tmp_path, capsys):
    config = {"session_timeout_minutes": 0.001, "obsidian_vault": str(tmp_path)}
    with patch("tinysteno.main.AudioRecorder") as rec_cls, \
         patch("tinysteno.main.load_persona", return_value=_make_list_only_persona()), \
         patch("tinysteno.main._process_audio") as process:
        rec_cls.return_value.start.return_value = "/tmp/x.wav"
        cmd_record(_record_args(), config)  # returns without KeyboardInterrupt
    rec_cls.return_value.stop.assert_called_once()
    process.assert_called_once()
    assert "timeout" in capsys.readouterr().out.lower()


def _timer_limit(tmp_path, *, persona_minutes, cli_minutes):
    """Run cmd_record with a 120-minute global config; return the limit given to SessionTimer."""
    config = {"session_timeout_minutes": 120, "obsidian_vault": str(tmp_path)}
    persona = _make_list_only_persona()
    persona.session_timeout_minutes = persona_minutes
    with patch("tinysteno.main.AudioRecorder") as rec_cls, \
         patch("tinysteno.main.load_persona", return_value=persona), \
         patch("tinysteno.main._process_audio"), \
         patch("tinysteno.main.SessionTimer") as timer_cls:
        rec_cls.return_value.start.return_value = "/tmp/x.wav"
        timer_cls.return_value.expired = True
        cmd_record(_record_args(timeout=cli_minutes), config)
    return timer_cls.call_args.args[0]


def test_global_timeout_applies_when_persona_sets_none(tmp_path):
    assert _timer_limit(tmp_path, persona_minutes=None, cli_minutes=None) == 7200


def test_persona_timeout_overrides_global(tmp_path):
    assert _timer_limit(tmp_path, persona_minutes=30, cli_minutes=None) == 1800


def test_persona_can_disable_the_timeout(tmp_path):
    assert _timer_limit(tmp_path, persona_minutes=0, cli_minutes=None) is None


def test_timeout_flag_overrides_persona_and_global(tmp_path):
    assert _timer_limit(tmp_path, persona_minutes=30, cli_minutes=5.0) == 300


def test_record_rejects_negative_timeout():
    with patch.object(sys, "argv", ["tinysteno", "record", "--timeout", "-1"]):
        with pytest.raises(SystemExit):
            main()
```

Append to `tests/test_personas.py` (it already has `_write_valid_persona_dir(d, slug, name=None)` at the bottom; read it and extend it with an optional `extra_yaml: str = ""` argument appended to the `persona.yaml` it writes):

```python
def test_persona_timeout_defaults_to_none(tmp_path):
    _write_valid_persona_dir(tmp_path / "personas" / "p", "p")
    with patch("tinysteno.personas._USER_DIR", tmp_path / "personas"):
        assert load_persona("p").session_timeout_minutes is None


@pytest.mark.parametrize("value, expected", [("30", 30), ("0", 0), ("22.5", 22.5)])
def test_persona_timeout_is_loaded_from_yaml(tmp_path, value, expected):
    _write_valid_persona_dir(
        tmp_path / "personas" / "p", "p", extra_yaml=f"session_timeout_minutes: {value}\n"
    )
    with patch("tinysteno.personas._USER_DIR", tmp_path / "personas"):
        assert load_persona("p").session_timeout_minutes == expected


@pytest.mark.parametrize("value", ['"two hours"', "-5", "true", "[30]"])
def test_persona_with_bad_timeout_is_invalid(tmp_path, value):
    _write_valid_persona_dir(
        tmp_path / "personas" / "p", "p", extra_yaml=f"session_timeout_minutes: {value}\n"
    )
    with patch("tinysteno.personas._USER_DIR", tmp_path / "personas"):
        with pytest.raises(PersonaInvalidError, match="session_timeout_minutes"):
            load_persona("p")


def test_no_builtin_persona_overrides_the_timeout(seeded_user_dir):
    with patch("tinysteno.personas._USER_DIR", seeded_user_dir):
        assert all(load_persona(s).session_timeout_minutes is None for s in BUILTIN_ORDER)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_personas.py -v -k "timeout" && uv run pytest tests/test_main.py -x -v -k "validate_config"`
Expected: FAIL (`AttributeError: 'Persona' object has no attribute 'session_timeout_minutes'`, then `DID NOT RAISE ValueError`). Do not run `test_cmd_record_stops_and_processes_when_timeout_elapses` before Step 3: against today's `while True` loop it hangs.

- [ ] **Step 3: Implement in `tinysteno/personas/__init__.py` and `tinysteno/main.py`**

- `Persona` gains `session_timeout_minutes: float | None = None`. `_validate_dir` rejects a present `session_timeout_minutes` that is not an `int`/`float`, is a `bool`, or is negative, with `PersonaInvalidError(f"Persona '{slug}': session_timeout_minutes must be a number >= 0 (0 disables)")`. `_load_from_dir` passes `data.get("session_timeout_minutes")` through. No built-in `persona.yaml` changes.

- `_validate_config`: `session_timeout_minutes`, when present, must be `int` or `float`, not `bool`, and `>= 0`; otherwise `ValueError("Config field 'session_timeout_minutes' must be a number >= 0 (0 disables)")`.
- `load_config` default dict gains `"session_timeout_minutes": DEFAULT_SESSION_TIMEOUT_MINUTES`.
- `record_parser.add_argument("--timeout", type=_non_negative_minutes, metavar="MINUTES", help="Auto-stop after MINUTES (0 disables; default from config)")`.
- `cmd_record`: build `SessionTimer(resolve_timeout_seconds(config, persona.session_timeout_minutes, args.timeout))` (use `getattr(args, "timeout", None)`), compute the printed `{minutes}` from that resolved limit, call `start()` after `recorder.start()`, replace `while True` with `while not timer.expired`. Printed lines, exactly:
  - at start with a limit: `Recording started...\nPress Ctrl+C to stop. Auto-stop after {minutes:g} min (--timeout 0 to disable).`
  - at start without a limit: the existing two lines unchanged
  - on expiry, before `Recording stopped.`: `Session timeout reached ({minutes:g} min).`
- `cmd_setup`: in the Audio section prompt `"Session timeout (minutes)"` with hint `"auto-stop a recording after this long; 0 disables"`, default from existing config or `75`; unparseable or negative input falls back to the default; include the key in the written dict.

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_main.py tests/test_personas.py -v && uv run pylint tinysteno/main.py tinysteno/personas/__init__.py tests/test_main.py tests/test_personas.py`
Expected: all PASS (including the pre-existing `cmd_setup` tests, which feed `""` to every prompt); pylint clean

- [ ] **Step 5: Commit**

```bash
git add tinysteno/main.py tinysteno/personas/__init__.py tests/test_main.py tests/test_personas.py
git commit -m "feat: 75-minute default session timeout with per-persona override and --timeout"
```

---

### Task 3: Extract the pipeline with stage reporting

**Files:**
- Create: `tinysteno/pipeline.py`
- Modify: `tinysteno/main.py` (`_process_audio` becomes a thin wrapper; `_format_duration` and `_extract_summary_text` move to `pipeline.py` and are re-imported into `main` under their current names)
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Produces, all in `tinysteno/pipeline.py`:
  - `class Stage(str, Enum)`: `TRANSCRIBE = "transcribe"`, `SUMMARIZE = "summarize"`, `METADATA = "metadata"`, `EXPORT = "export"`
  - `class PipelineReporter(Protocol)`: `stage_started(self, stage: Stage) -> None`, `stage_progress(self, stage: Stage, fraction: float) -> None`, `stage_finished(self, stage: Stage) -> None`, `stage_skipped(self, stage: Stage) -> None`
  - `class NullReporter` implementing it with no-ops
  - `class PipelineError(RuntimeError)` with `__init__(self, stage: Stage, message: str = "")` and attribute `stage: Stage`
  - `@dataclass(frozen=True) class PipelineResult`: `wav_path: str`, `note_path: str | None`, `title: str | None`, `duration_seconds: float`, `no_speech: bool`
  - `def run_pipeline(wav_path: str, name: str | None, config: dict, persona: Persona, timestamp: datetime, *, transcriber: Transcriber, orchestrator_cls: type, exporter_cls: type, reporter: PipelineReporter | None = None) -> PipelineResult`
- Behaviour that the tests pin:
  - Stages run in enum order. Each stage that runs gets `stage_started` then `stage_finished`; a stage that does not apply gets `stage_skipped` (SUMMARIZE when `config["api_key"]` is empty; METADATA when neither `auto_title` nor `auto_tags` is on, there is no orchestrator, or there is no summary text).
  - The transcriber is called with `on_progress=` wired to `reporter.stage_progress(Stage.TRANSCRIBE, fraction)`.
  - Empty/whitespace transcript: returns `PipelineResult(no_speech=True, note_path=None, …)`; the three later stages are reported skipped; the exporter class is never instantiated.
  - Title and tags still run concurrently (existing `ThreadPoolExecutor(max_workers=2)`); the existing `test_title_and_tags_generated_in_parallel` must keep passing through the wrapper.
  - Any exception inside a stage is re-raised as `PipelineError` carrying that stage, with the original as `__cause__`.
- `main._process_audio` keeps its exact signature. It calls `run_pipeline(..., transcriber=create_transcriber(config), orchestrator_cls=Orchestrator, exporter_cls=ObsidianExporter, reporter=<Rich reporter>)`, resolving those three names from `tinysteno.main`'s globals at call time (that is what keeps the existing `patch("tinysteno.main.…")` tests valid). It prints `No speech detected, skipping export.` on `no_speech`, `Error rendering note: {cause}` when a `PipelineError` has `stage is Stage.EXPORT`, re-raises the cause for other stages (today's behaviour: those exceptions propagate), and `Meeting: {note_path}` on success. The Rich reporter is a small private class in `main.py` that drives the existing transient `Progress`, now with a real bar for transcription.

- [ ] **Step 1: Write the failing tests** in `tests/test_pipeline.py`

Shared fakes at the top of the file: `_persona()` (copy the `Persona(...)` construction used by `_make_list_only_persona` in `tests/test_main.py`, but with one `string` field `summary`), a `RecordingReporter` that appends `(event, stage)` tuples to `.events`, a `FakeTranscriber` whose `transcribe(path, diarize=False, on_progress=None)` calls `on_progress(0.5)` then returns `{"text": text, "diarised_text": "", "duration_seconds": 61.0, "detected_language": "en"}`, and `MagicMock` orchestrator/exporter classes (`summarize` → `{"summary": "We met."}`, `generate_title` → `"Sync"`, `generate_tags` → `["a"]`, `export` → `"/vault/meetings/Sync.md"`).

```python
def test_happy_path_reports_every_stage_in_order(tmp_path):
    reporter = RecordingReporter()
    result = _run(tmp_path, reporter=reporter)
    assert result == PipelineResult(
        wav_path="x.wav", note_path="/vault/meetings/Sync.md", title="Sync",
        duration_seconds=61.0, no_speech=False,
    )
    assert reporter.events == [
        ("started", Stage.TRANSCRIBE), ("progress", Stage.TRANSCRIBE), ("finished", Stage.TRANSCRIBE),
        ("started", Stage.SUMMARIZE), ("finished", Stage.SUMMARIZE),
        ("started", Stage.METADATA), ("finished", Stage.METADATA),
        ("started", Stage.EXPORT), ("finished", Stage.EXPORT),
    ]


def test_explicit_name_wins_over_generated_title(tmp_path):
    assert _run(tmp_path, name="Board call").title == "Board call"


def test_no_speech_skips_everything_after_transcription(tmp_path):
    reporter = RecordingReporter()
    exporter_cls = MagicMock()
    result = _run(tmp_path, text="   ", reporter=reporter, exporter_cls=exporter_cls)
    assert result.no_speech and result.note_path is None
    exporter_cls.assert_not_called()
    assert reporter.events[-3:] == [
        ("skipped", Stage.SUMMARIZE), ("skipped", Stage.METADATA), ("skipped", Stage.EXPORT),
    ]


def test_no_api_key_skips_llm_stages_and_titles_from_filename(tmp_path):
    reporter = RecordingReporter()
    result = _run(tmp_path, config_overrides={"api_key": ""}, reporter=reporter)
    assert result.title == "x"
    assert ("skipped", Stage.SUMMARIZE) in reporter.events
    assert ("skipped", Stage.METADATA) in reporter.events


def test_stage_failure_becomes_pipeline_error_with_stage(tmp_path):
    orchestrator_cls = MagicMock()
    orchestrator_cls.return_value.summarize.side_effect = ConnectionError("endpoint down")
    with pytest.raises(PipelineError) as info:
        _run(tmp_path, orchestrator_cls=orchestrator_cls)
    assert info.value.stage is Stage.SUMMARIZE
    assert isinstance(info.value.__cause__, ConnectionError)


def test_export_template_error_is_pipeline_error_at_export(tmp_path):
    exporter_cls = MagicMock()
    exporter_cls.return_value.export.side_effect = RuntimeError("Template render error")
    with pytest.raises(PipelineError) as info:
        _run(tmp_path, exporter_cls=exporter_cls)
    assert info.value.stage is Stage.EXPORT
```

`_run(tmp_path, *, name=None, text="hello there", config_overrides=None, reporter=None, orchestrator_cls=None, exporter_cls=None)` is the file's helper: it builds the config `{"api_key": "k", "base_url": "u", "model": "m", "auto_title": True, "auto_tags": True, "obsidian_vault": str(tmp_path), "output_folder": "meetings", "diarization": False}`, merges overrides, and calls `run_pipeline("x.wav", name, config, _persona(), datetime(2024, 1, 1), …)`.

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: FAIL at import, `No module named 'tinysteno.pipeline'`

- [ ] **Step 3: Implement `tinysteno/pipeline.py`** by moving the body of `main._process_audio` (lines 154-245 today) and the two helpers, replacing prints and the Rich `Progress` with reporter calls, and wrapping each stage so failures raise `PipelineError(stage)`.

- [ ] **Step 4: Rewrite `main._process_audio` as the wrapper** described under Interfaces.

- [ ] **Step 5: Run the whole suite**

Run: `uv run pytest && uv run pylint $(git ls-files '*.py') tinysteno/pipeline.py tests/test_pipeline.py`
Expected: all PASS, including the untouched `test_auto_tags_called_for_list_only_persona` and `test_title_and_tags_generated_in_parallel`; pylint clean; `wc -l tinysteno/main.py` is lower than 736

- [ ] **Step 6: Commit**

```bash
git add tinysteno/pipeline.py tinysteno/main.py tests/test_pipeline.py
git commit -m "refactor: extract run_pipeline with stage reporting from main"
```

---

### Task 4: Recorder per-stream levels and discard

**Files:**
- Modify: `tinysteno/recorder.py` (`__init__`, `_write_mic_frame`, `_loopback_callback`, `_write_loopback_frame`, `stop`, new `mic_level` / `system_level` / `has_loopback` properties, new `discard`)
- Test: `tests/test_recorder.py`

**Interfaces:**
- Produces on `AudioRecorder`:
  - `mic_level: float` (property): RMS of the most recent microphone frame, clamped to `0.0..1.0`; `0.0` when not recording
  - `system_level: float` (property): the same for the most recent system-audio frame, updated by both loopback paths (`_loopback_callback` for PortAudio on Windows/Linux, `_write_loopback_frame` for ScreenCaptureKit on macOS); `0.0` when not recording or when there is no loopback
  - `has_loopback: bool` (property): public read of `_has_loopback`, which tells the Recording screen whether to show the System meter
  - `discard() -> None`: stop all streams, delete the temp raw files, write no WAV, leave the recorder not recording; safe to call when not recording

- [ ] **Step 1: Write the failing tests** (append; follow the file's existing `patch("tinysteno.recorder.sd")` setup from `test_stop_produces_valid_wav`)

```python
def test_mic_level_tracks_rms_of_latest_mic_frame(tmp_path):
    rec = _make_recorder(tmp_path)
    with patch("tinysteno.recorder.sd") as mock_sd:
        _stub_input_device(mock_sd)
        rec.start()
        assert rec.mic_level == 0.0
        rec._write_mic_frame(np.full((1024, 1), 0.5, dtype=np.float32))  # pylint: disable=protected-access
        assert rec.mic_level == pytest.approx(0.5, abs=1e-3)
        rec._write_mic_frame(np.full((1024, 1), 4.0, dtype=np.float32))  # pylint: disable=protected-access
        assert rec.mic_level == 1.0
        rec.stop()
    assert rec.mic_level == 0.0


@pytest.mark.parametrize("feed", ["_loopback_callback", "_write_loopback_frame"])
def test_system_level_tracks_both_loopback_paths_independently_of_mic(tmp_path, feed):
    rec = _make_recorder(tmp_path)
    frame = np.full((1024, 1), 0.25, dtype=np.float32)
    with patch("tinysteno.recorder.sd") as mock_sd:
        _stub_input_device(mock_sd)
        rec.start()
        assert rec.system_level == 0.0
        if feed == "_loopback_callback":
            rec._loopback_callback(frame, 1024, None, None)  # pylint: disable=protected-access
        else:
            rec._write_loopback_frame(frame)  # pylint: disable=protected-access
        assert rec.system_level == pytest.approx(0.25, abs=1e-3)
        assert rec.mic_level == 0.0
        rec.stop()
    assert rec.system_level == 0.0


def test_discard_writes_no_wav_and_removes_temp_files(tmp_path):
    rec = _make_recorder(tmp_path)
    with patch("tinysteno.recorder.sd") as mock_sd:
        _stub_input_device(mock_sd)
        wav_path = rec.start()
        rec._write_mic_frame(np.full((1024, 1), 0.5, dtype=np.float32))  # pylint: disable=protected-access
        temp_files = [rec._mic_raw_path, rec._loopback_raw_path]  # pylint: disable=protected-access
        rec.discard()
    assert not Path(wav_path).exists()
    assert not any(p.exists() for p in temp_files)
    assert rec.stop() is False


def test_discard_when_not_recording_is_a_no_op(tmp_path):
    _make_recorder(tmp_path).discard()
```

`_stub_input_device(mock_sd)` is a new helper in the test file holding the `query_devices` / `PortAudioError` stubbing that `test_stop_produces_valid_wav` already does inline; refactor that test to use it. On macOS the tests must also patch `AudioRecorder._start_macos_loopback` to return `False`, as the existing tests do (check lines 13-45 and mirror them).

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_recorder.py -v`
Expected: the new tests FAIL with `AttributeError` on `mic_level` / `system_level` / `discard`

- [ ] **Step 3: Implement** the three properties and `discard()` in `tinysteno/recorder.py`. `discard()` reuses `_stop_streams()` and `_cleanup_temp_files()`.

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_recorder.py -v && uv run pylint tinysteno/recorder.py tests/test_recorder.py`
Expected: all PASS; pylint clean

- [ ] **Step 5: Commit**

```bash
git add tinysteno/recorder.py tests/test_recorder.py
git commit -m "feat(recorder): expose live mic and system levels and add discard()"
```

---

### Task 5: TUI scaffold, Home screen, entry points

**Files:**
- Modify: `pyproject.toml` (dependency `textual>=8.2`; dev extra `pytest-asyncio>=1.0`; `[tool.pytest.ini_options] asyncio_mode = "auto"`; package-data `"tinysteno.tui" = ["*.tcss"]`), `uv.lock` (via `uv sync --extra dev`)
- Create: `tinysteno/tui/__init__.py`, `tinysteno/tui/deps.py`, `tinysteno/tui/app.py`, `tinysteno/tui/home.py`, `tinysteno/tui/styles.tcss`
- Modify: `tinysteno/main.py` (`cmd_tui`, `tui` subparser, no-subcommand dispatch)
- Test: `tests/tui_fakes.py`, `tests/test_tui_home.py`, `tests/test_main.py`

**Interfaces:**
- Consumes: `PipelineReporter`, `PipelineResult`, `run_pipeline` (Task 3); `_non_negative_minutes` (Task 2); `AudioRecorder.mic_level/system_level/has_loopback/discard` (Task 4)
- Produces:
  - `tinysteno/tui/deps.py`:
    ```python
    @dataclass
    class TuiDeps:
        make_recorder: Callable[[], Any]            # AudioRecorder-shaped: start(name)->str, stop()->bool, discard(), mic_level, system_level, has_loopback
        run_pipeline: Callable[[str, str | None, Persona, datetime, PipelineReporter], PipelineResult]
        list_personas: Callable[[], list[str]]
        load_persona: Callable[[str], Persona]
        recordings_dir: Path
        clock: Callable[[], float] = time.monotonic
    ```
  - `tinysteno/tui/app.py`: `class TinyStenoApp(App)` with `__init__(self, config: dict, deps: TuiDeps, *, autostart: bool = False, name: str | None = None, persona: str | None = None, timeout_override: float | None = None)`; attributes `config`, `deps`, `timeout_override` (minutes from `--timeout`, or `None`), `exit_message: str | None = None`; method `timeout_seconds_for(persona: Persona) -> float | None` returning `resolve_timeout_seconds(self.config, persona.session_timeout_minutes, self.timeout_override)`; method `start_recording(name: str | None, persona_slug: str) -> None` and `process_file(wav_path: str, persona_slug: str) -> None`. Both first append a tuple to `self.events` (a `list`, the app's navigation log: `("record", name, slug)` and `("process", wav_path, slug)`), which is what tests assert on. In this task that is all they do; Tasks 6 and 7 add the screen push after the append and keep the append.
  - `tinysteno/tui/home.py`: `class HomeScreen(Screen)` with widget ids `#name` (`Input`), `#persona` (`Select`), `#record` (`Button`), `#recent` (`OptionList`), `#timeout-hint` (`Static`); helper `recent_recordings(directory: Path, limit: int = 8) -> list[Path]` (newest first by mtime, `*.wav` only, empty list when the directory is missing)
  - `tinysteno/tui/__init__.py`: `run_tui(config: dict, deps: TuiDeps, *, autostart: bool = False, name: str | None = None, persona: str | None = None, timeout_override: float | None = None) -> str | None` (runs the app, returns `app.exit_message`)
  - `tests/tui_fakes.py`: `FakeClock` (as in Task 1), `FakeRecorder` (records calls in `.calls`; `start` returns `str(tmp/"Meeting.wav")` or raises `start_error` if set; `stop` writes an empty file at that path and returns `True`; `mic_level = 0.3`; `system_level = 0.6`; `has_loopback = True`, all plain attributes a test may reassign), and `make_deps(tmp_path, **overrides) -> TuiDeps` with `list_personas=lambda: ["default", "1on1"]`, a `load_persona(slug)` returning the Task 3 test persona with `slug` set to the argument and `session_timeout_minutes` taken from an overridable `persona_timeouts: dict[str, float]` (default `{}`, so `None`), and a `run_pipeline` fake returning a `PipelineResult` whose `note_path` is a real file under `tmp_path` containing `"---\ntitle: Sync\n---\n# Sync\n\nWe met.\n"`.
  - `main.cmd_tui(args, config) -> None`: applies `_apply_cli_overrides`, builds real `TuiDeps` (recorder factory from config as in `cmd_record`; `run_pipeline` as a lambda closing over `config`, `create_transcriber`, `Orchestrator`, `ObsidianExporter`), calls `run_tui` with `timeout_override=getattr(args, "timeout", None)`, prints the returned exit message if any.

- [ ] **Step 1: Add dependencies and confirm the toolkit works**

Edit `pyproject.toml` as listed, then:

Run: `uv sync --extra dev && uv run python -c "import textual; from textual.widgets import Digits, Sparkline, ProgressBar, Markdown, Select, OptionList, LoadingIndicator; print(textual.__version__)"`
Expected: prints a version `>= 8.2` with no ImportError

- [ ] **Step 2: Write the failing tests**

`tests/test_tui_home.py`:

```python
"""Home screen behaviour."""
import os

from tinysteno.tui.app import TinyStenoApp
from tinysteno.tui.home import recent_recordings
from tests.tui_fakes import make_deps

CONFIG = {"persona": "1on1", "session_timeout_minutes": 120}


async def test_record_button_has_focus_and_enter_starts_recording(tmp_path):
    app = TinyStenoApp(CONFIG, make_deps(tmp_path))
    async with app.run_test() as pilot:
        assert app.focused.id == "record"
        await pilot.press("enter")
        await pilot.pause()
        assert ("record", None, "1on1") in app.events


async def test_typed_name_is_passed_and_r_does_not_fire_inside_input(tmp_path):
    app = TinyStenoApp(CONFIG, make_deps(tmp_path))
    async with app.run_test() as pilot:
        app.screen.query_one("#name").focus()
        await pilot.press(*"Retro")          # contains "r": must type, not record
        assert app.events == []
        await pilot.press("enter")            # Enter in the name field starts recording
        await pilot.pause()
        assert ("record", "Retro", "1on1") in app.events


async def test_unknown_configured_persona_falls_back_to_first_available(tmp_path):
    app = TinyStenoApp({"persona": "gone"}, make_deps(tmp_path))
    async with app.run_test() as pilot:
        await pilot.press("r")
        await pilot.pause()
        assert ("record", None, "default") in app.events


async def test_selecting_recent_recording_processes_it(tmp_path):
    wav = tmp_path / "Old.wav"
    wav.write_bytes(b"")
    app = TinyStenoApp(CONFIG, make_deps(tmp_path))
    async with app.run_test() as pilot:
        app.screen.query_one("#recent").focus()
        await pilot.press("enter")
        await pilot.pause()
        assert ("process", str(wav), "1on1") in app.events


async def test_timeout_hint_follows_the_selected_persona(tmp_path):
    deps = make_deps(tmp_path, persona_timeouts={"1on1": 30, "default": 0})
    app = TinyStenoApp({"persona": "1on1"}, deps)
    async with app.run_test() as pilot:
        hint = app.screen.query_one("#timeout-hint")
        assert "30 min" in str(hint.renderable)
        app.screen.query_one("#persona").value = "default"
        await pilot.pause()
        assert "No time limit" in str(hint.renderable)


async def test_timeout_hint_shows_global_default_of_75(tmp_path):
    app = TinyStenoApp({"persona": "default"}, make_deps(tmp_path))
    async with app.run_test():
        assert "75 min" in str(app.screen.query_one("#timeout-hint").renderable)


def test_recent_recordings_newest_first_wav_only(tmp_path):
    for i, name in enumerate(["a.wav", "b.wav", "notes.md"]):
        path = tmp_path / name
        path.write_bytes(b"")
        os.utime(path, (i, i))
    assert [p.name for p in recent_recordings(tmp_path)] == ["b.wav", "a.wav"]
    assert recent_recordings(tmp_path / "missing") == []
```

Append to `tests/test_main.py`:

```python
def test_no_subcommand_launches_tui_on_a_tty(tmp_path):
    with patch.object(sys, "argv", ["tinysteno"]), \
         patch("tinysteno.main._is_interactive", return_value=True), \
         patch("tinysteno.main.load_config", return_value={"obsidian_vault": str(tmp_path)}), \
         patch("tinysteno.main.cmd_tui") as cmd_tui:
        main()
    cmd_tui.assert_called_once()


def test_no_subcommand_prints_help_when_not_a_tty(capsys):
    with patch.object(sys, "argv", ["tinysteno"]), \
         patch("tinysteno.main._is_interactive", return_value=False), \
         patch("tinysteno.main.cmd_tui") as cmd_tui:
        main()
    cmd_tui.assert_not_called()
    assert "usage:" in capsys.readouterr().out


def test_tui_subcommand_passes_now_flag(tmp_path):
    with patch.object(sys, "argv", ["tinysteno", "tui", "--now", "--timeout", "30"]), \
         patch("tinysteno.main.load_config", return_value={"obsidian_vault": str(tmp_path)}), \
         patch("tinysteno.main.cmd_tui") as cmd_tui:
        main()
    args = cmd_tui.call_args.args[0]
    assert args.now is True and args.timeout == 30.0
```

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest tests/test_tui_home.py tests/test_main.py -v`
Expected: FAIL, `No module named 'tinysteno.tui'` and `AttributeError: … has no attribute '_is_interactive'`

- [ ] **Step 4: Implement the TUI scaffold**

- `TinyStenoApp`: `CSS_PATH = "styles.tcss"`, `TITLE = "TinySteno"`, binding `q` → `quit`; `on_mount` sets `self.theme` from `config.get("tui_theme")` when it names a registered theme, pushes `HomeScreen`, and when `autostart` is true immediately calls `start_recording(name, persona or configured)`.
- `HomeScreen`: layout from the mockup; `#persona` options from `deps.list_personas()`, initial value = `app` persona argument, else `config["persona"]`, else the first option; `#record` focused on mount; binding `r` → record declared so it does not fire while an `Input` has focus (Textual's `Input` consumes printable keys, so a screen-level binding is enough: the second test proves it); `Input.Submitted` on `#name` and `Button.Pressed` on `#record` both call `app.start_recording(name or None, slug)`; `OptionList.OptionSelected` on `#recent` calls `app.process_file(path, slug)`. An empty recent list shows the single disabled line `No recordings yet`. `#timeout-hint` is set on mount and on `Select.Changed` to `Auto-stop after {minutes:g} min` or `No time limit`, from `app.timeout_seconds_for(deps.load_persona(slug))`; if the persona fails to load the hint shows `Persona error: {message}` in the error colour.
- `styles.tcss`: centre a fixed-width (max 64 columns) panel; layout rules only in this task.

- [ ] **Step 5: Implement the entry points in `main.py`**

`_is_interactive() -> bool` returns `sys.stdin.isatty() and sys.stdout.isatty()`. `tui` subparser: `--now` (store_true), `--name`, `--persona`, `--timeout` (`type=_non_negative_minutes`), `--verbose`, and the same five transcription override flags as `record`. In `main()`, `not args.command` becomes: interactive → load config and call `cmd_tui` with a namespace equivalent to `tui` with no flags; otherwise print help.

- [ ] **Step 6: Run to verify pass, then look at it**

Run: `uv run pytest && uv run pylint $(git ls-files '*.py') tinysteno/tui tests/tui_fakes.py tests/test_tui_home.py`
Expected: all PASS; pylint clean

Run: `uv run tinysteno tui` and confirm by eye: Record button focused, persona preselected, `q` quits cleanly.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock tinysteno/tui tinysteno/main.py tests/tui_fakes.py tests/test_tui_home.py tests/test_main.py
git commit -m "feat(tui): Textual app scaffold, home screen, and tui entry points"
```

---

### Task 6: Recording screen with abortable timeout

**Files:**
- Create: `tinysteno/tui/recording.py`
- Modify: `tinysteno/tui/app.py` (`start_recording`, quit handling), `tinysteno/tui/styles.tcss`
- Test: `tests/test_tui_recording.py`

**Interfaces:**
- Consumes: `SessionTimer`, `format_clock`, `TIMEOUT_WARNING_SECONDS` (Task 1); `TinyStenoApp.timeout_seconds_for` (Task 5); `TuiDeps`, `TinyStenoApp`, `FakeRecorder`, `FakeClock`, `make_deps` (Task 5)
- Produces:
  - `class RecordingScreen(Screen)` with `__init__(self, name: str | None, persona_slug: str)`; widget ids `#rec-indicator`, `#elapsed` (`Digits`), `#mic-level` and `#system-level` (`Sparkline`), `#system-unavailable` (`Static`), `#timeout-label` (`Static`), `#timeout-bar` (`ProgressBar`); public method `tick() -> None` (one UI refresh from the timer and recorder; also what `set_interval(0.1, self.tick)` calls) and attributes `timer: SessionTimer`, `recorder`
  - `class ConfirmDiscard(ModalScreen[bool])`
  - `TinyStenoApp.start_recording` pushes `RecordingScreen`; `TinyStenoApp.finish_recording(wav_path: str, name: str | None, persona_slug: str, started_at: datetime) -> None` is what the screen calls after a successful stop. In this task it appends `("process", wav_path, persona_slug)` to `self.events`; Task 7 adds the screen switch after the append.
- Behaviour:
  - On mount the persona is loaded with `deps.load_persona(persona_slug)` and `timer = SessionTimer(app.timeout_seconds_for(persona), clock=deps.clock)`; a persona that fails to load is handled exactly like a start failure (notify, pop to Home) and the recorder is never started. Then the recorder is created and started in a thread worker; until it returns the panel shows `Starting…`. `timer.start()` and `started_at = datetime.now()` happen when `start()` returns.
  - `tick()`: updates `#elapsed` with `format_clock(timer.elapsed)`; appends `recorder.mic_level` and `recorder.system_level` each to its own `deque(maxlen=120)` (12 seconds at the 0.1 s tick) and assigns them to `#mic-level` / `#system-level`, so both meters roll right-to-left independently; when `recorder.has_loopback` is false `#system-level` is hidden, `#system-unavailable` (`System audio unavailable`) is shown, and no system samples are collected; sets `#timeout-label` to `Auto-stop in {format_clock(remaining)}`, or `No time limit` when `remaining is None` (and hides `#timeout-bar`); toggles the `-warning` class on the screen from `timer.in_warning`; on the tick that first enters the warning window calls `self.notify("Auto-stopping in 60s. Press K to keep recording.", severity="warning")` and `self.app.bell()`; when `timer.expired`, stops exactly as the Stop key does.
  - Stop (keys `enter`, `space`, `s`, or expiry) is idempotent: the first call disables further ticks and runs `recorder.stop()` in a thread worker, then calls `app.finish_recording(...)`. If `stop()` returns `False` (no audio captured) notify `No audio was captured.` with `severity="error"` and pop back to Home.
  - `k`: `timer.cancel()`, notify `Timeout cancelled. Recording until you stop.` `+`: `timer.extend()`, notify `Extended by 15 minutes.`; a no-op with notify `No time limit is set.` after cancel.
  - `d`: push `ConfirmDiscard`; on `True` call `recorder.discard()` (thread worker) and pop to Home with notify `Recording discarded.`
  - Start failure (`RuntimeError` from `start()`): notify the error text with `severity="error"`, pop to Home.
  - App quit while this screen is recording: `TinyStenoApp.action_quit` asks the active `RecordingScreen` to stop without processing, sets `exit_message = f"Saved recording: {wav}\nProcess it with: tinysteno process {shlex.quote(wav)}"`, then exits.

- [ ] **Step 1: Write the failing tests** in `tests/test_tui_recording.py`

Helper at the top: `async def _recording(pilot, app)` presses `enter` on Home, awaits `app.workers.wait_for_complete()` and `pilot.pause()`, and returns `app.screen` (asserting it is a `RecordingScreen`). Each test builds `clock = FakeClock()`, `rec = FakeRecorder(tmp_path)`, `deps = make_deps(tmp_path, clock=clock, make_recorder=lambda: rec)`, `app = TinyStenoApp({"persona": "default", "session_timeout_minutes": 120}, deps)`.

```python
async def test_enter_stops_and_hands_wav_to_processing(tmp_path):
    ...
    async with app.run_test() as pilot:
        await _recording(pilot, app)
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert rec.calls == ["start", "stop"]
        assert ("process", rec.wav_path, "default") in app.events


async def test_tick_shows_elapsed_and_countdown(tmp_path):
    async with app.run_test() as pilot:
        screen = await _recording(pilot, app)
        clock.now += 2537                       # 42:17 in, 1:17:43 left of 2:00:00
        screen.tick()
        await pilot.pause()
        assert str(screen.query_one("#elapsed").value) == "42:17"
        assert "1:17:43" in str(screen.query_one("#timeout-label").renderable)
        assert not screen.has_class("-warning")


async def test_warning_window_then_auto_stop_processes_recording(tmp_path):
    async with app.run_test() as pilot:
        screen = await _recording(pilot, app)
        clock.now += 7200 - 30
        screen.tick()
        await pilot.pause()
        assert screen.has_class("-warning") and "stop" not in rec.calls
        clock.now += 31
        screen.tick()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert rec.calls == ["start", "stop"]
        assert ("process", rec.wav_path, "default") in app.events


async def test_k_cancels_timeout_so_recording_outlives_the_limit(tmp_path):
    async with app.run_test() as pilot:
        screen = await _recording(pilot, app)
        clock.now += 7200 - 30
        screen.tick()
        await pilot.press("k")
        clock.now += 3600
        screen.tick()
        await pilot.pause()
        assert "stop" not in rec.calls and not screen.has_class("-warning")
        assert "No time limit" in str(screen.query_one("#timeout-label").renderable)


async def test_plus_extends_by_15_minutes(tmp_path):
    async with app.run_test() as pilot:
        screen = await _recording(pilot, app)
        clock.now += 7200 - 30
        await pilot.press("plus")
        screen.tick()
        await pilot.pause()
        assert screen.timer.remaining == 930 and not screen.has_class("-warning")


async def test_stop_pressed_twice_stops_once(tmp_path):
    async with app.run_test() as pilot:
        await _recording(pilot, app)
        await pilot.press("s", "s")
        await app.workers.wait_for_complete()
        assert rec.calls.count("stop") == 1


async def test_discard_needs_confirmation(tmp_path):
    async with app.run_test() as pilot:
        await _recording(pilot, app)
        await pilot.press("d", "n")
        await pilot.pause()
        assert "discard" not in rec.calls and isinstance(app.screen, RecordingScreen)
        await pilot.press("d", "y")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert rec.calls == ["start", "discard"] and isinstance(app.screen, HomeScreen)


async def test_start_failure_returns_home_with_error(tmp_path):
    rec.start_error = RuntimeError("No audio input devices found")
    async with app.run_test() as pilot:
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert isinstance(app.screen, HomeScreen)


async def test_quit_while_recording_saves_wav_and_reports_path(tmp_path):
    async with app.run_test() as pilot:
        await _recording(pilot, app)
        await pilot.press("q")
        await pilot.pause()
    assert rec.calls == ["start", "stop"]
    assert rec.wav_path in app.exit_message and "tinysteno process" in app.exit_message


async def test_each_stream_has_its_own_rolling_sparkline(tmp_path):
    async with app.run_test() as pilot:
        screen = await _recording(pilot, app)
        for mic, system in [(0.1, 0.9), (0.2, 0.8), (0.3, 0.7)]:
            rec.mic_level, rec.system_level = mic, system
            screen.tick()
        await pilot.pause()
        assert list(screen.query_one("#mic-level").data)[-3:] == [0.1, 0.2, 0.3]
        assert list(screen.query_one("#system-level").data)[-3:] == [0.9, 0.8, 0.7]
        for _ in range(200):
            screen.tick()
        assert len(screen.query_one("#mic-level").data) == 120


async def test_without_system_audio_the_system_meter_is_replaced_by_a_label(tmp_path):
    rec.has_loopback = False
    async with app.run_test() as pilot:
        screen = await _recording(pilot, app)
        screen.tick()
        await pilot.pause()
        assert not screen.query_one("#system-level").display
        assert screen.query_one("#system-unavailable").display
        assert screen.query_one("#mic-level").display


async def test_persona_timeout_overrides_global_in_the_tui(tmp_path):
    deps = make_deps(tmp_path, clock=clock, make_recorder=lambda: rec, persona_timeouts={"default": 30})
    app = TinyStenoApp({"persona": "default", "session_timeout_minutes": 120}, deps)
    async with app.run_test() as pilot:
        screen = await _recording(pilot, app)
        assert screen.timer.limit_seconds == 1800


async def test_timeout_flag_overrides_persona_in_the_tui(tmp_path):
    deps = make_deps(tmp_path, clock=clock, make_recorder=lambda: rec, persona_timeouts={"default": 30})
    app = TinyStenoApp({"persona": "default"}, deps, timeout_override=5)
    async with app.run_test() as pilot:
        screen = await _recording(pilot, app)
        assert screen.timer.limit_seconds == 300


async def test_invalid_persona_never_starts_the_recorder(tmp_path):
    def broken(_slug):
        raise PersonaInvalidError("session_timeout_minutes must be a number >= 0")
    deps = make_deps(tmp_path, make_recorder=lambda: rec, load_persona=broken)
    app = TinyStenoApp({"persona": "default"}, deps)
    async with app.run_test() as pilot:
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert isinstance(app.screen, HomeScreen) and rec.calls == []


async def test_zero_timeout_config_means_no_limit(tmp_path):
    app = TinyStenoApp({"persona": "default", "session_timeout_minutes": 0}, deps)
    async with app.run_test() as pilot:
        screen = await _recording(pilot, app)
        clock.now += 10 ** 6
        screen.tick()
        assert "stop" not in rec.calls
```

(`...` in the first test stands for the per-test setup lines described above the block; every test has them.)

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_tui_recording.py -v`
Expected: FAIL, `No module named 'tinysteno.tui.recording'`

- [ ] **Step 3: Implement `RecordingScreen`, `ConfirmDiscard`, and the app changes** per Interfaces/Behaviour. The interval timer is created only after `start()` succeeds, so tests drive `tick()` deterministically through the fake clock.

- [ ] **Step 4: Add the recording styles and effects to `styles.tcss` / the screen**

- `#rec-indicator`: pulsing opacity, a looping `styles.animate("opacity", 0.25, duration=0.7)` back and forth started on mount.
- `#mic-level`, `#system-level`: `Sparkline` with `summary_function=max`, one row each behind a fixed-width `Mic` / `System` label; `.sparkline--min-color { color: $success; }` and `.sparkline--max-color { color: $error; }`.
- Panel: `border: round $error; border-title-color: $error;` with `transition: background 400ms, border 400ms;`; `RecordingScreen.-warning` panel gets `border: heavy $warning; background: $warning 15%;`.
- `ConfirmDiscard`: centred box over a dimmed (`background: $background 60%`) backdrop.

- [ ] **Step 5: Run to verify pass, then look at it**

Run: `uv run pytest && uv run pylint $(git ls-files '*.py') tinysteno/tui tests/test_tui_recording.py`
Expected: all PASS; pylint clean

Run: `uv run tinysteno tui --now --timeout 1.2` and confirm by eye: clock counts, the Mic meter rolls when you speak and the System meter rolls when the computer plays audio, each independently, the panel turns to the warning style 12 seconds in with a toast and bell, `k` clears it, and with a fresh run it auto-stops at 1:12.

- [ ] **Step 6: Commit**

```bash
git add tinysteno/tui tests/test_tui_recording.py
git commit -m "feat(tui): recording screen with live meter and abortable session timeout"
```

---

### Task 7: Processing screen

**Files:**
- Create: `tinysteno/tui/processing.py`
- Modify: `tinysteno/tui/app.py` (`finish_recording`, `process_file`), `tinysteno/tui/styles.tcss`
- Test: `tests/test_tui_processing.py`

**Interfaces:**
- Consumes: `Stage`, `PipelineReporter`, `PipelineResult`, `PipelineError` (Task 3); `TuiDeps.run_pipeline`, `TuiDeps.load_persona` (Task 5)
- Produces:
  - `class ProcessingScreen(Screen)` with `__init__(self, wav_path: str, name: str | None, persona_slug: str, timestamp: datetime)`; one row per stage with ids `#stage-transcribe`, `#stage-summarize`, `#stage-metadata`, `#stage-export`, each carrying exactly one of the classes `-pending`, `-active`, `-done`, `-skipped`, `-failed`; `#transcribe-bar` (`ProgressBar`, total 1.0); `#error` (`Static`, hidden until a failure)
  - Row labels: `Transcribing`, `Summarizing`, `Title & tags`, `Exporting note`
  - Each of these keeps its `self.events` append and then navigates: `TinyStenoApp.finish_recording(...)` → `switch_screen(ProcessingScreen(wav, name, slug, started_at))`; `TinyStenoApp.process_file(wav_path, slug)` → `push_screen(ProcessingScreen(wav_path, None, slug, datetime.fromtimestamp(mtime)))`; `TinyStenoApp.show_result(result: PipelineResult) -> None`, in this task appending `("result", result)` to `self.events` (Task 8 adds the screen switch after the append)
- Behaviour:
  - On mount a thread worker calls `deps.run_pipeline(wav_path, name, deps.load_persona(slug), timestamp, reporter)`, where the reporter marshals each callback onto the UI with `call_from_thread`.
  - Success with a note → `app.show_result(result)`.
  - `no_speech` → rows after Transcribing marked `-skipped`; `#error` shows `No speech detected. Audio kept at {wav_path}`; keys `n` (Home) and `q` remain.
  - `PipelineError` (or any other exception, including a persona that fails to load) → the stage's row gets `-failed`; `#error` shows `{stage label} failed: {cause}` and `Audio kept at {wav_path}`; key `r` restarts the pipeline from the top on the same WAV; `n` returns Home.
  - Quit during processing sets `app.exit_message` to the same two-line `Saved recording … / Process it with …` text as Task 6.

- [ ] **Step 1: Write the failing tests** in `tests/test_tui_processing.py`

Each test builds deps with a scripted `run_pipeline` fake and mounts the screen directly: `app = TinyStenoApp(CONFIG, deps)`, then inside `run_test()` call `app.process_file(str(wav), "default")`, `await app.workers.wait_for_complete()`, `await pilot.pause()`.

```python
async def test_success_marks_all_stages_done_and_shows_result(tmp_path):
    def fake(wav, name, persona, ts, reporter):
        for stage in Stage:
            reporter.stage_started(stage)
            if stage is Stage.TRANSCRIBE:
                reporter.stage_progress(stage, 0.5)
            reporter.stage_finished(stage)
        return RESULT
    ...
    assert ("result", RESULT) in app.events


async def test_progress_fraction_drives_the_transcription_bar(tmp_path):
    gate = threading.Event()
    def fake(wav, name, persona, ts, reporter):
        reporter.stage_started(Stage.TRANSCRIBE)
        reporter.stage_progress(Stage.TRANSCRIBE, 0.61)
        gate.wait(5)
        return RESULT
    ...  # before releasing the gate:
    assert screen.query_one("#stage-transcribe").has_class("-active")
    assert screen.query_one("#transcribe-bar").progress == pytest.approx(0.61)
    assert screen.query_one("#stage-summarize").has_class("-pending")
    gate.set()


async def test_no_speech_keeps_audio_and_offers_home(tmp_path):
    ...  # fake returns PipelineResult(wav_path=str(wav), note_path=None, title=None, duration_seconds=3.0, no_speech=True)
    assert "No speech detected" in str(screen.query_one("#error").renderable)
    assert str(wav) in str(screen.query_one("#error").renderable)
    await pilot.press("n")
    await pilot.pause()
    assert isinstance(app.screen, HomeScreen)


async def test_failure_shows_stage_cause_and_wav_then_retry_succeeds(tmp_path):
    attempts = []
    def fake(wav, name, persona, ts, reporter):
        attempts.append(1)
        if len(attempts) == 1:
            reporter.stage_started(Stage.TRANSCRIBE)
            reporter.stage_finished(Stage.TRANSCRIBE)
            reporter.stage_started(Stage.SUMMARIZE)
            raise PipelineError(Stage.SUMMARIZE) from ConnectionError("endpoint down")
        return RESULT
    ...
    error = str(screen.query_one("#error").renderable)
    assert "Summarizing failed" in error and "endpoint down" in error and str(wav) in error
    assert screen.query_one("#stage-summarize").has_class("-failed")
    await pilot.press("r")
    await app.workers.wait_for_complete()
    await pilot.pause()
    assert ("result", RESULT) in app.events and len(attempts) == 2


async def test_persona_load_failure_is_shown_not_raised(tmp_path):
    ...  # deps.load_persona raises PersonaInvalidError("bad schema")
    assert "bad schema" in str(screen.query_one("#error").renderable)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_tui_processing.py -v`
Expected: FAIL, `No module named 'tinysteno.tui.processing'`

- [ ] **Step 3: Implement `ProcessingScreen` and the three app methods.**

- [ ] **Step 4: Style the checklist**

Row prefix glyphs by class: `-pending` `·`, `-active` a spinner (one-line `LoadingIndicator`, or a `Static` cycling the braille frames `⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏` on a 0.1 s interval), `-done` `✓`, `-skipped` `–`, `-failed` `✗`. Colours: pending `$text-muted`, active `$accent` bold, done `$success`, skipped `$text-muted` italic, failed `$error`; `transition: color 300ms;` on rows.

- [ ] **Step 5: Run to verify pass**

Run: `uv run pytest && uv run pylint $(git ls-files '*.py') tinysteno/tui tests/test_tui_processing.py`
Expected: all PASS; pylint clean

- [ ] **Step 6: Commit**

```bash
git add tinysteno/tui tests/test_tui_processing.py
git commit -m "feat(tui): processing screen with per-stage progress, failure and retry"
```

---

### Task 8: Result screen

**Files:**
- Create: `tinysteno/tui/result.py`
- Modify: `tinysteno/tui/app.py` (`show_result`), `tinysteno/tui/styles.tcss`
- Test: `tests/test_tui_result.py`

**Interfaces:**
- Consumes: `PipelineResult` (Task 3); `TinyStenoApp.show_result` slot (Task 7)
- Produces:
  - `strip_frontmatter(text: str) -> str`: removes a leading `---\n…\n---\n` YAML block only when the text starts with it; anything else is returned unchanged
  - `obsidian_url(note_path: str) -> str`: `"obsidian://open?path=" + urllib.parse.quote(note_path, safe="")`
  - `class ResultScreen(Screen)` with `__init__(self, result: PipelineResult)`; ids `#note` (`Markdown` inside a `VerticalScroll`), `#note-path` (`Static`); bindings `o` open in Obsidian (`self.app.open_url(obsidian_url(path))`), `n` new (back to Home), `q` quit
  - `TinyStenoApp.show_result(result)` → `switch_screen(ResultScreen(result))`
- Behaviour: border title is the result title; border subtitle is `format_clock(duration_seconds)`; the note file is read as UTF-8 in a thread worker; if reading fails `#note` shows `Note written to {path} (could not be displayed: {error})`.

- [ ] **Step 1: Write the failing tests**

```python
def test_strip_frontmatter_removes_only_a_leading_block():
    assert strip_frontmatter("---\ntitle: A\ntags: [x]\n---\n# A\n\nBody\n") == "# A\n\nBody\n"
    assert strip_frontmatter("# A\n\n---\n\nrule above\n") == "# A\n\n---\n\nrule above\n"
    assert strip_frontmatter("---\nunterminated\n") == "---\nunterminated\n"


def test_obsidian_url_percent_encodes_the_path():
    assert obsidian_url("/v/meetings/Q3 sync (2026-11-14).md") == \
        "obsidian://open?path=%2Fv%2Fmeetings%2FQ3%20sync%20%282026-11-14%29.md"


async def test_full_flow_two_keypresses_from_launch_to_note(tmp_path):
    rec = FakeRecorder(tmp_path)
    app = TinyStenoApp(CONFIG, make_deps(tmp_path, make_recorder=lambda: rec))
    async with app.run_test() as pilot:
        await pilot.press("enter")                 # 1: record
        await app.workers.wait_for_complete()
        await pilot.pause()
        await pilot.press("enter")                 # 2: stop
        await app.workers.wait_for_complete()
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert isinstance(app.screen, ResultScreen)
        assert app.screen.query_one("#note").source == "# Sync\n\nWe met.\n"


async def test_o_opens_obsidian_and_n_returns_home(tmp_path):
    ...  # reach ResultScreen as above, with patch.object(app, "open_url") as open_url
    await pilot.press("o")
    assert open_url.call_args.args[0].startswith("obsidian://open?path=")
    await pilot.press("n")
    await pilot.pause()
    assert isinstance(app.screen, HomeScreen) and app.focused.id == "record"


async def test_unreadable_note_still_reports_its_path(tmp_path):
    ...  # run_pipeline fake returns note_path=str(tmp_path / "missing.md")
    assert "missing.md" in app.screen.query_one("#note").source
```

If the installed `Markdown` widget exposes its text under a different attribute than `source`, keep the assertion's intent and read it the way that version exposes it.

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_tui_result.py -v`
Expected: FAIL, `No module named 'tinysteno.tui.result'`

- [ ] **Step 3: Implement `tinysteno/tui/result.py` and `TinyStenoApp.show_result`.** The screen mounts at `opacity: 0` and animates to `1.0` over 0.4 s once the Markdown is loaded.

- [ ] **Step 4: Run to verify pass, then run the real thing end to end**

Run: `uv run pytest && uv run pylint $(git ls-files '*.py') tinysteno/tui tests/test_tui_result.py`
Expected: all PASS; pylint clean

Run: `uv run tinysteno`, press Enter, speak for ten seconds, press Enter. Expected: stages tick through, the note renders, `o` opens it in Obsidian, `n` returns to Home with Record focused and the new WAV at the top of Recent recordings.

- [ ] **Step 5: Commit**

```bash
git add tinysteno/tui tests/test_tui_result.py
git commit -m "feat(tui): result screen with rendered note and open-in-Obsidian"
```

---

### Task 9: Documentation

**Files:**
- Modify: `README.md`, `SPEC.md`, `TODO.md`, `CLAUDE.md`

- [ ] **Step 1: `README.md`**: replace the feature line `Record meetings (indefinite duration, Ctrl+C to stop)` with one stating the 75-minute default timeout and how to change or disable it; in the Personas section document the optional `session_timeout_minutes` key in `persona.yaml` and the precedence (flag, persona, global config, 75); add a "TUI" section with the launch commands (`tinysteno`, `tinysteno tui --now`), the key table from this plan's Design Decisions, and the `session_timeout_minutes` / `tui_theme` config keys; add `--timeout` to the `record` usage.
- [ ] **Step 2: `SPEC.md`**: add module entries for `tinysteno/session.py`, `tinysteno/pipeline.py` and `tinysteno/tui/` (one paragraph each: purpose and public names from this plan's Interfaces blocks); update the `main.py` entry (subcommand `tui`, `_process_audio` now delegates to `run_pipeline`) and the recorder entry (`mic_level`, `system_level`, `has_loopback`, `discard`).
- [ ] **Step 3: `TODO.md`**: tick `Improved TUI experience`; add unticked items for the stretch effects listed in this plan (inline mode, splash text effects, Line API waveform, snapshot tests).
- [ ] **Step 4: `CLAUDE.md`**: under Personas add that `persona.yaml` may set `session_timeout_minutes` (number `>= 0`, `0` disables) to override the global recording timeout; under Notes add: the TUI lives in `tinysteno/tui/`, takes all side effects through `TuiDeps`, must not import `tinysteno.main`, and is tested with `App.run_test()` (pytest-asyncio auto mode).
- [ ] **Step 5: Verify and commit**

Run: `uv run pytest && uv run pylint $(git ls-files '*.py')`
Expected: all PASS; pylint clean

```bash
git add README.md SPEC.md TODO.md CLAUDE.md
git commit -m "docs: document the TUI and the session timeout"
```
