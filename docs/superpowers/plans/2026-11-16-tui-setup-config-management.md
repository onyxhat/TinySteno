# TUI Setup, Settings and Management Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the SVG screenshot command from the running TUI, and let a user set up, configure and manage TinySteno (settings, personas, recordings, system check) without leaving the TUI.

**Architecture:** Configuration moves out of `tinysteno/main.py` into a UI-free `tinysteno/config.py` (defaults, load, validate, merge-and-write), described field by field by `tinysteno/settings.py` so the CLI wizard and the TUI form share one definition. The setup checks move into `tinysteno/doctor.py`. The TUI gains a Menu reached from Home and four screens (Settings, Setup, Personas, Recordings) plus a System check screen; every side effect they need is a new field on `TuiDeps`, wired in `cmd_tui`.

**Tech Stack:** Python 3, Textual 8.2, PyYAML, pytest + pytest-asyncio (auto mode), pylint, uv.

**Spec:** There is no separate spec document. The request is the spec, verbatim:
- "Remove the SVG screenshot functionality from the TUI app itself - that was only needed for testing."
- "Add setup, config, and management menus and functionality to the TUI."

Decisions this plan makes where the request is silent (each is a plan decision, not a quoted requirement; change them here before execution if they are wrong):
- "SVG screenshot functionality" is Textual's built-in **Screenshot** command in the command palette (`Ctrl+P`), which saves an SVG. `scripts/tui_screenshots.py` (regenerates the README images; not part of the app) and `docs/images/*.svg` stay.
- "Setup" = a first-run guide shown when no config file existed at launch, and re-openable from the Menu. "Config" = a Settings screen editing `~/.tinysteno/config.yaml`. "Management" = Personas (list, validity, set default, restore built-ins), Recordings (list all, process, transcribe again, delete) and a System check.
- Out of scope: editing persona files inside the TUI, creating personas, testing the LLM or transcription endpoint over the network, downloading the speaker-separation models from a menu, listing or deleting notes in the vault.

## Global Constraints

- Everything Python runs through `uv run`. `uv run pytest -q -W error` passes and `uv run pylint $(git ls-files '*.py')` stays at 10.00/10 (max line length 100) at every commit.
- `import-outside-toplevel` and `broad-exception-caught` are disabled on purpose; do not "fix" code to satisfy them.
- Nothing in `tinysteno/tui/` imports `tinysteno.main`. Every side effect a screen needs (writing config, seeding personas, deleting files, running checks) goes through `TuiDeps`. Reading the recordings directory listing directly is allowed (Home already does).
- TUI tests use `App.run_test()` with `tests/tui_fakes.make_deps`; no test touches `~/.tinysteno`, real audio hardware or the network. Tests that need a home directory use `monkeypatch.setenv("HOME", str(tmp_path))` plus `monkeypatch.setattr(Path, "home", lambda: tmp_path)`.
- Never run `tinysteno record`, `tinysteno tui`, `tinysteno.main.main()` or `cmd_record` without `AudioRecorder` stubbed: they record the real microphone.
- `sherpa_onnx` and `tinysteno.sherpa_diarizer` stay lazily imported; `tinysteno.config`, `tinysteno.settings` and `tinysteno.doctor` must import without them.
- Saving settings never drops a key it does not know about, and never writes values that came from CLI flags (`--backend`, `--whisper-api-key`, …) into the file.
- API keys are never logged, never put in a notification, and are shown masked in the TUI.
- Do not edit `CHANGELOG.md` or any version number. Do not change threshold constants in `hygiene.py`, `speaker_split.py`, `transcript.py` or `recorder.py`.
- Commit messages end with exactly: `Co-Authored-By: Claude <noreply@anthropic.com>`.
- Keep new files under 500 lines. `tinysteno/main.py` (958 lines) must end this plan shorter than it started.

## Review Focus

1. **A settings save that loses data.** A config file with keys the form does not show (`whisper_prompt`, `tui_theme`, `max_speakers`, anything hand-added) must keep them after a save from the TUI and after `tinysteno setup`. Today `cmd_setup` rewrites the file with 14 keys and silently drops the rest. Pinned in Task 2.
2. **CLI overrides leaking into the file.** `tinysteno tui --backend api --whisper-api-key sk-…` followed by a save in Settings must not write `api` or the key to `config.yaml`, and the override must still be in force afterwards. Pinned in Task 5.
3. **A malformed or unwritable config file.** Invalid YAML, a file that is a list, a read-only file or directory: Settings must show the reason and leave the file byte-for-byte unchanged; an interrupted write must not leave a truncated file. Pinned in Tasks 2 and 6.
4. **Deleting the wrong thing.** Delete must only ever remove a `.wav` directly inside the recordings directory plus its own `.transcript.json` / `.levels.json`, never follow a symlink out of it, and never delete the recording currently being processed. Pinned in Tasks 5 and 9.
5. **Invalid input in a field.** Text in a number field, a negative timeout, an empty vault path, a persona that no longer exists: the save is refused with the field named, focus moves to it, nothing is written. Pinned in Tasks 3 and 6.

---

## File Structure

| File | Responsibility |
|---|---|
| `tinysteno/config.py` (new) | Config path, `DEFAULT_CONFIG`, load, validate, read-stored, merge-and-write. No UI, no argparse. |
| `tinysteno/settings.py` (new) | `Setting` descriptors for the editable keys, grouped in sections; text ↔ value conversion. |
| `tinysteno/doctor.py` (new) | `run_checks(config) -> list[Check]`: the setup checks, UI-free. |
| `tinysteno/recordings.py` (new) | `RecordingInfo`, `list_recordings`, `delete_recording`. |
| `tinysteno/main.py` | CLI only: imports the above; `cmd_setup` and `cmd_test` become thin. |
| `tinysteno/tui/deps.py` | New `TuiDeps` fields. |
| `tinysteno/tui/menu.py` (new) | `MenuScreen` (modal list of destinations). |
| `tinysteno/tui/forms.py` (new) | `SettingsForm` widget: renders a list of `Setting`s, collects and validates values. |
| `tinysteno/tui/settings.py` (new) | `SettingsScreen`. |
| `tinysteno/tui/setup.py` (new) | `SetupScreen` (first-run guide). |
| `tinysteno/tui/personas.py` (new) | `PersonasScreen`. |
| `tinysteno/tui/recordings.py` (new) | `RecordingsScreen`. |
| `tinysteno/tui/checks.py` (new) | `ChecksScreen`. |
| `tinysteno/tui/confirm.py` (new) | `ConfirmScreen` (modal yes/no, returns bool). |
| `tinysteno/tui/app.py`, `home.py`, `processing.py`, `styles.tcss` | Navigation, bindings, palette commands, `retranscribe`. |

---

### Task 1: Remove the Screenshot command from the app

**Files:**
- Modify: `tinysteno/tui/app.py`
- Test: `tests/test_tui_app_commands.py` (new)

**Interfaces:**
- Produces: `TinyStenoApp.get_system_commands(screen)` — the override later tasks extend with their own commands.

- [ ] **Step 1: Write the failing tests**

```python
async def test_command_palette_has_no_screenshot_command(tmp_path):
    app = TinyStenoApp(CONFIG, make_deps(tmp_path))
    async with app.run_test():
        titles = [c.title for c in app.get_system_commands(app.screen)]
        assert "Screenshot" not in titles
        assert {"Theme", "Quit", "Keys"} <= set(titles)


async def test_screenshot_action_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    app = TinyStenoApp(CONFIG, make_deps(tmp_path))
    async with app.run_test() as pilot:
        await app.run_action("screenshot")
        await pilot.pause()
    assert not list(tmp_path.rglob("*.svg"))
```

- [ ] **Step 2: Run** `uv run pytest tests/test_tui_app_commands.py -q` — Expected: FAIL (`Screenshot` is in the list).

- [ ] **Step 3: Implement.** In `TinyStenoApp`: `get_system_commands(self, screen)` yields every command from `super()` except the one titled `Screenshot`; `action_screenshot(self, filename=None, path=None)` is overridden to do nothing (docstring: screenshots are made by `scripts/tui_screenshots.py`, which calls `save_screenshot` directly). Do not touch `save_screenshot` / `export_screenshot`: the script depends on them.

- [ ] **Step 4: Run** the new tests and `uv run python scripts/tui_screenshots.py "$(mktemp -d)"` — Expected: PASS, and the script lists six `tui-*.svg` files.

- [ ] **Step 5: Commit** — `feat(tui): remove the Screenshot command from the app`

---

### Task 2: `tinysteno/config.py` — one place for config, and saves that keep unknown keys

**Files:**
- Create: `tinysteno/config.py`
- Modify: `tinysteno/main.py` (delete the moved code; import the names so `tinysteno.main.load_config`, `_validate_config`, `_recordings_dir`, `_write_config`, `_is_timeout_minutes`, `_is_positive_number`, `_is_speaker_count` keep working for existing callers and tests)
- Test: `tests/test_config.py` (new)

**Interfaces:**
- Produces:
  - `config_path() -> Path` (`~/.tinysteno/config.yaml`, resolved on each call)
  - `DEFAULT_CONFIG: dict` (the dict currently inlined in `load_config`, unchanged values)
  - `load_config() -> dict` (behaviour unchanged: creates the default file and seeds personas on first run, validates, expands `obsidian_vault` / `recordings_path`)
  - `validate_config(config: dict) -> None` (raises `ValueError`; the current `_validate_config`)
  - `recordings_dir(config: dict) -> Path`
  - `read_stored() -> dict` — the file's own contents, unexpanded, `{}` when the file is missing; raises `ValueError` when it is not a mapping or not valid YAML
  - `update_config(changes: dict) -> dict` — `read_stored()`, apply `changes` (a value of `None` removes the key), `validate_config`, write, return the new stored dict; raises `ValueError` (validation, malformed file) or `OSError` (cannot write) and in both cases leaves the file unchanged
  - `write_config(path: Path, config: dict) -> None` — atomic: temp file in the same directory, then `os.replace`; a newly created file gets mode `0o600`, an existing file keeps its mode

- [ ] **Step 1: Write the failing tests** (all under a temp home)

```python
def test_update_keeps_keys_it_was_not_given(home):
    write_yaml(home, {"model": "a", "whisper_prompt": "Acme", "custom_thing": 7})
    stored = update_config({"model": "b"})
    assert stored == {"model": "b", "whisper_prompt": "Acme", "custom_thing": 7}
    assert yaml.safe_load(config_path().read_text()) == stored

def test_none_removes_a_key(home):
    write_yaml(home, {"recordings_path": "/x", "model": "a"})
    assert "recordings_path" not in update_config({"recordings_path": None})

def test_invalid_change_leaves_the_file_untouched(home):
    write_yaml(home, {"channels": 1})
    before = config_path().read_bytes()
    with pytest.raises(ValueError):
        update_config({"channels": 5})
    assert config_path().read_bytes() == before

def test_malformed_file_is_reported_and_untouched(home):
    config_path().parent.mkdir(parents=True)
    config_path().write_text("- just\n- a list\n")
    before = config_path().read_bytes()
    with pytest.raises(ValueError):
        update_config({"model": "b"})
    assert config_path().read_bytes() == before

def test_failed_write_leaves_the_old_file(home, monkeypatch):
    write_yaml(home, {"model": "a"})
    before = config_path().read_bytes()
    monkeypatch.setattr(os, "replace", raise_oserror)
    with pytest.raises(OSError):
        update_config({"model": "b"})
    assert config_path().read_bytes() == before
    assert not list(config_path().parent.glob("*.tmp*"))

def test_new_file_is_private_and_existing_mode_is_kept(home):
    update_config({"model": "a"})
    assert stat.S_IMODE(config_path().stat().st_mode) == 0o600
    config_path().chmod(0o644)
    update_config({"model": "b"})
    assert stat.S_IMODE(config_path().stat().st_mode) == 0o644

def test_setup_wizard_keeps_keys_it_does_not_ask_about(home, monkeypatch):
    write_yaml(home, {**DEFAULT_CONFIG, "whisper_prompt": "Acme", "tui_theme": "nord",
                      "transcription_backend": "api"})
    monkeypatch.setattr("builtins.input", lambda *_: "")      # Enter on every prompt
    main.cmd_setup(argparse.Namespace(reset_personas=False))
    stored = yaml.safe_load(config_path().read_text())
    assert stored["whisper_prompt"] == "Acme"
    assert stored["tui_theme"] == "nord"
    assert stored["transcription_backend"] == "api"

def test_importing_config_does_not_import_the_cli_or_the_diarizer():
    # subprocess: import tinysteno.config; assert "tinysteno.main" and "sherpa_onnx"
    # are not in sys.modules
```

- [ ] **Step 2: Run** `uv run pytest tests/test_config.py -q` — Expected: FAIL (module does not exist).

- [ ] **Step 3: Implement** `tinysteno/config.py` with the interface above, moving the code out of `main.py` unchanged where it exists. In `cmd_setup`, replace the final `config = {...}; _write_config(...)` with `update_config({...same keys...})`, passing `recordings_path: None` when the answer is blank.

- [ ] **Step 4: Run** `uv run pytest -q -W error` and `uv run pylint $(git ls-files '*.py')` — Expected: all pass (existing tests that patch `tinysteno.main.load_config` still do), 10.00/10.

- [ ] **Step 5: Commit** — `refactor: move configuration into tinysteno/config.py; saves keep unknown keys`

---

### Task 3: `tinysteno/settings.py` — the editable settings, described once

**Files:**
- Create: `tinysteno/settings.py`
- Test: `tests/test_settings.py` (new)

**Interfaces:**
- Consumes: `DEFAULT_CONFIG`, `validate_config` from Task 2.
- Produces:

```python
@dataclass(frozen=True)
class Setting:
    key: str
    label: str
    kind: str               # "text" | "path" | "secret" | "int" | "number" | "bool" | "choice"
    section: str
    help: str = ""
    choices: tuple[str, ...] = ()
    optional: bool = False  # blank is allowed and removes the key from the file
    essential: bool = False # shown in the first-run guide

SECTIONS: tuple[str, ...]            # display order
SETTINGS: tuple[Setting, ...]
def to_text(setting: Setting, value: object) -> str
def from_text(setting: Setting, text: str) -> object   # raises ValueError(f"{label}: {reason}")
def collect(values: dict[str, object]) -> dict         # {key: value or None-for-blank-optional}
```

The settings, in order (`*` = essential; choice lists in brackets):

| Section | key | kind | notes |
|---|---|---|---|
| Notes | `obsidian_vault`* | path | required |
| Notes | `output_folder`* | text | required |
| Notes | `recordings_path` | path | optional; help names the default (`<vault>/<folder>/audio`) |
| Notes | `persona`* | choice | choices filled at runtime from `list_personas`; see `SettingsForm` |
| Notes | `auto_title`, `auto_tags`, `transcript_timestamps` | bool | |
| Summary (LLM) | `base_url`*, `model`* | text | required |
| Summary (LLM) | `api_key`* | secret | required |
| Summary (LLM) | `llm_context_tokens` | int | > 0 |
| Transcription | `transcription_backend`* | choice | [`local`, `api`] |
| Transcription | `whisper_model` | choice | [`tiny`, `base`, `small`, `medium`, `large-v3`]; an existing value outside the list is kept and offered |
| Transcription | `whisper_device` | choice | [`auto`, `cpu`, `cuda`] |
| Transcription | `whisper_compute_type` | text | |
| Transcription | `whisper_vad` | bool | |
| Transcription | `whisper_prompt` | text | optional |
| Transcription | `whisper_base_url` | text | optional |
| Transcription | `whisper_api_key` | secret | optional |
| Transcription | `whisper_api_model` | text | |
| Transcription | `whisper_api_timeout` | number | > 0 |
| Speakers | `diarization`, `hallucination_filter`, `bleed_filter`, `speaker_split`, `whisper_api_diarize` | bool | |
| Speakers | `max_speakers` | int | optional; ≥ 1 |
| Recording | `session_timeout_minutes` | number | ≥ 0; help: "0 disables" |
| Recording | `sample_rate` | int | > 0 |
| Recording | `channels` | choice | [`1`, `2`], stored as int |
| Appearance | `tui_theme` | text | optional |

- [ ] **Step 1: Write the failing tests**

```python
def test_every_default_key_is_editable():
    assert set(DEFAULT_CONFIG) <= {s.key for s in SETTINGS}

def test_keys_are_unique_and_sections_known():
    keys = [s.key for s in SETTINGS]
    assert len(keys) == len(set(keys))
    assert {s.section for s in SETTINGS} == set(SECTIONS)

@pytest.mark.parametrize("key,text,expected", [
    ("session_timeout_minutes", "90", 90), ("session_timeout_minutes", "7.5", 7.5),
    ("session_timeout_minutes", "0", 0), ("channels", "2", 2),
    ("llm_context_tokens", "8192", 8192), ("max_speakers", "", None),
    ("recordings_path", "  ", None), ("whisper_api_timeout", "600", 600),
])
def test_from_text(key, text, expected): ...

@pytest.mark.parametrize("key,text", [
    ("session_timeout_minutes", "-1"), ("session_timeout_minutes", "soon"),
    ("session_timeout_minutes", "inf"), ("llm_context_tokens", "0"),
    ("llm_context_tokens", "8k"), ("max_speakers", "0"), ("sample_rate", "44.1"),
    ("obsidian_vault", ""), ("api_key", " "), ("whisper_api_timeout", "0"),
])
def test_from_text_rejects_and_names_the_field(key, text):
    setting = by_key(key)
    with pytest.raises(ValueError, match=re.escape(setting.label)):
        from_text(setting, text)

def test_round_trip_for_every_default():
    for s in SETTINGS:
        default = DEFAULT_CONFIG.get(s.key)
        if s.kind == "bool" or default is None:
            continue
        if s.optional and default == "":     # a blank optional reads back as None (key removed)
            assert from_text(s, to_text(s, default)) is None
        else:
            assert from_text(s, to_text(s, default)) == default

def test_whole_numbers_stay_ints():
    assert isinstance(from_text(by_key("session_timeout_minutes"), "75"), int)

def test_collected_defaults_pass_validation():
    validate_config(collect({s.key: DEFAULT_CONFIG[s.key] for s in SETTINGS
                             if s.key in DEFAULT_CONFIG}))
```

- [ ] **Step 2: Run** — Expected: FAIL (module does not exist).
- [ ] **Step 3: Implement** `tinysteno/settings.py`. `from_text` strips whitespace; a blank optional gives `None`; `path` values are stored as typed (no `expanduser`; `load_config` expands).
- [ ] **Step 4: Run** `uv run pytest tests/test_settings.py -q -W error` — Expected: PASS.
- [ ] **Step 5: Commit** — `feat: describe the editable settings in one place`

---

### Task 4: `tinysteno/doctor.py` and `tinysteno/recordings.py`

**Files:**
- Create: `tinysteno/doctor.py`, `tinysteno/recordings.py`
- Modify: `tinysteno/main.py` (`cmd_test` prints `run_checks`; same lines for the checks it has today, same exit codes)
- Test: `tests/test_doctor.py`, `tests/test_recordings.py` (new)

**Interfaces:**
- Produces:

```python
# doctor.py
@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str = ""          # one line; the fix when not ok
    required: bool = True     # a failed optional check is a warning, not a failure

def run_checks(config: dict | None) -> list[Check]
```

Checks, in order: the seven module imports `cmd_test` makes today (same names); `Config file` (exists); then, when `config` is given: `Vault folder` (`obsidian_vault` is an existing directory; detail names the path), `Recordings folder` (exists or its parent is writable), `Default persona` (`load_persona(config["persona"])` succeeds; detail is the error), `Speaker separation` (`required=False`, only when `speaker_split` is on: `importlib.util.find_spec("sherpa_onnx")` is not None — never import it).

```python
# recordings.py
@dataclass(frozen=True)
class RecordingInfo:
    path: Path
    size_bytes: int
    modified: datetime
    has_transcript: bool      # <wav>.transcript.json exists

def list_recordings(directory: Path) -> list[RecordingInfo]      # newest first; [] if missing
def delete_recording(directory: Path, wav: Path) -> list[Path]   # returns what was removed
```

`delete_recording` raises `ValueError` unless `wav` has suffix `.wav`, is not a symlink, and `wav.parent.resolve() == directory.resolve()`; it removes the WAV, `<wav>.transcript.json` and `<wav>.levels.json` when present (sidecars that are symlinks are unlinked, never followed), and nothing else.

- [ ] **Step 1: Write the failing tests**

```python
def test_checks_report_a_missing_vault(tmp_path): ...       # ok False, detail contains the path
def test_broken_default_persona_is_reported(...): ...
def test_speaker_check_only_when_split_is_on_and_never_imports(monkeypatch): ...
    # after run_checks({... "speaker_split": True}), "sherpa_onnx" not in sys.modules
def test_cmd_test_exit_code_and_output_unchanged(capsys, home): ...   # 0 with config, 1 without

def test_list_is_newest_first_with_transcript_flag(tmp_path): ...
def test_delete_removes_wav_and_its_sidecars_only(tmp_path):
    # a.wav, a.wav.transcript.json, a.wav.levels.json, b.wav, a.md -> only the first three go
def test_delete_refuses_outside_the_directory(tmp_path): ...     # ValueError, file still there
def test_delete_refuses_a_symlink_and_a_non_wav(tmp_path): ...
def test_delete_does_not_follow_a_symlinked_sidecar(tmp_path): ... # target file survives
def test_delete_of_a_missing_file_raises_filenotfound(tmp_path): ...
```

- [ ] **Step 2: Run** — Expected: FAIL.
- [ ] **Step 3: Implement** both modules; `cmd_test` loads the config if the file exists (catching `ValueError` into a failed `Config file` check) and prints `✓ name` / `✗ name: detail` / `! name: detail` (optional), then the existing summary lines.
- [ ] **Step 4: Run** `uv run pytest -q -W error`, pylint — Expected: pass, 10.00/10.
- [ ] **Step 5: Commit** — `refactor: setup checks and recording file operations as UI-free modules`

---

### Task 5: `TuiDeps` and `cmd_tui` wiring

**Files:**
- Modify: `tinysteno/tui/deps.py`, `tinysteno/main.py` (`cmd_tui`, `main`), `tinysteno/tui/processing.py`, `tinysteno/tui/app.py`, `tests/tui_fakes.py`
- Test: `tests/test_tui_wiring.py` (new), `tests/test_tui_processing.py`

**Interfaces:**
- Consumes: Tasks 2–4.
- Produces — new `TuiDeps` fields (all with defaults so existing constructions keep working):

```python
load_settings: Callable[[], dict] = dict                 # stored file contents (read_stored)
save_settings: Callable[[dict], None] = ...              # changes -> written; raises ValueError/OSError
run_checks: Callable[[], list[Check]] = list
seed_personas: Callable[[], dict[str, list[str]]] = ...  # restore built-ins (force=True)
delete_recording: Callable[[Path], list[Path]] = ...
first_run: bool = False
```

and `run_pipeline` gains a keyword: `run_pipeline(wav_path, name, persona, timestamp, reporter, *, retranscribe=False)`. `ProcessingScreen.__init__` gains `retranscribe: bool = False`, passed on its first run only (a Retry after a failure passes `False`); `TinyStenoApp.process_file(wav_path, persona_slug, *, retranscribe=False)`.

`save_settings` in `cmd_tui`: `update_config(changes)`, then rebuild the effective config **in place** (`config.clear(); config.update(load_config())`, then re-apply the CLI overrides from `args`), then `deps.recordings_dir = recordings_dir(config)`. Because the recorder factory and the pipeline closure read that same dict, the next recording and the next processing run use the new values. `main()` records whether `config_path()` existed before `load_config()` and passes it to `cmd_tui` as `first_run`.

- [ ] **Step 1: Write the failing tests** (`cmd_tui` is exercised with `run_tui` patched to capture `deps`; `AudioRecorder` patched)

```python
def test_save_updates_the_live_config_and_the_file(home, captured): ...
    # captured.deps.save_settings({"model": "new"}); config["model"] == "new";
    # yaml on disk has model: new

def test_save_does_not_persist_cli_overrides(home):
    # args: backend="api", whisper_api_key="sk-secret"
    # deps.save_settings({"model": "new"})
    # file: transcription_backend == "local", "sk-secret" not in file text
    # live config: transcription_backend == "api", whisper_api_key == "sk-secret"

def test_save_moves_the_recordings_dir(home): ...
    # save {"recordings_path": str(tmp/"rec")} -> deps.recordings_dir == tmp/"rec"
    # and deps.make_recorder() was built with that directory

def test_invalid_save_changes_nothing_in_memory(home): ...
    # save {"channels": 5} raises ValueError; config dict equals its copy from before

def test_first_run_flag(home): ...       # no config file -> deps.first_run True; second launch False

async def test_retranscribe_reaches_the_pipeline_once(tmp_path): ...
    # process_file(wav, slug, retranscribe=True): fake pipeline records retranscribe=True;
    # make it fail, press retry: second call has retranscribe=False
```

- [ ] **Step 2: Run** — Expected: FAIL.
- [ ] **Step 3: Implement.** `tests/tui_fakes.make_deps` gains keyword arguments for the new fields with in-memory fakes (a dict-backed `load_settings`/`save_settings` that records calls and can be given an error to raise, a `checks` list, a `seed_result`, a `delete_recording` that really deletes inside `tmp_path` via `tinysteno.recordings.delete_recording`).
- [ ] **Step 4: Run** `uv run pytest -q -W error`, pylint — Expected: pass, 10.00/10.
- [ ] **Step 5: Commit** — `feat(tui): dependencies for settings, checks, personas and recordings`

---

### Task 6: Menu and Settings screen

**Files:**
- Create: `tinysteno/tui/menu.py`, `tinysteno/tui/forms.py`, `tinysteno/tui/settings.py`, `tinysteno/tui/confirm.py`
- Modify: `tinysteno/tui/app.py`, `tinysteno/tui/home.py`, `tinysteno/tui/styles.tcss`
- Test: `tests/test_tui_menu.py`, `tests/test_tui_settings.py` (new)

**Interfaces:**
- Consumes: `SETTINGS`, `SECTIONS`, `to_text`, `from_text`, `collect` (Task 3); `deps.load_settings`, `deps.save_settings`, `deps.list_personas` (Task 5).
- Produces:
  - `MenuScreen(ModalScreen[str | None])` — an `OptionList#menu` with option ids `settings`, `setup`, `personas`, `recordings`, `checks`, labels `Settings`, `Setup guide`, `Personas`, `Recordings`, `System check`; `escape` dismisses with `None`.
  - `TinyStenoApp.action_open_menu()`, `open_destination(dest: str)`; `get_system_commands` also yields one command per destination (titles as above).
  - Home binding `("m", "app.open_menu", "Menu")`. The Home hint `NO_PERSONAS` becomes `"No personas found. Open the menu (m) and choose Personas."`
  - `SettingsForm(settings: Sequence[Setting], values: dict, personas: list[str])` — a widget; widget ids are `f"set-{key}"`; `Input` for text/path/int/number, `Input(password=True)` for secret, `Switch` for bool, `Select` for choice. `read() -> dict` returns `{key: value-or-None}` or raises `FormError(key, message)`.
  - `ConfirmScreen(ModalScreen[bool])(question: str, confirm_label: str)` — buttons `#confirm`, `#cancel`; `escape` = cancel; focus starts on Cancel.
  - `SettingsScreen(Screen)` — bindings `ctrl+s` Save, `escape` Back. One collapsible/tab per section in `SECTIONS` order.

Behaviour of `SettingsScreen`:
- On mount, `deps.load_settings()`; a key absent from the file shows its `DEFAULT_CONFIG` value. If loading raises `ValueError`, the screen shows only the message `Config file cannot be read: {reason}` and the path, with Save disabled.
- Save: `form.read()`; on `FormError` show the message in `#form-error` and focus `#set-{key}`; otherwise send **only the keys whose value differs from what was loaded** to `deps.save_settings`. `ValueError` → message in `#form-error`; `OSError` → `Could not save settings: {reason}`. On success: notify `Settings saved`, pop the screen. Nothing changed → pop without calling `save_settings`.
- Back with unsaved changes pushes `ConfirmScreen("Discard unsaved changes?", "Discard")`.
- If any of the saved keys is fixed by a CLI flag for this session (the app is given `overridden_keys: frozenset[str]` by `run_tui`; add that parameter, default empty), the field shows the suffix `(overridden by a command-line flag this session)`.
- The `persona` choice lists `deps.list_personas()`; a stored slug that no longer exists is shown as an extra option labelled `{slug} (missing)`.
- Secrets are never included in any notification, error text or `app.events` entry.
- Home, on resume, re-reads `app.config["persona"]` only if the user has not picked a persona in this session, re-reads the recordings list from `deps.recordings_dir`, and applies `tui_theme` if it changed.

- [ ] **Step 1: Write the failing tests**

```python
async def test_m_opens_the_menu_and_escape_closes_it(tmp_path): ...
async def test_menu_does_not_open_while_typing_in_the_name_field(tmp_path): ...
async def test_palette_lists_the_destinations(tmp_path): ...
    # titles include Settings, Setup guide, Personas, Recordings, System check
async def test_menu_is_not_reachable_from_recording_or_processing(tmp_path): ...
    # action_open_menu on those screens does nothing (check via check_action / no MenuScreen)

async def test_fields_show_stored_values_and_defaults(tmp_path): ...
async def test_save_sends_only_changed_keys(tmp_path):
    # change model -> deps.saved == [{"model": "gpt-4o-mini"}]
async def test_blank_optional_field_removes_the_key(tmp_path):
    # stored recordings_path "/x"; clear it; saved == [{"recordings_path": None}]
async def test_invalid_number_is_refused_and_focused(tmp_path):
    # session_timeout_minutes = "soon": saved == []; #form-error names "Session timeout";
    # app.focused.id == "set-session_timeout_minutes"
async def test_validation_error_from_save_is_shown_and_screen_stays(tmp_path): ...
async def test_write_failure_is_shown(tmp_path): ...      # OSError("read-only") -> message
async def test_malformed_file_disables_save(tmp_path): ...
async def test_api_key_is_masked_and_never_in_notifications(tmp_path): ...
async def test_escape_with_changes_asks_and_cancel_keeps_them(tmp_path): ...
async def test_escape_without_changes_leaves_at_once(tmp_path): ...
async def test_missing_persona_is_shown_as_missing(tmp_path): ...
async def test_overridden_key_is_marked(tmp_path): ...
async def test_home_uses_new_default_persona_after_save(tmp_path): ...
async def test_settings_fit_80x24(tmp_path): ...  # run_test(size=(80, 24)); Save button reachable
```

- [ ] **Step 2: Run** — Expected: FAIL.
- [ ] **Step 3: Implement** the four files and the app/home changes. `open_destination` only acts when the current screen is `HomeScreen` (or a menu/management screen above it). Destinations whose screens arrive in later tasks (`setup`, `personas`, `recordings`, `checks`) are wired in those tasks; until then `open_destination` ignores them and the menu lists only the destinations that exist (drive the menu's options from a dict on the app, so each later task adds one entry).
- [ ] **Step 4: Run** `uv run pytest -q -W error`, pylint — Expected: pass, 10.00/10.
- [ ] **Step 5: Commit** — `feat(tui): menu and settings screen`

---

### Task 7: System check screen and first-run Setup guide

**Files:**
- Create: `tinysteno/tui/checks.py`, `tinysteno/tui/setup.py`
- Modify: `tinysteno/tui/app.py`, `tinysteno/tui/styles.tcss`
- Test: `tests/test_tui_checks.py`, `tests/test_tui_setup.py` (new)

**Interfaces:**
- Consumes: `SettingsForm`, `ConfirmScreen` (Task 6); `deps.run_checks`, `deps.first_run`, `deps.seed_personas`, `deps.save_settings`.
- Produces:
  - `ChecksScreen(Screen)` — runs `deps.run_checks()` in a thread worker on mount and on `r` (Run again); one row per check: `✓`, `✗` (required failed) or `!` (optional failed), the name, the detail; a summary line `All checks passed` / `{n} problem(s) found`; `escape` Back.
  - `SetupScreen(Screen)` — the settings with `essential=True` in one form, a short intro line, buttons `#finish` (`Save and continue`) and `#skip` (`Skip for now`). Finish saves (same validation and error display as Settings, and it sends every essential key, changed or not, so a first-run file holds the user's answers), then replaces itself with `ChecksScreen`. Skip and `escape` return to Home with no write.
  - On mount of the app: when `deps.first_run` is true and `autostart` is false, `SetupScreen` is pushed over Home. With `--now`, recording starts and the guide is not shown.

- [ ] **Step 1: Write the failing tests**

```python
async def test_checks_are_listed_with_marks_and_summary(tmp_path): ...
    # checks: ok, required-failed, optional-failed -> "✓", "✗", "!", "1 problem(s) found"
async def test_run_again_reruns(tmp_path): ...
async def test_a_check_that_raises_is_shown_as_a_failure_not_a_crash(tmp_path): ...

async def test_first_run_opens_the_guide_over_home(tmp_path): ...
async def test_not_first_run_goes_straight_to_home(tmp_path): ...
async def test_now_skips_the_guide(tmp_path): ...
async def test_guide_shows_only_essential_settings(tmp_path): ...
async def test_finish_saves_every_essential_key_then_shows_checks(tmp_path): ...
async def test_skip_writes_nothing_and_shows_home(tmp_path): ...
async def test_invalid_vault_is_refused(tmp_path): ...    # blank vault: nothing saved, field named
async def test_guide_can_be_reopened_from_the_menu(tmp_path): ...
```

- [ ] **Step 2: Run** — Expected: FAIL.
- [ ] **Step 3: Implement.** Add `checks` and `setup` to the app's destinations.
- [ ] **Step 4: Run** `uv run pytest -q -W error`, pylint — Expected: pass, 10.00/10.
- [ ] **Step 5: Commit** — `feat(tui): first-run setup guide and system check`

---

### Task 8: Personas screen

**Files:**
- Create: `tinysteno/tui/personas.py`
- Modify: `tinysteno/tui/app.py`, `tinysteno/tui/styles.tcss`
- Test: `tests/test_tui_personas.py` (new)

**Interfaces:**
- Consumes: `deps.list_personas`, `deps.load_persona`, `deps.seed_personas`, `deps.save_settings`, `ConfirmScreen`.
- Produces: `PersonasScreen(Screen)` — a `DataTable#personas` with columns `Persona`, `Name`, `Auto-stop`, `Status`; bindings `d` Set as default, `b` Restore built-ins, `escape` Back.

Behaviour: one row per slug from `list_personas()`; `load_persona(slug)` failing with `PersonaInvalidError`/`PersonaNotFoundError` gives status `Invalid: {reason}` (one line, the row stays selectable); the current default is marked `default`. Auto-stop shows the persona's own `session_timeout_minutes` (`{n} min`, `off` for 0) or `global` when it has none. `d` on a valid persona calls `save_settings({"persona": slug})` and refreshes; on an invalid one it notifies `Cannot use an invalid persona as the default` and saves nothing. `b` asks `ConfirmScreen("Restore the built-in personas? Your changes to them will be overwritten; personas you added are kept.", "Restore")`; on yes, `seed_personas()` in a thread worker, then notify `Restored: {comma list}` and refresh; an exception notifies `Could not restore personas: {reason}`. An empty list shows `No personas found. Press b to restore the built-in ones.`

- [ ] **Step 1: Write the failing tests**

```python
async def test_rows_show_validity_default_and_timeout(tmp_path): ...
async def test_d_sets_the_default_and_marks_it(tmp_path): ...     # saved == [{"persona": "rca"}]
async def test_d_on_an_invalid_persona_saves_nothing(tmp_path): ...
async def test_restore_asks_first_and_cancel_does_nothing(tmp_path): ...
async def test_restore_runs_and_reports(tmp_path): ...
async def test_restore_failure_is_reported(tmp_path): ...
async def test_empty_list_offers_restore_and_home_enables_record_afterwards(tmp_path): ...
async def test_save_failure_is_reported_and_default_unchanged(tmp_path): ...
```

- [ ] **Step 2: Run** — Expected: FAIL.
- [ ] **Step 3: Implement**; add `personas` to the destinations. Home, on resume, rebuilds its persona `Select` when `list_personas()` changed.
- [ ] **Step 4: Run** `uv run pytest -q -W error`, pylint — Expected: pass, 10.00/10.
- [ ] **Step 5: Commit** — `feat(tui): personas screen`

---

### Task 9: Recordings screen

**Files:**
- Create: `tinysteno/tui/recordings.py`
- Modify: `tinysteno/tui/app.py`, `tinysteno/tui/home.py`, `tinysteno/tui/styles.tcss`
- Test: `tests/test_tui_recordings.py` (new)

**Interfaces:**
- Consumes: `list_recordings`, `RecordingInfo` (Task 4); `deps.delete_recording`, `deps.recordings_dir`; `app.process_file(..., retranscribe=)` (Task 5); `ConfirmScreen`.
- Produces: `RecordingsScreen(Screen)` — `DataTable#recordings` with columns `Recording`, `Date`, `Size`, `Transcript`; bindings `enter` Process, `t` Transcribe again, `d` Delete, `escape` Back. A `Select#persona` at the top chooses the persona used for processing (initial value: the configured default).

Behaviour: lists every WAV in `deps.recordings_dir`, newest first; `Transcript` shows `saved` or `—`; size as `12.3 MB`; an empty directory shows `No recordings yet` and the folder path. `enter` → `app.process_file(path, slug)`; `t` → `app.process_file(path, slug, retranscribe=True)` (on a row without a saved transcript it behaves as `enter`). `d` asks `ConfirmScreen(f"Delete {name}? The audio and its saved transcript are removed. Notes in your vault are kept.", "Delete")`; on yes `deps.delete_recording(path)`; `ValueError`/`OSError` notifies `Could not delete {name}: {reason}`; success notifies `Deleted {name}` and refreshes, keeping the cursor on the next row. Delete is refused with `{name} is being processed` when the app ledger shows that WAV as the one handed to a pipeline that has not finished (add `SessionLedger.is_busy(path) -> bool` if the ledger cannot answer that today). Returning from Processing/Result refreshes the table. Home's "Recent recordings" label gains the hint `(m → Recordings for all)`.

- [ ] **Step 1: Write the failing tests**

```python
async def test_all_recordings_are_listed_newest_first(tmp_path): ...   # 12 files, all 12 shown
async def test_enter_processes_with_the_chosen_persona(tmp_path): ...
async def test_t_requests_a_new_transcription(tmp_path): ...
    # events contain ("process", wav, slug); fake pipeline saw retranscribe=True
async def test_delete_asks_and_cancel_keeps_the_file(tmp_path): ...
async def test_delete_removes_audio_and_sidecars_and_keeps_others(tmp_path): ...
async def test_delete_failure_is_reported_and_row_stays(tmp_path): ...
async def test_cannot_delete_the_recording_being_processed(tmp_path): ...
async def test_file_removed_outside_the_app_is_reported_not_a_crash(tmp_path): ...
async def test_empty_folder_message_names_the_folder(tmp_path): ...
async def test_no_personas_disables_processing_but_not_delete(tmp_path): ...
```

- [ ] **Step 2: Run** — Expected: FAIL.
- [ ] **Step 3: Implement**; add `recordings` to the destinations.
- [ ] **Step 4: Run** `uv run pytest -q -W error`, pylint — Expected: pass, 10.00/10.
- [ ] **Step 5: Commit** — `feat(tui): recordings screen with re-transcribe and delete`

---

### Task 10: Documentation and screenshots

**Files:**
- Modify: `README.md`, `SPEC.md`, `CLAUDE.md`, `TODO.md`, `scripts/tui_screenshots.py`, `docs/images/` (regenerated)

- [ ] **Step 1: Extend `scripts/tui_screenshots.py`** to also write `tui-menu.svg`, `tui-settings.svg`, `tui-personas.svg` and `tui-recordings.svg` (stand-in deps; a sample API key must be a placeholder and is masked anyway). Run `uv run python scripts/tui_screenshots.py` — Expected: ten `tui-*.svg` files listed, none containing a real path under the user's home.

- [ ] **Step 2: README.** In the TUI section: the Menu (`m`, and the command palette), the first-run guide, Settings (what saving does: keeps keys it does not show, rewrites the file so comments in `config.yaml` are not preserved, does not write values given as command-line flags, takes effect for the next recording and the next processing run), Personas, Recordings (what Delete removes and what it keeps), System check; the four new images. Remove any statement that the command palette can save a screenshot; keep the sentence about `scripts/tui_screenshots.py`. In the configuration section: `tinysteno setup` now keeps settings it does not ask about; a new config file is created with mode `600`.

- [ ] **Step 3: SPEC.md** — entries for `config.py`, `settings.py`, `doctor.py`, `recordings.py` and the new TUI screens; the `TuiDeps` field list. **CLAUDE.md** — in Notes: config lives in `tinysteno/config.py`; a new editable setting is added to `SETTINGS` in `tinysteno/settings.py` (and to `DEFAULT_CONFIG` when it has a default); TUI screens reach the filesystem only through `TuiDeps`. **TODO.md** — tick or remove items this plan completes; add unticked: edit personas in the TUI, test the LLM/transcription connection from Setup, download speaker-separation models from the menu.

- [ ] **Step 4: Verify** every new passage against the code; `uv run pytest -q -W error`; `uv run pylint $(git ls-files '*.py')`; `wc -l tinysteno/main.py` is below 958.

- [ ] **Step 5: Commit** — `docs: TUI menu, settings, personas, recordings and system check`
