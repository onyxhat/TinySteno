# Persona Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate all persona configs (built-ins and user-defined) to `~/.tinysteno/personas/`, seeding built-ins on first run and via `tinysteno setup`.

**Architecture:** Add a `seed_builtin_personas()` function to `tinysteno/personas/__init__.py` that copies built-in dirs to `~/.tinysteno/personas/`. Update `load_persona` and `list_personas` to look only in `~/.tinysteno/personas/`. Wire the seeding into `load_config()` (silent, first-run) and `cmd_setup()` (interactive + `--reset-personas` flag).

**Tech Stack:** Python 3.12+, `shutil.copytree`, `pytest` with `monkeypatch` / `unittest.mock.patch`, `rich` for CLI output.

**Spec:** `docs/superpowers/specs/2026-03-20-persona-consolidation-design.md`

---

## File Map

| File | Change |
|---|---|
| `tinysteno/personas/__init__.py` | Add `seed_builtin_personas`; update `list_personas` and `load_persona` |
| `tinysteno/main.py` | Add seeding to `load_config()`; add `--reset-personas` flag and seeding to `cmd_setup()`; import `seed_builtin_personas` |
| `tests/conftest.py` | Add `seeded_user_dir` fixture |
| `tests/test_personas.py` | Add seeding tests; update broken tests |
| `tests/test_main.py` | Create; tests for `load_config()` seeding trigger and `cmd_setup()` changes |

---

## Task 1: Add `seed_builtin_personas` to `personas/__init__.py`

**Files:**
- Modify: `tinysteno/personas/__init__.py`
- Test: `tests/test_personas.py`

### Background

`_BUILTIN_DIR = Path(__file__).parent` already points to `tinysteno/personas/` in the package — the built-in dirs are siblings of `__init__.py`. We're adding a new exported function that copies them to `_USER_DIR`. The existing code and tests are untouched in this task.

- [ ] **Step 1: Write the failing tests**

First, update the **top-of-file imports** in `tests/test_personas.py`:
- Add `import shutil` to the stdlib imports block at the top (alongside any existing stdlib imports)
- Add `seed_builtin_personas, _BUILTIN_DIR` to the existing `from tinysteno.personas import (...)` block

Then add the following test functions to the bottom of `tests/test_personas.py` (before the `# --- helpers ---` section). Do **not** add the import lines again — they're already at the top.

```python
# --- seed_builtin_personas ---


def test_seed_copies_all_builtins_to_empty_dir(tmp_path):
    personas_dir = tmp_path / "personas"
    with patch("tinysteno.personas._USER_DIR", personas_dir):
        result = seed_builtin_personas()
    assert set(result["copied"]) == set(BUILTIN_ORDER)
    assert result["skipped"] == []
    for slug in BUILTIN_ORDER:
        assert (personas_dir / slug / "persona.yaml").exists()


def test_seed_skips_existing_when_not_forced(tmp_path):
    personas_dir = tmp_path / "personas"
    _shutil.copytree(_BUILTIN_DIR / "default", personas_dir / "default")
    with patch("tinysteno.personas._USER_DIR", personas_dir):
        result = seed_builtin_personas()
    assert "default" in result["skipped"]
    assert "default" not in result["copied"]


def test_seed_overwrites_when_forced(tmp_path):
    personas_dir = tmp_path / "personas"
    _shutil.copytree(_BUILTIN_DIR / "default", personas_dir / "default")
    with patch("tinysteno.personas._USER_DIR", personas_dir):
        result = seed_builtin_personas(force=True)
    assert "default" in result["copied"]
    assert "default" not in result["skipped"]


def test_seed_interactive_overwrites_on_yes(tmp_path, monkeypatch):
    personas_dir = tmp_path / "personas"
    _shutil.copytree(_BUILTIN_DIR / "default", personas_dir / "default")
    monkeypatch.setattr("builtins.input", lambda _: "y")
    with patch("tinysteno.personas._USER_DIR", personas_dir):
        result = seed_builtin_personas(interactive=True)
    assert "default" in result["copied"]
    assert "default" not in result["skipped"]


def test_seed_interactive_skips_on_no(tmp_path, monkeypatch):
    personas_dir = tmp_path / "personas"
    _shutil.copytree(_BUILTIN_DIR / "default", personas_dir / "default")
    monkeypatch.setattr("builtins.input", lambda _: "n")
    with patch("tinysteno.personas._USER_DIR", personas_dir):
        result = seed_builtin_personas(interactive=True)
    assert "default" in result["skipped"]
    assert "default" not in result["copied"]


def test_seed_warns_and_skips_missing_source(tmp_path, caplog):
    import logging
    fake_builtin = tmp_path / "fake_builtin"
    fake_builtin.mkdir()
    for slug in BUILTIN_ORDER[1:]:  # all except "default"
        _shutil.copytree(_BUILTIN_DIR / slug, fake_builtin / slug)
    personas_dir = tmp_path / "personas"
    with patch("tinysteno.personas._BUILTIN_DIR", fake_builtin):
        with patch("tinysteno.personas._USER_DIR", personas_dir):
            with caplog.at_level(logging.WARNING, logger="tinysteno.personas"):
                result = seed_builtin_personas()
    assert "default" not in result["copied"]
    assert "default" not in result["skipped"]
    assert any("default" in r.message for r in caplog.records)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_personas.py::test_seed_copies_all_builtins_to_empty_dir -v
```

Expected: `ImportError: cannot import name 'seed_builtin_personas'`

- [ ] **Step 3: Add `import shutil` and implement `seed_builtin_personas`**

At the top of `tinysteno/personas/__init__.py`, add `import shutil` after the existing stdlib imports.

Then add this function after `_load_from_dir` and before `load_persona`:

```python
def seed_builtin_personas(interactive: bool = False, force: bool = False) -> dict:
    """Copy built-in persona dirs to _USER_DIR.

    Returns {"copied": [...], "skipped": [...]} for callers to display or discard.
    """
    _USER_DIR.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    skipped: list[str] = []

    for slug in BUILTIN_ORDER:
        src = _BUILTIN_DIR / slug
        dst = _USER_DIR / slug

        if not src.exists():
            logger.warning(f"Built-in persona source missing: {src}")
            continue

        if dst.exists():
            if force:
                shutil.copytree(src, dst, dirs_exist_ok=True)
                copied.append(slug)
            elif interactive:
                answer = input(
                    f"  Built-in persona '{slug}' already exists. Overwrite? [y/N] "
                ).strip().lower()
                if answer in ("y", "yes"):
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                    copied.append(slug)
                else:
                    skipped.append(slug)
            else:
                skipped.append(slug)
        else:
            shutil.copytree(src, dst)
            copied.append(slug)

    return {"copied": copied, "skipped": skipped}
```

Also add `seed_builtin_personas` to the imports list in the existing `from tinysteno.personas import (...)` block — but that's in `main.py`. For now, make sure it's reachable from the module. No `__all__` exists, so it will be importable automatically.

- [ ] **Step 4: Run all new seeding tests**

```bash
uv run pytest tests/test_personas.py -k "test_seed" -v
```

Expected: all 6 new tests PASS.

- [ ] **Step 5: Run full test suite to confirm no regressions**

```bash
uv run pytest tests/test_personas.py -v
```

Expected: all existing tests PASS (nothing changed in `load_persona`/`list_personas` yet).

- [ ] **Step 6: Commit**

```bash
git add tinysteno/personas/__init__.py tests/test_personas.py
git commit -m "feat: add seed_builtin_personas function"
```

---

## Task 2: Add shared fixture and update `list_personas` to scan only `_USER_DIR`

**Files:**
- Modify: `tests/conftest.py`
- Modify: `tinysteno/personas/__init__.py`
- Modify: `tests/test_personas.py`

### Background

Currently `list_personas` always starts with `result = list(BUILTIN_ORDER)` — it returns all 6 built-in slugs regardless of `_USER_DIR` content. After this task it will only return slugs that exist and are valid inside `_USER_DIR`. Several existing tests call `list_personas()` without patching `_USER_DIR` and will break; we fix them here.

- [ ] **Step 1: Add `seeded_user_dir` fixture to `conftest.py`**

```python
"""Shared pytest fixtures for TinySteno tests."""
import shutil

import pytest

from tinysteno.personas import _BUILTIN_DIR, BUILTIN_ORDER


@pytest.fixture
def seeded_user_dir(tmp_path):
    """A tmp personas dir pre-seeded with all built-in personas."""
    personas_dir = tmp_path / "personas"
    for slug in BUILTIN_ORDER:
        shutil.copytree(_BUILTIN_DIR / slug, personas_dir / slug)
    return personas_dir
```

- [ ] **Step 2: Update failing `list_personas` tests before changing the implementation**

The following tests in `tests/test_personas.py` will break after the implementation change. Update them now so they'll pass once the new implementation lands.

**Replace** `test_list_personas_returns_all_builtins`:
```python
def test_list_personas_returns_all_builtins(seeded_user_dir):
    with patch("tinysteno.personas._USER_DIR", seeded_user_dir):
        slugs = list_personas()
    assert slugs[:len(BUILTIN_ORDER)] == BUILTIN_ORDER
```

**Replace** `test_list_personas_no_duplicates`:
```python
def test_list_personas_no_duplicates(seeded_user_dir):
    with patch("tinysteno.personas._USER_DIR", seeded_user_dir):
        slugs = list_personas()
    assert len(slugs) == len(set(slugs))
```

**Delete** `test_list_personas_malformed_user_with_builtin_slug_keeps_builtin` entirely (remove the whole function). Then **add** this new test in its place:
```python
def test_list_personas_malformed_builtin_slug_is_excluded(tmp_path):
    # Malformed user dir for a built-in slug: no fallback to package, slug excluded
    user_dir = tmp_path / "personas" / "default"
    user_dir.mkdir(parents=True)
    # malformed: no files

    with patch("tinysteno.personas._USER_DIR", tmp_path / "personas"):
        slugs = list_personas()

    assert "default" not in slugs
```

- [ ] **Step 3: Verify old test is gone and run updated tests against old implementation**

First, confirm the deleted test name no longer exists in the file:

```bash
grep -n "malformed_user_with_builtin_slug_keeps_builtin" tests/test_personas.py
```

Expected: no output (the old test is gone).

Then run the updated tests:

```bash
uv run pytest tests/test_personas.py::test_list_personas_returns_all_builtins tests/test_personas.py::test_list_personas_no_duplicates tests/test_personas.py::test_list_personas_malformed_builtin_slug_is_excluded -v
```

Expected:
- `test_list_personas_returns_all_builtins` — PASS (seeded_user_dir patches `_USER_DIR` with all built-ins; old code still returns `BUILTIN_ORDER` from the result prefix)
- `test_list_personas_no_duplicates` — PASS
- `test_list_personas_malformed_builtin_slug_is_excluded` — **FAIL** (old code returns all BUILTIN_ORDER including "default"). This is the test that will pass once we change the implementation.

- [ ] **Step 4: Update `list_personas` in `tinysteno/personas/__init__.py`**

Replace the existing `list_personas` function body entirely:

```python
def list_personas() -> list[str]:
    """Return available persona slugs in canonical order.

    Order: built-ins in BUILTIN_ORDER (only those present and valid in _USER_DIR),
    then user-only personas sorted case-insensitively.
    Malformed persona directories are skipped with a warning.
    """
    if not _USER_DIR.exists():
        return []

    valid_slugs: set[str] = set()
    user_only: list[str] = []

    for entry in sorted(_USER_DIR.iterdir(), key=lambda p: p.name.lower()):
        if not entry.is_dir():
            continue
        slug = entry.name
        try:
            _validate_dir(entry, slug)
        except PersonaInvalidError as e:
            logger.warning(str(e))
            continue
        valid_slugs.add(slug)
        if slug not in BUILTIN_ORDER:
            user_only.append(slug)

    result = [s for s in BUILTIN_ORDER if s in valid_slugs]
    result.extend(user_only)
    return result
```

- [ ] **Step 5: Run the full `list_personas` test set**

```bash
uv run pytest tests/test_personas.py -k "list_personas" -v
```

Expected: all 6 list_personas tests PASS.

- [ ] **Step 6: Run full test suite**

```bash
uv run pytest tests/test_personas.py -v
```

Expected: all tests PASS. (`load_persona` tests that call built-ins directly still pass because `load_persona` still has the `_BUILTIN_DIR` fallback — that's removed in the next task.)

- [ ] **Step 7: Commit**

```bash
git add tests/conftest.py tinysteno/personas/__init__.py tests/test_personas.py
git commit -m "feat: list_personas scans only _USER_DIR; add seeded_user_dir fixture"
```

---

## Task 3: Remove `_BUILTIN_DIR` fallback from `load_persona` and fix tests

**Files:**
- Modify: `tinysteno/personas/__init__.py`
- Modify: `tests/test_personas.py`

### Background

Currently `load_persona` checks `_USER_DIR` first, then `_BUILTIN_DIR`. After this task it only checks `_USER_DIR`. Two existing tests call `load_persona("default")` without patching `_USER_DIR` and will fail — we update them first.

- [ ] **Step 1: Update the three tests that call `load_persona` or `list_personas` without patching `_USER_DIR`**

**Replace** `test_load_persona_default_returns_persona` (add `seeded_user_dir` fixture):

```python
def test_load_persona_default_returns_persona(seeded_user_dir):
    with patch("tinysteno.personas._USER_DIR", seeded_user_dir):
        p = load_persona("default")
    assert isinstance(p, Persona)
    assert p.slug == "default"
    assert p.name
    assert p.description
    assert "overview" in p.schema
    assert p.system_prompt
    assert p.template
```

**Replace** `test_load_persona_unknown_lists_available_in_error` (needs seeded dir so "default" appears in error):

```python
def test_load_persona_unknown_lists_available_in_error(seeded_user_dir):
    with patch("tinysteno.personas._USER_DIR", seeded_user_dir):
        with pytest.raises(PersonaNotFoundError) as exc_info:
            load_persona("no-such-persona")
    assert "default" in str(exc_info.value)
```

**Replace** `test_load_persona_unknown_slug_raises` (make hermetic by patching `_USER_DIR`):

```python
def test_load_persona_unknown_slug_raises(seeded_user_dir):
    with patch("tinysteno.personas._USER_DIR", seeded_user_dir):
        with pytest.raises(PersonaNotFoundError, match="unknown-slug"):
            load_persona("unknown-slug")
```

- [ ] **Step 2: Run updated tests against old implementation to confirm they still pass**

```bash
uv run pytest tests/test_personas.py::test_load_persona_default_returns_persona tests/test_personas.py::test_load_persona_unknown_lists_available_in_error -v
```

Expected: both PASS (old `load_persona` still has the `_BUILTIN_DIR` fallback, seeded_user_dir provides the persona).

- [ ] **Step 3: Remove the `_BUILTIN_DIR` fallback from `load_persona`**

Replace the existing `load_persona` function body:

```python
def load_persona(slug: str) -> Persona:
    """Load persona by slug from _USER_DIR.

    Raises PersonaNotFoundError if no directory for the slug exists in _USER_DIR.
    Raises PersonaInvalidError if the directory exists but is malformed.
    """
    user_path = _USER_DIR / slug

    if user_path.exists() and user_path.is_dir():
        return _load_from_dir(slug, user_path)

    available = list_personas()
    raise PersonaNotFoundError(
        f"Unknown persona '{slug}'. Available: {', '.join(available)}"
    )
```

- [ ] **Step 4: Run the full test suite**

```bash
uv run pytest tests/test_personas.py -v
```

Expected: all tests PASS. Verify specifically:
- `test_load_persona_default_returns_persona` — PASS (seeded_user_dir fixture)
- `test_load_persona_malformed_user_shadows_builtin_raises` — still PASS (user dir has malformed default, no fallback, raises PersonaInvalidError — correct)
- `test_load_persona_user_overrides_builtin` — still PASS (patches _USER_DIR with custom persona)

- [ ] **Step 5: Commit**

```bash
git add tinysteno/personas/__init__.py tests/test_personas.py
git commit -m "feat: load_persona uses _USER_DIR only, remove _BUILTIN_DIR fallback"
```

---

## Task 4: Add seeding trigger to `load_config()` and create `tests/test_main.py`

**Files:**
- Create: `tests/test_main.py`
- Modify: `tinysteno/main.py`

### Background

`load_config()` in `main.py` currently creates `~/.tinysteno/config.yaml` on first run. We add a check: immediately after the `mkdir` for `~/.tinysteno/`, if `~/.tinysteno/personas/` doesn't exist, call `seed_builtin_personas()` silently. This fires only for `record` and `process` commands (the only ones that call `load_config`).

- [ ] **Step 1: Create `tests/test_main.py` with the failing tests**

```python
"""Tests for tinysteno.main module."""
from pathlib import Path
from unittest.mock import patch, call

import pytest
import yaml

from tinysteno.main import load_config


def _write_config(path: Path) -> None:
    """Write a minimal valid config file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "obsidian_vault": str(path.parent),
        "api_key": "ollama",
        "base_url": "http://localhost:11434/v1",
        "model": "llama3.2:3b",
        "whisper_model": "small",
        "diarization": False,
        "auto_title": True,
        "tags": ["meeting"],
        "output_folder": "meetings",
        "sample_rate": 44100,
        "channels": 1,
        "persona": "default",
    }
    path.write_text(yaml.dump(config))


def test_load_config_seeds_personas_when_dir_absent(tmp_path):
    _write_config(tmp_path / ".tinysteno" / "config.yaml")
    # personas dir does NOT exist

    with patch.object(Path, "home", return_value=tmp_path):
        with patch("tinysteno.main.seed_builtin_personas") as mock_seed:
            load_config()

    mock_seed.assert_called_once_with()


def test_load_config_does_not_seed_when_dir_exists(tmp_path):
    _write_config(tmp_path / ".tinysteno" / "config.yaml")
    (tmp_path / ".tinysteno" / "personas").mkdir(parents=True)  # dir exists

    with patch.object(Path, "home", return_value=tmp_path):
        with patch("tinysteno.main.seed_builtin_personas") as mock_seed:
            load_config()

    mock_seed.assert_not_called()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_main.py -v
```

Expected: both tests FAIL — `seed_builtin_personas` not yet imported in `main.py` and no seeding logic in `load_config()`.

- [ ] **Step 3: Import `seed_builtin_personas` in `main.py`**

Find the existing personas import block in `tinysteno/main.py` (line 17) and add `seed_builtin_personas`:

```python
from tinysteno.personas import (  # noqa: E501
    load_persona, list_personas, seed_builtin_personas, PersonaNotFoundError, PersonaInvalidError, Persona,
)
```

- [ ] **Step 4: Add seeding trigger inside `load_config()`**

In `load_config()`, the current `mkdir` call is on line 49:
```python
config_path.parent.mkdir(parents=True, exist_ok=True)
```
This is inside the `if not config_path.exists():` block. Add the seeding check **after** that block (outside the `if`), right before `config = yaml.safe_load(...)`:

```python
    # Seed built-in personas on first run (silent; only when dir absent)
    personas_dir = config_path.parent / "personas"
    if not personas_dir.exists():
        seed_builtin_personas()

    config = yaml.safe_load(config_path.read_text())
```

- [ ] **Step 5: Run the new tests**

```bash
uv run pytest tests/test_main.py -v
```

Expected: both tests PASS.

- [ ] **Step 6: Run full test suite to confirm no regressions**

```bash
uv run pytest -v
```

Expected: all tests PASS.

- [ ] **Step 7: Commit**

```bash
git add tinysteno/main.py tests/test_main.py
git commit -m "feat: seed built-in personas on first run via load_config"
```

---

## Task 5: Add `--reset-personas` flag and seeding to `cmd_setup()`

**Files:**
- Modify: `tinysteno/main.py`
- Modify: `tests/test_main.py`

### Background

`cmd_setup()` needs two changes:
1. When `--reset-personas` is passed: immediately call `seed_builtin_personas(force=True)`, print summary, return (no config wizard).
2. In the normal path: call `seed_builtin_personas(interactive=True)` after `_write_config()`, then print the summary.

The `setup` subparser in `main()` needs a `--reset-personas` flag added.

- [ ] **Step 1: Add the failing tests to `tests/test_main.py`**

```python
import argparse
from tinysteno.main import cmd_setup


def test_cmd_setup_reset_personas_forces_seed_without_wizard(monkeypatch):
    args = argparse.Namespace(reset_personas=True)
    calls = []
    monkeypatch.setattr("builtins.input", lambda _: calls.append("input") or "")

    with patch("tinysteno.main.seed_builtin_personas") as mock_seed:
        mock_seed.return_value = {"copied": ["default"], "skipped": []}
        cmd_setup(args)

    mock_seed.assert_called_once_with(force=True)
    assert calls == [], "input() must not be called when --reset-personas is set"


def test_cmd_setup_normal_seeds_interactively_after_config_write(monkeypatch):
    args = argparse.Namespace(reset_personas=False)
    monkeypatch.setattr("builtins.input", lambda _: "")  # accept all defaults

    with patch("tinysteno.main.seed_builtin_personas") as mock_seed:
        mock_seed.return_value = {"copied": ["default", "rca"], "skipped": []}
        with patch("tinysteno.main._write_config"):
            with patch("tinysteno.main.list_personas", return_value=["default"]):
                cmd_setup(args)

    mock_seed.assert_called_once_with(interactive=True)


def test_setup_argparser_has_reset_personas_flag():
    """Verify the real main() parser wires --reset-personas correctly."""
    import sys
    from io import StringIO
    from tinysteno.main import main

    # Run `tinysteno setup --reset-personas` through the real parser.
    # Patch seed_builtin_personas so no filesystem side-effects occur.
    with patch("tinysteno.main.seed_builtin_personas") as mock_seed:
        mock_seed.return_value = {"copied": [], "skipped": []}
        with patch.object(sys, "argv", ["tinysteno", "setup", "--reset-personas"]):
            main()

    mock_seed.assert_called_once_with(force=True)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_main.py::test_cmd_setup_reset_personas_forces_seed_without_wizard tests/test_main.py::test_cmd_setup_normal_seeds_interactively_after_config_write -v
```

Expected: both FAIL — `cmd_setup` doesn't check `reset_personas` yet and doesn't call `seed_builtin_personas`.

- [ ] **Step 3: Add `--reset-personas` to the `setup` subparser in `main()`**

Find the setup subparser line in `main()` (currently just `subparsers.add_parser("setup", ...)`). Replace it:

```python
setup_parser = subparsers.add_parser("setup", help="Create or update config interactively")
setup_parser.add_argument(
    "--reset-personas",
    action="store_true",
    default=False,
    help="Re-copy all built-in personas to ~/.tinysteno/personas/ (overwrites existing)",
)
```

- [ ] **Step 4: Add `--reset-personas` branch at the top of `cmd_setup()`**

The existing `cmd_setup()` already has `import yaml`, `from rich.console import Console`, `from rich.rule import Rule`, and `console = Console()`. Do **not** re-add those lines.

Insert the following block immediately **after** the existing `console = Console()` line (do not modify anything above or below it):

```python
    # --reset-personas: re-seed built-ins without running the config wizard
    if getattr(_args, "reset_personas", False):
        result = seed_builtin_personas(force=True)
        _print_seed_summary(console, result)
        return
```

- [ ] **Step 5: Add `seed_builtin_personas(interactive=True)` call at the end of the normal `cmd_setup()` path**

After the `_write_config(config_path, config)` call (near the end of `cmd_setup()`), insert the following two lines. Do **not** duplicate the `_write_config` call itself — it already exists:

```python
    # Seed built-in personas, prompting on conflicts
    seed_result = seed_builtin_personas(interactive=True)
    _print_seed_summary(console, seed_result)
```

The existing `console.print()`, `console.print(Rule())`, and success message lines that follow `_write_config` remain unchanged.

- [ ] **Step 6: Add `_print_seed_summary()` helper to `main.py`**

Add this function near the other private helpers (`_prompt`, `_prompt_bool`, `_write_config`):

```python
def _print_seed_summary(console, result: dict) -> None:
    """Print persona seeding summary using rich."""
    user_dir = Path.home() / ".tinysteno" / "personas"
    if result["copied"]:
        console.print(f"[green]✓[/green] Personas seeded in [bold]{user_dir}[/bold]")
        console.print(f"  Copied: {', '.join(result['copied'])}")
    if result["skipped"]:
        console.print(f"  Skipped (kept yours): {', '.join(result['skipped'])}")
```

- [ ] **Step 7: Run all new tests**

```bash
uv run pytest tests/test_main.py -v
```

Expected: all tests PASS.

- [ ] **Step 8: Run full test suite**

```bash
uv run pytest -v
```

Expected: all tests PASS.

- [ ] **Step 9: Commit**

```bash
git add tinysteno/main.py tests/test_main.py
git commit -m "feat: add --reset-personas flag and interactive seeding to cmd_setup"
```

---

## Done

After all 5 tasks:
- `~/.tinysteno/personas/` is the single source of truth for all personas
- Built-ins are seeded silently on first `record`/`process` run
- `tinysteno setup` seeds interactively, prompting on conflicts
- `tinysteno setup --reset-personas` force-resets all built-ins without the config wizard
- All tests pass
