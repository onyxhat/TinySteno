# Persona Consolidation Design

**Date:** 2026-03-20
**Branch:** fix/persona-configs
**Status:** Approved

## Goal

Consolidate all persona configurations (built-ins and user-defined) to `~/.tinysteno/personas/`. Built-in personas are seeded to that location on first run and via `tinysteno setup`. This gives users a single, discoverable place to view and edit all personas.

## Background

Currently, built-in personas live inside the installed package at `tinysteno/personas/<slug>/` and user personas live at `~/.tinysteno/personas/<slug>/`. `load_persona` checks the user dir first, then falls back to the package dir. Users have no visibility into the built-in files without inspecting the package.

## Architecture

### Files changed

**`tinysteno/personas/__init__.py`**
- `_BUILTIN_DIR` remains — points to the package directory, used only as seed source
- `_USER_DIR` (`~/.tinysteno/personas/`) becomes the sole runtime lookup
- `load_persona` removes the `_BUILTIN_DIR` fallback — only checks `_USER_DIR`
- `list_personas` scans only `_USER_DIR` (ordering: built-in slugs in `BUILTIN_ORDER` first if present, then user-only extras alphabetically; only slugs actually on disk are listed)
- New public function: `seed_builtin_personas(interactive=False, force=False)` — returns `{"copied": [...], "skipped": [...]}` for callers to display or discard

**`tinysteno/main.py`**
- `load_config()` — immediately after the `mkdir` call that ensures `~/.tinysteno/` exists, checks if `~/.tinysteno/personas/` is absent; if so, calls `seed_builtin_personas()` and discards the return value (silent). Note: `setup`, `test`, `config`, and `list` commands return before `load_config()` is called, so this path only fires for `record` and `process`.
- `cmd_setup()` — branches on `--reset-personas` at the top (see below); in the normal path, calls `seed_builtin_personas(interactive=True)` unconditionally after writing config, then prints the returned summary.
- `setup` subparser — gains `--reset-personas` flag

### `--reset-personas` branching in `cmd_setup()`

When `args.reset_personas` is `True`, `cmd_setup()` immediately:
1. Calls `seed_builtin_personas(force=True)`
2. Prints the summary (using the `{"copied": [...], "skipped": [...]}` return value)
3. Returns — does **not** prompt for config values or write the config file

When `args.reset_personas` is `False` (default), `cmd_setup()` runs the full interactive wizard as today, then calls `seed_builtin_personas(interactive=True)` before returning.

## Seeding Function: `seed_builtin_personas(interactive=False, force=False)`

```python
_USER_DIR.mkdir(parents=True, exist_ok=True)
copied = []
skipped = []

for slug in BUILTIN_ORDER:
    src = _BUILTIN_DIR / slug
    dst = _USER_DIR / slug

    if not src.exists():
        logger.warning(f"Built-in persona source missing: {src}")
        continue  # skip this slug entirely

    if dst.exists():
        if force:
            shutil.copytree(src, dst, dirs_exist_ok=True)  # requires Python 3.8+; project requires 3.12+
            copied.append(slug)
        elif interactive:
            # prompt using input() consistent with existing _prompt() style
            answer = input(f"  Built-in persona '{slug}' already exists. Overwrite? [y/N] ").strip().lower()
            if answer in ("y", "yes"):
                shutil.copytree(src, dst, dirs_exist_ok=True)
                copied.append(slug)
            else:
                skipped.append(slug)
        else:
            skipped.append(slug)  # silent skip
    else:
        shutil.copytree(src, dst)
        copied.append(slug)

return {"copied": copied, "skipped": skipped}
```

If `shutil.copytree` raises (e.g. permissions, disk full), the exception propagates to the caller with no swallowing.

### Conflict behavior matrix

| Context | `~/.tinysteno/personas/` state | interactive | force | Result |
|---|---|---|---|---|
| First run via `record`/`process` | Does not exist | False | False | Create dir, copy all slugs silently |
| Subsequent `record`/`process` | Exists (any state) | False | False | Dir already existed — `load_config()` does not call seed |
| `tinysteno setup` (normal) | Any | True | False | Prompt per existing slug; copy missing slugs silently |
| `tinysteno setup --reset-personas` | Any | False | True | Overwrite all slugs silently, print summary, return |

> Note: `load_config()` triggers seeding only when `~/.tinysteno/personas/` is absent. Once it exists (even if partially populated), `load_config()` does not call seed again. Users should run `tinysteno setup` to fill in any missing slugs interactively.

## CLI Output

After seeding in interactive or reset contexts, print a summary using `rich` consistent with existing `cmd_setup` style. Use "Personas seeded" in all contexts:

```
✓ Personas seeded in ~/.tinysteno/personas/
  Copied: default, rca, irm, sprint, kickoff, executive-summary
```

With skipped entries:

```
✓ Personas seeded in ~/.tinysteno/personas/
  Copied: rca, irm
  Skipped (kept yours): default, sprint, kickoff, executive-summary
```

The "Skipped" line is omitted entirely when `skipped` is empty (i.e. all slugs were copied or overwritten).

`load_config()`'s silent seeding path prints nothing.

## Error Handling

- Missing built-in source dir (package integrity issue): log a warning with `logger.warning`, skip that slug via `continue`, do not abort
- `shutil.copytree` failure (permissions, disk full): let the exception propagate; the caller (`load_config` or `cmd_setup`) handles it or crashes with a traceback
- `load_persona` error message: calls `list_personas()` to populate "Available:" list. After this change, `list_personas()` scans only `_USER_DIR`. If seeding failed silently (e.g. a missing source slug), the Available list may be incomplete — this is acceptable; the user is directed to run `tinysteno setup` to reseed.

## Testing

### Existing tests
Tests in `tests/test_personas.py` that load built-in personas must be updated: monkeypatch `_USER_DIR` to a temp directory pre-populated with built-in content (copy from `_BUILTIN_DIR` in the fixture), or monkeypatch `_USER_DIR` to point directly at `_BUILTIN_DIR` for read-only tests.

### New tests for `seed_builtin_personas`
- Seeds all built-in dirs into an empty temp `_USER_DIR`
- Skips existing dirs when `interactive=False, force=False`
- Overwrites all dirs when `force=True`
- Prompts correctly per-conflict when `interactive=True` (monkeypatch `input`)
- Logs a warning and skips a slug when its source dir is absent, without raising

### New tests for `load_config()` seeding trigger
- When `~/.tinysteno/personas/` does not exist, `load_config()` calls `seed_builtin_personas()` with no arguments (monkeypatch `seed_builtin_personas` and assert it was called)
- When `~/.tinysteno/personas/` already exists, `load_config()` does not call `seed_builtin_personas()`

### New tests for `--reset-personas` CLI path
- `tinysteno setup --reset-personas` calls `seed_builtin_personas(force=True)` and returns without prompting for config values
- Argument parser correctly wires `--reset-personas` to `args.reset_personas`

### Updated `list_personas` tests
- Monkeypatch `_USER_DIR` to a tmp dir with known contents
- Verify ordering: built-in slugs in `BUILTIN_ORDER` order, user-only extras appended alphabetically
- Verify a built-in slug absent from `_USER_DIR` is not in the returned list
