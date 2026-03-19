# Pylint Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve all pylint issues across the codebase, organized by severity, bringing the score to 10/10.

**Architecture:** Work task-by-task, highest severity first. After each task, run pylint to verify the target warnings are gone and run pytest to confirm no regressions. Use `.pylintrc` for legitimate project-wide suppressions; use inline `# pylint: disable=` comments only in test files (per-file, at the top) where test-specific conventions apply.

**Tech Stack:** Python, pylint, pytest, uv

---

## Verification commands (run after every task)

```bash
uv run pylint $(git ls-files '*.py')
uv run pytest -q
```

---

## Task 1: Fix E1101 false positives in recorder.py (wave.Wave_write)

**Files:**
- Modify: `tinysteno/recorder.py:253-257`

Pylint cannot resolve the mode-conditional return type of `wave.open()` — it infers `Wave_read` regardless of mode. The code correctly opens in write mode `"w"`. Fix with targeted `# pylint: disable=no-member` on each affected line.

- [ ] **Step 1: Add targeted disable comments to the four flagged lines**

In `tinysteno/recorder.py`, update the `wave.open` block:
```python
with wave.open(str(self.output_path), "w") as wav_file:
    wav_file.setnchannels(out_channels)   # pylint: disable=no-member
    wav_file.setsampwidth(2)              # pylint: disable=no-member
    wav_file.setframerate(self.sample_rate)  # pylint: disable=no-member
    wav_file.writeframes(audio_data.tobytes())  # pylint: disable=no-member
```

- [ ] **Step 2: Verify E1101 errors are gone**

```bash
uv run pylint tinysteno/recorder.py 2>&1 | grep E1101
```
Expected: no output

- [ ] **Step 3: Run tests**

```bash
uv run pytest -q
```
Expected: all pass

- [ ] **Step 4: Commit**

```bash
git add tinysteno/recorder.py
git commit -m "fix: suppress E1101 false positives for wave.Wave_write methods"
```

---

## Task 2: Fix unused import and import order (test_obsidian.py, recorder.py)

**Files:**
- Modify: `tests/test_obsidian.py:1-5`
- Modify: `tinysteno/recorder.py:1-8`

Rules: W0611 (unused import), C0411 (stdlib before third-party).

- [ ] **Step 1: Fix test_obsidian.py — remove unused `patch`, reorder imports**

Read the current imports at the top of the file, then apply:
- Remove `from unittest.mock import patch` entirely (it is unused — confirm with `grep -n "patch" tests/test_obsidian.py` before removing)
- Move stdlib imports (`pathlib`, `unittest.mock`) above third-party (`pytest`)

The corrected import block should look like:
```python
from pathlib import Path

import pytest
```
(Only add `from unittest.mock import ...` back if grep confirms something is actually used.)

- [ ] **Step 2: Fix recorder.py import order — stdlib before third-party**

Current:
```python
import platform
import wave
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional
```
Replace with:
```python
import platform
import wave
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
```

- [ ] **Step 3: Verify**

```bash
uv run pylint tests/test_obsidian.py tinysteno/recorder.py 2>&1 | grep -E "W0611|C0411"
uv run pytest -q
```
Expected: no W0611/C0411, all tests pass

- [ ] **Step 4: Commit**

```bash
git add tests/test_obsidian.py tinysteno/recorder.py
git commit -m "fix: remove unused import, fix stdlib-before-third-party import order"
```

---

## Task 3: Fix exception chaining (raise-missing-from, W0707)

**Files:**
- Modify: `tinysteno/personas/__init__.py:56`
- Modify: `tinysteno/recorder.py:166, 263`

Python requires `raise X(...) from e` when re-raising inside an `except` block to preserve the exception chain.

- [ ] **Step 1: Fix personas/__init__.py**

Find the `raise PersonaInvalidError(...)` inside an `except` block (around line 56) and add `from e`:
```python
raise PersonaInvalidError(f"Persona '{slug}': failed to parse persona.yaml: {e}") from e
```

- [ ] **Step 2: Fix recorder.py line 166**

```python
raise RuntimeError(f"Failed to start audio recording: {e}") from e
```

- [ ] **Step 3: Fix recorder.py line 263**

```python
raise RuntimeError(f"Failed to save recording: {e}") from e
```

- [ ] **Step 4: Verify**

```bash
uv run pylint tinysteno/personas/__init__.py tinysteno/recorder.py 2>&1 | grep W0707
uv run pytest -q
```
Expected: no W0707, all tests pass

- [ ] **Step 5: Commit**

```bash
git add tinysteno/personas/__init__.py tinysteno/recorder.py
git commit -m "fix: add raise-from to preserve exception chains (W0707)"
```

---

## Task 4: Fix logging f-strings (W1203)

**Files:**
- Modify: `tinysteno/orchestrator.py` (lines 55, 65, 75, 104, 126)
- Modify: `tinysteno/main.py` (all W1203 occurrences)

Pylint requires lazy `%`-style formatting in logging calls so the string is only formatted if the log level is active. Pattern: `logger.warning(f"foo {x}")` → `logger.warning("foo %s", x)`.

- [ ] **Step 1: Fix orchestrator.py**

Line 55: `logger.error(f"Summarization failed: {e}")` → `logger.error("Summarization failed: %s", e)`

Line 65-68 (multiline warning):
```python
logger.warning(
    "Field '%s': expected string, got %s; using empty string",
    field_name,
    type(value).__name__,
)
```

Line 75-78 (multiline warning):
```python
logger.warning(
    "Field '%s': expected list, got %s; using empty list",
    field_name,
    type(value).__name__,
)
```

Line 104: `logger.warning(f"Title generation failed: {e}")` → `logger.warning("Title generation failed: %s", e)`

Line 126-128 (multiline warning):
```python
logger.warning(
    "Transcript truncated from %d to %d characters for summarization.",
    len(transcript),
    _TRANSCRIPT_MAX_CHARS,
)
```

- [ ] **Step 2: Fix main.py**

Run to find all occurrences:
```bash
uv run pylint tinysteno/main.py 2>&1 | grep W1203
```
Apply the same pattern (`f"... {x}"` → `"... %s", x`) to each reported line.

- [ ] **Step 3: Verify**

```bash
uv run pylint tinysteno/orchestrator.py tinysteno/main.py 2>&1 | grep W1203
uv run pytest -q
```
Expected: no W1203, all tests pass

- [ ] **Step 4: Commit**

```bash
git add tinysteno/orchestrator.py tinysteno/main.py
git commit -m "fix: use lazy % formatting in logging calls (W1203)"
```

---

## Task 5: Fix unspecified encoding in Path.read_text() calls (W1514)

**Files:**
- Modify: `tests/test_obsidian.py` (lines 68, 86, 100, 113, 133)

The W1514 findings are on `Path.read_text()` calls (not `open()`) — pylint fires W1514 on both. Add `encoding="utf-8"` to each call.

- [ ] **Step 1: Find all Path.read_text() calls missing encoding**

```bash
grep -n "read_text()" tests/test_obsidian.py
```

Change each `.read_text()` to `.read_text(encoding="utf-8")`.

- [ ] **Step 2: Verify**

```bash
uv run pylint tests/test_obsidian.py 2>&1 | grep W1514
uv run pytest -q
```
Expected: no W1514, all tests pass

- [ ] **Step 3: Commit**

```bash
git add tests/test_obsidian.py
git commit -m "fix: specify encoding='utf-8' in Path.read_text() calls (W1514)"
```

---

## Task 6: Fix unused arguments and unnecessary lambda (recorder.py, main.py)

**Files:**
- Modify: `tinysteno/recorder.py:48, 53, 197`
- Modify: `tinysteno/main.py:361`

- [ ] **Step 1: Prefix unused callback args with underscore**

`_audio_callback` at line 48:
```python
def _audio_callback(self, indata, _frames, _time, status):
```

`_loopback_callback` at line 53:
```python
def _loopback_callback(self, indata, _frames, _time, status):
```

- [ ] **Step 2: Replace unnecessary lambda with direct method reference (line 197)**

Current:
```python
callback=lambda data: self._loopback_buffer.append(data),
```
Replace with a direct bound-method reference (no new method needed):
```python
callback=self._loopback_buffer.append,
```

- [ ] **Step 3: Fix unused `args` in cmd_setup (main.py:361)**

Change:
```python
def cmd_setup(args):
```
to:
```python
def cmd_setup(_args):
```

- [ ] **Step 4: Verify**

```bash
uv run pylint tinysteno/recorder.py tinysteno/main.py 2>&1 | grep -E "W0613|W0108"
uv run pytest -q
```
Expected: no W0613/W0108, all tests pass

- [ ] **Step 5: Commit**

```bash
git add tinysteno/recorder.py tinysteno/main.py
git commit -m "fix: prefix unused args with _, replace lambda with bound method ref (W0613, W0108)"
```

---

## Task 7: Fix attribute defined outside __init__ (recorder.py, W0201)

**Files:**
- Modify: `tinysteno/recorder.py:26-44`

`self.output_path` is set in `start()` but never initialized in `__init__`. Add it with a `None` sentinel.

- [ ] **Step 1: Initialize output_path in __init__**

In the `__init__` method, add after the existing attributes:
```python
self.output_path: Optional[Path] = None
```

- [ ] **Step 2: Verify**

```bash
uv run pylint tinysteno/recorder.py 2>&1 | grep W0201
uv run pytest -q
```
Expected: no W0201, all tests pass

- [ ] **Step 3: Commit**

```bash
git add tinysteno/recorder.py
git commit -m "fix: initialize output_path in __init__ (W0201)"
```

---

## Task 8: Add .pylintrc to configure legitimate suppressions

**Files:**
- Create: `.pylintrc`
- Modify: `tests/test_obsidian.py`, `tests/test_orchestrator.py`, `tests/test_personas.py`

Some pylint findings are valid conventions in this codebase and don't represent defects:

| Code | Reason to suppress |
|------|--------------------|
| `C0415` (import-outside-toplevel) | Intentional lazy imports for optional heavy deps (`sounddevice`, `jinja2`) |
| `W0718` (broad-exception-caught) | All 5 occurrences are deliberate degraded-mode fallbacks (LLM errors, hardware errors) — tested by existing tests |
| `W0212` (protected-access in tests) | Standard test pattern — tests must access internals to verify behavior |
| `C0116` (missing-function-docstring in tests) | pytest test functions use the function name as documentation |
| `R0903` (too-few-public-methods) | False positive on thin wrapper classes (`ObsidianExporter`, `WhisperTranscriber`) |
| `R0902` (too-many-instance-attributes) | `AudioRecorder` manages hardware state; 12 attributes is justified |

- [ ] **Step 1: Create .pylintrc**

```ini
[FORMAT]
max-line-length=100

[MESSAGES CONTROL]
disable=
    # Intentional lazy imports for optional/heavy platform-specific dependencies
    import-outside-toplevel,
    # Deliberate broad catches: LLM and hardware errors use fallback/degraded mode
    broad-exception-caught,
    # Hardware interface class (AudioRecorder) legitimately needs many attributes
    too-many-instance-attributes,
    # Thin wrapper classes don't need artificial extra public methods
    too-few-public-methods

[TYPECHECK]
# platform-conditional; don't error on missing members at import time
ignored-modules=sounddevice

[DESIGN]
max-attributes=12
```

- [ ] **Step 2: Add inline disables at the top of each test file**

`tests/test_obsidian.py` — add after imports:
```python
# pylint: disable=missing-function-docstring
```

`tests/test_orchestrator.py` — add after imports:
```python
# pylint: disable=missing-function-docstring,protected-access
```

`tests/test_personas.py` — add after imports:
```python
# pylint: disable=missing-function-docstring,protected-access
```

- [ ] **Step 3: Verify suppressions**

```bash
uv run pylint $(git ls-files '*.py') 2>&1 | grep -E "C0415|W0718|W0212|C0116|R0903|R0902"
```
Expected: no output

- [ ] **Step 4: Run full test suite**

```bash
uv run pytest -q
```
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add .pylintrc tests/test_obsidian.py tests/test_orchestrator.py tests/test_personas.py
git commit -m "chore: add .pylintrc with justified suppressions, add per-file test disables"
```

---

## Task 9: Fix remaining C/R issues (docstrings, line length, complexity)

**Files:**
- Modify: `tinysteno/personas/__init__.py` (C0115 — missing class docstring)
- Modify: `tinysteno/orchestrator.py:95` (C0301 — line too long)
- Modify: `tinysteno/recorder.py:240` (C0301 — line too long)
- Modify: `tinysteno/obsidian.py:50` (C0301 — line too long)
- Modify: `tinysteno/main.py:361` (R0915 — too many statements)

- [ ] **Step 1: Add docstring to Persona dataclass (personas/__init__.py)**

```python
@dataclass
class Persona:
    """Encapsulates a meeting note persona: schema, system prompt, and Jinja2 template."""
```

- [ ] **Step 2: Shorten long lines**

`orchestrator.py:95` — break the content string:
```python
"content": (
    f"Generate a brief title "
    f"(max {_TITLE_MAX_WORDS} words, no special characters)."
),
```

`recorder.py:240` and `obsidian.py:50` — read the actual lines, then wrap at a natural boundary (before an operator or after a comma). Keep the logic identical.

- [ ] **Step 3: Check if R0914 (too-many-locals) remains in obsidian.py**

```bash
uv run pylint tinysteno/obsidian.py 2>&1 | grep R0914
```
`export()` has well under 15 locals; this was already suppressed by prior tasks. If it still appears, add `# pylint: disable=too-many-locals` as a targeted inline comment on the method — do not refactor.

- [ ] **Step 4: Reduce too-many-statements in cmd_setup (main.py)**

Extract the config-writing into a helper function at module level:
```python
def _write_config(config_path: Path, config: dict) -> None:
    """Persist config dict to YAML file."""
    import yaml
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.dump(config, default_flow_style=False), encoding="utf-8")
```
Call `_write_config(config_path, config)` in `cmd_setup` where the write currently happens.

- [ ] **Step 5: Verify**

```bash
uv run pylint $(git ls-files '*.py') 2>&1 | grep -E "C0115|C0301|R0914|R0915"
uv run pytest -q
```
Expected: no remaining C/R issues, all tests pass

- [ ] **Step 6: Commit**

```bash
git add tinysteno/personas/__init__.py tinysteno/orchestrator.py tinysteno/recorder.py tinysteno/obsidian.py tinysteno/main.py
git commit -m "fix: add missing docstrings, shorten long lines, reduce function complexity"
```

---

## Task 10: Final verification — target score 10/10

- [ ] **Step 1: Run full pylint**

```bash
uv run pylint $(git ls-files '*.py')
```
Expected: score 10.00/10

- [ ] **Step 2: Run tests**

```bash
uv run pytest -q
```
Expected: 39 passed

- [ ] **Step 3: If any remaining issues**

Use `# pylint: disable=<code>  # <reason>` inline — never suppress without a comment explaining why.

---

## Issue Reference Table

| Severity | Code | Count | File(s) | Task |
|----------|------|-------|---------|------|
| E | E1101 | 4 | recorder.py | 1 |
| W | W0611 | 1 | test_obsidian.py | 2 |
| C | C0411 | 5 | test_obsidian.py, recorder.py | 2 |
| W | W0707 | 3 | personas/__init__.py, recorder.py | 3 |
| W | W1203 | 8+ | orchestrator.py, main.py | 4 |
| W | W1514 | 5 | test_obsidian.py | 5 |
| W | W0613 | 4 | recorder.py, main.py | 6 |
| W | W0108 | 1 | recorder.py | 6 |
| W | W0201 | 1 | recorder.py | 7 |
| W | W0212 | 13 | test_orchestrator.py, test_personas.py | 8 (suppress) |
| W | W0718 | 5 | orchestrator.py, recorder.py | 8 (suppress) |
| C | C0116 | 20+ | test files | 8 (suppress) |
| C | C0415 | 8 | recorder.py, obsidian.py | 8 (suppress) |
| C | C0115 | 1 | personas/__init__.py | 9 |
| C | C0301 | 3 | recorder.py, orchestrator.py, obsidian.py | 9 |
| R | R0902 | 1 | recorder.py | 8 (suppress) |
| R | R0903 | 2 | obsidian.py, transcriber.py | 8 (suppress) |
| R | R0914 | 1 | obsidian.py | 9 |
| R | R0915 | 1 | main.py | 9 |
