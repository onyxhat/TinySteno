# MLX Transcription Backend + Beam-Size Tweak Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make local transcription fast on Apple Silicon by adding an `mlx` backend (GPU via mlx-whisper) and a configurable beam size for the existing faster-whisper backend, with zero behavior change on Linux, Windows, CUDA, and Intel Macs.

**Architecture:** Extract the channel-splitting/assemble orchestration from `WhisperTranscriber` into a `LocalTranscriber` base class whose subclasses implement only `_decode()`. A new `MlxWhisperTranscriber` (new file, lazy `mlx_whisper` import, same pattern as `sherpa_diarizer`) plugs in as `transcription_backend: mlx`. The factory falls back to the faster-whisper backend with a logged warning whenever mlx is unavailable, so a synced config never breaks another machine.

**Tech Stack:** mlx-whisper (optional extra, darwin/arm64-only marker), faster-whisper (unchanged), existing `Transcriber` protocol.

**Spec:** No separate spec document; the Background section below is the agreed scope (from the performance investigation in this session).

## Background

- faster-whisper/CTranslate2 has no Metal backend: on Apple Silicon, `_detect_device()` (tinysteno/transcriber.py:120) resolves to CPU. mlx-whisper runs Whisper on the GPU via Apple MLX and is a plain pip dependency.
- Current decode uses `beam_size=5` hardcoded (tinysteno/transcriber.py:239); greedy (`1`) is markedly faster on CPU at a small accuracy cost. Default stays 5 — behavior-preserving.
- mlx-whisper does not install or import off Apple Silicon, has no `vad_filter` and no `beam_size` parameter. Those settings apply to the `local` backend only; the hygiene filters in `tinysteno/hygiene.py` still run for every backend inside `transcript.assemble()`.

## Global Constraints

- All dependency management goes through `uv`: edit `pyproject.toml` then `uv sync` (or `uv add`/`uv sync --extra mlx`); every Python command runs as `uv run …`. Never `pip install`, bare `python`, or a manually activated venv.
- `uv run pytest` and `uv run pylint $(git ls-files '*.py')` must pass on every commit; max line length 100.
- `tinysteno.main`, `transcriber`, `transcript` must import without `mlx_whisper` installed (lazy import inside functions, like `sherpa_onnx`; `.pylintrc` already allows `import-outside-toplevel` and `broad-exception-caught`).
- New editable settings go in `SETTINGS` (tinysteno/settings.py) and `DEFAULT_CONFIG` (tinysteno/config.py) — the Settings screen and first-run guide are generated from them; no TUI changes.
- Do not tune hygiene/speaker-split thresholds.
- Default behavior on non-mac platforms, Intel Macs, and CUDA machines must be byte-for-byte unchanged (same backend, same beam size, same fingerprints).
- Work on branch `feat/mlx-backend`.

## Review Focus

1. Config says `transcription_backend: mlx` on Linux/Windows/Intel Mac (synced `~/.tinysteno/config.yaml`) → recording must still transcribe via faster-whisper with a logged warning, never crash. → test in Task 4.
2. Hand-edited `whisper_beam_size: 0`, `-3`, or `"abc"` → `validate_config` raises a clear ValueError at load, not a crash mid-transcription. → test in Task 1.
3. Existing saved transcripts must not become stale: `transcript_fingerprint` output is unchanged when `whisper_beam_size` is 5 (the default). → test in Task 1.
4. `whisper_model` set to an unknown name or a full HF repo id with the mlx backend → passed through to mlx-whisper, no KeyError. → test in Task 3.
5. Dual-channel recordings with a silent channel must behave identically on the mlx backend (silent channel skipped, bleed/hallucination filters applied). → test in Task 3.

---

### Task 1: `whisper_beam_size` setting

**Files:**
- Modify: `tinysteno/settings.py` (SETTINGS tuple, Transcription group, after `whisper_vad`; and the positive-number check near line 161)
- Modify: `tinysteno/config.py` (DEFAULT_CONFIG; `validate_config`)
- Modify: `tinysteno/transcriber.py:153-184` (`WhisperTranscriber.__init__`), `:230-252` (`_decode`)
- Modify: `tinysteno/main.py:70-101` (`create_transcriber`)
- Modify: `tinysteno/pipeline.py:157-183` (`transcript_fingerprint`)
- Test: `tests/test_config.py`, `tests/test_main.py`, `tests/test_pipeline.py`

**Interfaces:**
- Produces: `WhisperTranscriber.__init__(..., compute_type="auto", *, beam_size: int = 5, vad=True, prompt="", options=None)` storing `self.beam_size`; config key `"whisper_beam_size"` (int, default 5).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_config.py
def test_validate_config_rejects_bad_beam_size():
    for bad in (0, -3, "abc", 2.5):
        with pytest.raises(ValueError, match="whisper_beam_size"):
            validate_config({**DEFAULT_CONFIG, "whisper_beam_size": bad})

def test_default_config_includes_beam_size():
    assert DEFAULT_CONFIG["whisper_beam_size"] == 5

# tests/test_main.py  (follow test_create_transcriber_reads_vad_and_prompt_from_config's
# patching style: patch tinysteno.transcriber.WhisperModel, clear _MODEL_CACHE)
def test_create_transcriber_passes_beam_size():
    # config {"whisper_beam_size": 1} -> transcriber.beam_size == 1
    # config without the key -> beam_size == 5

# tests/test_pipeline.py
def test_fingerprint_unchanged_at_default_beam_size():
    # beam 5 (or key absent): "beam_size" not in fingerprint
def test_fingerprint_includes_non_default_beam_size():
    # local backend, beam 1: fingerprint["beam_size"] == 1
    # api backend, beam 1: "beam_size" not in fingerprint
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_config.py tests/test_main.py tests/test_pipeline.py -k beam -v`
Expected: FAIL (KeyError / AttributeError / assertion)

- [ ] **Step 3: Implement**

- `DEFAULT_CONFIG["whisper_beam_size"] = 5`; `validate_config` rejects non-int or `< 1` (message names the field; `bool` is not a valid int here).
- `Setting("whisper_beam_size", "Whisper beam size", "int", "Transcription", "Decoding beams for local transcription; 1 is fastest, 5 is most accurate.")` placed after `whisper_vad`; add the key to the settings-side positive-number guard beside `whisper_api_timeout`.
- `WhisperTranscriber` stores `beam_size` and `_decode` passes `beam_size=self.beam_size` instead of the literal 5.
- `create_transcriber` passes `beam_size=config["whisper_beam_size"]` when the value is a valid int ≥ 1, else 5.
- `transcript_fingerprint`: after building the dict, `if backend != "api" and beam != 5: fingerprint["beam_size"] = beam` (same only-while-in-effect pattern as `whisper_api_diarize`, so existing sidecars stay fresh).

- [ ] **Step 4: Run the full suite and lint**

Run: `uv run pytest && uv run pylint $(git ls-files '*.py')`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add tinysteno tests
git commit -m "feat: configurable whisper_beam_size for local transcription"
```

---

### Task 2: Extract `LocalTranscriber` base class (pure refactor)

**Files:**
- Modify: `tinysteno/transcriber.py:150-252`
- Test: existing `tests/test_transcriber.py` (no new tests; green before and after)

**Interfaces:**
- Produces: `class LocalTranscriber` in `tinysteno/transcriber.py` holding the current `transcribe()` body (tinysteno/transcriber.py:186-228) and `__init__(self, *, vad: bool = True, prompt: str = "", options: TranscriptOptions | None = None)`; subclasses implement `_decode(self, audio: np.ndarray, on_progress=None) -> tuple[list[Segment], str]` (base raises `NotImplementedError`).
- `WhisperTranscriber(LocalTranscriber)` keeps its exact public signature (incl. `beam_size` from Task 1) and its `_decode`, model cache, and device detection.

- [ ] **Step 1: Verify the suite is green before refactoring**

Run: `uv run pytest tests/test_transcriber.py -v`
Expected: PASS

- [ ] **Step 2: Move `transcribe()` into `LocalTranscriber`; make `WhisperTranscriber` extend it**

No behavior change: same method bodies, `WhisperTranscriber.__init__` calls `super().__init__(vad=..., prompt=..., options=...)` then does model resolution exactly as today.

- [ ] **Step 3: Run the full suite and lint**

Run: `uv run pytest && uv run pylint $(git ls-files '*.py')`
Expected: all pass, zero test edits needed

- [ ] **Step 4: Commit**

```bash
git add tinysteno/transcriber.py
git commit -m "refactor: extract LocalTranscriber base from WhisperTranscriber"
```

---

### Task 3: `MlxWhisperTranscriber`

**Files:**
- Create: `tinysteno/mlx_transcriber.py`
- Test: `tests/test_mlx_transcriber.py` (new)

**Interfaces:**
- Consumes: `LocalTranscriber` and `Segment` from Task 2.
- Produces: `class MlxWhisperTranscriber(LocalTranscriber)` with `__init__(self, model_size: str = "small", *, prompt: str = "", options: TranscriptOptions | None = None)`; module function `mlx_available() -> bool`; module dict `_MLX_MODELS: dict[str, str]`.

- [ ] **Step 1: Write the failing tests**

`mlx_whisper` is never installed in CI (Linux), so every test injects a fake: `monkeypatch.setitem(sys.modules, "mlx_whisper", fake_module)` where `fake_module.transcribe(audio, **kw)` returns `{"segments": [{"start": 0.0, "end": 0.5, "text": " hi "}], "language": "en"}`.

```python
def test_decode_adapts_mlx_result_to_segments():
    # _decode returns ([Segment(start=0.0, end=0.5, text="hi")], "en")
    # and passed path_or_hf_repo=_MLX_MODELS["small"], initial_prompt=None

def test_known_sizes_map_and_unknown_pass_through():
    # "tiny","base","small","medium","large-v3" -> "mlx-community/whisper-<size>-mlx"
    # ("large-v3" -> "mlx-community/whisper-large-v3-mlx")
    # "mlx-community/whisper-large-v3-turbo" and "my-model" -> unchanged

def test_prompt_forwarded():
    # prompt="Acme" -> fake called with initial_prompt="Acme"

def test_dual_channel_skips_silent_channel():
    # reuse the WAV-writing pattern of tests/test_transcriber.py::_transcribe:
    # two channels, one silent; fake raises if called with a silent-channel array;
    # result has text only for the audible channel and warnings list present

def test_mlx_available_false_without_module(monkeypatch):
    # monkeypatch.setitem(sys.modules, "mlx_whisper", None) / ensure ImportError -> False

def test_module_imports_without_mlx_whisper():
    # importlib.import_module("tinysteno.mlx_transcriber") succeeds with mlx absent
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_mlx_transcriber.py -v`
Expected: FAIL with "No module named 'tinysteno.mlx_transcriber'"

- [ ] **Step 3: Implement `tinysteno/mlx_transcriber.py`**

- `_MLX_MODELS` maps the five sizes from the `whisper_model` setting choices to `mlx-community/whisper-<size>-mlx` repos; `__init__` resolves `self.model = _MLX_MODELS.get(model_size, model_size)`.
- `_decode` lazily does `import mlx_whisper` (function scope), calls `mlx_whisper.transcribe(audio, path_or_hf_repo=self.model, initial_prompt=self.prompt or None)`, converts segment dicts to `Segment`, strips/drops empty text like `WhisperTranscriber._decode`, reports `on_progress(1.0)` when done (mlx gives no per-segment stream; coarse per-channel progress is accepted).
- `mlx_available()` = try `import mlx_whisper`; `except Exception: return False` (also False on non-arm64 where the wheel doesn't exist).

- [ ] **Step 4: Run the new tests and lint**

Run: `uv run pytest tests/test_mlx_transcriber.py -v && uv run pylint tinysteno/mlx_transcriber.py tests/test_mlx_transcriber.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tinysteno/mlx_transcriber.py tests/test_mlx_transcriber.py
git commit -m "feat: MLX (Metal) local transcription backend"
```

---

### Task 4: Wire the `mlx` backend with safe fallback

**Files:**
- Modify: `tinysteno/settings.py:62-64` (`transcription_backend` Setting)
- Modify: `tinysteno/main.py:70-101` (`create_transcriber`)
- Modify: `pyproject.toml` (`[project.optional-dependencies]`)
- Test: `tests/test_main.py`, `tests/test_settings.py`

**Interfaces:**
- Consumes: `MlxWhisperTranscriber`, `mlx_available()` from Task 3.
- Produces: config value `transcription_backend: "mlx"`; extra `uv sync --extra mlx`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_settings.py
def test_transcription_backend_offers_mlx():
    assert setting("transcription_backend").choices == ("local", "mlx", "api")

# tests/test_main.py (same WhisperModel-patch + _MODEL_CACHE.clear() style as neighbors)
def test_create_transcriber_mlx_backend_when_available():
    # fake mlx_whisper in sys.modules; backend "mlx"
    # -> isinstance(result, MlxWhisperTranscriber), model from whisper_model

def test_create_transcriber_mlx_falls_back_to_local(caplog):
    # mlx import failing (sys.modules["mlx_whisper"] absent/raising); backend "mlx"
    # -> isinstance(result, WhisperTranscriber); caplog has one warning naming "mlx"
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_main.py tests/test_settings.py -k mlx -v`
Expected: FAIL

- [ ] **Step 3: Implement**

- Setting: `choices=("local", "mlx", "api")`, description "Transcribe on this computer (local), on an Apple Silicon GPU (mlx), or through a Whisper API (api)."
- `create_transcriber`: `elif backend == "mlx":` lazily import `MlxWhisperTranscriber` and check `mlx_available()`; on success return it with `model_size=config.get("whisper_model", "small")`, `prompt`, `options`; otherwise `logger.warning("mlx backend unavailable on this machine; using local faster-whisper")` and fall through to the existing `WhisperTranscriber` return.
- pyproject: `mlx = ["mlx-whisper>=0.4.1; sys_platform == 'darwin' and platform_machine == 'arm64'"]` — the marker makes `uv sync --extra mlx` a no-op on every other platform instead of a resolution failure.

- [ ] **Step 4: Run the full suite and lint**

Run: `uv run pytest && uv run pylint $(git ls-files '*.py')`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add tinysteno/settings.py tinysteno/main.py pyproject.toml tests
git commit -m "feat: wire mlx transcription backend with cross-platform fallback"
```

---

### Task 5: On-hardware smoke test (this Mac only)

**Files:** none (verification)

- [ ] **Step 1: Install the extra and confirm the five model repos resolve**

Run: `uv sync --extra mlx`, then for each value in `_MLX_MODELS`, `uv run python -c "from huggingface_hub import model_info; model_info('<repo>')"` (hf-hub ships with mlx-whisper).
Expected: no 404. If a repo name is wrong, fix `_MLX_MODELS` to the existing `mlx-community` repo for that size and re-run Task 3's tests.

- [ ] **Step 2: Transcribe a real recording on both backends**

Run: generate a short spoken-word WAV (or reuse one from `~/.tinysteno`), then time `backend: mlx` vs `backend: local` through `create_transcriber(config).transcribe(path)` in `uv run python`.
Expected: mlx output text is sane, wall-clock clearly faster; note both times in the PR description.

- [ ] **Step 3: Confirm fallback end-to-end**

Run: `uv run python -c` snippet that blocks `mlx_whisper` (`sys.modules['mlx_whisper']=None`) with backend `mlx` and transcribes the same WAV.
Expected: warning logged, faster-whisper result returned.

- [ ] **Step 4: Final full gate and push**

Run: `uv run pytest && uv run pylint $(git ls-files '*.py')`
Expected: all pass. Then commit anything outstanding and open the PR.

---

## Self-Review (done at planning time)

- Coverage: both agreed changes (mlx backend, beam-size tweak) have tasks; the zero-code option (external whisper.cpp server) needs no code and is out of scope.
- Cross-platform safety is enforced three ways: environment-marked optional extra (no resolution break), lazy import (modules import everywhere), factory fallback (synced configs keep working) — each pinned by a test.
- Type consistency: `LocalTranscriber._decode` signature identical in Tasks 2 and 3; `beam_size` lives only on `WhisperTranscriber`.
- Fingerprint compatibility covered in Task 1 (Review Focus 3).
- Known uncertainty: exact `mlx-community` repo ids are verified against the hub in Task 5 Step 1 before the PR, on the one platform that can run them.
