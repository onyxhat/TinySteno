# Speaker Split Implementation Plan (3 of 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Optionally split the remote side of a call into `Others`, `Speaker 2`, `Speaker 3`… (and the microphone of an in-person meeting into `You`, `Speaker 2`…), off by default, never worse than today's channel labels when it fails.

**Architecture:** Two interchangeable sources of speaker turns, a local `sherpa-onnx` diarizer (optional extra) and the maintainer's WhisperX server (speaker labels on returned segments), feed one pure policy module, `tinysteno/speaker_split.py`, that cleans the turns and relabels transcript segments. The policy decides when to give up and keep plain channel labels; every failure path does the same. It plugs into `transcript.assemble()`, so both backends share it.

**Tech Stack:** Python 3.12, uv, `sherpa-onnx` (Apache-2.0, optional), openai SDK, numpy, pytest, pylint 4.0.5.

**Spec:** `docs/reports/AI note taker diarization comparison.md`, rows 11-12 of "A prioritised adoption plan", and the sections on OpenWhispr, StenoAI, anarlog and "The backend answer…". Requires plans 1 and 2 to be merged first.

## Global Constraints

- Run tools as `uv run pytest -q -W error` and `uv run pylint $(git ls-files '*.py')`; both clean. Max line length 100.
- `sherpa_onnx` is imported lazily, inside functions, and only when `speaker_split` is on. `import tinysteno.main`, `tinysteno.transcriber` and `tinysteno.transcript` must work without it installed.
- No test downloads a model, runs a real diarizer, loads a Whisper model, opens an audio device or calls a network API.
- The feature is off by default (`speaker_split: false`). With it off, behaviour and output are byte-identical to plan 2.
- Any failure (extra not installed, download failed, model error, server refusal, policy collapse) leaves the `You` / `Others` labels exactly as they were. Only genuine errors add a warning; a policy decision that there is nothing to split does not.
- The speaker count is never passed to the clusterer (`num_clusters=-1` always); a `--speakers` hint is applied afterwards as a cap.
- Do not use the `sherpa-onnx-reverb-diarization-v1` model (non-commercial licence).
- openminutes is AGPL-3.0: do not copy its code. OpenWhispr and StenoAI are MIT; behaviour ported from them is credited in a module docstring.
- Policy constants are fixed by this plan and named in code: threshold `0.6` up to 15 minutes rising linearly to `0.7` at 60 minutes; phantom cluster under `1.0` s total, or under `10%` of speech with mean turn under `3.0` s; collapse at `93%` dominance; minimum audio `30.0` s; nearest-turn fallback `2.0` s.
- Do not edit `CHANGELOG.md` or the version.

## What this plan leaves out

Voiceprints and automatic naming, word-level splitting of a segment between speakers, centroid merging (the sherpa-onnx Python result does not expose embeddings), Parakeet, and signal-level echo cancellation: all row 13 "defer" items. The honest description of this feature in the docs is "a best-effort split".

## Review Focus

1. **`speaker_split: true` without the extra installed**: the user expects their note with normal labels and one clear line telling them what to install, not a crash after the meeting. Test in Task 4.
2. **No network on first use** (model download fails): same expectation. Test in Task 3.
3. **A one-to-one call**: the user expects `You` and `Others` only, never a phantom `Speaker 2`. Tests in Task 2.
4. **A very short recording**: the user expects no split attempted and no warning. Test in Task 2.
5. **An API server that rejects the `diarize` field**: the user expects the transcript anyway, with channel labels and a warning. Test in Task 5.

---

### Task 1: Add the optional dependency and pin what it actually exposes

**Files:**
- Modify: `pyproject.toml`, `uv.lock`
- Create: `tests/test_sherpa_api.py`

**Interfaces:**
- Produces: extra `diarize = ["sherpa-onnx"]` under `[project.optional-dependencies]` (for users: `pip install "tinysteno[diarize]"`), and `sherpa-onnx` also in the `dev` dependency group so `uv run pytest` exercises it. The lower bound is whatever version uv resolves today, written as `>=X.Y.Z`.

- [ ] **Step 1: Add the dependency**

Run: `uv add --optional diarize sherpa-onnx && uv add --group dev sherpa-onnx && uv sync`
Then: `uv run python -c "import sherpa_onnx; print(sherpa_onnx.__version__)"`
Expected: a version prints. Set the lower bound in both places to that version.

- [ ] **Step 2: Write the test that pins the API this plan relies on**

```python
"""The parts of sherpa-onnx's Python API that tinysteno.sherpa_diarizer uses."""
import pytest

sherpa_onnx = pytest.importorskip("sherpa_onnx")

@pytest.mark.parametrize("name", [
    "OfflineSpeakerDiarization", "OfflineSpeakerDiarizationConfig",
    "OfflineSpeakerSegmentationModelConfig", "OfflineSpeakerSegmentationPyannoteModelConfig",
    "SpeakerEmbeddingExtractorConfig", "FastClusteringConfig",
])
def test_class_exists(name):
    assert hasattr(sherpa_onnx, name)

def test_config_accepts_the_fields_we_set(tmp_path):
    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model="seg.onnx")),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model="emb.onnx", num_threads=2),
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=-1, threshold=0.6),
        min_duration_on=0.2, min_duration_off=0.5,
    )
    assert config.clustering.threshold == pytest.approx(0.6)
```

- [ ] **Step 3: Run it**

Run: `uv run pytest tests/test_sherpa_api.py -q`
Expected: PASS. If a name or field differs in the installed version, correct this test to the real name, and carry that name into Task 3; note the difference in the commit message.

- [ ] **Step 4: Confirm the model downloads exist** (no download, headers only)

Run:
```bash
curl -sIL https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2 | grep -i '^HTTP' | tail -1
curl -sIL https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx | grep -i '^HTTP' | tail -1
```
Expected: `200` for both (the misspelt `recongition` is the upstream release tag). If either is not 200, stop and report: Task 3's URLs depend on them.

- [ ] **Step 5: Commit**: `git add pyproject.toml uv.lock tests/test_sherpa_api.py && git commit -m "build: optional diarize extra (sherpa-onnx)"`

---

### Task 2: The speaker-split policy

**Files:**
- Create: `tinysteno/speaker_split.py`
- Test: `tests/test_speaker_split.py`

**Interfaces:**
- Consumes: `Segment` from `tinysteno.transcript`
- Produces:
  - `@dataclass(frozen=True) class Turn`: `start: float`, `end: float`, `speaker: int`
  - `MIN_AUDIO_SECONDS = 30.0`
  - `cluster_threshold(duration_seconds: float) -> float`: `0.6` at or below 900 s, `0.7` at or above 3600 s, linear between.
  - `apply_policy(turns: list[Turn], *, max_speakers: int | None = None) -> list[Turn] | None`:
    1. Drop phantom clusters: total speech under 1.0 s, or under 10% of all speech with a mean turn under 3.0 s.
    2. If `max_speakers` is set and more clusters remain, the clusters with the least speech are folded away: each of their turns takes the speaker of the kept turn nearest in time.
    3. Return `None` (keep channel labels) when no turns remain, one cluster remains, `max_speakers == 1`, or the largest cluster holds 93% or more of the speech.
    4. Otherwise renumber speakers by total speech, most first (`0` is dominant), and return the turns in time order.
  - `label_segments(segments: list[Segment], turns: list[Turn], base_label: str) -> list[Segment]`: every segment whose speaker is `base_label` gets the speaker of the turn it overlaps most; with no overlap, the turn whose midpoint is nearest within 2.0 s; otherwise the dominant speaker. Speaker `0` keeps `base_label`; speaker `k` becomes `f"Speaker {k + 1}"`. Other segments are returned unchanged.

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.parametrize("seconds, expected", [(60, 0.6), (900, 0.6), (2250, 0.65), (3600, 0.7), (9000, 0.7)])
def test_cluster_threshold_ramps(seconds, expected):
    assert cluster_threshold(seconds) == pytest.approx(expected)

def test_one_to_one_call_keeps_channel_labels():                    # Review Focus 3
    assert apply_policy([Turn(0, 50, 0), Turn(52, 110, 0)]) is None

def test_a_phantom_cluster_does_not_become_a_speaker():             # Review Focus 3
    turns = [Turn(0, 120, 0), Turn(121, 121.6, 1), Turn(125, 240, 0)]
    assert apply_policy(turns) is None                               # 0.6 s phantom dropped → one cluster

def test_near_total_dominance_collapses():
    assert apply_policy([Turn(0, 950, 0), Turn(950, 1000, 1)]) is None      # 95% / 5%

def test_two_real_speakers_are_kept_and_ranked_by_speech():
    turns = [Turn(0, 30, 4), Turn(30, 110, 7), Turn(110, 140, 4)]      # 7 spoke 80 s, 4 spoke 60 s
    assert apply_policy(turns) == [Turn(0, 30, 1), Turn(30, 110, 0), Turn(110, 140, 1)]

def test_speaker_cap_folds_the_smallest_into_its_neighbour():
    turns = [Turn(0, 60, 0), Turn(60, 100, 1), Turn(100, 112, 2), Turn(112, 170, 1)]
    out = apply_policy(turns, max_speakers=2)
    assert {t.speaker for t in out} == {0, 1} and len(out) == 4

def test_cap_of_one_means_no_split():
    assert apply_policy([Turn(0, 60, 0), Turn(60, 120, 1)], max_speakers=1) is None

def test_empty_turns():
    assert apply_policy([]) is None

def test_label_segments_by_overlap_and_leaves_other_channel_alone():
    segs = [Segment(1, 5, "a", "Others"), Segment(31, 40, "b", "Others"), Segment(6, 8, "c", "You")]
    turns = [Turn(0, 30, 1), Turn(30, 100, 0)]
    assert [s.speaker for s in label_segments(segs, turns, "Others")] == ["Speaker 2", "Others", "You"]

def test_segment_outside_every_turn_goes_to_the_nearest_or_dominant(): ...
```

- [ ] **Step 2: Run to verify failure**: `uv run pytest tests/test_speaker_split.py -q`. Expected: `No module named 'tinysteno.speaker_split'`.
- [ ] **Step 3: Implement `tinysteno/speaker_split.py`.** Module docstring credits OpenWhispr's `diarizationPolicy.js` (MIT) for the threshold ramp, phantom and dominance rules and StenoAI (MIT) for keeping the base label on the dominant voice.
- [ ] **Step 4: Run to verify pass**: `uv run pytest tests/test_speaker_split.py -q -W error && uv run pylint tinysteno/speaker_split.py tests/test_speaker_split.py`.
- [ ] **Step 5: Commit**: `git commit -m "feat: policy that decides when a speaker split is trustworthy"`

---

### Task 3: The local diarizer

**Files:**
- Create: `tinysteno/sherpa_diarizer.py`
- Test: `tests/test_sherpa_diarizer.py`

**Interfaces:**
- Consumes: `Turn`, `cluster_threshold` (Task 2); the API pinned in Task 1
- Produces:
  - `MODEL_DIR = Path.home() / ".tinysteno" / "models" / "diarize"`, `SEGMENTATION_URL`, `EMBEDDING_URL` (the two URLs checked in Task 1).
  - `class DiarizerUnavailable(RuntimeError)`: raised with a message fit to show the user.
  - `ensure_models(model_dir: Path = MODEL_DIR, fetch: Callable[[str, Path], None] = _download) -> tuple[Path, Path]`: returns the segmentation `model.onnx` and the embedding `.onnx`, downloading whichever is missing (to a temporary name, then renamed; the archive is extracted with `tarfile` using `filter="data"`). Any failure raises `DiarizerUnavailable("Could not download the speaker-separation models: {reason}")` and leaves no partial file.
  - `diarize(audio: np.ndarray, *, on_progress: Callable[[float], None] | None = None, num_threads: int = 2) -> list[Turn]`: `audio` is 16 kHz mono float32. Imports `sherpa_onnx` lazily; a missing import raises `DiarizerUnavailable('Speaker separation needs the optional extra: pip install "tinysteno[diarize]"')`. Runs `OfflineSpeakerDiarization` with `num_clusters=-1`, `threshold=cluster_threshold(len(audio) / 16000)`, `min_duration_on=0.2`, `min_duration_off=0.5`, and returns the result sorted by start time as `Turn`s. Progress from the library's callback is passed on as a 0-1 fraction.

- [ ] **Step 1: Write the failing tests**

```python
def test_missing_extra_is_reported_with_the_install_command():
    with patch.dict(sys.modules, {"sherpa_onnx": None}):
        with pytest.raises(DiarizerUnavailable, match=r'pip install "tinysteno\[diarize\]"'):
            diarize(np.zeros(16000 * 40, dtype=np.float32))

def test_models_are_downloaded_once_then_reused(tmp_path):
    fetched = []
    paths = ensure_models(tmp_path, fetch=_fake_fetch(fetched))
    assert all(p.exists() for p in paths) and len(fetched) == 2
    ensure_models(tmp_path, fetch=_fake_fetch(fetched))
    assert len(fetched) == 2

def test_failed_download_leaves_nothing_behind(tmp_path):            # Review Focus 2
    def offline(_url, _dest):
        raise OSError("network unreachable")
    with pytest.raises(DiarizerUnavailable, match="Could not download"):
        ensure_models(tmp_path, fetch=offline)
    assert not list(tmp_path.rglob("*"))

def test_diarize_maps_library_output_to_turns_and_sets_the_ramped_threshold(tmp_path):
    fake = _fake_sherpa(segments=[(12.0, 20.0, 1), (0.0, 11.5, 0)])
    with patch.dict(sys.modules, {"sherpa_onnx": fake}), \
         patch("tinysteno.sherpa_diarizer.ensure_models", return_value=(tmp_path / "s", tmp_path / "e")):
        turns = diarize(np.zeros(16000 * 1800, dtype=np.float32))    # 30 minutes
    assert turns == [Turn(0.0, 11.5, 0), Turn(12.0, 20.0, 1)]
    assert fake.FastClusteringConfig.call_args.kwargs == {"num_clusters": -1,
                                                          "threshold": pytest.approx(0.6333, abs=1e-3)}
```

`_fake_fetch` writes a tiny valid `.tar.bz2` containing `sherpa-onnx-pyannote-segmentation-3-0/model.onnx` for the archive URL and a small file for the other; `_fake_sherpa` is a `MagicMock` module whose diarizer's `process(...).sort_by_start_time()` returns objects with `start`, `end`, `speaker`.

- [ ] **Step 2: Run to verify failure**: `uv run pytest tests/test_sherpa_diarizer.py -q`.
- [ ] **Step 3: Implement.** `_download` uses `urllib.request` with a 60 s timeout.
- [ ] **Step 4: Run to verify pass**: `uv run pytest tests/test_sherpa_diarizer.py -q -W error && uv run pylint tinysteno/sherpa_diarizer.py tests/test_sherpa_diarizer.py`.
- [ ] **Step 5: Commit**: `git commit -m "feat: local speaker diarization through the optional sherpa-onnx extra"`

---

### Task 4: Wire the split into both backends

**Files:**
- Modify: `tinysteno/transcript.py` (`TranscriptOptions`, `assemble`), `tinysteno/transcriber.py`, `tinysteno/api_transcriber.py`, `tinysteno/main.py` (`load_config` defaults, `create_transcriber`, `--speakers` on `record`, `process`, `tui`), `tinysteno/orchestrator.py` (`PromptContext`, the context block), `tinysteno/pipeline.py`
- Test: `tests/test_transcript.py`, `tests/test_main.py`, `tests/test_orchestrator.py`

**Interfaces:**
- Consumes: `apply_policy`, `label_segments`, `MIN_AUDIO_SECONDS` (Task 2); `diarize`, `DiarizerUnavailable` (Task 3)
- Produces:
  - `TranscriptOptions` gains `speaker_split: bool = False`, `max_speakers: int | None = None`.
  - `assemble(..., options=None, turns_for: Callable[[str, np.ndarray], list[Turn]] | None = None, on_progress=None)`. `turns_for(label, audio)` returns raw turns for one channel; the default calls `sherpa_diarizer.diarize`. After the plan-2 filters and only when `options.speaker_split`:
    - **Which channel:** `OTHERS` when it has segments; otherwise (system audio silent or absent, or a mono file) the microphone channel, with `You` as the base label (mono segments are first labelled `You`). The microphone of a call that has system audio is never split.
    - **Skip quietly** when that channel's audio is shorter than `MIN_AUDIO_SECONDS` or `max_speakers == 1`.
    - Turns go through `apply_policy(turns, max_speakers=...)`; `None` leaves labels alone; otherwise `label_segments`.
    - `DiarizerUnavailable` adds its message as a warning; any other exception adds `"Speaker separation failed; channel labels were used."` Labels are untouched in both cases.
  - Progress: when the split is on, the transcriber maps decoding to 0-0.7 of its progress and the diarizer's fraction to 0.7-1.0.
  - Config keys `speaker_split: false`; flag `--speakers N` (integer ≥ 1, argparse-validated): the most voices to expect on the split channel; it sets `max_speakers` for that run and implies nothing when `speaker_split` is off.
  - `PromptContext.has_numbered_speakers: bool = False`, set by the pipeline when any segment's speaker starts with `Speaker `. The context block then adds: `Speaker 2, Speaker 3 and so on are further voices separated automatically. The separation is approximate: do not assign an action to a numbered speaker unless the transcript names them.`

- [ ] **Step 1: Write the failing tests**

```python
ON = TranscriptOptions(speaker_split=True)

def test_split_relabels_the_remote_channel_only():
    channels = {YOU: [S(5, "mine")], OTHERS: [S(1, "first voice"), S(40, "second voice")]}
    result = assemble(channels, LONG_BOTH, diarize=True, language="en", options=ON,
                      turns_for=lambda label, audio: [Turn(0, 30, 0), Turn(30, 90, 1)])
    assert [s.speaker for s in result["segments"]] == ["Speaker 2", "You", "Others"]   # 2nd voice spoke most

def test_off_by_default_never_calls_the_diarizer():
    assemble(CHANNELS, LONG_BOTH, diarize=True, language="en", turns_for=_must_not_be_called)

def test_in_person_meeting_splits_the_microphone():
    channels = {YOU: [S(1, "host"), S(70, "guest")], OTHERS: []}
    result = assemble(channels, {YOU: LONG, OTHERS: SILENT_LONG}, diarize=True, language="en",
                      options=ON, turns_for=lambda label, audio: [Turn(0, 60, 0), Turn(60, 90, 1)])
    assert [s.speaker for s in result["segments"]] == ["You", "Speaker 2"]

def test_missing_extra_keeps_labels_and_says_what_to_install():       # Review Focus 1
    def unavailable(_label, _audio):
        raise DiarizerUnavailable('Speaker separation needs the optional extra: pip install "tinysteno[diarize]"')
    result = assemble(CHANNELS, LONG_BOTH, diarize=True, language="en", options=ON, turns_for=unavailable)
    assert {s.speaker for s in result["segments"]} == {"You", "Others"}
    assert result["warnings"] == ['Speaker separation needs the optional extra: pip install "tinysteno[diarize]"']

def test_any_other_failure_keeps_labels_with_a_generic_warning(): ...
def test_policy_collapse_is_silent(): ...                             # one cluster → no warning, labels unchanged
def test_short_audio_is_not_split(): ...                              # Review Focus 4: 20 s arrays, turns_for not called
```

`LONG` is 120 s of 0.1 at 16 kHz, `SILENT_LONG` the same length of zeros, `LONG_BOTH` maps both labels to `LONG`; `S` is the plan-1 helper.

```python
# tests/test_main.py
def test_speakers_flag_rejects_zero(): ...                            # SystemExit from argparse
def test_create_transcriber_passes_split_options(): ...
```

- [ ] **Step 2: Run to verify failure**: `uv run pytest tests/test_transcript.py tests/test_main.py tests/test_orchestrator.py -q`.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run to verify pass**: `uv run pytest -q -W error && uv run pylint $(git ls-files '*.py')`. Also confirm the lazy import: `uv run python -c "import sys, tinysteno.main, tinysteno.transcript; assert 'sherpa_onnx' not in sys.modules"`.
- [ ] **Step 5: Commit**: `git commit -m "feat: optional speaker split of the remote channel, off by default"`

---

### Task 5: Speaker labels from the WhisperX server

**Files:**
- Modify: `tinysteno/api_transcriber.py`, `tinysteno/main.py` (`load_config` defaults, `create_transcriber`)
- Test: `tests/test_api_transcriber.py`

**Interfaces:**
- Consumes: `Turn`, the `turns_for` hook of `assemble` (Task 4)
- Produces:
  - Config keys `whisper_api_diarize: false` and `whisper_api_timeout: 600` (seconds; replaces the fixed 180). `ApiTranscriber(api_key, base_url, model, prompt="", options=None, server_diarize: bool = False, timeout: float = 600.0)`.
  - With `server_diarize` and `options.speaker_split` both on, the upload of the channel that will be split (same choice rule as Task 4) carries `extra_body={"diarize": True}`, plus `"max_speakers": N` when a hint was given. No other upload carries these fields, and none do when either switch is off.
  - Segments from that upload are read with their `speaker` field (for example `SPEAKER_01`); `_decode` returns them alongside the segments, and `transcribe` passes `turns_for` = a function returning one `Turn` per labelled segment, with each distinct server label mapped to an integer in order of first appearance. The local diarizer is not called.
  - If that upload fails with an HTTP 400 or 422, it is retried once without the extra fields and the warning `"The transcription server rejected speaker separation; channel labels were used."` is added. If the server answers but no segment has a `speaker`, the labels stay as they are, silently.

- [ ] **Step 1: Write the failing tests**

```python
def test_diarize_field_is_sent_on_the_remote_channel_only(mock_openai, tmp_path):
    _responses(mock_openai, left=[(5.0, 6.0, "me")],            # (start, end, text[, speaker])
               right=[(1.0, 31.0, "a", "SPEAKER_00"), (40.0, 100.0, "b", "SPEAKER_01")])
    result = _run(tmp_path, stereo=True, diarize=True, long=True, server_diarize=True,
                  options=TranscriptOptions(speaker_split=True, max_speakers=4))
    calls = mock_openai.return_value.audio.transcriptions.create.call_args_list
    assert "extra_body" not in calls[0].kwargs
    assert calls[1].kwargs["extra_body"] == {"diarize": True, "max_speakers": 4}
    assert [s.speaker for s in result["segments"]] == ["Speaker 2", "You", "Others"]   # SPEAKER_01 spoke most

def test_nothing_extra_is_sent_when_the_switch_is_off(mock_openai, tmp_path): ...

def test_server_that_rejects_the_field_still_yields_a_transcript(mock_openai, tmp_path):   # Review Focus 5
    # second upload raises openai.BadRequestError once, then succeeds without extra_body
    ...
    assert {s.speaker for s in result["segments"]} == {"You", "Others"}
    assert result["warnings"] == ["The transcription server rejected speaker separation; channel labels were used."]

def test_server_without_speaker_labels_changes_nothing(mock_openai, tmp_path): ...
def test_timeout_comes_from_config(mock_openai): ...                  # OpenAI(..., timeout=600.0)
```

- [ ] **Step 2: Run to verify failure**: `uv run pytest tests/test_api_transcriber.py -q`.
- [ ] **Step 3: Implement.** Update `test_init`, which pins `timeout=180.0`.
- [ ] **Step 4: Run to verify pass**: `uv run pytest -q -W error && uv run pylint tinysteno/api_transcriber.py tests/test_api_transcriber.py`.
- [ ] **Step 5: Commit**: `git commit -m "feat(api): use a server's speaker labels to split the remote channel"`

---

### Task 6: Documentation

**Files:**
- Modify: `README.md`, `SPEC.md`, `CLAUDE.md`, `TODO.md`

- [ ] **Step 1: `README.md`**: a "Speaker split (optional)" section covering: what it does and that it is a best-effort split, off by default; `pip install "tinysteno[diarize]"` and the roughly 40 MB of models fetched on first use into `~/.tinysteno/models/diarize/`; `speaker_split`, `--speakers N`, `whisper_api_diarize`, `whisper_api_timeout`; the rules a user will notice (only the remote side is split on a call; the microphone is split when no system audio was captured; a one-to-one call stays `You` / `Others`; any failure falls back to channel labels with a warning in the note); and that local separation adds processing time after the meeting.
- [ ] **Step 2: `SPEC.md`**: entries for `tinysteno/speaker_split.py` and `tinysteno/sherpa_diarizer.py`; update `transcript.py` and the API transcriber.
- [ ] **Step 3: `CLAUDE.md`**: under Notes, add that `sherpa_onnx` is optional and imported lazily, and that the policy constants in `speaker_split.py` are deliberate.
- [ ] **Step 4: `TODO.md`**: add unticked items for what this plan left out: speaker naming with human confirmation, word-level splitting, a Parakeet backend, and measuring the split against hand-labelled recordings.
- [ ] **Step 5: Verify and commit**: `uv run pytest -q -W error && uv run pylint $(git ls-files '*.py')`, then `git commit -m "docs: optional speaker split"`.
