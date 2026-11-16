# Transcript Correctness Implementation Plan (1 of 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every recording produce a complete, correctly ordered transcript and a summary that never loses data silently: both channels are always transcribed, both backends interleave, and failures are reported in the note instead of being swallowed.

**Architecture:** A new UI-free `tinysteno/transcript.py` owns the `Segment` type and one `assemble()` function that turns per-channel segments into the transcriber result. Both transcription backends decode each channel once and hand their segments to it, so they cannot drift apart again. Warnings travel from transcriber and orchestrator through `PipelineResult` into the note and the UI.

**Tech Stack:** Python 3.12, uv, faster-whisper, openai SDK, numpy, pytest, pylint 4.0.5.

**Spec:** `docs/reports/AI note taker diarization comparison.md`, rows 1-5 of "A prioritised adoption plan", with the section "TinySteno's own code hides defects…" as the defect list. Plans 2 (`2026-11-16-transcript-quality-and-prompts.md`, rows 6-10) and 3 (`2026-11-16-speaker-split.md`, rows 11-12) build on this one; rows 13-14 are deliberately not planned.

## Global Constraints

- Run tools as `uv run pytest` and `uv run pylint $(git ls-files '*.py')`; both must be clean, with `-W error` for pytest. Max line length 100.
- Do not "fix" `import-outside-toplevel` or `broad-exception-caught`; both are disabled on purpose.
- No test may open an audio device, load a Whisper model, or call a network API: patch `tinysteno.transcriber.WhisperModel`, `tinysteno.api_transcriber.OpenAI` and `tinysteno.recorder.sd` as the existing tests do. Never call `tinysteno.main.main()` or `cmd_record` without stubbing `AudioRecorder`.
- Speaker labels stay exactly `You` and `Others`; the labelled transcript format stays `[You] text`, one segment per line.
- The transcriber result keeps its four existing keys (`text`, `diarised_text`, `duration_seconds`, `detected_language`) with unchanged meaning; this plan adds `segments` and `warnings`.
- `tinysteno/tui/*` still must not import `tinysteno.main`; `tinysteno/main.py` must not import `textual` at module level.
- Do not edit `CHANGELOG.md` or the version. Commit messages follow the repo's conventional style.
- openminutes is AGPL-3.0: nothing in this plan copies its code.

## Deliberate deviation from the report

Row 3 says to "raise instead of exporting an all-empty note". This plan instead exports the note with the transcript, empty summary fields and a visible warning. Reason: TinySteno already exports transcript-only notes when no LLM is configured, and raising would discard a transcript that took minutes to produce until plan 2 persists transcripts separately.

## Review Focus

1. **Stereo recording made before this change, one channel very quiet** (joint normalisation left it near silent): the user expects it still transcribed, not skipped as silence. Test in Task 2.
2. **API server that returns `text` but no `segments`**: the user expects a transcript with both speakers, in two blocks, not an empty one. Test in Task 3.
3. **Both channels silent**: the user expects "no speech" and no note, on both backends. Tests in Tasks 2 and 3.
4. **Mono file with diarization turned on**: the user expects a plain transcript and no crash. Test in Task 2.
5. **Two people starting to speak at the same instant**: the user expects a stable order (You first), not one that changes between runs. Test in Task 1.

---

### Task 1: `transcript.py`: segments and assembly

**Files:**
- Create: `tinysteno/transcript.py`
- Test: `tests/test_transcript.py`

**Interfaces:**
- Produces:
  - `YOU = "You"`, `OTHERS = "Others"`, `SILENCE_RMS = 0.001`
  - `@dataclass(frozen=True) class Segment`: `start: float`, `end: float`, `text: str`, `speaker: str = ""`
  - `is_silent(samples: np.ndarray) -> bool`: RMS of the samples below `SILENCE_RMS`; an empty array is silent
  - `interleave(segments: Iterable[Segment]) -> list[Segment]`: sorted by `start`; ties keep `You` before `Others`, then input order
  - `plain_text(segments) -> str`: segment texts joined with single spaces
  - `labelled_text(segments) -> str | None`: `[speaker] text` per line; `None` when there are no segments or none has a speaker
  - `assemble(channels: dict[str, list[Segment]], audio: dict[str, np.ndarray], *, diarize: bool, language: str) -> dict`: `channels` and `audio` are keyed by speaker label (`YOU` / `OTHERS`, or `""` for a mono file) and `audio` holds the 16 kHz mono array of each. Returns the transcriber result: `text`, `diarised_text` (labelled text when `diarize` and any segment has a speaker, else `None`), `duration_seconds` (longest audio array / 16000), `detected_language`, `segments` (interleaved), `warnings`.
  - Warning rule inside `assemble`: for each labelled channel whose audio is not silent but which has no segments, append `"Microphone audio was present but no speech was recognised."` (for `YOU`) or `"System audio was present but no speech was recognised."` (for `OTHERS`).

- [ ] **Step 1: Write the failing tests**

```python
def test_interleave_orders_by_start_and_puts_you_first_on_ties():
    segs = [S(2.0, "b", OTHERS), S(1.0, "a", OTHERS), S(1.0, "c", YOU)]
    assert [s.text for s in interleave(segs)] == ["c", "a", "b"]

def test_assemble_stereo_with_labels():
    result = assemble({YOU: [S(0.0, "hi")], OTHERS: [S(1.0, "hello")]}, LOUD_BOTH,
                      diarize=True, language="en")
    assert result["text"] == "hi hello"
    assert result["diarised_text"] == "[You] hi\n[Others] hello"
    assert result["duration_seconds"] == 2.0 and result["warnings"] == []

def test_assemble_without_diarize_still_includes_both_channels():
    result = assemble({YOU: [S(0.0, "hi")], OTHERS: [S(1.0, "hello")]}, LOUD_BOTH,
                      diarize=False, language="en")
    assert result["text"] == "hi hello" and result["diarised_text"] is None

def test_assemble_nothing_recognised_is_empty_not_labels():
    result = assemble({YOU: [], OTHERS: []}, SILENT_BOTH, diarize=True, language="en")
    assert result["text"] == "" and result["diarised_text"] is None and result["warnings"] == []

def test_assemble_warns_when_a_loud_channel_produced_no_text():
    result = assemble({YOU: [S(0.0, "hi")], OTHERS: []}, LOUD_BOTH, diarize=True, language="en")
    assert result["warnings"] == ["System audio was present but no speech was recognised."]

def test_assemble_mono_has_no_labels_even_with_diarize():
    result = assemble({"": [S(0.0, "hi")]}, {"": LOUD}, diarize=True, language="en")
    assert result["text"] == "hi" and result["diarised_text"] is None

@pytest.mark.parametrize("level, silent", [(0.0, True), (0.0005, True), (0.002, False)])
def test_is_silent(level, silent): ...
```

`S(start, text, speaker="")` is a test helper building `Segment(start, start + 1.0, text, speaker)` (the speaker is set by `assemble` from the dict key when the segment's own is empty); `LOUD` is 32,000 samples of 0.1; `LOUD_BOTH` / `SILENT_BOTH` map both labels to loud / all-zero arrays of that length.

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_transcript.py -q`
Expected: collection error, `No module named 'tinysteno.transcript'`

- [ ] **Step 3: Implement `tinysteno/transcript.py`** with the interface above. `assemble` stamps each segment with its channel's label when the segment has none.

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_transcript.py -q -W error && uv run pylint tinysteno/transcript.py tests/test_transcript.py`
Expected: all pass; 10.00/10

- [ ] **Step 5: Commit**

```bash
git add tinysteno/transcript.py tests/test_transcript.py
git commit -m "feat: transcript segments and a single assembly path for both backends"
```

---

### Task 2: Local backend decodes every channel once

**Files:**
- Modify: `tinysteno/transcriber.py` (`WhisperTranscriber.transcribe`, `_run_whisper`, `_run_whisper_segments`, `_diarize`; `Transcriber` protocol docstring)
- Test: `tests/test_transcriber.py`

**Interfaces:**
- Consumes: `Segment`, `YOU`, `OTHERS`, `is_silent`, `assemble` (Task 1)
- Produces:
  - `split_channels(data: np.ndarray, sr: int) -> dict[str, np.ndarray]` in `tinysteno/transcriber.py`: `{"": mono16k}` for 1-channel audio, `{YOU: left16k, OTHERS: right16k}` for 2 or more channels (first two used). Task 3 imports it.
  - `WhisperTranscriber._decode(audio: np.ndarray, on_progress) -> tuple[list[Segment], str]`: one Whisper pass returning segments (with `start`, `end`, stripped non-empty `text`) and the detected language. It replaces `_run_whisper`, `_run_whisper_segments` and `_diarize`.
- Behaviour:
  - A stereo file is decoded once per channel whether or not `diarize` is set; a channel for which `is_silent` is true is not decoded and contributes no segments.
  - Progress is continuous across the passes: with two channels to decode, the first reports 0.0-0.5 and the second 0.5-1.0; a final `1.0` is always sent.
  - Language is the microphone channel's when it produced segments, otherwise the other channel's, otherwise `"en"`.
  - The result comes from `assemble`.

- [ ] **Step 1: Write the failing tests** (follow the file's existing pattern of patching `tinysteno.transcriber.WhisperModel` and writing a WAV with `soundfile` into `tmp_path`; the fake model's `transcribe` returns segments chosen by which array it was given)

```python
def test_stereo_without_diarize_transcribes_both_channels(tmp_path):
    result = _transcribe(tmp_path, left="hello from me", right="hello from them", diarize=False)
    assert result["text"] == "hello from me hello from them"
    assert result["diarised_text"] is None

def test_stereo_with_diarize_interleaves_by_time_in_two_passes(tmp_path):
    result, model = _transcribe(tmp_path, left=[(5.0, "second")], right=[(1.0, "first")],
                                diarize=True, return_model=True)
    assert result["diarised_text"] == "[Others] first\n[You] second"
    assert model.transcribe.call_count == 2          # was three

def test_silent_channel_is_not_decoded(tmp_path):
    result, model = _transcribe(tmp_path, left=[(0.0, "only me")], right_silent=True,
                                diarize=True, return_model=True)
    assert model.transcribe.call_count == 1 and result["warnings"] == []

def test_quiet_but_not_silent_channel_is_still_decoded(tmp_path):   # Review Focus 1
    _result, model = _transcribe(tmp_path, left=[(0.0, "me")], right=[(1.0, "them")],
                                 right_level=0.004, diarize=True, return_model=True)
    assert model.transcribe.call_count == 2

def test_both_channels_silent_gives_no_speech(tmp_path):            # Review Focus 3
    result = _transcribe(tmp_path, left_silent=True, right_silent=True, diarize=True)
    assert result["text"] == "" and result["diarised_text"] is None

def test_mono_with_diarize_is_plain(tmp_path):                      # Review Focus 4
    result = _transcribe(tmp_path, mono=[(0.0, "just me")], diarize=True)
    assert result["text"] == "just me" and result["diarised_text"] is None

def test_progress_is_monotonic_across_both_passes(tmp_path):
    seen = []
    _transcribe(tmp_path, left=[(0.0, "a"), (1.0, "b")], right=[(0.5, "c")],
                diarize=True, on_progress=seen.append)
    assert seen == sorted(seen) and seen[-1] == 1.0 and any(0.5 <= p < 1.0 for p in seen)
```

`_transcribe(...)` is a new helper in the test file that builds the WAV and the fake model from those keyword arguments. Update or delete existing tests that pin `_run_whisper`'s signature (`test_run_whisper_accepts_array`, `test_transcribe_calls_progress_callback`) so they assert the same behaviour through `_decode` / `transcribe`.

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_transcriber.py -q`
Expected: the new tests fail (`text` holds only the left channel; three `transcribe` calls)

- [ ] **Step 3: Implement** `split_channels`, `_decode` and the new `transcribe` body; delete `_run_whisper`, `_run_whisper_segments`, `_diarize`. `convert_to_16khz_array` keeps its signature and is applied per channel.

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_transcriber.py tests/test_pipeline.py tests/test_main.py -q -W error && uv run pylint tinysteno/transcriber.py tests/test_transcriber.py`
Expected: all pass; 10.00/10

- [ ] **Step 5: Commit**

```bash
git add tinysteno/transcriber.py tests/test_transcriber.py
git commit -m "fix(transcribe): transcribe both channels of a stereo recording, once each"
```

---

### Task 3: API backend reads segments and interleaves

**Files:**
- Modify: `tinysteno/api_transcriber.py` (`transcribe`, `_transcribe_diarized`, `_transcribe_array`, `_parse_response`)
- Modify: `README.md` (the API backend feature list and the diarization example text near lines 339-345 and 398-403)
- Test: `tests/test_api_transcriber.py`

**Interfaces:**
- Consumes: `Segment`, `YOU`, `OTHERS`, `is_silent`, `assemble` (Task 1); `split_channels` (Task 2)
- Produces: `ApiTranscriber._decode(audio: np.ndarray) -> tuple[list[Segment], str]`: one upload returning segments and language. Segments come from the response's `segments` (each has `start`, `end`, `text`; accept attribute or dict access; skip empty text). When the response has text but no usable segments, return a single `Segment(0.0, duration, text)`.
- Behaviour: identical channel rules to Task 2 (both channels always, silent channels not uploaded, language preference, result from `assemble`). Progress: `0.0` at the start, `0.5` between two uploads, `1.0` at the end.

- [ ] **Step 1: Write the failing tests** (use the file's `mock_openai` fixture; a response is a `MagicMock` with `text`, `language`, `duration`, `segments`)

```python
def test_stereo_diarize_interleaves_segments_from_both_uploads(mock_openai, tmp_path):
    _responses(mock_openai, left=[(5.0, "second")], right=[(1.0, "first")])
    result = _run(tmp_path, stereo=True, diarize=True)
    assert result["diarised_text"] == "[Others] first\n[You] second"

def test_stereo_without_diarize_uploads_both_channels(mock_openai, tmp_path):
    _responses(mock_openai, left=[(0.0, "me")], right=[(1.0, "them")])
    result = _run(tmp_path, stereo=True, diarize=False)
    assert result["text"] == "me them" and result["diarised_text"] is None
    assert mock_openai.return_value.audio.transcriptions.create.call_count == 2

def test_server_without_segments_falls_back_to_one_block_per_channel(mock_openai, tmp_path):
    _responses(mock_openai, left_text="all of me", right_text="all of them", segments=False)
    result = _run(tmp_path, stereo=True, diarize=True)           # Review Focus 2
    assert result["diarised_text"] == "[You] all of me\n[Others] all of them"

def test_silent_stereo_recording_is_no_speech(mock_openai, tmp_path):  # Review Focus 3
    result = _run(tmp_path, stereo=True, silent=True, diarize=True)
    assert result["text"] == "" and result["diarised_text"] is None
    mock_openai.return_value.audio.transcriptions.create.assert_not_called()

def test_segments_given_as_dicts_are_accepted(mock_openai, tmp_path): ...
```

Update `test_transcribe_stereo_diarize`, whose expected string was the old two-block format with a blank line between.

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_api_transcriber.py -q`
Expected: the new tests fail (blocks not interleaved; one upload when not diarizing; `"[You] \n\n[Others] "` for silence)

- [ ] **Step 3: Implement** `_decode` and the new `transcribe`; delete `_transcribe_diarized` and `_parse_response`. The request itself is unchanged (`response_format="verbose_json"`, no new fields).

- [ ] **Step 4: Correct the README**: the API backend now interleaves like the local one when the server returns segments, and falls back to one block per speaker when it does not; both backends transcribe system audio whether or not `diarization` is on (the setting only controls the `[You]` / `[Others]` labels).

- [ ] **Step 5: Run to verify pass**

Run: `uv run pytest tests/test_api_transcriber.py -q -W error && uv run pylint tinysteno/api_transcriber.py tests/test_api_transcriber.py`
Expected: all pass; 10.00/10

- [ ] **Step 6: Commit**

```bash
git add tinysteno/api_transcriber.py tests/test_api_transcriber.py README.md
git commit -m "fix(api): interleave speakers from returned segments; always upload both channels"
```

---

### Task 4: Recorder writes channels that mean what the transcriber assumes

**Files:**
- Modify: `tinysteno/recorder.py` (`stop` normalisation block, `_build_audio_data`)
- Modify: `tinysteno/main.py` (`cmd_setup` diarization and channels prompts, about lines 630-660)
- Test: `tests/test_recorder_lifecycle.py`, `tests/test_main.py`

**Interfaces:**
- Produces:
  - `NORMALISE_FLOOR = 0.003` and `_to_int16(audio: np.ndarray) -> np.ndarray` in `tinysteno/recorder.py`: each column is scaled so its own peak is 32767, unless that column's peak is below `NORMALISE_FLOOR`, in which case it is converted at unit gain (a near-silent channel is not amplified into noise).
  - A WAV written by the recorder is 2-channel only when system audio was captured (left = microphone, right = system). With no system audio the microphone is written mono, averaging its channels when `channels: 2`.

- [ ] **Step 1: Write the failing tests** (use the `rec` and `sd` fixtures already in `tests/test_recorder_lifecycle.py`; read the WAV back with `soundfile`)

```python
def test_each_channel_is_normalised_on_its_own(rec, sd):
    rec.start(); rec._has_loopback = True
    rec._write_mic_frame(np.full((1024, 1), 0.02, dtype=np.float32))       # quiet me
    rec._write_loopback_frame(np.full((1024, 1), 0.8, dtype=np.float32))   # loud them
    rec.stop()
    data, _ = sf.read(str(rec.output_path))
    assert abs(data[:, 0]).max() > 0.99 and abs(data[:, 1]).max() > 0.99

def test_a_near_silent_channel_is_not_amplified(rec, sd):
    rec.start(); rec._has_loopback = True
    rec._write_mic_frame(np.full((1024, 1), 0.5, dtype=np.float32))
    rec._write_loopback_frame(np.full((1024, 1), 0.0005, dtype=np.float32))
    rec.stop()
    data, _ = sf.read(str(rec.output_path))
    assert abs(data[:, 1]).max() < 0.001

def test_two_channel_mic_without_system_audio_is_written_mono(tmp_path, sd):
    rec = AudioRecorder(sample_rate=44100, channels=2, recordings_dir=tmp_path / "out")
    sd.query_devices.return_value = {"index": 0, "max_input_channels": 2}
    rec.start()
    rec._write_mic_frame(np.full((1024, 2), 0.25, dtype=np.float32))
    rec.stop()
    assert sf.info(str(rec.output_path)).channels == 1
```

In `tests/test_main.py`, extend the existing `cmd_setup` prompt tests: the diarization prompt's hint no longer contains `channels = 2`, and answering yes to diarization no longer changes the channels default from `1`.

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_recorder_lifecycle.py tests/test_main.py -q`
Expected: the new tests fail

- [ ] **Step 3: Implement** `_to_int16`, the mono downmix in `_build_audio_data` when there is no loopback, and the wizard text. Diarization hint: `"labels come from recording system audio; no microphone setting is needed"`. Channels hint: `"microphone input channels (1 is right for almost every microphone)"`.

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_recorder.py tests/test_recorder_lifecycle.py tests/test_main.py -q -W error && uv run pylint tinysteno/recorder.py tinysteno/main.py`
Expected: all pass; 10.00/10

- [ ] **Step 5: Commit**

```bash
git add tinysteno/recorder.py tinysteno/main.py tests/test_recorder_lifecycle.py tests/test_main.py
git commit -m "fix(recorder): per-channel normalisation; stereo only when system audio was captured"
```

---

### Task 5: Summaries stop losing data silently

**Files:**
- Modify: `tinysteno/orchestrator.py` (`summarize`, `_extract`, `_validate`; new `_merge_locally`)
- Test: `tests/test_orchestrator.py`

**Interfaces:**
- Produces:
  - `Orchestrator.warnings: list[str]`, reset at the start of every `summarize` call.
  - `_extract` makes one retry when the call raises or every field comes back empty.
  - `_validate`: a non-empty string returned for a list field becomes a one-item list.
  - `_merge_locally(partials: list[dict], persona) -> dict`: string fields are the non-empty partial values joined with a blank line; list fields are concatenated with exact duplicates removed, first occurrence kept.
  - Warning texts, exact:
    - whole summary failed (single chunk empty after retry, or every chunk empty): `"The summary could not be generated; this note contains the transcript only."`
    - some chunks failed: `"{failed} of {total} transcript sections could not be summarized; the summary is incomplete."`
    - merge call returned nothing: `"The section summaries could not be merged by the model; they were combined without de-duplication."`
  - `summarize` still returns a dict with every schema field; it never raises for an LLM failure.

- [ ] **Step 1: Write the failing tests** (use the file's `_make_orchestrator` / `_make_persona` helpers and a scripted `side_effect` on `_client.chat.completions.create`)

```python
def test_empty_response_is_retried_once():
    orch = _scripted(["{}", '{"summary": "ok", "items": []}'])
    assert orch.summarize("short", PERSONA)["summary"] == "ok" and orch.warnings == []

def test_failed_summary_warns_and_returns_defaults():
    orch = _scripted([Exception("down"), Exception("down")])
    assert orch.summarize("short", PERSONA) == {"summary": "", "items": []}
    assert orch.warnings == ["The summary could not be generated; this note contains the transcript only."]

def test_failed_chunks_are_counted_in_a_warning():
    # 3 chunks: second fails twice; merge succeeds
    ...
    assert orch.warnings == ["1 of 3 transcript sections could not be summarized; the summary is incomplete."]

def test_failed_merge_keeps_the_partials():
    # 2 chunks succeed with items ["a","b"] and ["b","c"], summaries "one" and "two"; merge returns {} twice
    ...
    assert result == {"summary": "one\n\ntwo", "items": ["a", "b", "c"]}
    assert orch.warnings == ["The section summaries could not be merged by the model; they were combined without de-duplication."]

def test_string_for_a_list_field_becomes_one_item():
    assert _validate({"items": "just one thing"})["items"] == ["just one thing"]

def test_warnings_reset_between_calls(): ...
```

Update `test_summarize_returns_defaults_on_llm_exception` and `test_summarize_all_chunks_fail_returns_defaults` to expect two attempts per call and the warning; update call-count assertions in `test_summarize_single_chunk_makes_one_llm_call` only if they now see a retry (a non-empty first response must still mean exactly one call).

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_orchestrator.py -q`
Expected: the new tests fail (`AttributeError: 'Orchestrator' object has no attribute 'warnings'`)

- [ ] **Step 3: Implement** the changes above.

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_orchestrator.py -q -W error && uv run pylint tinysteno/orchestrator.py tests/test_orchestrator.py`
Expected: all pass; 10.00/10

- [ ] **Step 5: Commit**

```bash
git add tinysteno/orchestrator.py tests/test_orchestrator.py
git commit -m "fix(summary): retry once, keep partial results, and say what was lost"
```

---

### Task 6: Warnings reach the note and the user

**Files:**
- Modify: `tinysteno/pipeline.py` (`PipelineResult`, `run_pipeline`)
- Modify: `tinysteno/obsidian.py` (`export`)
- Modify: `tinysteno/personas/__init__.py` (`_RESERVED_NAMES`)
- Modify: `tinysteno/main.py` (`_process_audio`), `tinysteno/tui/result.py` (`on_mount`)
- Test: `tests/test_pipeline.py`, `tests/test_obsidian.py`, `tests/test_personas.py`, `tests/test_main.py`, `tests/test_tui_result.py`

**Interfaces:**
- Consumes: transcriber result key `warnings` (Tasks 1-3); `Orchestrator.warnings` (Task 5)
- Produces:
  - `PipelineResult.warnings: tuple[str, ...] = ()` (new last field): transcriber warnings first, then the orchestrator's (read with `getattr(orchestrator, "warnings", [])`).
  - Metadata key `processing_warnings: list[str]` passed to the exporter. When non-empty the exporter appends, after the rendered template:

    ```
    
    ## Processing warnings
    
    - {warning}
    ```
    one bullet per warning. Nothing is appended when the list is empty.
  - `_RESERVED_NAMES` gains `generated_tags`, `tags` and `processing_warnings`.
  - Headless `_process_audio` prints `Warning: {text}` for each warning before the `Meeting:` line. The TUI Result screen shows each as a `severity="warning"` notification.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pipeline.py
def test_warnings_from_transcriber_and_orchestrator_reach_result_and_note(tmp_path):
    # FakeTranscriber returns warnings=["System audio was present but no speech was recognised."];
    # orchestrator mock has .warnings = ["1 of 3 transcript sections could not be summarized; the summary is incomplete."]
    ...
    assert result.warnings == (TRANSCRIBER_WARNING, ORCHESTRATOR_WARNING)
    assert exporter_cls.return_value.export.call_args.args[2]["processing_warnings"] == [...]

# tests/test_obsidian.py
def test_processing_warnings_are_appended_to_the_note(tmp_path):
    path = _make_exporter(tmp_path).export({"summary": "s"}, _make_persona(),
                                           {**_make_metadata(), "processing_warnings": ["lost a bit"]})
    assert Path(path).read_text().rstrip().endswith("## Processing warnings\n\n- lost a bit")

def test_no_warnings_section_without_warnings(tmp_path): ...

# tests/test_personas.py
@pytest.mark.parametrize("name", ["generated_tags", "tags", "processing_warnings"])
def test_newly_reserved_field_names_are_rejected(tmp_path, name): ...
```

Plus one test each in `tests/test_main.py` (stdout contains `Warning: …` before `Meeting:`) and `tests/test_tui_result.py` (with `run_test(notifications=True)`, one warning notification per entry).

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_pipeline.py tests/test_obsidian.py tests/test_personas.py -q`
Expected: the new tests fail

- [ ] **Step 3: Implement** the plumbing. Check that no built-in persona schema uses a newly reserved name (`grep -n "tags\|warnings" tinysteno/personas/*/persona.yaml` shows only the top-level `tags:` key, never a schema field).

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest -q -W error && uv run pylint $(git ls-files '*.py')`
Expected: all pass; 10.00/10

- [ ] **Step 5: Commit**

```bash
git add tinysteno tests
git commit -m "feat: processing warnings are written into the note and shown on exit"
```

---

### Task 7: Small defects and documentation drift

**Files:**
- Modify: `tinysteno/obsidian.py` (`export` file naming), `tinysteno/pipeline.py` (`_extract_summary_text`), `tinysteno/personas/rca/system_prompt.md` (the Timeline rule), `pyproject.toml` + `uv.lock` (drop `pywhispercpp`), `README.md`, `SPEC.md`, `CLAUDE.md`
- Test: `tests/test_obsidian.py`, `tests/test_pipeline.py`

**Interfaces:**
- Produces:
  - Export never overwrites: when `<title> (<date>).md` exists the note is written as `<title> (<date>) 2.md`, then ` 3`, and so on.
  - `_extract_summary_text` returns the longest non-empty string field (so the `sprint` persona titles from `overview`, not the one-word `ceremony_type`); the list-only fallback is unchanged.
  - The `rca` Timeline rule reads: `Timeline: one entry per event as 'TIME | EVENT | SOURCE', in the order events happened. Use a time only when the transcript states one; otherwise write 'time not stated'. Prefix approximate times with '~'.`

- [ ] **Step 1: Write the failing tests**

```python
def test_export_does_not_overwrite_an_existing_note(tmp_path):
    exporter = _make_exporter(tmp_path)
    first = exporter.export({"summary": "one"}, _make_persona(), _make_metadata())
    second = exporter.export({"summary": "two"}, _make_persona(), _make_metadata())
    third = exporter.export({"summary": "three"}, _make_persona(), _make_metadata())
    assert Path(first).name.endswith(").md")
    assert Path(second).name.endswith(") 2.md") and Path(third).name.endswith(") 3.md")
    assert "one" in Path(first).read_text()

def test_summary_text_is_the_longest_string_field():
    persona = _persona_with_fields(ceremony_type="string", overview="string")
    assert _extract_summary_text(persona, {"ceremony_type": "Review",
                                           "overview": "We reviewed the sprint."}) \
        == "We reviewed the sprint."
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_obsidian.py tests/test_pipeline.py -q`
Expected: both new tests fail

- [ ] **Step 3: Implement** the two code changes and the `rca` prompt edit.

- [ ] **Step 4: Remove the unused dependency**

Run: `grep -rn "pywhispercpp" tinysteno tests scripts` (expected: no matches), then delete the `pywhispercpp` line from `pyproject.toml` and run `uv lock && uv sync`.

- [ ] **Step 5: Check the Apple Silicon claim before editing it**

Run: `uv run python -c "import ctranslate2; print(ctranslate2.get_supported_compute_types('mps'))"`
If it raises an unsupported-device error, change the README's three "MPS" statements (feature list, config comment, GPU section) to say that Apple Silicon runs faster-whisper on the CPU. If it prints a list, leave the README alone and say so in the commit message.

- [ ] **Step 6: Bring `SPEC.md` and `CLAUDE.md` in line with the code**: new `transcript.py` module entry; transcriber entries (one decode per channel, both channels always, `segments` and `warnings` keys); API backend interleaving and fallback; export collision suffix is ` 2`, ` 3` (the old `_001` claim was never true); summarizer retry, local merge and warnings; remove the claim of programmatic tag de-duplication if the code has none; reserved names list in `CLAUDE.md`.

- [ ] **Step 7: Run to verify pass**

Run: `uv run pytest -q -W error && uv run pylint $(git ls-files '*.py')`
Expected: all pass; 10.00/10

- [ ] **Step 8: Commit**

```bash
git add -A tinysteno tests pyproject.toml uv.lock README.md SPEC.md CLAUDE.md
git commit -m "fix: no note overwrite, better title source, rca timeline; docs match the code"
```
