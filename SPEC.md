# TinySteno Architecture

## Summary

TinySteno is a minimal meeting recorder that captures audio (microphone + system audio), transcribes it via Whisper, summarizes the transcript through a configurable LLM, and exports the result as Markdown notes into an Obsidian vault. It follows a linear pipeline architecture: **Record -> Transcribe -> Orchestrate (LLM) -> Export**.

- **Language:** Python 3.12+
- **Package manager:** uv
- **Lines of code:** ~1,100 core + ~1,200 tests
- **Modules:** 10 core modules (including the `tui/` package), 7 built-in personas
- **CLI entry:** `tinysteno` command (installed via `pip install`)

---

## Pipeline Overview

```
 ┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────────────┐
 │ Recorder  │───>│ Transcriber  │───>│ Orchestrator │───>│ Obsidian      │
 │ (audio)   │    │ (Whisper)    │    │ (LLM)        │    │ Exporter      │
 └──────────┘    └──────────────┘    └──────────────┘    └───────────────┘
       │                │                    │                    │
       │ WAV file       │ text               │ structured dict    │ .md file
       v                v                    v                    v
  recordings/      transcript text       summary dict        Obsidian vault
```

Each stage is independent and communicates through well-defined data contracts (Python dicts). The pipeline is linear but not strictly serialized -- `tinysteno process` can accept a pre-recorded WAV to skip the recording step.

---

## Module Breakdown

### 1. `tinysteno/main.py` -- CLI Entry Point & Config

**Purpose:** Parse CLI args, load config, orchestrate the pipeline, provide interactive setup wizard.

**Key functions:**
- `main()` -- argparse CLI dispatcher. Subcommands: `tui`, `record`, `process`, `list`, `config`, `setup`, `test`. With no subcommand it launches the TUI when stdin and stdout are both terminals, otherwise it prints help
- `load_config()` -- loads/saves `~/.tinysteno/config.yaml`. Seeds built-in personas on first run.
- `cmd_record()` -- record audio, then immediately transcribe + summarize + export. Stops on Ctrl+C or when the session timeout elapses (`--timeout MINUTES`; resolved by `resolve_timeout_seconds`)
- `cmd_tui()` -- imports `tinysteno.tui` lazily, builds a `TuiDeps` (recorder factory, `run_pipeline` bound to the config, persona loaders, recordings directory) and calls `run_tui`. While the app runs, logging goes to `~/.tinysteno/tui.log` instead of the terminal (`log_file.log_to_file`; DEBUG with `--verbose`) and the previous root handlers are restored afterwards. Prints the returned exit message (stderr when the exit code is non-zero), plus a final `Log: <path>` line when the exit code is non-zero or the session logged a warning, and exits with that code. The `tui` subcommand takes `--now`, `--name`, `--persona`, `--timeout` and the transcription override flags
- `cmd_process()` -- transcribe + summarize + export a pre-recorded WAV
- `cmd_setup()` -- interactive wizard to set config values, including the session timeout in minutes
- `_process_audio()` -- shared by `record` and `process`: delegates to `run_pipeline` with a Rich progress reporter. A template (`RuntimeError`) failure at export prints `Error rendering note`; any other stage failure re-raises the underlying exception

**Configuration** lives at `~/.tinysteno/config.yaml`:
- `obsidian_vault`, `output_folder` -- Obsync output destination
- `api_key`, `base_url`, `model` -- LLM provider (default: ollama/llama3.2:3b)
- `whisper_model` -- model size for transcription (default: small)
- `diarization` -- stereo-based speaker diarization toggle
- `auto_title`, `auto_tags` -- LLM-generated title and tags
- `sample_rate`, `channels` -- audio capture settings
- `persona` -- active persona slug
- `session_timeout_minutes` -- maximum recording length in minutes, a finite number `>= 0` (default 75; `0` disables). `inf` and `nan` are rejected here, by `--timeout` and in `persona.yaml`
- `tui_theme` -- optional starting theme for the TUI (a Textual theme name)

### 2. `tinysteno/recorder.py` -- Audio Recording

**Class:** `AudioRecorder`

Captures microphone audio via `sounddevice` (PortAudio) and, on macOS, system audio loopback via ScreenCaptureKit. Output is a 16-bit PCM WAV file.

**Key behaviors:**
- Stereo output when loopback available: left = mic, right = system audio (enables diarization)
- Mono output when loopback unavailable
- Generates timestamped filenames: `meeting_20250801_093000.wav`
- Fallback handling: macOS uses ScreenCaptureKit via `_macos_loopback.py`; Windows uses WASAPI loopback; Linux uses PulseAudio/PipeWire monitor
- Records into `recordings/` subdirectory (configurable)
- Live levels for the TUI: `mic_level` and `system_level` are the RMS (0.0 to 1.0) of the most recent frame from each stream, and `has_loopback` says whether system audio is being captured. Audio threads write them; the UI thread reads them
- Lifecycle is an explicit `RecorderState` (`idle` → `recording` → `idle`, or → `save_failed` when the WAV cannot be written). `start()` refuses to run twice, and a `start()` that fails part-way closes whatever it opened and leaves the recorder idle; `stop()` after a failed save raises instead of reporting an empty recording; `discard()` is the only thing that deletes kept raw audio
- The stereo mix pads the shorter stream with silence, so one capture ending early (or the microphone delivering nothing) never shortens the WAV
- `stop()` returns `False` when no audio was captured. The raw temp files are deleted only after the WAV has been written; if saving fails they are kept, listed in `raw_paths` (a tuple of paths, empty otherwise), and the `RuntimeError` names them
- `loopback_error` holds why macOS system-audio capture failed to start (usually Screen Recording permission), or `None`; the reason is also printed for headless use
- `discard()` stops the streams and throws the audio away: no WAV is written and the temp files are deleted

### 3. `tinysteno/transcriber.py` -- Whisper Transcription

**Class:** `WhisperTranscriber`

Transcribes audio files using `faster-whisper` (CTranslate2-optimized Whisper). Runs on CPU with int8 quantization.

**Key behaviors:**
- Converts input audio to 16kHz mono float32 for Whisper
- Checks stereo before conversion to support diarization
- Module-level model cache `_MODEL_CACHE` -- one shared model instance per (size, device, compute_type)
- `transcribe()` returns: `text`, `diarised_text`, `duration_seconds`, `detected_language`
- `_diarize()` -- splits stereo channels, transcribes left as "You" and right as "Others", interleaves results by timestamp

### 4. `tinysteno/orchestrator.py` -- LLM Orchestration

**Class:** `Orchestrator`

Handles all LLM communication via the OpenAI-compatible API. Transforms raw transcripts into structured data matching a persona's schema.

**Key behaviors:**
- Chunking strategy: splits transcripts over 12,000 characters into overlapping chunks (500-char overlap)
- Summarizes each chunk independently, then merges partial results
- Merge deduplicates: for `list` fields, checks if summary already exists before appending
- Graceful degradation: if LLM returns invalid JSON, attempts recovery. If all chunks fail, returns defaults (empty string / empty list)
- Also generates: titles (`generate_title`) and tags (`generate_tags`) via separate LLM calls with shorter context windows

**Constants:**
- Chunk size: 12,000 chars
- Chunk overlap: 500 chars
- LLM timeout: 120 seconds
- Title context: first 500 chars of overview
- Tag context: first 500 chars of overview

### 5. `tinysteno/obsidian.py` -- Obsidian Export

**Class:** `ObsidianExporter`

Renders meeting notes using Jinja2 templates and writes Markdown files to the Obsidian vault.

**Key behaviors:**
- Renders the persona's `template.md` with merged context (metadata + schema fields)
- Uses Jinja2 `StrictUndefined` -- template references to missing fields raise errors
- Generates YAML frontmatter with title, date, duration, tags, detected_language
- Tag deduplication: persona static tags first, LLM-generated tags appended (duplicates removed)
- Sanitizes filenames (replaces non-alphanumeric with `_`, limits to 100 chars)
- Timestamps to avoid overwrites: appends `_001`, `_002` suffix when file exists

### 6. `tinysteno/personas/__init__.py` -- Persona System

**Dataclass:** `Persona`

Encapsulates all persona configuration: schema definition, system prompt, and Jinja2 template. The persona system allows users to define custom output formats without modifying code.

**Key functions:**
- `load_persona(slug)` -- loads a persona from `~/.tinysteno/personas/<slug>/`
- `list_personas()` -- discovers installed personas
- `seed_builtin_personas()` -- copies built-in personas to user dir on first run

**Validation:**
- Required files: `persona.yaml`, `system_prompt.md`, `template.md`
- `persona.yaml` must declare `name`, `description`, `schema`
- Schema fields validated: only `string` and `list` types allowed
- Reserved field names cannot be overridden: `title`, `date`, `duration`, `transcript`, `detected_language`, `generated_tags`
- Optional `session_timeout_minutes` (number `>= 0`, `0` = no limit) becomes `Persona.session_timeout_minutes` (`None` when absent) and overrides the global recording timeout for that persona

**Built-in personas (7), in `BUILTIN_ORDER`:**

| Slug | Purpose |
|------|---------|
| default | General meeting summary |
| 1on1 | Manager/report 1-on-1: goals, needs, struggles, recent wins, per-side actions |
| rca | Root cause analysis |
| irm | Incident response management |
| sprint | Sprint ceremony (planning / review / retrospective) |
| kickoff | Project kickoff notes |
| executive-summary | Decision-ready executive briefing |

Each persona directory contains:
- `persona.yaml` -- name, tags, description, schema definition
- `system_prompt.md` -- LLM system prompt for extraction
- `template.md` -- Jinja2 template for Obsidian output

### 7. `tinysteno/_macos_loopback.py` -- macOS System Audio

Platform-specific module for capturing system audio on macOS 12.3+ using ScreenCaptureKit via `pyobjc-framework-ScreenCaptureKit`.

**Key behaviors:**
- Uses CoreMedia ctypes wrappers for audio buffer extraction
- Falls back to AVAudioEngine-based capture for macOS versions without ScreenCaptureKit support
- Handles CoreMedia audio buffers with format conversion to float32
- Thread-safe with callback-based audio delivery

### 8. `tinysteno/session.py` -- Recording Session Timeout

A pure countdown with no threads or sleeping. `SessionTimer(limit_seconds, clock=time.monotonic)` reads the injected clock when polled and exposes `start()`, `cancel()` (remove the limit), `extend(seconds=900)`, and the properties `elapsed`, `limit_seconds`, `remaining`, `expired` and `in_warning` (true in the last `TIMEOUT_WARNING_SECONDS` = 60 seconds). `resolve_timeout_seconds(config, persona_minutes, cli_minutes)` picks the limit in seconds by precedence: `--timeout` flag, then the persona's value, then `session_timeout_minutes` from config, then `DEFAULT_SESSION_TIMEOUT_MINUTES` (75); it returns `None` when the chosen value is `0`. `format_clock(seconds)` renders `MM:SS` (or `H:MM:SS` from one hour). Used by both `cmd_record` and the TUI.

### 9. `tinysteno/pipeline.py` -- UI-free Pipeline

`run_pipeline(wav_path, name, config, persona, timestamp, *, transcriber, orchestrator_cls, exporter_cls, reporter=None) -> PipelineResult` runs transcribe, summarize, title and tags, and export, reporting each `Stage` (`TRANSCRIBE`, `SUMMARIZE`, `METADATA`, `EXPORT`) to a `PipelineReporter` (`stage_started`, `stage_progress`, `stage_finished`, `stage_skipped`; `NullReporter` ignores them). A failure in any stage is re-raised as `PipelineError(stage)` with the original exception as `__cause__`. `PipelineResult` carries `wav_path`, `note_path`, `title`, `duration_seconds` and `no_speech`. Nothing here prints or imports a UI library, so the CLI (Rich) and the TUI (Textual) share it.

### 10. `tinysteno/tui/` -- Textual Interface

The only package that imports `textual`; `main.py` imports it lazily inside `cmd_tui`, and nothing under `tui/` imports `tinysteno.main`. Everything with a side effect arrives through the `TuiDeps` dataclass (`make_recorder`, `run_pipeline`, `list_personas`, `load_persona`, `recordings_dir`, `clock`, and `live_updates`, which tests turn off to keep interval timers out of pilot tests).

- `__init__.py` -- `run_tui(config, deps, *, autostart, name, persona, timeout_override) -> tuple[str | None, int]`: runs the app and returns its exit message and exit code. Its `finally` is the safety net: a recorder that was started and never stopped or discarded (the app crashed, or ended some other way) is stopped there and the message says where the WAV is (or, if saving failed, where the raw audio is, with exit code 1).
- `app.py` -- `TinyStenoApp`: pushes Home, applies `tui_theme`, resolves the per-persona timeout (`timeout_seconds_for`), and moves between screens (`start_recording`, `finish_recording`, `process_file`, `show_result`; the last two first close any dialog, such as the command palette, above the screen they replace). Its `action_quit` stops and saves a recording in progress (exit message: the saved path and the `tinysteno process` command; a failed save exits with code 1 and the error as the message), abandons a running pipeline without waiting for it, and gives the same saved-recording message when quitting a Processing screen that ended without a note. SIGHUP and SIGTERM are routed to `action_quit` on the app's event loop while it runs (where the platform has them) and the previous handlers are restored on exit. `recorder_guard` holds the started recorder until the UI has reported its outcome; `final_result()` is what `run_tui` returns. Navigation is logged in `events` for tests.
- `ledger.py` -- `SessionLedger`: what became of every recording this session touched (`saved`, `processed`, `save_failed`). Every exit message and the exit code are read from its `summary()`, on quit, on a signal and after a crash alike, so they do not depend on which screen is showing.
- `guard.py` -- `RecorderGuard`: wraps the started recorder so `stop()` / `discard()` reach it at most once whichever thread gets there first, and `final_message()` for the safety net; `saved_message` and `save_failed_message` build the user-facing text.
- `home.py` -- `HomeScreen`: name and persona on one line each, a hint with the timeout that applies, a one-row Record button and the recent recordings list (`recent_recordings`), which takes the remaining height and scrolls; fits 80x24. An unknown `--persona` falls back to the configured persona (else the first) with an error toast; with no personas at all the hint says to run `tinysteno setup` and Record is disabled.
- `recording.py` -- `RecordingScreen` and the `ConfirmDiscard` modal (with its own footer: `y` discard, `n`/`Esc` keep recording, `q` quit and save): the clock, one `LevelMeter` per stream (Mic, System; the System row reads "System audio unavailable" when there is no loopback, and the recorder's `loopback_error` is shown once as a warning toast), the auto-stop countdown with a warning in the last 60 seconds, and the `k`, `+`, `d` keys. Stop keys are ignored with a `Still starting…` notice until `recorder.start()` has returned. `recorder.start()` and `stop()` run in thread workers; `tick()` is one UI refresh.
- `meter.py` -- `LevelMeter`: a two-row, fixed-scale meter (-60 dBFS empty to 0 dBFS full, 16 steps, colour from `$success` through `$warning` to `$error`) with 120 samples of history in `data`, newest at the right; `level_fraction`, `level_steps`, `column_steps` and `meter_rows` are the pure parts.
- `processing.py` -- `ProcessingScreen`: a row per stage (`-pending`, `-active`, `-done`, `-skipped`, `-failed`), a transcription progress bar, retry and home; `check_action` shows `r` only after a failed stage and `n` only once the pipeline has stopped. The pipeline runs in a daemon thread because it cannot be cancelled.
- `result.py` -- `ResultScreen`: renders the exported note (front matter removed) with `Markdown`, and opens it in Obsidian through an `obsidian://open?path=` URL (`obsidian_url`).
- `styles.tcss` -- shared styles and the warning transition (the border animates; the panel background changes in one step, because Textual 8.2 does not repaint children while an ancestor's background transitions).

---

## Data Flow (Detailed)

```
User runs: tinysteno record --persona rca

1. CLI (main.py:main)
   ├── Parse args (record subcommand)
   ├── Load config (~/.tinysteno/config.yaml)
   ├── Load persona (rca)
   └── Call process_audio()

2. Recorder (recorder.py:AudioRecorder)
   ├── Detect audio devices
   ├── Start mic + loopback capture
   ├── Write stereo WAV to recordings/
   └── Return WAV path

3. Transcriber (transcriber.py:WhisperTranscriber)
   ├── Read WAV file via soundfile
   ├── Convert to 16kHz mono
   ├── Run faster-whisper (CPU, int8)
   ├── Diarize if stereo + enabled
   └── Return {text, diarised_text, duration, language}

4. Orchestrator (orchestrator.py:Orchestrator)
   ├── Chunk transcript if > 12k chars
   ├── Build user message (persona system prompt + transcript)
   ├── Call LLM (OpenAI-compatible API)
   ├── Parse JSON response, validate against schema
   ├── Merge partial results if chunked
   ├── Generate title (separate LLM call)
   ├── Generate tags (separate LLM call)
   └── Return {schema_fields, title, generated_tags}

5. Exporter (obsidian.py:ObsidianExporter)
   ├── Merge metadata + schema data into context
   ├── Render persona's Jinja2 template
   ├── Generate YAML frontmatter
   ├── Write .md file to Obsidian vault
   └── Print file path to console
```

---

## Dependency Graph

```
tinysteno.main
  ├── tinysteno.recorder
  │     ├── sounddevice          (mic capture via PortAudio)
  │     ├── numpy                (audio buffer manipulation)
  │     └── tinysteno._macos_loopback  (macOS ScreenCaptureKit)
  ├── tinysteno.transcriber
  │     ├── faster-whisper       (Whisper transcription)
  │     ├── soundfile            (WAV I/O)
  │     ├── numpy
  │     └── scipy                (audio resampling)
  ├── tinysteno.orchestrator
  │     ├── openai               (LLM API client)
  │     └── tinysteno.personas
  ├── tinysteno.obsidian
  │     ├── jinja2               (template rendering)
  │     └── tinysteno.personas
  └── tinysteno.personas
        ├── pyyaml               (persona.yaml parsing)
        └── (dataclasses, pathlib)
```

---

## Configuration

User config stored at `~/.tinysteno/config.yaml` (auto-created with defaults):

```yaml
obsidian_vault: ~/Obsidian/Vault
output_folder: meetings
api_key: ollama
base_url: http://localhost:11434/v1
model: llama3.2:3b
whisper_model: small
diarization: false
auto_title: true
auto_tags: true
sample_rate: 44100
channels: 1
persona: default
session_timeout_minutes: 75
# tui_theme: tokyo-night   # optional
```

Personas are discovered from `~/.tinysteno/personas/<slug>/`. Built-in personas are seeded automatically on first run.

---

## Testing

Tests live in `tests/` and use `pytest` with `pylint` linting (CI via GitHub Actions). Each core module has a dedicated test file:

| Test File | Module Under Test | Key Coverage |
|-----------|------------------|--------------|
| `test_main.py` | CLI commands, config loading, pipeline orchestration | Argument parsing, config seeding, process_audio integration |
| `test_recorder.py` | AudioRecorder | Device detection, recording lifecycle, path generation |
| `test_transcriber.py` | WhisperTranscriber | Transcription, diarization, resampling |
| `test_orchestrator.py` | Orchestrator | Chunking, LLM calls, merge logic, error handling |
| `test_obsidian.py` | ObsidianExporter | Template rendering, file writing, frontmatter |
| `test_personas.py` | Persona loading/validation | Directory validation, schema checking, error cases |
| `test_session.py` | `SessionTimer`, `resolve_timeout_seconds` | Countdown, warning window, cancel/extend, precedence |
| `test_pipeline.py` | `run_pipeline` | Stage reporting, skipped stages, `PipelineError` |
| `test_tui_*.py` (`tui_fakes.py` holds the doubles) | `tinysteno/tui/` | Home, Recording, Processing and Result screens, driven with `App.run_test()` |

---

## Directory Structure

```
tinysteno/
├── pyproject.toml              # Project config, dependencies, entry point
├── AGENTS.md                   # GitNexus code intelligence rules
├── CLAUDE.md                   # Performance fixes implementation plan
├── PLAN.md                     # Architecture and implementation plan
├── README.md                   # User-facing documentation
├── TODO.md                     # Known issues and remaining work
├── SPEC.md                     # This file
├── graphify-out/               # Knowledge graph outputs
│   ├── graph.html              # Interactive architecture graph
│   ├── GRAPH_REPORT.md         # Graph analysis report
│   └── graph.json              # Raw graph data
├── tinysteno/
│   ├── __init__.py             # Package version (0.2.0)
│   ├── main.py                 # CLI entry point, config, pipeline orchestration
│   ├── recorder.py             # Audio capture (mic + loopback)
│   ├── transcriber.py          # Whisper transcription
│   ├── orchestrator.py         # LLM orchestration and summarization
│   ├── obsidian.py             # Obsidian Markdown export
│   ├── _macos_loopback.py      # macOS ScreenCaptureKit audio capture
│   ├── session.py              # Session timeout countdown and resolution
│   ├── pipeline.py             # UI-free transcribe -> summarize -> export with stage reporting
│   ├── log_file.py             # Logging to ~/.tinysteno/tui.log while the TUI runs
│   ├── signals.py              # Hold SIGHUP/SIGTERM back while a recording is being saved
│   ├── tui/                    # Textual interface (app, guard, ledger, home, recording, meter, processing, result)
│   └── personas/
│       ├── __init__.py         # Persona loading, validation, discovery
│       ├── default/            # Built-in personas (each has: persona.yaml, system_prompt.md, template.md)
│       ├── 1on1/
│       ├── executive-summary/
│       ├── irm/
│       ├── kickoff/
│       ├── leadership/
│       ├── meeting/
│       ├── rca/
│       └── sprint/
└── tests/
    ├── __init__.py
    ├── conftest.py             # Shared fixtures
    ├── test_main.py            # CLI tests
    ├── test_obsidian.py        # Exporter tests
    ├── test_orchestrator.py    # LLM orchestration tests
    ├── test_personas.py        # Persona loading tests
    ├── test_pipeline.py        # Pipeline tests
    ├── test_recorder.py        # Recorder tests
    ├── test_session.py         # Session timeout tests
    ├── test_transcriber.py     # Transcriber tests
    └── test_tui_*.py           # TUI screen tests (pilot-driven)
```

---

## Design Decisions and Rationale

**Why linear pipeline instead of event-driven?**
The recording -> transcription -> summarization -> export flow is inherently sequential. Each stage depends on the previous stage's output. An event-driven architecture would add complexity without benefit for this use case.

**Why persona system over fixed output formats?**
Users need different meeting note formats (RCA, 1-on-1, sprint retrospective). Hard-coding these would require code changes. The persona system makes the output format data-driven: each persona is a directory with a schema, system prompt, and Jinja2 template. Adding a new persona requires zero code changes.

**Why chunked LLM processing?**
Whisper transcripts of long meetings (1+ hours) can exceed 50,000 characters, beyond most LLM context windows for structured extraction. The chunking strategy splits transcripts into overlapping 12,000-char segments, extracts from each, then merges. The merge step deduplicates list fields to avoid repeated items.

**Why faster-whisper over openai-whisper?**
CTranslate2-optimized Whisper is 4x faster on CPU with int8 quantization, making real-time or near-real-time transcription viable on consumer hardware without a GPU.

**Why stereo loopback for diarization?**
True speaker diarization requires an ML model. By placing mic on the left channel and system audio on the right, the transcriber can separate "You" from "Others" without a diarization model. This is a pragmatic trade-off: it works well when the user speaks through their mic and others speak through the call (Zoom/Teams).

---

## Knowledge Graph (graphify)

An interactive architecture graph is available at `graphify-out/graph.html`. The graph encodes all code relationships (imports, calls, references, conceptual links) extracted from the codebase. Key findings from the analysis:

- **18 communities** identified by community detection, aligning closely with module boundaries
- **375 nodes** and **727 edges** capturing structural and semantic relationships
- **God nodes** (most connected): `_make_orchestrator()` (35), `Orchestrator` (24), `AudioRecorder` (24), `load_persona()` (21), `Persona` (18)
- **Bridge nodes** cross module boundaries: `Orchestrator` connects the orchestrator module to CLI and tests; `AudioRecorder` connects recording to dependencies, CLI, and macOS loopback; `Persona` connects the persona system to CLI, orchestrator tests, and obsidian export
- **Cohesion warning:** The `Main CLI & Setup` community has low cohesion (0.054) -- nodes in this community are weakly interconnected, suggesting the CLI module may benefit from splitting as it grows