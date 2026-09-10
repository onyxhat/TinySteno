# TinySteno Architecture

## Summary

TinySteno is a minimal meeting recorder that captures audio (microphone + system audio), transcribes it via Whisper, summarizes the transcript through a configurable LLM, and exports the result as Markdown notes into an Obsidian vault. It follows a linear pipeline architecture: **Record -> Transcribe -> Orchestrate (LLM) -> Export**.

- **Language:** Python 3.12+
- **Package manager:** uv
- **Lines of code:** ~1,100 core + ~1,200 tests
- **Modules:** 7 core modules, 7 built-in personas
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
- `main()` -- argparse CLI dispatcher. Subcommands: `record`, `process`, `list`, `config`, `setup`, `test`
- `load_config()` -- loads/saves `~/.tinysteno/config.yaml`. Seeds built-in personas on first run.
- `cmd_record()` -- record audio, then immediately transcribe + summarize + export
- `cmd_process()` -- transcribe + summarize + export a pre-recorded WAV
- `cmd_setup()` -- interactive wizard to set config values
- `process_audio()` -- core pipeline: transcribe -> summarize -> export, shared by record and process

**Configuration** lives at `~/.tinysteno/config.yaml`:
- `obsidian_vault`, `output_folder` -- Obsync output destination
- `api_key`, `base_url`, `model` -- LLM provider (default: ollama/llama3.2:3b)
- `whisper_model` -- model size for transcription (default: small)
- `diarization` -- stereo-based speaker diarization toggle
- `auto_title`, `auto_tags` -- LLM-generated title and tags
- `sample_rate`, `channels` -- audio capture settings
- `persona` -- active persona slug

### 2. `tinysteno/recorder.py` -- Audio Recording

**Class:** `AudioRecorder`

Captures microphone audio via `sounddevice` (PortAudio) and, on macOS, system audio loopback via ScreenCaptureKit. Output is a 16-bit PCM WAV file.

**Key behaviors:**
- Stereo output when loopback available: left = mic, right = system audio (enables diarization)
- Mono output when loopback unavailable
- Generates timestamped filenames: `meeting_20250801_093000.wav`
- Fallback handling: macOS uses ScreenCaptureKit via `_macos_loopback.py`; Windows uses WASAPI loopback; Linux uses PulseAudio/PipeWire monitor
- Records into `recordings/` subdirectory (configurable)

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
    ├── test_recorder.py        # Recorder tests
    └── test_transcriber.py     # Transcriber tests
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