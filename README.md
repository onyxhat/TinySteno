# TinySteno

Inspired by [StenoAI](https://github.com/ruzin/stenoai).
Minimal meeting recorder with Obsidian export (local filesystem vault).

## Features

- Record meetings (indefinite duration, Ctrl+C to stop)
- Captures mic + system audio simultaneously — no virtual audio device required
- Transcribe with faster-whisper (local CPU/GPU) or OpenAI Whisper API
- GPU auto-detection — CUDA, Apple Silicon (MPS), and CPU fallback
- Stereo diarization — speaker labels when system audio is captured
- Summarize via OpenAI-compatible API (Ollama, OpenAI, OpenRouter, etc.)
- Handles long transcripts automatically — splits into chunks, summarizes each, then merges
- Persona system — choose how recordings are summarized and formatted
- Auto-generate note titles and tags from content
- Export structured markdown notes to Obsidian vault

## Prerequisites
- [Python 3.12+](https://www.python.org/downloads/)
- [UV](https://docs.astral.sh/uv/getting-started/installation/)
- An [OpenAI](https://platform.openai.com/) Compatible API
  - [Ollama](https://ollama.com)
  - [LMStudio](https://lmstudio.ai)
  - [OpenRouter](https://openrouter.ai)
  - MORE!

## Installation

**UV:**
```bash
uv tool install git+https://github.com/onyxhat/TinySteno
```

**Manually:**
```bash
git clone https://github.com/onyxhat/TinySteno.git
cd TinySteno
uv sync
uv run tinysteno --help
```

## Usage

```bash
# Update config interactively (also seeds built-in personas to ~/.tinysteno/personas/)
tinysteno setup

# Reset built-in personas to defaults (overwrites any edits you've made to them)
tinysteno setup --reset-personas

# Record a meeting (uses default persona)
tinysteno record

# Record with custom name
tinysteno record --name "Budget Review"

# Record using a specific persona
tinysteno record --persona rca

# Record with Whisper API backend
tinysteno record --backend api --whisper-api-key sk-...

# Record with explicit GPU device override
tinysteno record --whisper-device cuda --whisper-compute-type float16

# Process existing audio file
tinysteno process recordings/Meeting.wav

# Process with API backend and specific model
tinysteno process recordings/Meeting.wav --backend api --whisper-api-model whisper-large-v3

# Process with a specific persona
tinysteno process recordings/Meeting.wav --persona executive-summary

# Verify setup
tinysteno test

# List processed meetings
tinysteno list --vault /path/to/vault

# Show current config
tinysteno config

# Edit config in $EDITOR
tinysteno config --edit
```

## Personas

Personas control what the LLM extracts from a transcript and how the Obsidian note is rendered. Each persona defines a system prompt, an output schema, and a Jinja2 note template.

### Built-in personas

Built-in personas are seeded to `~/.tinysteno/personas/` on first run, so you can inspect and edit them directly. To restore them to their defaults, run `tinysteno setup --reset-personas`.

| Slug | Name | Use for |
|------|------|---------|
| `default` | Meeting Summary | General meetings — overview, participants, key points, action items |
| `1on1` | 1-on-1 Meeting Analyst | Manager/report 1-on-1s — goals, needs, struggles, recent wins, per-side actions |
| `rca` | Root Cause Analysis | Postmortems — timeline, root cause, contributing factors, corrective actions |
| `irm` | Incident Response & Management | Incident calls — severity, impact, responders, mitigations, follow-ups |
| `sprint` | Sprint Ceremony | Planning/review/retros — completed work, blockers, retrospective notes |
| `kickoff` | Project Kickoff | Project kickoffs — objectives, stakeholders, scope, risks, decisions |
| `executive-summary` | Executive Summary | Decision-ready briefing — bottom line, key decisions, risks, asks |

### Custom personas

Drop a directory into `~/.tinysteno/personas/<slug>/` with three files:

```
~/.tinysteno/personas/
└── my-persona/
    ├── persona.yaml      # name, description, schema
    ├── system_prompt.md  # LLM system message
    └── template.md       # Jinja2 Obsidian note template
```

**`persona.yaml`** example:

```yaml
name: My Persona
description: What this persona does.
schema:
  summary:
    type: string
    description: A brief summary
  highlights:
    type: list
    description: Key highlights as strings
```

Schema field types are `string` or `list` (list of strings). Field names must be valid Python identifiers and cannot collide with reserved metadata variables: `title`, `date`, `duration`, `transcript`, `detected_language`, `generated_tags`.

**`template.md`** receives all schema fields plus the metadata variables above as Jinja2 context.

## Configuration

Config file: `~/.tinysteno/config.yaml`

Run `tinysteno setup` to create it interactively, or create it manually:

```yaml
# Obsidian vault path (~ is expanded)
obsidian_vault: "~/Obsidian/Vault"

# Output settings
output_folder: "meetings"   # subfolder inside vault

# Where to store raw audio recordings
# Defaults to <obsidian_vault>/<output_folder>/audio if not set
# recordings_path: "~/recordings"

# Persona to use when --persona is not specified
persona: "default"

# OpenAI-compatible API settings
# Ollama (local):  base_url: "http://localhost:11434/v1", api_key: "ollama"
# OpenAI (cloud):  base_url: "https://api.openai.com/v1", api_key: "sk-..."
api_key: "ollama"
base_url: "http://localhost:11434/v1"
model: "llama3.2:3b"
auto_title: true            # generate note titles from content
auto_tags: true             # generate tags from content

# Transcription
# Model sizes (speed ↔ accuracy): tiny · base · small · medium · large
whisper_model: "small"

# Backend: "local" (default, faster-whisper) or "api" (OpenAI Whisper API)
transcription_backend: "local"

# Device override for local backend: "auto" (default, auto-detect), "cpu", "cuda"
# Architecture is auto-detected when set to "auto":
#   - CUDA available → cuda + float16
#   - Apple Silicon → auto + auto (CTranslate2 MPS)
#   - Fallback → cpu + int8
whisper_device: "auto"
whisper_compute_type: "auto"

# API endpoint for remote transcription
# Uses the same base_url/api_key as the summarizer if not set
# whisper_base_url: "https://api.openai.com/v1"
# whisper_api_key: "sk-..."
# whisper_api_model: "whisper-1"

# Feature flags
diarization: false          # enable [You]/[Others] speaker labels
                            # when system audio is captured, output is automatically
                            # stereo (L=mic, R=system audio) and diarization will work

# Audio recording
sample_rate: 44100
channels: 1                 # mic input channels; output is automatically stereo
                            # when system audio loopback is available
```

## Output Format

Note format is controlled by the active persona's template. The default persona produces:

```markdown
---
created: 2024-01-15 14:30
type: meeting
tags: [meeting, budget, planning]
duration: 00:45:12
participants: Alice, Bob
---

# Budget Review

## Overview
AI-generated summary of the meeting.

## Participants
- Alice
- Bob

## Key Points
1. Point one
2. Point two

## Action Items
- [ ] Follow up with the team (Alice)

## Transcript
\```
Full transcript text here...
\```
```

When diarization is enabled and system audio is captured, the transcript is interleaved chronologically with speaker labels:

```
[You] I think we should go with option A.
[Others] That makes sense, let's proceed.
[You] Great, I'll follow up with the team.
```

## Requirements

- Python 3.12+
- Optional: Ollama or OpenAI API key

### macOS — System Audio Capture

TinySteno captures system audio via ScreenCaptureKit (macOS 12.3+). To enable it:

1. Open **System Settings → Privacy & Security → Screen Recording**
2. Enable permission for your terminal application (e.g. Terminal, iTerm2, Ghostty)

Without this permission, only the microphone will be recorded.

## Transcription Backends

TinySteno supports two transcription backends, selected via `transcription_backend` in config or `--backend` on the CLI.

### Local (default) — faster-whisper

Uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2) to transcribe entirely on-device. No data leaves your machine.

**GPU auto-detection:** The first time a local transcriber is created with `device=auto`, TinySteno probes the hardware:

- **CUDA** — detected via CTranslate2's `get_supported_devices()`; uses `cuda` + `float16`
- **Apple Silicon (M1+)** — detected via `platform.machine() == "arm64"`; uses `auto` + `auto` (delegates to CTranslate2 MPS backend)
- **CPU fallback** — no GPU found; uses `cpu` + `int8`

Override detection explicitly in config:

```yaml
whisper_device: cuda
whisper_compute_type: float16
```

Or on the CLI:

```bash
tinysteno record --whisper-device cuda --whisper-compute-type float16
```

### API — OpenAI Whisper API

Transmits audio to a remote Whisper-compatible API endpoint. Useful for machines without a GPU or when you need a larger model than local memory supports.

```yaml
transcription_backend: api
whisper_api_key: "sk-..."                # falls back to api_key if unset
whisper_api_model: "whisper-large-v3"    # defaults to whisper-1
```

The API backend supports the same features as the local backend:

- Mono transcription
- Stereo diarization (two API calls, one per channel)
- Language detection fallback
- Progress callbacks

Audio files are uploaded to the configured API endpoint. At ~30-40 MB per hour of recording, consider privacy implications when using a cloud API.

## License

MIT
