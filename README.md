# TinySteno

Inspired by [StenoAI](https://github.com/ruzin/stenoai).
Minimal meeting recorder with Obsidian export (local filesystem vault).

## Features

- Record meetings (indefinite duration, Ctrl+C to stop)
- Captures mic + system audio simultaneously — no virtual audio device required
- Transcribe with faster-whisper (local, runs on CPU)
- Summarize via OpenAI-compatible API (Ollama, OpenAI, OpenRouter, etc.)
- Persona system — choose how recordings are summarized and formatted
- Export structured markdown notes to Obsidian vault

## Prerequisites
- [Python 3.12+](https://www.python.org/downloads/)
- [UV](https://docs.astral.sh/uv/getting-started/installation/)

## Installation

**UVX tool:**
```bash
uvx tool install git+https://github.com/onyxhat/TinySteno
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
# Update config interactively
tinysteno setup

# Record a meeting (uses default persona)
tinysteno record

# Record with custom name
tinysteno record --name "Budget Review"

# Record using a specific persona
tinysteno record --persona rca

# Process existing audio file
tinysteno process recordings/Meeting.wav

# Process with a specific persona
tinysteno process recordings/Meeting.wav --persona executive-summary

# Verify setup
tinysteno test

# List processed meetings
tinysteno list --vault /path/to/vault

# Show current config
tinysteno config
```

## Personas

Personas control what the LLM extracts from a transcript and how the Obsidian note is rendered. Each persona defines a system prompt, an output schema, and a Jinja2 note template.

### Built-in personas

| Slug | Name | Use for |
|------|------|---------|
| `default` | Meeting Summary | General meetings — overview, participants, key points, action items |
| `rca` | Root Cause Analysis | Postmortems — timeline, root cause, contributing factors, corrective actions |
| `irm` | Incident Response & Management | Incident calls — severity, impact, responders, mitigations, follow-ups |
| `sprint` | Sprint Ceremony | Planning/review/retros — completed work, blockers, retrospective notes |
| `kickoff` | Project Kickoff | Project kickoffs — objectives, stakeholders, scope, risks, decisions |
| `executive-summary` | Executive Summary | Concise summaries — key decisions, risks, asks |

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

Schema field types are `string` or `list` (list of strings). Field names must be valid Python identifiers and cannot collide with reserved metadata variables: `title`, `date`, `duration`, `transcript`, `detected_language`.

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

# Transcription
# Model sizes (speed ↔ accuracy): tiny · base · small · medium · large
whisper_model: "small"

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
tags: [meeting]
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

## License

MIT
