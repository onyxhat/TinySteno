# TinySteno

Inspired by [StenoAI](https://github.com/ruzin/stenoai).
Minimal meeting recorder with Obsidian export (local filesystem vault).

## Features

- Record meetings (indefinite duration, Ctrl+C to stop)
- Captures mic + system audio simultaneously — no virtual audio device required
- Transcribe with faster-whisper (local, runs on CPU)
- Summarize via OpenAI-compatible API (Ollama, OpenAI, Groq)
- Export structured markdown notes to Obsidian vault

## Installation

```bash
uv sync
uv run tinysteno --help
```

## Usage

```bash
# Update config interactively
tinysteno setup

# Record a meeting
tinysteno record

# Record with custom name
tinysteno record --name "Budget Review"

# Process existing audio file
tinysteno process recordings/Meeting.wav

# Verify setup
tinysteno test

# List processed meetings
tinysteno list --vault /path/to/vault

# Show current config
tinysteno config
```

## Configuration

Config file: `~/.tinysteno/config.yaml`

Run `tinysteno setup` to create it interactively, or create it manually:

```yaml
# Obsidian vault path (~ is expanded)
obsidian_vault: "~/Obsidian/Vault"

# Output settings
output_folder: "meetings"   # subfolder inside vault
tags: ["meeting"]           # default YAML frontmatter tags

# Where to store raw audio recordings
# Defaults to <obsidian_vault>/<output_folder>/audio if not set
# recordings_path: "~/recordings"

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

### Meeting Note

```yaml
---
created: 2024-01-15T14:30:00
type: meeting
tags: [meeting]
duration: "45m"
participants: [Alice, Bob]
---

# Budget-Review

## Overview
AI-generated summary...

## Key Points
1. Point 1
2. Point 2

## Action Items
- [ ] Task → @Alice

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
