# Personas Feature Design

**Date:** 2026-03-18
**Status:** Approved
**Branch:** feat/personas

---

## Overview

Add a modular, user-extensible persona system to TinySteno. A persona defines how a recording is processed — the LLM system prompt, the expected output schema, and the Obsidian note template. The current "meeting summarizer" becomes the `default` persona. Users can add custom personas by dropping a directory into `~/.tinysteno/personas/`.

---

## Goals

- Replace hardcoded summarization with a configurable persona system
- Ship built-in personas for common meeting/event types
- Allow users to create and override personas without modifying source code
- Preserve existing behavior for users who don't configure a persona

---

## Non-Goals

- Python plugin API for personas (file-based only)
- Persona chaining or multi-step processing pipelines
- UI for managing personas

---

## Persona Format

Each persona is a directory containing three files.

### Directory locations

```
<package>/personas/        # built-ins, shipped with TinySteno
~/.tinysteno/personas/     # user personas; override built-ins by slug
```

A persona's **slug** is its directory name. A user persona with the same slug as a built-in overrides it.

### Files

#### `persona.yaml`

Metadata and output schema definition.

```yaml
name: Root Cause Analysis
description: Extracts timeline, root cause, and corrective actions from incident postmortems.
schema:
  overview:             { type: string, description: "Brief incident summary" }
  timeline:             { type: list,   description: "Chronological list of events" }
  root_cause:           { type: string, description: "Primary root cause" }
  contributing_factors: { type: list,   description: "Contributing factors" }
  corrective_actions:   { type: list,   description: "Follow-up actions with owners" }
```

The `schema` is used to:
1. Build the JSON schema injected into the LLM prompt (tells the model what to return)
2. Define the variables available in the Jinja2 template

#### `system_prompt.md`

Plain text/markdown LLM system prompt. The transcript is appended by the engine after this content — the prompt should not reference how the transcript will be provided.

#### `template.md`

Jinja2 template for the Obsidian note. Available variables:

| Variable | Source |
|----------|--------|
| `{{ title }}` | Auto-generated or filename |
| `{{ date }}` | Recording timestamp |
| `{{ duration }}` | Recording duration |
| `{{ transcript }}` | Full transcript text |
| `{{ <field> }}` | Any field defined in `persona.yaml` schema |

If `{{ transcript }}` is absent from the template, the transcript is **not** included in the note.

---

## Built-in Personas

| Slug | Name | Key Schema Fields |
|------|------|-------------------|
| `default` | Meeting Summary | `overview`, `participants`, `key_points`, `action_items` |
| `rca` | Root Cause Analysis | `overview`, `timeline`, `root_cause`, `contributing_factors`, `corrective_actions` |
| `irm` | Incident Response & Management | `overview`, `timeline`, `severity`, `impact`, `responders`, `mitigations`, `follow_ups` |
| `sprint` | Sprint Ceremony | `ceremony_type`, `overview`, `completed_items`, `incomplete_items`, `blockers`, `retrospective_notes`, `action_items` |
| `kickoff` | Project Kickoff | `overview`, `objectives`, `stakeholders`, `scope`, `risks`, `decisions`, `next_steps` |
| `executive-summary` | Executive Summary | `summary`, `key_decisions`, `risks`, `asks` |

The `default` persona replicates current behavior exactly — no breaking change for existing users.

---

## Configuration

### `~/.tinysteno/config.yaml`

New key:

```yaml
persona: default    # slug of persona to use; defaults to "default" if omitted
```

### CLI

```
tinysteno record [--persona <slug>]
tinysteno process <file> [--persona <slug>]
```

`--persona` overrides the config value for that run.

### `tinysteno setup`

New prompt added after existing prompts:

```
Available personas: default, rca, irm, sprint, kickoff, executive-summary
Default persona [default]:
```

---

## Architecture

### New module: `tinysteno/personas.py`

Responsible for discovering and loading personas.

```python
@dataclass
class Persona:
    slug: str
    name: str
    description: str
    schema: dict          # field name → {type, description}
    system_prompt: str    # raw text from system_prompt.md
    template: str         # raw Jinja2 from template.md

def list_personas() -> list[str]: ...       # returns all available slugs
def load_persona(slug: str) -> Persona: ... # raises if not found
```

Load order: scan built-in package directory, then `~/.tinysteno/personas/`. User personas override built-ins by slug.

### Modified: `tinysteno/summarizer.py`

- `Summarizer.summarize(transcript, persona: Persona)` — builds prompt dynamically from `persona.system_prompt` + JSON schema derived from `persona.schema`
- Returns `dict` of field values (instead of hardcoded `MeetingData`)
- `MeetingData` retained as internal implementation detail for `default` persona only, or removed in favour of plain dict

### Modified: `tinysteno/obsidian.py`

- `ObsidianExporter.export(data: dict, persona: Persona, metadata: dict)` — renders `persona.template` via Jinja2 with schema fields + built-in variables (`title`, `date`, `duration`, `transcript`)
- Replaces hardcoded section-by-section markdown generation

### Modified: `tinysteno/main.py`

- `load_config()` reads new `persona` key (default: `"default"`)
- `--persona` flag added to `record` and `process` commands
- `_process_audio()` resolves persona slug, loads `Persona`, passes to `Summarizer` and `ObsidianExporter`
- `cmd_setup()` prompts for default persona

---

## Data Flow

```
tinysteno record --persona rca
        ↓
[recorder.py]  capture audio → WAV file
        ↓
[transcriber.py]  transcribe → transcript text
        ↓
[personas.py]  load_persona("rca") → Persona
        ↓
[summarizer.py]  summarize(transcript, persona)
   • builds: system_prompt + JSON schema from persona
   • LLM returns JSON dict matching schema
        ↓
[obsidian.py]  export(data_dict, persona, metadata)
   • renders persona.template via Jinja2
   • writes .md to vault
```

---

## Error Handling

- **Unknown slug**: `load_persona()` raises `PersonaNotFoundError` with list of available slugs; surfaced as a user-friendly CLI error before processing begins
- **Missing persona files**: validated at load time; clear error messages indicate which file is missing
- **LLM returns unexpected fields**: extra fields are ignored; missing required fields are set to empty string/list with a warning
- **Template render error**: Jinja2 errors surfaced with the template file path and line number

---

## Backward Compatibility

- Users with no `persona` config key get `default`, which reproduces current behavior
- `default` persona schema matches current `MeetingData` fields exactly
- No changes to recording, transcription, or file storage behavior
