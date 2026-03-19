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
- Partial override of built-in persona files (user personas must supply all three files)

---

## Persona Format

Each persona is a directory containing exactly three files. All three are required — a user persona missing any file is treated as malformed.

### Directory locations

```
<package>/personas/        # built-ins, shipped with TinySteno
~/.tinysteno/personas/     # user personas; override built-ins by slug
```

A persona's **slug** is its directory name. A user persona with the same slug as a built-in replaces the built-in entirely — there is no partial override or inheritance of individual files.

### Files

#### `persona.yaml`

Metadata and output schema definition.

```yaml
name: Root Cause Analysis
description: Extracts timeline, root cause, and corrective actions from incident postmortems.
schema:
  overview:             { type: string, description: "Brief incident summary" }
  timeline:             { type: list,   description: "Chronological list of events as strings" }
  root_cause:           { type: string, description: "Primary root cause" }
  contributing_factors: { type: list,   description: "Contributing factors as strings" }
  corrective_actions:   { type: list,   description: "Follow-up actions, each formatted as 'task (owner)'" }
```

**Valid field types:**

| Type | JSON equivalent | Template variable type |
|------|----------------|------------------------|
| `string` | JSON string | Python `str` |
| `list` | JSON array of strings | Python `list[str]` |

All list fields are lists of strings. To represent structured items (e.g., action with owner), use a string format convention described in the field's `description` (e.g., `"Fix the auth bug (alice)"`). The LLM prompt instructs the model to follow the description's format guidance.

**Field name constraints:** Schema field names must satisfy all of:
- Pass `str.isidentifier()` (valid Python identifier characters)
- Not be a Python keyword (checked via `keyword.iskeyword()`)
- Not collide with the reserved metadata variable names: `title`, `date`, `duration`, `transcript`, `detected_language`

Violations are caught at `load_persona()` time with a `PersonaInvalidError`.

The `schema` is used to:
1. Construct a JSON example block injected into the LLM user message (see LLM Prompt Construction below)
2. Define the variables available in the Jinja2 template

#### `system_prompt.md`

Plain text LLM system prompt sent as the `system` role message. The transcript is appended to the `user` role message by the engine after the JSON format instruction — the system prompt should not reference how the transcript or output format will be provided.

#### `template.md`

Jinja2 template for the Obsidian note. Available variables:

| Variable | Type | Source |
|----------|------|--------|
| `{{ title }}` | `str` | See Title Generation below |
| `{{ date }}` | `str` | Recording timestamp, formatted as `YYYY-MM-DD HH:MM` (see Date Source below) |
| `{{ duration }}` | `str` | Recording duration as `HH:MM:SS`, from transcriber output |
| `{{ transcript }}` | `str` | Full (untruncated) transcript text |
| `{{ detected_language }}` | `str` | Language code detected by Whisper |
| `{{ <field> }}` | `str` or `list[str]` | Any field defined in `persona.yaml` schema |

List fields are passed as Python `list[str]` into the Jinja2 context. Template authors are responsible for iterating them with `{% for %}` loops. The built-in persona templates demonstrate this pattern.

If `{{ transcript }}` is absent from the template, the transcript is **not** included in the note.

---

## LLM Prompt Construction

The summarizer builds the LLM call as follows:

- **System message**: contents of `persona.system_prompt`
- **User message**: a JSON format instruction block followed by the transcript

The user message format instruction is constructed from `persona.schema` as a prose block plus a JSON example:

```
Return a JSON object with exactly these fields:
{
  "overview": "string",
  "timeline": ["string", "..."],
  "root_cause": "string",
  "contributing_factors": ["string", "..."],
  "corrective_actions": ["string", "..."]
}

Field descriptions:
- overview: Brief incident summary
- timeline: Chronological list of events as strings
- root_cause: Primary root cause
- contributing_factors: Contributing factors as strings
- corrective_actions: Follow-up actions, each formatted as 'task (owner)'

Transcript:
<transcript text, truncated to 15000 chars>
```

The LLM call uses `response_format={"type": "json_object"}` (same as current code). No retry logic. If the response is not valid JSON, the existing regex-extraction fallback in `summarizer.py` is applied. Missing required fields default to `""` (string) or `[]` (list) with a logged warning. Extra fields in the LLM response are ignored.

The 15,000-character truncation applies only to the transcript text sent to the LLM. The full transcript is stored separately and passed to `ObsidianExporter` as the `transcript` metadata key.

---

## Title Generation

Title generation behavior is controlled by the existing `auto_title` config key (default: `true`):

- **`auto_title: true`**: `main.py` calls `summarizer.generate_title(field_value)` after summarization, where `field_value` is the value of the first `string`-type field in the persona's schema (in YAML document order — see YAML Field Ordering below). `generate_title()` handles internal truncation to 500 chars (unchanged from current `_title_prompt` logic). If the LLM call fails (timeout, API error, etc.) or if the persona's schema contains no `string`-type fields, title silently falls back to the filename stem.
- **`auto_title: false`**: Title is set to the recording filename stem (without extension).

Title resolution happens in `main.py` after `summarizer.summarize()` returns, before building the `metadata` dict.

---

## Date Source

The `date` metadata variable is formatted as `YYYY-MM-DD HH:MM`:

- **`tinysteno record`**: system clock at the moment recording begins
- **`tinysteno process <file>`**: file modification time (`mtime`) of the input audio file

## YAML Field Ordering

Schema field order follows insertion order as loaded by `yaml.safe_load()`, which preserves document order in Python 3.7+ (where `dict` preserves insertion order). TinySteno requires Python 3.7+. This ordering determines which field is "first" for title generation.

---

## Output File Naming

The Obsidian note filename follows the same convention as the current implementation: `{title} ({YYYY-MM-DD}).md`, placed in `{vault}/{output_folder}/`. `ObsidianExporter` derives the date portion by taking the first 10 characters of `metadata["date"]` (the `YYYY-MM-DD` prefix).

---

## Built-in Personas

Built-in personas are returned by `list_personas()` in this fixed order:

| # | Slug | Name | Key Schema Fields |
|---|------|------|-------------------|
| 1 | `default` | Meeting Summary | `overview` (string), `participants` (list), `key_points` (list), `action_items` (list) |
| 2 | `rca` | Root Cause Analysis | `overview` (string), `timeline` (list), `root_cause` (string), `contributing_factors` (list), `corrective_actions` (list) |
| 3 | `irm` | Incident Response & Management | `overview` (string), `timeline` (list), `severity` (string), `impact` (string), `responders` (list), `mitigations` (list), `follow_ups` (list) |
| 4 | `sprint` | Sprint Ceremony | `ceremony_type` (string), `overview` (string), `completed_items` (list), `incomplete_items` (list), `blockers` (list), `retrospective_notes` (list), `action_items` (list) |
| 5 | `kickoff` | Project Kickoff | `overview` (string), `objectives` (list), `stakeholders` (list), `scope` (string), `risks` (list), `decisions` (list), `next_steps` (list) |
| 6 | `executive-summary` | Executive Summary | `summary` (string), `key_decisions` (list), `risks` (list), `asks` (list) |

The `default` persona replicates current behavior exactly:
- Schema fields match the former `MeetingData` fields: `overview` (string), `participants` (list of strings), `key_points` (list of strings), `action_items` (list of strings)
- Title is generated from `overview` (the first string field) when `auto_title: true`

---

## Configuration

### `~/.tinysteno/config.yaml`

One new key. `auto_title` already exists; shown here for reference:

```yaml
persona: default    # slug of persona to use; defaults to "default" if omitted
auto_title: true    # existing key; controls title generation behavior
```

### CLI

```
tinysteno record [--persona <slug>]
tinysteno process <file> [--persona <slug>]
```

`--persona` overrides the config value for that run (in-memory only; `config.yaml` is never written by this flag). The slug is validated — and `PersonaNotFoundError` raised — **before** recording starts (for `record`) and before transcription begins (for `process`), so no work is lost to an invalid slug.

### `tinysteno setup`

New prompt added after existing prompts. The list is built dynamically by calling `list_personas()`, so user-added personas appear. The prompt accepts any non-empty string; no validation is performed at setup time. The entered value is written to `config.yaml` as-is and persists across sessions. An invalid slug will produce a `PersonaNotFoundError` or `PersonaInvalidError` at the next `record` or `process` run.

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
    schema: dict          # field name → {type: str, description: str}; ordered (insertion order)
    system_prompt: str    # raw text from system_prompt.md
    template: str         # raw Jinja2 from template.md

class PersonaNotFoundError(Exception):
    """Raised when no persona directory exists for the requested slug."""

class PersonaInvalidError(Exception):
    """Raised when a persona directory exists but is malformed (missing file,
    invalid field name, unknown field type, or reserved field name collision).
    Exception message identifies the specific file path or field name."""

def list_personas() -> list[str]:
    """Returns available slugs.

    Ordering rules:
    - A user persona that overrides a built-in slug appears in the built-in's original
      position (the built-in is replaced in-place, not appended).
    - User-only personas (no built-in counterpart) are appended after built-ins, sorted
      case-insensitively (str.lower() key).

    Malformed user persona directories (missing files, invalid schema, etc.) are
    excluded after logging a warning to stderr. The warning message includes the
    directory path and the specific reason (same text that would appear in a
    PersonaInvalidError for that directory).

    If a malformed user persona shadows a built-in slug, the built-in is exposed
    in the list (the malformed directory does NOT suppress it). A warning is logged.

    A valid user persona with a built-in slug appears only once (user wins)."""

def load_persona(slug: str) -> Persona:
    """Loads and returns the Persona for the given slug.
    If a user persona directory exists for the slug, it is used exclusively —
    there is NO fallback to the built-in, even if the user persona is malformed.
    (This means list_personas() may show a slug as available via the built-in,
    while load_persona() raises PersonaInvalidError for that slug if the user
    directory is present but malformed. Users must fix or remove the malformed
    directory to restore access to the built-in.)
    Raises PersonaNotFoundError if no directory for the slug exists (in either location).
    Raises PersonaInvalidError if a directory exists but is malformed."""
```

### Modified: `tinysteno/summarizer.py`

- `Summarizer.summarize(transcript: str, persona: Persona) -> dict` — builds prompt dynamically (see LLM Prompt Construction above); returns a plain `dict` mapping field names to values
- `Summarizer.generate_title(field_value: str) -> str | None` — truncates `field_value` to 500 chars internally, sends to LLM, returns title string on success; returns `None` on any exception (timeout, API error, etc.). `main.py` substitutes the filename stem when `None` is returned.
- `MeetingData` is removed; `summarize()` returns `dict` throughout

### Modified: `tinysteno/obsidian.py`

- `ObsidianExporter.export(data: dict, persona: Persona, metadata: dict) -> None`
- `metadata` keys (all strings): `title`, `date`, `duration`, `transcript` (full text), `detected_language`
- Renders `persona.template` via Jinja2 with the merged context: `metadata` values + all `data` field values
- Output file named `{title} ({YYYY-MM-DD}).md` in `{vault}/{output_folder}/`, derived from `metadata["title"]` and `metadata["date"]`
- On `jinja2.TemplateError`: wraps in a `RuntimeError` with the template file path and original error (including line number), then re-raises; `main.py` catches `RuntimeError` and displays the message to the user
- Replaces hardcoded section-by-section markdown generation

### Modified: `tinysteno/main.py`

- `load_config()` reads new `persona` key (default: `"default"`)
- `--persona` flag added to `record` and `process` commands
- Slug validation (via `load_persona()`) happens at the start of both `cmd_record()` and `cmd_process()`, before any audio or transcription work; `PersonaNotFoundError` and `PersonaInvalidError` are caught separately and display distinct user-facing messages (see Error Handling table)
- After `summarizer.summarize()` returns, `main.py` resolves `title`: if `auto_title: true` and persona has at least one `string`-type field, calls `summarizer.generate_title(first_string_field_value)`; otherwise uses recording filename stem
- Builds `metadata` dict: `{title, date, duration, transcript, detected_language}`, where `duration` and `detected_language` come from the transcriber's return dict (keys: `text`, `duration_seconds`, `detected_language`; `duration_seconds` is a `float` formatted to `HH:MM:SS` using `total = int(duration_seconds); h, rem = divmod(total, 3600); m, s = divmod(rem, 60)`; values where `h > 99` are clamped to `99:59:59`)
- Passes `Persona`, `data` dict, and `metadata` dict to `ObsidianExporter.export()`
- `cmd_setup()` calls `list_personas()` to display available options and prompts for default persona

---

## Data Flow

Both `record` and `process` follow the same pipeline from persona load onward:

```
tinysteno record --persona rca          tinysteno process file.wav --persona rca
        ↓                                           ↓
[main.py]  load_persona("rca") → Persona
           or raise PersonaNotFoundError / PersonaInvalidError (displayed; exit)
        ↓
[recorder.py]  capture audio → WAV      [skip: file already exists]
        ↓
[transcriber.py]  transcribe
   → {text: str, duration_seconds: float, detected_language: str}
        ↓
[summarizer.py]  summarize(transcript, persona)
   • system message: persona.system_prompt
   • user message: JSON format instruction + transcript truncated to 15000 chars
   • response_format: json_object
   • returns: dict of schema field values
        ↓
[main.py]  resolve title:
   • auto_title=true + schema has string field → generate_title(first_string_value)
   • otherwise → filename stem
[main.py]  build metadata: {title, date, duration (HH:MM:SS), transcript=full_text, detected_language}
   • date source: start-of-recording clock (record) or file mtime (process)
        ↓
[obsidian.py]  export(data_dict, persona, metadata)
   • renders persona.template via Jinja2
   • output: {vault}/{output_folder}/{title} ({YYYY-MM-DD}).md
```

---

## Error Handling

| Scenario | Exception | Where caught | User message |
|----------|-----------|--------------|--------------|
| Slug not found | `PersonaNotFoundError` | `cmd_record` / `cmd_process` | "Unknown persona 'X'. Available: default, rca, ..." |
| Persona dir malformed (missing file, bad field, etc.) | `PersonaInvalidError` | `cmd_record` / `cmd_process` | Exception message (includes specific file path or field name) |
| LLM returns invalid JSON | — (caught internally) | `summarizer.py` | Warning logged; fields default to `""` / `[]` |
| Template render error | `jinja2.TemplateError` wrapped in `RuntimeError` | `main.py` | Template file path + line number from original error |

---

## Implementation Notes

**Slug naming:** Persona slugs are directory names — no format constraints beyond what the OS allows for directory names. Slug naming rules are entirely separate from schema field name rules. `executive-summary` is a valid slug despite containing a hyphen.

**`persona.yaml` required keys:** `name`, `description`, and `schema` are all required top-level keys. A missing `name` or `description` raises `PersonaInvalidError`. An empty `schema` (zero fields) is permitted.

**`tinysteno setup --persona`:** `tinysteno setup` does not accept a `--persona` flag. No changes to the `setup` command's flags.

**Setup prompt validation:** Silent acceptance of unrecognized slugs at setup time is intentional. No warning is displayed. The user will see an error at the next `record`/`process` run.

**Jinja2 context merge order:** `metadata` dict is merged first, then `data` dict values are added. Since field name validation prevents collisions at load time, no runtime collision is expected. If a collision occurs despite validation (implementation bug), the `data` value silently overwrites the `metadata` value — no exception is raised.

**`ObsidianExporter` date slicing:** `metadata["date"][:10]` is used without validation. If the value is shorter than 10 characters (a bug in upstream date formatting), the filename will contain a partial date string. No special handling is required.

---

## Backward Compatibility

- Users with no `persona` config key get `default`, which reproduces current behavior
- `default` persona schema fields match the former `MeetingData` fields exactly
- `MeetingData` is removed; it was internal with no public API
- No changes to recording, transcription, or file storage behavior
