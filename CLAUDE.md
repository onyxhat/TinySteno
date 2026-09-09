# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

TinySteno is a minimal CLI meeting recorder: it records mic + system audio, transcribes
(faster-whisper locally or an OpenAI Whisper-compatible API), summarizes via an
OpenAI-compatible API, and exports structured Markdown notes to an Obsidian vault.
Single Python package `tinysteno`; CLI entry point is `tinysteno.main:main`.

## Commands

Everything Python runs through `uv` — the project is uv-managed and CI uses it. There is
no Node/`npm` build here; ignore the Node/claude-flow workflow in the global config.

```bash
uv sync --extra dev                    # install with dev extras (pytest, pylint)
uv run tinysteno --help               # run the CLI
uv run pytest                         # run all tests
uv run pytest -k "test_name"          # run a single test
uv run pylint $(git ls-files '*.py')  # lint — exactly what CI runs on every push
```

## Linting

- CI (`.github/workflows/pylint.yml`) runs `pylint` over all tracked `.py` files on every
  push and must pass. Max line length is 100 (`.pylintrc`).
- Some checks are disabled on purpose in `.pylintrc` — do not "fix" the code to satisfy them:
  - `import-outside-toplevel`: heavy / platform-specific deps (`sounddevice`, `objc`,
    `ScreenCaptureKit`) are lazily imported by design.
  - `broad-exception-caught`: LLM and hardware calls deliberately catch broadly and fall
    back to degraded mode.

## Personas

Built-in personas live in `tinysteno/personas/<slug>/`, each with `persona.yaml`,
`system_prompt.md`, and `template.md` (Jinja2). They are packaged via `package-data` in
`pyproject.toml` and seeded to `~/.tinysteno/personas/` at runtime.

- Persona schema field types are `string` or `list` (list of strings) only.
- Field names must be valid Python identifiers and must not collide with the reserved
  metadata vars: `title`, `date`, `duration`, `transcript`, `detected_language`,
  `generated_tags`.
- `tests/conftest.py` seeds personas from `tinysteno.personas._BUILTIN_DIR` / `BUILTIN_ORDER`.

## Notes

- Runtime config lives at `~/.tinysteno/config.yaml` (created by `tinysteno setup`).
- Transcription has two backends, selected by `transcription_backend` / `--backend`:
  `local` (faster-whisper, GPU auto-detected) and `api` (Whisper-compatible endpoint).
- This repo is indexed by both `graft` and the GitNexus MCP for code navigation and
  impact analysis.
