# Graph Report - .  (2026-08-01)

## Corpus Check
- Corpus is ~19,226 words - fits in a single context window. You may not need a graph.

## Summary
- 375 nodes · 727 edges · 18 communities (14 shown, 4 thin omitted)
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 45 edges (avg confidence: 0.87)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Main CLI & Setup
- Personas System & Tests
- Orchestrator Tests
- Persona Definitions
- Audio Recorder
- LLM Orchestrator
- Obsidian Export
- Audio Transcriber
- Dependencies & CI
- macOS Loopback Audio
- Env & MCP Config
- Test Fixtures
- Claude Flow Integration
- GitNexus Integration
- Package Init
- Package Config

## God Nodes (most connected - your core abstractions)
1. `_make_orchestrator()` - 35 edges
2. `Orchestrator` - 24 edges
3. `AudioRecorder` - 24 edges
4. `load_persona()` - 21 edges
5. `_make_persona()` - 19 edges
6. `Persona` - 18 edges
7. `_make_persona()` - 16 edges
8. `_make_exporter()` - 16 edges
9. `_process_audio()` - 16 edges
10. `_make_metadata()` - 14 edges

## Surprising Connections (you probably didn't know these)
- `CLAUDE.md — Performance Fixes Implementation Plan` --references--> `TinySteno Project`  [INFERRED]
  CLAUDE.md → pyproject.toml
- `test_cmd_setup_reset_personas_forces_seed_without_wizard()` --calls--> `cmd_setup()`  [EXTRACTED]
  tests/test_main.py → tinysteno/main.py
- `test_cmd_setup_normal_seeds_interactively_after_config_write()` --calls--> `cmd_setup()`  [EXTRACTED]
  tests/test_main.py → tinysteno/main.py
- `_make_list_only_persona()` --references--> `Persona`  [EXTRACTED]
  tests/test_main.py → tinysteno/personas/__init__.py
- `test_title_and_tags_generated_in_parallel()` --calls--> `Persona`  [EXTRACTED]
  tests/test_main.py → tinysteno/personas/__init__.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Performance Fixes Implementation Plan** — tinysteno_recorder, tinysteno_transcriber, tinysteno_orchestrator, tinysteno_main, tinysteno_macos_loopback [EXTRACTED 1.00]
- **Audio Recording and Transcription Pipeline** — tinysteno_recorder, tinysteno_transcriber, tinysteno_orchestrator, tinysteno_main [INFERRED 0.95]
- **Project Configuration and CI** — tinysteno_project, github_workflows_pylint_ci, tinysteno_mcp, tinysteno_agents [INFERRED 0.85]
- **Persona Three-File Structure** — tinysteno_personas_1on1_persona_doc, tinysteno_personas_1on1_system_prompt_doc, tinysteno_personas_1on1_template_doc [EXTRACTED 1.00]
- **All Personas Share Three-File Architecture** — persona_1on1, persona_default, persona_executive_summary, persona_irm, persona_kickoff, persona_rca, persona_sprint [INFERRED 0.95]
- **Common Extraction Principles Across Personas** — attribution_rule, schema_definition, jinja2_templating, yaml_frontmatter [INFERRED 0.95]

## Communities (18 total, 4 thin omitted)

### Community 0 - "Main CLI & Setup"
Cohesion: 0.05
Nodes (62): datetime, Logger, soundfile, Stream Recording to Disk, _make_list_only_persona(), Path, Tests for tinysteno.main module., Verify the real main() parser wires --reset-personas correctly. (+54 more)

### Community 1 - "Personas System & Tests"
Cohesion: 0.08
Nodes (44): Path, Tests for tinysteno.personas module., Malformed user dir for a built-in slug: load raises, no built-in fallback., test_list_personas_malformed_builtin_slug_is_excluded(), test_list_personas_no_duplicates(), test_list_personas_returns_all_builtins(), test_list_personas_skips_malformed_user_persona(), test_list_personas_valid_user_only_appended_alphabetically() (+36 more)

### Community 2 - "Orchestrator Tests"
Cohesion: 0.12
Nodes (39): Exception, _llm_response(), _make_orchestrator(), _make_persona(), Tests for tinysteno.orchestrator (Orchestrator)., Every character of the transcript appears in at least one chunk., test_build_merge_message_includes_dedup_instruction(), test_build_merge_message_includes_schema_fields() (+31 more)

### Community 3 - "Persona Definitions"
Cohesion: 0.17
Nodes (33): Explicit Attribution Rule — never guess owner, use unassigned, Jinja2 Template Rendering Engine, 1-on-1 Meeting Analyst Persona, Meeting Summary Persona, Executive Summary Persona, Incident Response & Management Persona, Project Kickoff Persona, Persona-Based Meeting Transcription Pipeline (+25 more)

### Community 4 - "Audio Recorder"
Cohesion: 0.09
Nodes (15): AudioRecorder, ndarray, Path, Return (device_index, max_input_channels) for the system loopback device. Only…, Find a WASAPI loopback device on Windows., Find a PulseAudio/PipeWire monitor source on Linux., Start recording and return the output WAV path., Record mic + system audio loopback to WAV files. On Windows and Linux the… (+7 more)

### Community 5 - "LLM Orchestrator"
Cohesion: 0.09
Nodes (13): Orchestrator, Generate tags from a field value string. Returns a list of lowercase…, Split transcript into overlapping chunks of _CHUNK_SIZE_CHARS. Overlap…, Build the LLM user message: JSON format instruction + transcript., Build the LLM merge prompt for combining partial chunk results., Call the LLM and validate/normalize the response against the persona schema., Normalize a parsed LLM response against the persona schema., Orchestrate LLM calls for transcript extraction and title generation. (+5 more)

### Community 6 - "Obsidian Export"
Cohesion: 0.20
Nodes (23): _make_exporter(), _make_metadata(), _make_persona(), Path, Tests for updated ObsidianExporter., test_export_creates_file(), test_export_data_and_metadata_merged_in_context(), test_export_file_in_correct_output_folder() (+15 more)

### Community 7 - "Audio Transcriber"
Cohesion: 0.10
Nodes (22): _make_transcriber(), Tests for WhisperTranscriber., _convert_to_16khz_array should resample without calling np.interp., No resampling performed when input is already 16kHz., transcribe() should not write any .wav temp files., _run_whisper should accept numpy array, not require a file path., Two WhisperTranscriber instances with same model_size share one WhisperModel., on_progress callback should be called with values between 0.0 and 1.0. (+14 more)

### Community 8 - "Dependencies & CI"
Cohesion: 0.10
Nodes (21): actions/checkout@v4, actions/setup-python@v6.2.0, astral-sh/setup-uv@v7.6.0, faster-whisper Transcription, Pylint CI Workflow, jinja2, numpy, Obsidian Vault Export (+13 more)

### Community 9 - "macOS Loopback Audio"
Cohesion: 0.13
Nodes (14): CDLL, _AudioBuffer, _detect_sr(), _get_delegate_class(), _load_coremedia(), MacOSLoopback, ndarray, macOS system audio capture via ScreenCaptureKit (macOS 12.3+). Requires… (+6 more)

### Community 10 - "Env & MCP Config"
Cohesion: 0.20
Nodes (9): CLAUDE_FLOW_HOOKS_ENABLED, CLAUDE_FLOW_MAX_AGENTS, CLAUDE_FLOW_MEMORY_BACKEND, CLAUDE_FLOW_MODE, CLAUDE_FLOW_TOPOLOGY, npm_config_update_notifier, npx, claude-flow (+1 more)

### Community 11 - "Test Fixtures"
Cohesion: 0.40
Nodes (4): fixture, Shared pytest fixtures for TinySteno tests., A tmp personas dir pre-seeded with all built-in personas., seeded_user_dir()

## Knowledge Gaps
- **23 isolated node(s):** `npx`, `@claude-flow/cli`, `npm_config_update_notifier`, `CLAUDE_FLOW_MODE`, `CLAUDE_FLOW_HOOKS_ENABLED` (+18 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **4 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Orchestrator` connect `LLM Orchestrator` to `Main CLI & Setup`, `Orchestrator Tests`?**
  _High betweenness centrality (0.160) - this node is a cross-community bridge._
- **Why does `AudioRecorder` connect `Audio Recorder` to `Dependencies & CI`, `Main CLI & Setup`, `macOS Loopback Audio`?**
  _High betweenness centrality (0.159) - this node is a cross-community bridge._
- **Why does `Persona` connect `Personas System & Tests` to `Main CLI & Setup`, `Orchestrator Tests`, `Obsidian Export`?**
  _High betweenness centrality (0.135) - this node is a cross-community bridge._
- **What connects `npx`, `@claude-flow/cli`, `npm_config_update_notifier` to the rest of the system?**
  _23 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Main CLI & Setup` be split into smaller, more focused modules?**
  _Cohesion score 0.053821800090456805 - nodes in this community are weakly interconnected._
- **Should `Personas System & Tests` be split into smaller, more focused modules?**
  _Cohesion score 0.07770582793709528 - nodes in this community are weakly interconnected._
- **Should `Orchestrator Tests` be split into smaller, more focused modules?**
  _Cohesion score 0.12179487179487179 - nodes in this community are weakly interconnected._