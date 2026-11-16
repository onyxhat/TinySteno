# TinySteno Architecture

## Summary

TinySteno is a minimal meeting recorder that captures audio (microphone + system audio), transcribes it via Whisper, summarizes the transcript through a configurable LLM, and exports the result as Markdown notes into an Obsidian vault. It follows a linear pipeline architecture: **Record -> Transcribe -> Orchestrate (LLM) -> Export**.

- **Language:** Python 3.12+
- **Package manager:** uv
- **Lines of code:** ~1,100 core + ~1,200 tests
- **Modules:** 13 core modules (including the `tui/` package), 7 built-in personas
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
- `cmd_process()` -- transcribe + summarize + export a pre-recorded WAV; `--retranscribe` ignores a saved transcript
- `create_transcriber(config)` -- builds the local or API transcriber with `whisper_vad` (local only), `whisper_prompt` and the `TranscriptOptions` from `hallucination_filter` / `bleed_filter` / `speaker_split` / `max_speakers`; the API transcriber also gets `server_diarize` (`whisper_api_diarize`) and `timeout` (`whisper_api_timeout`, default 600 s). `--speakers N` (`record`, `process`, `tui`) overrides `max_speakers` for the run
- `_read_notes(args)` -- reads the `--notes PATH` file (`record`, `process`, `tui`) as UTF-8 before recording or processing starts; an unreadable file prints an error and exits 1. The text goes to `run_pipeline(notes=)`
- `cmd_setup()` -- interactive wizard to set config values, including the session timeout in minutes
- `_process_audio()` -- shared by `record` and `process`: delegates to `run_pipeline` with a Rich progress reporter. A template (`RuntimeError`) failure at export prints `Error rendering note`; any other stage failure re-raises the underlying exception

**Configuration** lives at `~/.tinysteno/config.yaml`:
- `obsidian_vault`, `output_folder` -- Obsync output destination
- `api_key`, `base_url`, `model` -- LLM provider (default: ollama/llama3.2:3b)
- `whisper_model` -- model size for transcription (default: small)
- `diarization` -- adds `[You]` / `[Others]` labels to a stereo recording's transcript; both channels are transcribed whether or not it is on
- `auto_title`, `auto_tags` -- LLM-generated title and tags
- `sample_rate`, `channels` -- audio capture settings (`channels` is the microphone's input channels, 1 or 2; it does not control whether the WAV is stereo)
- `persona` -- active persona slug
- `session_timeout_minutes` -- maximum recording length in minutes, a finite number `>= 0` (default 75; `0` disables). `inf` and `nan` are rejected here, by `--timeout` and in `persona.yaml`
- `llm_context_tokens` -- context window the model is served with (default 8192; an int, validated by `_validate_config`); the orchestrator sizes its chunks from it
- `whisper_vad` (default true, local backend only), `whisper_prompt` (default empty, both backends) -- passed to Whisper as `vad_filter` / `initial_prompt` (`prompt` for the API)
- `hallucination_filter`, `bleed_filter` (default true) -- switch the two `hygiene` filters; `transcript_timestamps` (default true) -- `[hh:mm:ss]` on each transcript line
- `speaker_split` (default false) -- relabel the remote channel as `Others` / `Speaker 2` / ... (a split microphone or single-channel file as `Speaker 1` / `Speaker 2` / ..., never `You`) with the optional diarizer; only acts when `diarization` is also on. `max_speakers` (optional int >= 1, validated) caps the voices kept
- `whisper_api_diarize` (default false) -- API backend: ask a WhisperX-style server for speaker labels instead of running the local diarizer. `whisper_api_timeout` (default 600; a positive number of seconds, validated) -- per-request timeout of the API client
- `tui_theme` -- optional starting theme for the TUI (a Textual theme name)

### 2. `tinysteno/recorder.py` -- Audio Recording

**Class:** `AudioRecorder`

Captures microphone audio via `sounddevice` (PortAudio) and, on macOS, system audio loopback via ScreenCaptureKit. Output is a 16-bit PCM WAV file.

**Key behaviors:**
- Stereo output only when system audio was actually captured: left = mic, right = system audio. Otherwise the microphone is written mono (a 2-channel microphone is averaged to one), because a 2-channel file is read by the transcribers as mic + system
- Each channel is normalised to its own peak when the WAV is written (`_to_int16`); a channel whose peak is below `NORMALISE_FLOOR` (0.003) is converted at unit gain instead of being boosted into noise. `_to_int16` also returns each channel's gain relative to full scale (`1/peak`, or 1.0 for a channel left at unit gain), and once the WAV is written `stop()` saves them with `levels.save_levels` as `<wav filename>.levels.json` (`{"version": 1, "gains": [...]}`, one entry per WAV channel), atomically; a failure to write it is logged and never fails the save
- Old recordings made with a 2-channel microphone and no system audio are still read as mic/system
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

Transcribes audio files using `faster-whisper` (CTranslate2-optimized Whisper). With `device=auto`, `_detect_device()` picks CUDA (float16) when `ctranslate2.get_cuda_device_count()` is above zero (any error from that probe means no CUDA), otherwise CPU with int8; on Apple Silicon it passes `auto` + `auto` to CTranslate2, which has no MPS support and so runs on the CPU.

**Key behaviors:**
- `split_channels()` turns the file into 16kHz mono float32 channels keyed by speaker: one-channel audio gives a single unlabelled channel, two or more give `You` (left) and `Others` (right)
- One Whisper decode per channel, so both channels of a stereo file are always transcribed. A channel whose RMS is below `SILENCE_RMS` (`is_silent`) is skipped; with a valid levels file (`recorded_gains(path, data)`, shared with the API backend, reads it through `levels.load_levels`) the RMS is first divided by the channel's recorded gain (`audible_channels`), so a quiet but live microphone the recorder boosted is still skipped. The gains are passed to `assemble`; the audio handed to the decoder stays normalised
- `WhisperTranscriber(..., vad=True, prompt="", options=None)`: `vad` is faster-whisper's `vad_filter`, `prompt` its `initial_prompt`, `options` the `TranscriptOptions` handed to `assemble`
- The `diarization` setting only controls the `[You]` / `[Others]` labels (`diarised_text`); it never decides whether a channel is transcribed. A mono file with diarization on gives a plain transcript
- Module-level model cache `_MODEL_CACHE` -- one shared model instance per (size, device, compute_type)
- `transcribe()` returns: `text`, `diarised_text`, `duration_seconds`, `detected_language`, plus `segments` (time-ordered `Segment`s with a speaker) and `warnings` (for example a channel that had audio but no recognised speech). The result is built by `transcript.assemble`

### 3a. `tinysteno/api_transcriber.py` -- API Transcription

**Class:** `ApiTranscriber`

The same `transcribe()` contract against an OpenAI Whisper-compatible endpoint. Each non-silent channel is uploaded once as a 16kHz WAV with `response_format="verbose_json"`; segment times are read from the response and the channels are interleaved by `transcript.assemble`. When the server returns no usable segments, or any segment has a missing or non-finite start time, that channel falls back to a single block (`text`, or the joined segment texts) spanning the whole channel, so a stereo file still yields both speakers, one block each. `ApiTranscriber(..., prompt="", options=None, server_diarize=False, timeout=600.0)` sends `prompt` with each upload when set and has no VAD switch; `timeout` is the OpenAI client's per-request timeout.

Server-side speaker labels (only when the split is in effect, i.e. `options.speaker_split` and `diarize`, and `server_diarize` is on): the remote channel if uploaded, else the first uploaded channel (the only channel of a single-channel recording, which `_server_turns_for` maps to the `You` label `assemble` hands to `turns_for`, or the microphone when the system channel is silent), is uploaded with the non-standard `extra_body` fields `diarize: true` and, when the speaker hint is above 1, `max_speakers`. Each response segment's `speaker` string becomes a `Turn` (numbered in order of first appearance), supplied to `assemble` as `turns_for`, so the local diarizer is not used and the same `apply_policy` runs. On `BadRequestError` / `UnprocessableEntityError` the upload is repeated without the fields, the channel keeps its plain labels and `REJECTED_WARNING` is appended to the result's warnings. A response with no usable speaker labels leaves the labels unchanged; when that channel has segments, `NO_LABELS_WARNING` ("The transcription server returned no speaker labels; channel labels were used.") is appended. Segment times are clamped so `start >= 0` and `end >= start`; a time that is not a finite number (including one too large for a float) counts as missing. When the system channel has audio but no recognised speech, the server's labels belong to that channel, so the microphone is not split and the local diarizer is not used as a fallback. Progress: decoding takes the first `SPLIT_DECODE_SHARE` (0.7) of the bar when a split is in effect

### 3b. `tinysteno/transcript.py` -- Segments and Assembly

The one place both backends build their result. `Segment(start, end, text, speaker)`; `YOU` / `OTHERS` labels; `is_silent(samples, gain=1.0)`; `audible_channels(audio, gains)`; `interleave()` sorts by start time with ties going You, then Others, so the order is stable between runs; `plain_text()`; `labelled_text()` (one `[You] text` line per segment, `None` when no segment has a speaker); `render_transcript(segments, *, labels, timestamps)`, one `[hh:mm:ss] [Speaker] text` line per segment with each prefix only when enabled (plain text when neither applies), which is what the note's transcript is built from; and `assemble(channels, audio, *, diarize, language, options=None, turns_for=None, on_progress=None, gains=None)`, which stamps the channel label on each segment, runs the `hygiene` filters selected by `TranscriptOptions(drop_hallucinations=True, suppress_bleed=True, speaker_split=False, max_speakers=None)`, runs the speaker split (below), interleaves, adds a warning per non-silent channel that produced no speech, and returns the transcriber result dict. `gains` (label to recorder gain, from the levels file) makes the cross-channel decisions use the levels as recorded: the RMS callables given to `suppress_bleed` and the silence test behind the loud-but-empty warnings divide by the channel's gain; `drop_hallucinations` compares levels within one channel and is unaffected. Without gains the levels in the file are used, as before.

The levels file (`tinysteno/levels.py`): `levels_path(wav)` (`Meeting-1.wav.levels.json`), `save_levels(wav, gains)` and `load_levels(wav, channels)`, which returns the gains or `None` (treated as absent) when the file is missing, older than the recording, unreadable, of another version, or not a list of finite positive numbers with exactly one entry per channel of the WAV. Without it (a recording made before this version, moved or processed without the file, or a third-party file) everything behaves as before, and a listen-only call on speakers can then mislabel the remote speaker's line as `[You]` and drop the `[Others]` copy, because the bleed was boosted to full scale on the microphone.

The speaker split step (`split_enabled(options, diarize)`, then `_split_speakers`) runs after the filters, only when `options.speaker_split` and `diarize` are both on. It splits one channel: `Others` if it has segments, else `You`, else a single-channel file's unlabelled channel (handed to `turns_for` as `You`); the microphone of a recording whose system channel has recognised speech is never split. On `Others` the most-heard voice keeps `Others`; on the microphone or a single-channel file it is labelled `Speaker 1` (`label_segments(..., first_label="Speaker 1")`), so no segment of a split microphone or single channel is labelled `You`; this applies equally to local and server turns. It is skipped without a warning when no channel qualifies, `max_speakers == 1`, the channel has under `MIN_AUDIO_SECONDS` (30 s) of audio, or `apply_policy` returns `None`. `turns_for(label, audio)` supplies the raw turns (default: a lazy call to `sherpa_diarizer.diarize`, so `sherpa_onnx` is imported only here). A `DiarizerUnavailable` adds its message to the result's warnings; any other exception raised outside the diarizer is logged and adds "Speaker separation failed; channel labels were used." (an ordinary exception inside `sherpa_diarizer.diarize` is already a `DiarizerUnavailable`: "Speaker separation could not run: <reason>"); either way the channel keeps its labels and the result's `split_failed` is true (it is false for a split that ran or was skipped quietly), so the pipeline does not save it. `on_progress` receives the diarizer's 0-1 fraction.

The transcript sidecar: `sidecar_path(wav)` (the full recording filename plus `.transcript.json`: `Meeting-1.wav.transcript.json`), `save_transcript(wav, result, fingerprint)` (atomic write of `version` (`SIDECAR_VERSION`, now 2), `fingerprint`, `detected_language`, `duration_seconds`, `warnings` and `segments`; nothing is written for a result with no segment list or a recording that does not exist; any failure is logged as a warning, never raised) and `load_transcript(wav, fingerprint)`, which returns the result dict or `None` when the recording or sidecar is missing, the sidecar is older than the recording, of another `SIDECAR_VERSION`, saved under a different fingerprint, or fails validation (malformed JSON, segment times or types; logged at INFO). The file holds the full meeting text and sits in the recordings folder, which by default is inside the vault.

### 3c. `tinysteno/hygiene.py` -- Transcript Clean-up

Pure functions over `Segment` lists, called by `transcript.assemble`; every removal is logged at INFO with reason, speaker, start and text. Thresholds are module constants fixed deliberately.
- `drop_hallucinations(segments, rms=None)` -- per channel, in time order: removes caption credits only when they are the whole line or start it (`Subtitles by ...`, `Thanks for watching`, `Please subscribe`, any `amara.org`); collapses a run of 5 or more identical lines to the first; and removes a line of at most 5 words that occurs at least 3 times only when it is isolated (no neighbour within 10 s) and its `rms(start, end)` level is under 0.25 of the channel's median segment level. With `rms=None` the last rule never applies
- `suppress_bleed(mic, system, mic_rms, system_rms)` -- for each system segment, the nearest unpaired mic segment starting within 3 s; when both have at least 15 characters and their word sets overlap (Jaccard >= 0.5) the copy with the lower level goes, the mic copy on a tie. Levels are measured on the channel a segment came from, not its speaker label
- Known limits: two people genuinely saying the same long sentence at once lose one copy; a real, quiet, isolated short line repeated three times can be dropped. `hallucination_filter: false` / `bleed_filter: false` switch them off

### 3d. `tinysteno/speaker_split.py` -- Speaker Split Policy

Pure policy with no diarization-library dependency (the clustering-threshold ramp, phantom-cluster and dominance rules are ported from OpenWhispr, MIT; keeping the base label on the dominant voice from StenoAI, MIT). `Turn(start, end, speaker)`; `DiarizerUnavailable(RuntimeError)`, whose message is shown to the user as a warning; `cluster_threshold(seconds)` (0.6 up to 15 minutes, rising linearly to 0.7 at 60 minutes); `apply_policy(turns, *, max_speakers=None)`, which drops phantom voices (under 1.0 s of speech in total, or under 10% of the speech with a mean turn under 3.0 s), folds extra voices into the nearest kept turn when capped, and returns `None` (keep the channel labels) when nothing is left, the cap is 1, only one voice remains or one voice has 93% or more of the speech; otherwise the turns renumbered by speech share, voice 0 the most heard; `label_segments(segments, turns, base_label, *, first_label=None)`, which gives each segment on the base channel the speaker it overlaps most (when none overlaps, the turn whose midpoint is nearest the segment's midpoint, if within 2.0 s, else the base voice): voice 0 is labelled `first_label` (default `base_label`), the rest become `Speaker N` (N = index + 1). Segments on other channels are untouched. The constants are fixed deliberately and the clusterer is never given a speaker count (`--speakers` is applied afterwards as a cap).

### 3e. `tinysteno/sherpa_diarizer.py` -- Local Diarizer Adapter

A thin adapter over the optional `sherpa-onnx` package (the `diarize` extra in `pyproject.toml`), imported only inside `diarize`. `default_model_dir()` is `~/.tinysteno/models/diarize`, resolved at call time. `ensure_models()` downloads whichever of the two models is missing (the pyannote segmentation archive `SEGMENTATION_URL`, extracted to `sherpa-onnx-pyannote-segmentation-3-0/model.onnx`, and the 3D-Speaker embedding `EMBEDDING_URL`), HTTPS from `github.com` only, at most `MAX_DOWNLOAD_BYTES` (100 MB) each, into a scratch directory moved into place only when complete, each file verified against its pinned SHA-256 (`SEGMENTATION_SHA256`, `EMBEDDING_SHA256`, of the file as downloaded). `diarize(audio, *, on_progress=None, num_threads=2)` runs the offline diarizer on 16 kHz mono float32 audio (clustering threshold from `cluster_threshold`, `num_clusters=-1`) and returns `Turn`s sorted by start time. Guarantee: `diarize` and `ensure_models` raise only `DiarizerUnavailable` (original as `__cause__`) for any ordinary `Exception`; `KeyboardInterrupt` and `SystemExit` propagate. A missing extra gives the message `Speaker separation needs the optional extra: pip install "tinysteno[diarize]"`.

### 4. `tinysteno/orchestrator.py` -- LLM Orchestration

**Class:** `Orchestrator`

Handles all LLM communication via the OpenAI-compatible API. Transforms raw transcripts into structured data matching a persona's schema.

**Key behaviors:**
- `Orchestrator(api_key, base_url, model, context_tokens=8192)`: chunk size is `max(context_tokens, 500) * 2 * 0.6` characters (2 characters per token, 60% of the window), so the default is 9,830 characters and the floor 600
- Chunking: a transcript that fits is one chunk; otherwise whole lines are packed with no overlap, and a line longer than a chunk is split at the last space within the limit
- `summarize(transcript, persona, context=None)` takes a `PromptContext(meeting_date, has_speaker_labels, has_you_label, has_numbered_speakers, has_timestamps, notes)` (every flag defaults to false). Every user message carries a `Context:` block: with labels, either that `[You]` is the person who recorded the meeting and `[Others]` everyone else on the call (only when `has_you_label`: some rendered line is labelled `You`) or that no label marks the recorder and `[Others]`, where it appears, is the other side of the call; with `has_numbered_speakers` as well, that numbered speakers are separated automatically and approximately and are not to be given actions unless the transcript names them (`Speaker 2, Speaker 3 ...` as further voices when `[You]` is present, `Speaker 1, Speaker 2 ...` numbered from the most heard otherwise); without labels, that there are none and owners must not be guessed; the `[hh:mm:ss]` format when timestamped, the meeting date for resolving relative dates, an instruction to keep proposals, open points and decisions distinct, `This is part n of m` for chunks, and the notes between `<notes>` tags as information, never instructions (a `</notes>` inside them is defanged)
- Summarizes each chunk independently, then merges partial results in at most two rounds (`_reduce`): partials are packed into groups that fit one chunk, each group is merged by the LLM, and what remains after the second round is combined locally with a warning
- The merge call asks the LLM to combine the partial results and drop duplicates; the code itself does no semantic de-duplication
- Each LLM call is tried twice (`_extract`): a call that raises, or returns no usable fields, is retried once
- Graceful degradation: if LLM returns invalid JSON, attempts recovery. Chunks that fail twice are left out and counted. If every chunk fails, the result is the defaults (empty string / empty list)
- If the merge call fails, the partial results are combined locally (`_merge_locally`: string fields joined with a blank line, list fields concatenated with exact duplicates removed) instead of being thrown away
- `Orchestrator.warnings` is a public list, reset by each `summarize()`, that says what was lost (the summary could not be generated, N of M sections failed, the merge fell back to local combination)
- Also generates: titles (`generate_title`) and tags (`generate_tags`) via separate LLM calls on the summary digest

**Constants:**
- Chunk size: `context_tokens * 2 * 0.6` chars (`_CHARS_PER_TOKEN`, `_CHUNK_CONTEXT_SHARE`, floor `_MIN_CONTEXT_TOKENS` = 500); no overlap
- Merge rounds: 2 (`_REDUCE_ROUNDS`)
- LLM timeout: 120 seconds
- Title context: first 4,000 chars of the text passed in (the pipeline passes `summary_digest`)
- Tag context: first 4,000 chars of the text passed in

### 5. `tinysteno/obsidian.py` -- Obsidian Export

**Class:** `ObsidianExporter`

Renders meeting notes using Jinja2 templates and writes Markdown files to the Obsidian vault.

**Key behaviors:**
- Renders the persona's `template.md` with merged context (metadata + schema fields)
- Uses Jinja2 `StrictUndefined` -- template references to missing fields raise errors
- Frontmatter is whatever the persona's template renders (the exporter adds none itself)
- Tag deduplication: persona static tags first, LLM-generated tags appended (`dict.fromkeys`, so duplicates are removed)
- Names the file `<title> (<date>).md`, sanitising filenames (`: \ / * ? " | < >` become `-`, runs of `-` collapse, leading and trailing `-` are stripped; there is no length limit)
- Never overwrites: if the name exists the note is written as `<title> (<date>) 2.md`, then ` 3`, and so on, taking the first free name
- When processing warnings exist, a `## Processing warnings` section is appended (a fenced code block left open by the template is closed first)

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
- Reserved field names cannot be overridden: `title`, `date`, `duration`, `transcript`, `detected_language`, `generated_tags`, `tags`, `processing_warnings`
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

`run_pipeline(wav_path, name, config, persona, timestamp, *, transcriber, orchestrator_cls, exporter_cls, reporter=None, notes="", retranscribe=False) -> PipelineResult` runs transcribe, summarize, title and tags, and export, reporting each `Stage` (`TRANSCRIBE`, `SUMMARIZE`, `METADATA`, `EXPORT`) to a `PipelineReporter` (`stage_started`, `stage_progress`, `stage_finished`, `stage_skipped`; `NullReporter` ignores them). The transcribe stage computes `transcript_fingerprint(wav_path, config)`, a dict of the recording's byte `size`, `backend`, `model` (`whisper_api_model` for the API backend, else `whisper_model`), `prompt`, `vad`, `hallucination_filter`, `bleed_filter`, `speaker_split` (true only when `speaker_split` and `diarization` are both on) and `max_speakers` (only while the split is in effect, else `None`), plus, on the API backend only, `whisper_base_url` (the effective transcription server: `whisper_base_url`, else `base_url`, else empty) and `whisper_api_diarize: true` only when the split is in effect on the API backend with that option on (absent otherwise, so setting the option while it has no effect does not make a saved transcript stale). Every fingerprint carries the `speaker_split` and `max_speakers` keys, so a transcript saved before this feature does not match and is transcribed again once; with no recording on disk there is no fingerprint, so nothing is reused or saved. Unless `retranscribe` is set it tries `load_transcript(wav_path, fingerprint)` and reuses a match (reporting full progress); otherwise it transcribes and calls `save_transcript`, except for a result whose `split_failed` is true (the speaker split was attempted and failed), which is not saved so the next run tries the split again. The note's transcript is `render_transcript(segments, labels=diarization, timestamps=transcript_timestamps)`, so `transcript_timestamps` applies at render time and needs no new transcription, and neither does `diarization` unless `speaker_split` is on (then it is part of the fingerprint and toggling it transcribes again); a result without segments falls back to `diarised_text` or `text`. `llm_context_tokens` is passed to the orchestrator as `context_tokens` when present, and `summarize` receives a `PromptContext` built from the meeting timestamp, whether labels and timestamps are shown, and `notes`; `PromptContext.has_you_label` is true when labels are shown and some segment is labelled `You` (for a result without segments: some `diarised_text` line starts with `[You]`); `PromptContext.has_numbered_speakers` is true when labels are shown and some segment is labelled `Speaker N`, and then the model is told that numbered speakers are approximate and not to assign actions to them unless the transcript names them. Title and tag generation are fed `summary_digest(persona, data, limit=4000)`: the non-empty string fields, then every list item, in schema order, one per line, cut to the limit (the stage is skipped when there is nothing to digest). A failure in any stage is re-raised as `PipelineError(stage)` with the original exception as `__cause__`. `PipelineResult` carries `wav_path`, `note_path`, `title`, `duration_seconds`, `no_speech`, `warnings` (transcriber and orchestrator warnings merged by `_merge_warnings`, each made one line (whitespace runs collapsed to single spaces, ends trimmed, an empty one dropped) and cut to 300 characters ending in `…`, without duplicates) and `reused_transcript` (true when a saved transcript was used; the CLI prints `Using the saved transcript for this recording (--retranscribe to transcribe again).` and the TUI Result screen notifies `Used the saved transcript for this recording.`; it is not written into the note). The warnings are also passed to the exporter as `processing_warnings`; the CLI prints them as `Warning:` lines and the TUI shows them as notifications on the Result screen and as lines on the no-speech screen. Nothing here prints or imports a UI library, so the CLI (Rich) and the TUI (Textual) share it.

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
   ├── Write WAV to recordings/ (stereo only if system audio was captured, else mono)
   └── Return WAV path

3. Transcriber (transcriber.py:WhisperTranscriber)
   ├── Read WAV file via soundfile
   ├── Split into 16kHz mono channels (You / Others for stereo)
   ├── Run faster-whisper once per non-silent channel
   ├── Drop hallucinated lines and speaker bleed (hygiene, in transcript.assemble)
   ├── Interleave segments by start time (transcript.assemble)
   ├── Label speakers if diarization is enabled
   └── Return {text, diarised_text, duration, language, segments, warnings}
   (pipeline.py then saves <wav>.transcript.json (matched to the recording size and settings), or reuses it on a retry, and renders
    the transcript one [hh:mm:ss] [Speaker] line per segment)

4. Orchestrator (orchestrator.py:Orchestrator)
   ├── Chunk transcript on line boundaries if over 0.6 x llm_context_tokens x 2 chars (each LLM call retried once)
   ├── Build user message (persona system prompt + Context: block + transcript)
   ├── Call LLM (OpenAI-compatible API)
   ├── Parse JSON response, validate against schema
   ├── Merge partial results if chunked (two bounded rounds; local merge if a call fails)
   ├── Generate title from the summary digest (separate LLM call)
   ├── Generate tags from the summary digest (separate LLM call)
   └── Return {schema_fields, title, generated_tags}

5. Exporter (obsidian.py:ObsidianExporter)
   ├── Merge metadata + schema data into context
   ├── Render persona's Jinja2 template
   ├── (frontmatter comes from the template)
   ├── Append processing warnings, if any
   ├── Write .md file to Obsidian vault (never overwriting an existing note)
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
  │     ├── scipy                (audio resampling)
  │     └── tinysteno.transcript (segments, interleaving, result assembly)
  │           ├── tinysteno.hygiene (hallucination and bleed filters)
  │           ├── tinysteno.speaker_split (policy, no heavy deps)
  │           └── tinysteno.sherpa_diarizer (lazy; optional sherpa-onnx)
  ├── tinysteno.api_transcriber
  │     ├── openai               (Whisper API client)
  │     ├── tinysteno.transcriber (split_channels)
  │     └── tinysteno.transcript
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
llm_context_tokens: 8192   # the context window the model is served with
whisper_model: small
whisper_vad: true         # local backend only
whisper_prompt: ""
hallucination_filter: true
bleed_filter: true
diarization: false        # only adds [You]/[Others] labels
speaker_split: false      # optional; needs diarization and the `diarize` extra
# max_speakers: 4          # optional cap on voices kept
# whisper_api_diarize: false  # API backend: server-side speaker labels
# whisper_api_timeout: 600    # seconds per transcription request
transcript_timestamps: true
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
| `test_transcriber.py` | WhisperTranscriber | Per-channel decoding, silent channels, labels, resampling |
| `test_api_transcriber.py` | ApiTranscriber | One upload per channel, segment interleaving, single-block fallback |
| `test_transcript.py` | `tinysteno/transcript.py` | Interleave order, labelled text, rendering, assembly, speaker-split step, warnings, transcript sidecar |
| `test_levels.py` | `tinysteno/levels.py`, recorder gains, level-aware transcription | Gains written on save, strict validation, listen-only bleed, quiet live microphone, missing or invalid file |
| `test_hygiene.py` | `tinysteno/hygiene.py` | Caption credits, stuck and stranded repeats, bleed suppression |
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
│   ├── transcriber.py          # Local Whisper transcription (one decode per channel)
│   ├── api_transcriber.py      # Whisper API transcription (one upload per channel)
│   ├── transcript.py           # Segment type, interleaving, rendering, result assembly, transcript sidecar
│   ├── levels.py               # The recorder's per-channel gains file (<wav>.levels.json)
│   ├── hygiene.py              # Hallucination and speaker-bleed filters
│   ├── speaker_split.py        # Speaker-split policy: phantom voices, dominance, relabelling
│   ├── sherpa_diarizer.py      # Optional sherpa-onnx diarizer adapter and model download
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
    ├── test_api_transcriber.py # API transcriber tests
    ├── test_transcript.py      # Segment and assembly tests
    ├── test_hygiene.py         # Transcript filter tests
    ├── test_speaker_split.py   # Speaker-split policy tests
    ├── test_sherpa_diarizer.py # Diarizer adapter and download tests
    └── test_tui_*.py           # TUI screen tests (pilot-driven)
```

---

## Design Decisions and Rationale

**Why linear pipeline instead of event-driven?**
The recording -> transcription -> summarization -> export flow is inherently sequential. Each stage depends on the previous stage's output. An event-driven architecture would add complexity without benefit for this use case.

**Why persona system over fixed output formats?**
Users need different meeting note formats (RCA, 1-on-1, sprint retrospective). Hard-coding these would require code changes. The persona system makes the output format data-driven: each persona is a directory with a schema, system prompt, and Jinja2 template. Adding a new persona requires zero code changes.

**Why chunked LLM processing?**
Whisper transcripts of long meetings (1+ hours) can exceed 50,000 characters, beyond most LLM context windows for structured extraction. The chunking strategy splits transcripts on line boundaries into chunks sized from `llm_context_tokens` (60% of the window at 2 characters per token), extracts from each, then merges in at most two bounded rounds. The merge step asks the model to deduplicate list fields; if it fails, the local merge removes exact duplicates.

**Why faster-whisper over openai-whisper?**
CTranslate2-optimized Whisper is 4x faster on CPU with int8 quantization, making real-time or near-real-time transcription viable on consumer hardware without a GPU.

**Why stereo loopback for diarization?**
True speaker diarization requires an ML model. By placing mic on the left channel and system audio on the right, the transcriber can separate "You" from "Others" without a diarization model. Both channels are always transcribed; the `diarization` setting only decides whether the labels are shown. This is a pragmatic trade-off: it works well when the user speaks through their mic and others speak through the call (Zoom/Teams).

---

## Knowledge Graph (graphify)

An interactive architecture graph is available at `graphify-out/graph.html`. The graph encodes all code relationships (imports, calls, references, conceptual links) extracted from the codebase. Key findings from the analysis:

- **18 communities** identified by community detection, aligning closely with module boundaries
- **375 nodes** and **727 edges** capturing structural and semantic relationships
- **God nodes** (most connected): `_make_orchestrator()` (35), `Orchestrator` (24), `AudioRecorder` (24), `load_persona()` (21), `Persona` (18)
- **Bridge nodes** cross module boundaries: `Orchestrator` connects the orchestrator module to CLI and tests; `AudioRecorder` connects recording to dependencies, CLI, and macOS loopback; `Persona` connects the persona system to CLI, orchestrator tests, and obsidian export
- **Cohesion warning:** The `Main CLI & Setup` community has low cohesion (0.054) -- nodes in this community are weakly interconnected, suggesting the CLI module may benefit from splitting as it grows