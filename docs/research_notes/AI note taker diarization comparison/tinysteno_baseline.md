# TinySteno baseline: speaker diarization and transcript processing/summarization

State described: branch `main`, HEAD `61b8881` ("chore: bump version to 0.2.4", committed 2026-11-14), read on 2026-11-16. Package version 0.2.4 ([pyproject.toml:3](../../../pyproject.toml)). All findings come from reading the code; nothing was executed. Citations are `file:lines` relative to the repo root `/Users/idspringer/Repos/TinySteno`.

## Diarization: exactly how are speakers distinguished?

### Takeaway
TinySteno has no acoustic speaker diarization. It labels speech by recording channel only: left channel (microphone) is `[You]`, right channel (system audio) is `[Others]`, so it can never produce more than two labels and never a person's name. The local faster-whisper backend interleaves the two channels by Whisper segment start time; the API backend does not interleave at all and emits one `[You]` block followed by one `[Others]` block.

### Cited Findings

How the stereo WAV is built (recorder)
- The recorder documents the contract: when a loopback source is found the WAV is stereo, left = microphone, right = system audio, "left = You, right = Others" — [tinysteno/recorder.py:54-58](../../../tinysteno/recorder.py)
- Mic and system audio are captured by two independent streams, each appending raw float32 to its own temp `.f32` file; the WAV is only assembled in `stop()` — [tinysteno/recorder.py:118-141, 266-274, 345-385](../../../tinysteno/recorder.py)
- `_mix_stereo` reduces the mic to its first channel, averages loopback channels to mono, pads the shorter stream with trailing silence, and column-stacks them as `[mic, loopback]` — [tinysteno/recorder.py:468-479](../../../tinysteno/recorder.py)
- The two streams are started one after the other (mic first, then loopback; on macOS the ScreenCaptureKit thread may wait up to 5 s for shareable content) and are joined sample-index-for-sample-index from sample 0. There is no cross-stream timestamping, start-offset compensation or drift correction; only end padding — [tinysteno/recorder.py:276-305, 468-479](../../../tinysteno/recorder.py); [tinysteno/_macos_loopback.py:270-278](../../../tinysteno/_macos_loopback.py)
- The loopback is resampled to the mic rate only if the detected macOS loopback rate differs by more than 1 Hz (whole-signal `scipy.signal.resample`) — [tinysteno/recorder.py:463-465](../../../tinysteno/recorder.py)
- macOS system audio is requested as 1 channel, capturing all apps on the first display (not per-app/per-participant), and any multi-channel buffer is averaged to mono — [tinysteno/_macos_loopback.py:153-155, 284-297](../../../tinysteno/_macos_loopback.py)
- Both channels are peak-normalized jointly using a single global maximum (`np.max(np.abs(audio_data))`), then written as 16-bit PCM at the configured rate (default 44100 Hz) — [tinysteno/recorder.py:375-385](../../../tinysteno/recorder.py); [tinysteno/main.py:60](../../../tinysteno/main.py)
- If no loopback data exists, the WAV is just the mic data with the mic's channel count (mono by default, `channels: 1`) — [tinysteno/recorder.py:455-456](../../../tinysteno/recorder.py); [tinysteno/main.py:61](../../../tinysteno/main.py)
- System loopback discovery off macOS is name-based: a WASAPI device containing "loopback" (or the default output device) on Windows, the first device containing "monitor" on Linux; failure silently leaves the recording mic-only — [tinysteno/recorder.py:166-209, 292-305](../../../tinysteno/recorder.py)

Local backend (faster-whisper)
- Diarization runs only if `diarize` is true and the file has 2 or more channels — [tinysteno/transcriber.py:151-160](../../../tinysteno/transcriber.py)
- `_diarize` resamples channel 0 and channel 1 separately to 16 kHz, runs Whisper on each independently, tags left segments `You` and right segments `Others`, sorts all segments by segment start time, and emits one line per segment as `[speaker] text` — [tinysteno/transcriber.py:191-207](../../../tinysteno/transcriber.py)
- The only timing information used is the Whisper segment `start`; segment end, word timestamps and the timestamps themselves are discarded after sorting (output lines carry no time) — [tinysteno/transcriber.py:186-189, 206-207](../../../tinysteno/transcriber.py)
- Whisper is invoked as `self._model.transcribe(audio, beam_size=5)` with no other arguments: no `vad_filter`, no `word_timestamps`, no `language`, no `initial_prompt` — [tinysteno/transcriber.py:176, 188](../../../tinysteno/transcriber.py). A grep of `tinysteno/` for `vad_filter|word_timestamps|pyannote|embedding|initial_prompt|BatchedInference` returns no matches.
- So output is chronologically interleaved in the local backend, at Whisper-segment granularity — [tinysteno/transcriber.py:202-207](../../../tinysteno/transcriber.py); matches the README example — [README.md:339-345](../../../README.md)

API backend (OpenAI-compatible Whisper endpoint)
- Diarization runs only if the file has exactly 2 channels and `diarize` is true — [tinysteno/api_transcriber.py:54-58](../../../tinysteno/api_transcriber.py)
- `_transcribe_diarized` uploads the left channel, then the right channel, and builds `"[You] {left_text}\n\n[Others] {right_text}"`: two monolithic blocks, not interleaved. The plain `text` is the two channel texts concatenated — [tinysteno/api_transcriber.py:73-107](../../../tinysteno/api_transcriber.py)
- The API is asked for `response_format="verbose_json"`, but `_parse_response` reads only `text`, `language` and `duration`; the segment list that verbose_json returns is never used — [tinysteno/api_transcriber.py:126-149](../../../tinysteno/api_transcriber.py)
- README says the API backend "supports the same features as the local backend" including "Stereo diarization (two API calls, one per channel)" and shows an interleaved example; the code does not interleave for the API backend — [README.md:339-345, 398-403](../../../README.md); contradicted by [tinysteno/api_transcriber.py:101](../../../tinysteno/api_transcriber.py)

Label count and wiring
- The only speaker strings in the codebase are the literals `"You"` and `"Others"` — [tinysteno/transcriber.py:203-204](../../../tinysteno/transcriber.py); [tinysteno/api_transcriber.py:101](../../../tinysteno/api_transcriber.py). Maximum labels: 2.
- Diarization is off by default (`"diarization": False`) and is passed to the transcriber from config — [tinysteno/main.py:56](../../../tinysteno/main.py); [tinysteno/pipeline.py:131-135](../../../tinysteno/pipeline.py)
- The pipeline uses `diarised_text` if present, otherwise `text`, as the single transcript for both the LLM and the note — [tinysteno/pipeline.py:140, 155, 191](../../../tinysteno/pipeline.py)
- SPEC states the design intent: "True speaker diarization requires an ML model. By placing mic on the left channel and system audio on the right, the transcriber can separate 'You' from 'Others' without a diarization model... it works well when the user speaks through their mic and others speak through the call" — [SPEC.md:376-377](../../../SPEC.md)

Edge cases, as the code behaves
- Mono recording (no loopback captured): `is_stereo` is false, `diarised_text` is `None`, transcript is unlabeled — [tinysteno/transcriber.py:152, 158-160](../../../tinysteno/transcriber.py); [tinysteno/api_transcriber.py:55-71](../../../tinysteno/api_transcriber.py)
- Stereo recording with diarization off (the default): `convert_to_16khz_array` keeps only channel 0 (`data = data[:, 0]`), and both backends pass the full stereo array through it for the plain transcript. The right channel (system audio, i.e. the remote participants) is therefore not transcribed at all — [tinysteno/transcriber.py:56-57, 154-155](../../../tinysteno/transcriber.py); [tinysteno/api_transcriber.py:62, 116](../../../tinysteno/api_transcriber.py)
- Several remote participants: all arrive in the single mono system-audio channel and all get the one label `[Others]` — [tinysteno/_macos_loopback.py:153-155, 297](../../../tinysteno/_macos_loopback.py); [tinysteno/transcriber.py:204](../../../tinysteno/transcriber.py)
- Several people in one room on the mic: all are on the left channel and all get `[You]` — [tinysteno/recorder.py:471](../../../tinysteno/recorder.py); [tinysteno/transcriber.py:203](../../../tinysteno/transcriber.py)
- Echo/bleed (speakers audible to the mic): there is no echo cancellation, cross-channel deduplication or energy gating anywhere in recorder or transcribers; each channel is transcribed independently and both results are kept — [tinysteno/transcriber.py:196-207](../../../tinysteno/transcriber.py); [tinysteno/recorder.py:468-479](../../../tinysteno/recorder.py)
- A stereo file whose channels are not mic/system (e.g. `tinysteno process` on any stereo audio, or a 2-channel mic with no loopback, which the recorder writes as a 2-channel WAV) is still labeled left = `You`, right = `Others`; the transcribers only check channel count — [tinysteno/recorder.py:455-456](../../../tinysteno/recorder.py); [tinysteno/transcriber.py:152](../../../tinysteno/transcriber.py); [tinysteno/main.py:355-376](../../../tinysteno/main.py). The setup wizard tells the user diarization "requires stereo recording (channels = 2)" and defaults `channels` to 2 when diarization is enabled, although `channels` sets mic input channels — [tinysteno/main.py:630-654](../../../tinysteno/main.py); [README.md:300-301](../../../README.md)
- In the API backend a silent diarized recording yields the non-empty string `"[You] \n\n[Others] "`, so the pipeline's "no speech" check (`transcript.strip()`) does not trigger — [tinysteno/api_transcriber.py:101](../../../tinysteno/api_transcriber.py); [tinysteno/pipeline.py:140-144](../../../tinysteno/pipeline.py)

### Inferences
- With echo/bleed (laptop speakers, no headphones), remote speech will likely be transcribed twice: once correctly as `[Others]` and again, degraded, as `[You]`. Nothing in the code prevents it.
- Whisper segments can span many seconds, and each line is ordered only by its start, so rapid back-and-forth or overlapping speech will be ordered coarsely; a long `[You]` segment that started first is printed whole before an `[Others]` interjection that happened mid-way.
- Because Whisper is run without VAD on each channel, the long silences on a channel while the other side talks are fed to the model as-is; Whisper is known to hallucinate text on silence, which would then appear as spurious labeled lines. This is a known Whisper trait, not something measured here.
- Any start offset between mic and loopback streams (sequential start, ScreenCaptureKit setup) shifts one channel's timestamps relative to the other, which can misorder adjacent turns in the interleave.
- Joint peak normalization means a loud channel leaves the other channel quiet in the 16-bit file; this may reduce accuracy on the quieter side.
- The "diarization" of TinySteno is better described as two-channel source separation by capture path. Tools using pyannote-style embeddings/clustering, or per-participant streams from a meeting bot, are doing a categorically different thing.

### Gaps
- No measurement of diarization accuracy (DER) or interleave ordering error exists in the repo; tests only cover mechanics with mocks.
- Actual size of the mic/loopback start offset and clock drift on real hardware is not determinable from code.
- Whether bleed duplicates occur in practice depends on the user's audio setup (headphones vs speakers); not testable by reading code.

## Transcription performance: models, devices, batch vs live, resampling, passes

### Takeaway
Transcription is strictly post-meeting and whole-file: the WAV is loaded fully into memory, resampled to 16 kHz in one FFT, and decoded sequentially by faster-whisper (default model `small`, beam 5). Local diarization costs three full Whisper passes (the mic channel is transcribed twice); the API path costs two uploads with no audio chunking.

### Cited Findings
- Two backends chosen by `transcription_backend` / `--backend`: `local` → `WhisperTranscriber`, `api` → `ApiTranscriber` — [tinysteno/main.py:129-142, 749](../../../tinysteno/main.py)
- Local defaults: `whisper_model: small`, `whisper_device: auto`, `whisper_compute_type: auto`; README lists sizes tiny, base, small, medium, large (the string is passed straight to `WhisperModel`) — [tinysteno/main.py:49-52](../../../tinysteno/main.py); [README.md:266-267](../../../README.md); [tinysteno/transcriber.py:129-131](../../../tinysteno/transcriber.py)
- Device detection: CUDA if CTranslate2's `get_supported_devices()` lists it → `("cuda", "float16")`; Darwin arm64 → `("auto", "auto")` with a comment claiming CTranslate2 4+ MPS support; otherwise `("cpu", "int8")` — [tinysteno/transcriber.py:75-97](../../../tinysteno/transcriber.py)
- Models are cached per process by `(model_size, device, compute_type)` — [tinysteno/transcriber.py:72, 127-132](../../../tinysteno/transcriber.py)
- Decoding call is `WhisperModel.transcribe(audio, beam_size=5)`, the sequential API; faster-whisper's batched pipeline is not used — [tinysteno/transcriber.py:176, 188](../../../tinysteno/transcriber.py)
- Language is auto-detected per call; only the mono pass's `info.language` is kept — [tinysteno/transcriber.py:176, 184](../../../tinysteno/transcriber.py)
- Progress for the local backend is `seg.start / duration` from the first (mono) pass only; the two diarization passes report no progress — [tinysteno/transcriber.py:178-183, 186-189](../../../tinysteno/transcriber.py)
- Passes over the audio, local + diarize: (1) `_run_whisper` on channel 0 for `text`, (2) `_run_whisper_segments` on channel 0, (3) `_run_whisper_segments` on channel 1. Pass 1 and 2 decode the identical signal — [tinysteno/transcriber.py:154-160, 196-200](../../../tinysteno/transcriber.py)
- Passes, API + diarize: two sequential uploads (left then right), progress reported as 0.0 / 0.5 / 1.0 — [tinysteno/api_transcriber.py:80-99](../../../tinysteno/api_transcriber.py)
- Resampling: `convert_to_16khz_array` takes channel 0, casts to float32, and uses `scipy.signal.resample` (FFT-based) over the entire signal at once to 16 kHz — [tinysteno/transcriber.py:50-65](../../../tinysteno/transcriber.py)
- The whole file is read into memory with `sf.read(str(path))` (local backend, soundfile default dtype) or `dtype="float32"` (API backend) before anything else — [tinysteno/transcriber.py:151](../../../tinysteno/transcriber.py); [tinysteno/api_transcriber.py:54](../../../tinysteno/api_transcriber.py)
- API backend: default model `whisper-1`, client timeout 180 s, each channel written to a temp 16 kHz WAV and uploaded in a single `audio.transcriptions.create` request; no splitting of long audio — [tinysteno/api_transcriber.py:31-37, 109-135](../../../tinysteno/api_transcriber.py)
- API credentials fall back to the summarizer's `api_key`/`base_url` — [tinysteno/main.py:133-137](../../../tinysteno/main.py)
- Transcription is not live: the recorder only writes raw floats during the meeting and the WAV at `stop()`; `run_pipeline` takes a finished `wav_path`; the TUI recording screen shows level meters only (a grep for "transcript" in `tinysteno/tui/recording.py` returns nothing) — [tinysteno/recorder.py:345-385](../../../tinysteno/recorder.py); [tinysteno/pipeline.py:115-135](../../../tinysteno/pipeline.py); [README.md:11](../../../README.md)
- Default session cap is 75 minutes — [tinysteno/session.py:8](../../../tinysteno/session.py)
- `pywhispercpp>=1.4.1` is a declared dependency but is not imported anywhere under `tinysteno/` or `tests/` — [pyproject.toml:10](../../../pyproject.toml)
- SPEC still says the transcriber "Runs on CPU with int8 quantization" and justifies faster-whisper as making "real-time or near-real-time transcription viable"; the code has GPU detection and no real-time path — [SPEC.md:81, 373-374](../../../SPEC.md); contradicted by [tinysteno/transcriber.py:75-97](../../../tinysteno/transcriber.py)

### Inferences
- Enabling diarization roughly triples local decode time versus a plain transcript, and one third of that work is redundant (channel 0 decoded twice).
- Memory: a 75-minute 44.1 kHz stereo WAV read as float64 (soundfile's default) is about 198 M frames x 2 channels x 8 bytes, roughly 3.2 GB, before the per-channel copies and full-length FFT resamples. The recorder also reads the full raw capture into memory at stop ([tinysteno/recorder.py:358-361](../../../tinysteno/recorder.py)). Long meetings may be slow or memory-heavy; not measured.
- On Apple Silicon the code passes `device="auto"`. To my knowledge CTranslate2 has no Metal/MPS backend, so this most likely runs on CPU (Accelerate) despite the README's "Apple Silicon (MPS)" claim ([README.md:14, 275](../../../README.md)). This comes from outside knowledge, not from this repo, and should be verified before being stated as fact.
- A 16 kHz mono 16-bit WAV is about 115 MB per hour, so single-request uploads will exceed the 25 MB per-file limit I recall for OpenAI's hosted transcription endpoint after roughly 13 minutes. The limit figure is from memory and unverified here; the absence of audio chunking is from the code.
- No VAD means silence is decoded rather than skipped, costing time as well as accuracy.

### Gaps
- No benchmarks (real-time factor, WER, memory) exist in the repo; none were run.
- What `device="auto"`, `compute_type="auto"` resolves to on this machine was not checked (would require running code).

## Processing and summarization: chunking, prompts, structured output, title/tags, context, parameters, export

### Takeaway
Summarization is a single-level map-reduce over fixed 12,000-character slices with 500-character overlap, each slice sent with a persona system prompt and a "return JSON with exactly these fields" user message, then merged by one more LLM call. Output is coerced to a flat schema of strings and string lists, with empty values as the fallback for every failure; the LLM sees only the transcript text and the persona prompt.

### Cited Findings

Pipeline shape
- Stages in order: TRANSCRIBE, SUMMARIZE, METADATA (title/tags), EXPORT — [tinysteno/pipeline.py:18-24](../../../tinysteno/pipeline.py)
- Summarization is skipped entirely if `api_key` is falsy; the note is then exported with empty fields and the transcript — [tinysteno/pipeline.py:146-157](../../../tinysteno/pipeline.py)
- Empty transcript short-circuits to `no_speech=True` with no note — [tinysteno/pipeline.py:140-144](../../../tinysteno/pipeline.py)
- Any stage exception becomes a `PipelineError` for that stage; the TUI offers a manual retry (`r`) that re-runs the pipeline from the top, including transcription — [tinysteno/pipeline.py:104-112](../../../tinysteno/pipeline.py); [tinysteno/tui/processing.py:82, 245, 262](../../../tinysteno/tui/processing.py); [README.md:148, 164](../../../README.md)

Chunking
- Constants: `_CHUNK_SIZE_CHARS = 12000`, `_CHUNK_OVERLAP_CHARS = 500` — [tinysteno/orchestrator.py:16-17](../../../tinysteno/orchestrator.py)
- `_chunk_transcript` slices by raw character offset (`transcript[start:end]`, next `start = end - 500`); no sentence, line, speaker-turn or token awareness — [tinysteno/orchestrator.py:129-146](../../../tinysteno/orchestrator.py)
- Transcripts of 12,000 characters or fewer are sent in one call with no merge — [tinysteno/orchestrator.py:44-47, 135-136](../../../tinysteno/orchestrator.py)
- Chunks are processed sequentially in a `for` loop; partials whose every field is empty are dropped; if exactly one partial survives it is returned as the final result; if none survive, schema defaults are returned — [tinysteno/orchestrator.py:55-67](../../../tinysteno/orchestrator.py)

Prompts
- Per-chunk messages: system = the persona's `system_prompt.md` verbatim; user = "Return a JSON object with exactly these fields:" + a JSON skeleton (`"string"` or `["string", "..."]` per field) + "Field descriptions:" (from `persona.yaml`) + "Transcript:" + the chunk — [tinysteno/orchestrator.py:148-168, 203-208](../../../tinysteno/orchestrator.py)
- The per-chunk prompt is identical to the whole-transcript prompt: a chunk is not told it is a fragment, its index, or given any summary of preceding chunks — [tinysteno/orchestrator.py:47, 58](../../../tinysteno/orchestrator.py)
- Merge prompt (user message): "The following are partial summaries extracted from sequential sections of a single transcript. Merge them into one cohesive JSON result with exactly these fields" + skeleton + descriptions + "Merge rules: String fields: write a single comprehensive synthesis from all parts. List fields: combine all unique items; remove exact duplicates and near-duplicates that convey the same point. Do not invent information not present in the partial summaries." + all partials as indented JSON. It is sent with the same persona system prompt, in a single call for all partials — [tinysteno/orchestrator.py:69-70, 170-197](../../../tinysteno/orchestrator.py)
- SPEC says "Merge deduplicates: for list fields, checks if summary already exists before appending"; in the code deduplication is delegated entirely to the merge LLM call, there is no programmatic dedupe — [SPEC.md:99](../../../SPEC.md); contradicted by [tinysteno/orchestrator.py:170-197](../../../tinysteno/orchestrator.py)

Structured output and validation
- Each extraction call uses `response_format={"type": "json_object"}` (JSON mode, not a JSON-schema/structured-outputs constraint) — [tinysteno/orchestrator.py:203-211](../../../tinysteno/orchestrator.py)
- `_parse_json` strips Markdown code fences, tries `json.loads`, then falls back to a greedy first-`{`-to-last-`}` regex; on failure returns `{}` with a warning — [tinysteno/orchestrator.py:278-301](../../../tinysteno/orchestrator.py)
- `_validate` iterates the persona schema: a `string` field keeps the value only if it is a `str` (else `""`); a `list` field keeps it only if it is a `list`, coercing every item with `str(item)` and flattening newlines (else `[]`). Extra keys are ignored. Type mismatches log a warning — [tinysteno/orchestrator.py:218-256](../../../tinysteno/orchestrator.py)
- Any exception in the LLM call or parse is caught and logged, and `_validate({})` returns all-empty fields; there is no re-prompt or repair attempt — [tinysteno/orchestrator.py:199-216](../../../tinysteno/orchestrator.py)
- No application-level retry logic exists in the orchestrator. The `OpenAI` client is constructed without `max_retries`, so the SDK default applies; the installed SDK defines `DEFAULT_MAX_RETRIES = 2` (`.venv/lib/python3.12/site-packages/openai/_constants.py:10`), which covers transport-level failures only — [tinysteno/orchestrator.py:27-33](../../../tinysteno/orchestrator.py)
- Schema field types are restricted to `string` and `list`; names must be identifiers, non-keywords, and not in `{"title", "date", "duration", "transcript", "detected_language"}` — [tinysteno/personas/__init__.py:17-18, 85-105](../../../tinysteno/personas/__init__.py)
- `generated_tags` (named as reserved in the project CLAUDE.md) and `tags` are not in `_RESERVED_NAMES`; the exporter merges `{**metadata, **data}` so a schema field named `generated_tags` would overwrite metadata, and `tags` is overwritten by the exporter afterwards — [tinysteno/personas/__init__.py:17](../../../tinysteno/personas/__init__.py); [tinysteno/obsidian.py:45-51](../../../tinysteno/obsidian.py)

LLM parameters and defaults
- Default LLM config: `api_key: ollama`, `base_url: http://localhost:11434/v1`, `model: llama3.2:3b` — [tinysteno/main.py:46-48](../../../tinysteno/main.py)
- `temperature=0.3` on all three call types (extract/merge, title, tags); client timeout 120 s — [tinysteno/orchestrator.py:21, 32, 91, 121, 209](../../../tinysteno/orchestrator.py)
- No `max_tokens`, no seed, no context-window option, no streaming, no token counting; chunk size is in characters — [tinysteno/orchestrator.py:16, 203-211](../../../tinysteno/orchestrator.py)

Title and tags
- Input for both is `_extract_summary_text`: the value of the first `string` field in schema order; if that is empty or absent, all list items joined with ". " — [tinysteno/pipeline.py:88-101, 159-161](../../../tinysteno/pipeline.py)
- Title: system "Generate a brief title (max 6 words, no special characters)."; user "Based on this content, generate a 3-6 word title (no special chars): " + first 500 characters; cleaned by stripping quotes/colons/special characters and truncating to 6 words; `None` on failure — [tinysteno/orchestrator.py:18-19, 72-97, 272-276, 315-320](../../../tinysteno/orchestrator.py)
- Tags: system asks for "1-5 single-word tags... comma-separated... lowercase letters, digits, and underscores only"; user prompt carries the first 500 characters; output is split on commas, normalized to `[a-z0-9_]`, deduplicated; `[]` on failure. The 1-5 count is not enforced in code — [tinysteno/orchestrator.py:20, 99-127, 265-270, 303-313](../../../tinysteno/orchestrator.py)
- Title and tag calls run concurrently in a 2-worker thread pool, gated by `auto_title` / `auto_tags` (both default true) — [tinysteno/pipeline.py:164-180](../../../tinysteno/pipeline.py); [tinysteno/main.py:57-58](../../../tinysteno/main.py)
- Final title precedence: user-supplied name, generated title, WAV filename stem — [tinysteno/pipeline.py:184](../../../tinysteno/pipeline.py)
- For the `sprint` persona the first string field is `ceremony_type` ("Planning, Review, or Retrospective"), so title and tags are generated from that one word rather than from the overview — [tinysteno/personas/sprint/persona.yaml:5-6](../../../tinysteno/personas/sprint/persona.yaml); [tinysteno/pipeline.py:90-94](../../../tinysteno/pipeline.py)

Context given to the LLM
- `Orchestrator.summarize(transcript, persona)` receives only the transcript string and the persona; the messages contain only persona system prompt, schema skeleton, field descriptions and transcript text — [tinysteno/pipeline.py:155](../../../tinysteno/pipeline.py); [tinysteno/orchestrator.py:148-168](../../../tinysteno/orchestrator.py)
- Speaker labels reach the LLM only as inline `[You]` / `[Others]` prefixes when diarization produced them. No prompt (orchestrator or any built-in `system_prompt.md`) explains what `[You]` and `[Others]` mean or who the user is — [tinysteno/pipeline.py:140](../../../tinysteno/pipeline.py); [tinysteno/personas/default/system_prompt.md:1-10](../../../tinysteno/personas/default/system_prompt.md)
- Not passed: timestamps (dropped at [tinysteno/transcriber.py:207](../../../tinysteno/transcriber.py)), meeting date/duration/title, detected language, user name, attendee list, calendar data, prior meetings, glossary/vocabulary, user-typed notes. The metadata dict is built after summarization and only for the template — [tinysteno/pipeline.py:186-194](../../../tinysteno/pipeline.py)

Built-in personas
- Seven built-ins: `default`, `1on1`, `rca`, `irm`, `sprint`, `kickoff`, `executive-summary`, seeded to `~/.tinysteno/personas/` and loaded from there — [tinysteno/personas/__init__.py:15, 22, 131-186](../../../tinysteno/personas/__init__.py)
- `default` schema: `overview` (string, "2-4 sentence summary"), `participants` (list), `key_points` (list, "3-7 major discussion points"), `action_items` (list, each "task (assignee)", firm commitments only, "unassigned" if no clear owner) — [tinysteno/personas/default/persona.yaml:4-8](../../../tinysteno/personas/default/persona.yaml)
- `default` system prompt rules: participants only if they spoke or are confirmed present; concise; action items only for explicit commitments; "never guess" ownership — [tinysteno/personas/default/system_prompt.md:4-10](../../../tinysteno/personas/default/system_prompt.md)
- Other schemas: `1on1` is list-only (participants, goals, needs, struggles, recent_wins, action_items_employee, action_items_manager); `rca` has overview, timeline ("TIME | EVENT | SOURCE"), root_cause, contributing_factors, resolution, action_items ("ACTION | OWNER ROLE | PRIORITY"), open_questions; `irm` has overview, timeline, severity, impact, responders, mitigations, follow_ups; `sprint` has ceremony_type, overview, completed_items, incomplete_items, blockers, retrospective_notes, action_items; `kickoff` has overview, objectives, stakeholders, scope, risks, decisions, next_steps; `executive-summary` has bottom_line, summary, key_decisions, risks, asks — [tinysteno/personas/1on1/persona.yaml:4-11](../../../tinysteno/personas/1on1/persona.yaml); [tinysteno/personas/rca/persona.yaml:4-11](../../../tinysteno/personas/rca/persona.yaml); [tinysteno/personas/irm/persona.yaml:4-11](../../../tinysteno/personas/irm/persona.yaml); [tinysteno/personas/sprint/persona.yaml:4-11](../../../tinysteno/personas/sprint/persona.yaml); [tinysteno/personas/kickoff/persona.yaml:4-11](../../../tinysteno/personas/kickoff/persona.yaml); [tinysteno/personas/executive-summary/persona.yaml:4-9](../../../tinysteno/personas/executive-summary/persona.yaml)
- Structured items are encoded as formatted strings (e.g. "task (assignee)", pipe-delimited rows that the `rca` template drops straight into a Markdown table), because nested objects are not allowed by the schema — [tinysteno/personas/rca/template.md:15-18, 35-38](../../../tinysteno/personas/rca/template.md); [tinysteno/personas/__init__.py:18](../../../tinysteno/personas/__init__.py)
- The `rca` prompt asks for real timestamps ("TIME | EVENT | SOURCE", "Normalize all timestamps to a single timezone") although the transcript carries none — [tinysteno/personas/rca/system_prompt.md:8](../../../tinysteno/personas/rca/system_prompt.md); [tinysteno/transcriber.py:207](../../../tinysteno/transcriber.py)
- A persona may also set static `tags` and `session_timeout_minutes` — [tinysteno/personas/__init__.py:36-37, 107-110](../../../tinysteno/personas/__init__.py)

Export
- Metadata passed to the template: `title`, `date` ("%Y-%m-%d %H:%M"), `duration` (HH:MM:SS), `transcript`, `detected_language`, `generated_tags` — [tinysteno/pipeline.py:187-194](../../../tinysteno/pipeline.py)
- `ObsidianExporter.export` renders the persona's `template.md` with Jinja2 (`StrictUndefined`, `trim_blocks`, `lstrip_blocks`), sets `tags` = persona static tags + generated tags deduplicated, collapses blank lines outside code fences, and writes `<vault>/<output_folder>/<title> (<YYYY-MM-DD>).md` — [tinysteno/obsidian.py:26-72](../../../tinysteno/obsidian.py)
- The `default` template emits YAML frontmatter (`created`, `type`, `tags`, `duration`, `participants`), an H1 title, optional Overview / Key Points (numbered) / Action Items (`- [ ]` checkboxes), and the full transcript inside a fenced code block — [tinysteno/personas/default/template.md:1-33](../../../tinysteno/personas/default/template.md)
- The file is written with `filepath.write_text(content)` and no existence check; SPEC's claims of `_001` suffixes on collision, 100-character filename limit and `detected_language` in frontmatter are not implemented in the exporter or the built-in templates — [tinysteno/obsidian.py:68-72, 83-88](../../../tinysteno/obsidian.py); contradicted by [SPEC.md:119-122](../../../SPEC.md)
- The README sample shows a "## Participants" body section; the `default` template puts participants only in frontmatter — [README.md:322-324](../../../README.md); [tinysteno/personas/default/template.md:6](../../../tinysteno/personas/default/template.md)

### Inferences
- Number of LLM calls for a long transcript is N chunk calls + 1 merge + up to 2 metadata calls, where N is about `len / 11,500`. A 50,000-character transcript (SPEC's figure for 1+ hour meetings, [SPEC.md:371](../../../SPEC.md)) gives 5 sequential chunk calls.
- Diarized local transcripts are one line per segment with a label prefix, so a 12,000-character boundary will usually fall mid-line, and the 500-character overlap repeats a few turns, some without their label at the head of the slice.
- Running the merge under the persona system prompt (which describes a transcript-extraction role) with a user message containing JSON summaries is workable but not purpose-built; there is no separate reduce prompt per field type and no verification of merged output against sources.
- With the default Ollama model `llama3.2:3b` and no context-size option set, a 12,000-character chunk plus prompt (very roughly 3,000+ tokens) may exceed Ollama's default context window and be silently truncated server-side. Ollama's default window size is from outside knowledge and should be verified; the code sets none.
- JSON mode depends on the configured OpenAI-compatible server honouring `response_format`; servers that reject it raise, and the catch-all turns that into empty fields.

### Gaps
- No evaluation of summary quality, hallucination rate or chunk-merge fidelity exists in the repo.
- Behaviour of user-authored personas beyond the seven built-ins was not examined.
- Actual token counts per chunk depend on language and model tokenizer; not computed.

## Limitations: concrete weaknesses substantiated from the code

### Takeaway
The weaknesses cluster in three places: speaker attribution is capped at two capture-path labels with no acoustic modelling; transcription is offline, whole-file and unfiltered (no VAD, timestamps discarded); and summarization is a blind fixed-size map-reduce where every failure degrades silently to empty fields. Several README/SPEC statements overstate what the code does.

### Cited Findings

Diarization
- D1. No real speaker diarization: at most two labels, hard-coded `You` / `Others`, assigned by channel index — [tinysteno/transcriber.py:202-205](../../../tinysteno/transcriber.py); [tinysteno/api_transcriber.py:101](../../../tinysteno/api_transcriber.py)
- D2. No speaker embeddings, clustering, voice enrolment, or name resolution; no dependency capable of it (dependency list has no pyannote/speechbrain/NeMo/whisperx) — [pyproject.toml:7-20](../../../pyproject.toml)
- D3. Multiple remote participants are indistinguishable (system audio forced to mono, one label) — [tinysteno/_macos_loopback.py:153-155, 297](../../../tinysteno/_macos_loopback.py)
- D4. Multiple in-room speakers are indistinguishable (mic reduced to channel 0, one label) — [tinysteno/recorder.py:471](../../../tinysteno/recorder.py)
- D5. Mono recordings get no labels at all; diarization silently does nothing — [tinysteno/transcriber.py:158-160](../../../tinysteno/transcriber.py)
- D6. API backend does not interleave; conversation order between the two sides is lost — [tinysteno/api_transcriber.py:101](../../../tinysteno/api_transcriber.py)
- D7. API backend requests `verbose_json` but discards the segments that would allow interleaving — [tinysteno/api_transcriber.py:129, 138-149](../../../tinysteno/api_transcriber.py)
- D8. Local interleave uses segment start only: no word-level timestamps, no forced alignment, no overlap handling — [tinysteno/transcriber.py:186-189, 206](../../../tinysteno/transcriber.py)
- D9. No echo cancellation or cross-channel duplicate suppression — [tinysteno/recorder.py:468-479](../../../tinysteno/recorder.py); [tinysteno/transcriber.py:196-207](../../../tinysteno/transcriber.py)
- D10. No synchronization of mic and loopback streams beyond end padding — [tinysteno/recorder.py:276-305, 474-479](../../../tinysteno/recorder.py)
- D11. With diarization off (default) and a stereo recording, only the mic channel is transcribed; remote participants are absent from transcript and summary — [tinysteno/transcriber.py:56-57, 154](../../../tinysteno/transcriber.py); [tinysteno/api_transcriber.py:62, 116](../../../tinysteno/api_transcriber.py); [tinysteno/main.py:56](../../../tinysteno/main.py)
- D12. Any 2-channel file is treated as mic/system regardless of origin; a stereo mic with no loopback yields wrong labels — [tinysteno/transcriber.py:152](../../../tinysteno/transcriber.py); [tinysteno/recorder.py:455-456](../../../tinysteno/recorder.py)
- D13. The LLM is never told what `[You]` / `[Others]` mean, and the user's identity is unknown to it — [tinysteno/orchestrator.py:148-168](../../../tinysteno/orchestrator.py)

Transcription
- T1. No live transcript; transcription starts only after recording stops — [tinysteno/recorder.py:345-385](../../../tinysteno/recorder.py); [tinysteno/pipeline.py:130-135](../../../tinysteno/pipeline.py)
- T2. No VAD / silence filtering — [tinysteno/transcriber.py:176, 188](../../../tinysteno/transcriber.py)
- T3. Diarization triples local Whisper passes, one of them redundant — [tinysteno/transcriber.py:155, 199-200](../../../tinysteno/transcriber.py)
- T4. Sequential decoding only; no batched inference — [tinysteno/transcriber.py:176](../../../tinysteno/transcriber.py)
- T5. Whole-file in-memory load and single-shot FFT resample; no streaming — [tinysteno/transcriber.py:50-65, 151](../../../tinysteno/transcriber.py)
- T6. No custom vocabulary / `initial_prompt` / language hint; language kept from the mic-channel pass only — [tinysteno/transcriber.py:176, 184](../../../tinysteno/transcriber.py)
- T7. API path uploads each channel as one uncompressed WAV with no chunking of long audio — [tinysteno/api_transcriber.py:119-130](../../../tinysteno/api_transcriber.py)
- T8. Timestamps never leave the transcriber: notes and LLM input have none — [tinysteno/transcriber.py:207](../../../tinysteno/transcriber.py); [tinysteno/personas/default/template.md:28-33](../../../tinysteno/personas/default/template.md)
- T9. Progress reporting covers only the first of three local passes, and is 0 / 0.5 / 1 for the API — [tinysteno/transcriber.py:178-189](../../../tinysteno/transcriber.py); [tinysteno/api_transcriber.py:80-99](../../../tinysteno/api_transcriber.py)
- T10. The transcript is not persisted separately; it exists only inside the exported note, and a failed later stage followed by retry re-transcribes from scratch — [tinysteno/pipeline.py:130-201](../../../tinysteno/pipeline.py); [README.md:164](../../../README.md)

Summarization
- S1. Chunk boundaries are raw character offsets, not sentence/turn/token aware — [tinysteno/orchestrator.py:138-146](../../../tinysteno/orchestrator.py)
- S2. Chunks are summarized without position, prior-chunk context or awareness of being partial — [tinysteno/orchestrator.py:56-58, 148-168](../../../tinysteno/orchestrator.py)
- S3. Chunks run sequentially, not in parallel — [tinysteno/orchestrator.py:55-60](../../../tinysteno/orchestrator.py)
- S4. Single-level merge of all partials in one prompt; no hierarchical reduce, no size guard on the merge input — [tinysteno/orchestrator.py:69-70, 196](../../../tinysteno/orchestrator.py)
- S5. Failed or empty chunks are dropped silently; the note gives no indication that part of the meeting was not summarized — [tinysteno/orchestrator.py:59-60](../../../tinysteno/orchestrator.py)
- S6. If the merge call itself fails, `_extract` returns all-empty fields and the successful partials are discarded — [tinysteno/orchestrator.py:70, 199-216](../../../tinysteno/orchestrator.py)
- S7. No application-level retries, re-prompting, or JSON repair beyond fence stripping and a greedy brace regex — [tinysteno/orchestrator.py:199-216, 278-301](../../../tinysteno/orchestrator.py)
- S8. No map-reduce quality controls: no grounding/citation check, no coverage check, no programmatic dedupe, no confidence — [tinysteno/orchestrator.py:35-70](../../../tinysteno/orchestrator.py)
- S9. JSON mode only, no schema-constrained output; schema limited to flat strings and string lists, so owners, due dates and priorities are free-text conventions — [tinysteno/orchestrator.py:210](../../../tinysteno/orchestrator.py); [tinysteno/personas/__init__.py:18](../../../tinysteno/personas/__init__.py)
- S10. Wrong-typed fields are zeroed rather than coerced (a string where a list is expected becomes `[]`) — [tinysteno/orchestrator.py:234-245](../../../tinysteno/orchestrator.py)
- S11. No context beyond transcript and persona: no user notes, user identity, attendee list, calendar, prior meetings, glossary, or meeting date — [tinysteno/orchestrator.py:35, 148-168](../../../tinysteno/orchestrator.py); [tinysteno/pipeline.py:155](../../../tinysteno/pipeline.py)
- S12. Participant names can only come from names spoken aloud, since labels are `You`/`Others`; yet `default`, `1on1` and `irm` personas ask for named participants/responders — [tinysteno/personas/default/persona.yaml:6](../../../tinysteno/personas/default/persona.yaml); [tinysteno/personas/irm/persona.yaml:9](../../../tinysteno/personas/irm/persona.yaml)
- S13. Title and tags derive from only the first 500 characters of the first string field; for `sprint` that field is the ceremony type — [tinysteno/orchestrator.py:19-20, 269, 275](../../../tinysteno/orchestrator.py); [tinysteno/personas/sprint/persona.yaml:5](../../../tinysteno/personas/sprint/persona.yaml)
- S14. No token budgeting, `max_tokens` or context-window setting; default model is a 3B local model — [tinysteno/orchestrator.py:16, 203-211](../../../tinysteno/orchestrator.py); [tinysteno/main.py:48](../../../tinysteno/main.py)
- S15. No streaming of LLM output and no progress within the summarize stage (only transcription reports a fraction) — [tinysteno/tui/processing.py:143](../../../tinysteno/tui/processing.py); [tinysteno/orchestrator.py:203-211](../../../tinysteno/orchestrator.py)
- S16. The `rca` persona requests timestamped timelines the transcript cannot support — [tinysteno/personas/rca/system_prompt.md:8](../../../tinysteno/personas/rca/system_prompt.md)

Export
- E1. Same title and date overwrites the existing note without warning — [tinysteno/obsidian.py:68-71](../../../tinysteno/obsidian.py)
- E2. Transcript is embedded as one fenced code block in the note: no per-turn structure, timestamps, or audio links — [tinysteno/personas/default/template.md:28-33](../../../tinysteno/personas/default/template.md)
- E3. `detected_language` is available to templates but no built-in template uses it — [tinysteno/pipeline.py:192](../../../tinysteno/pipeline.py); [tinysteno/personas/default/template.md:1-7](../../../tinysteno/personas/default/template.md)

Documentation drift (claims a comparison should not take at face value)
- README implies interleaved diarization for both backends; only the local backend interleaves — [README.md:339-345, 398-401](../../../README.md)
- README advertises "Apple Silicon (MPS)"; the code only passes `device="auto"` — [README.md:14](../../../README.md); [tinysteno/transcriber.py:92-94](../../../tinysteno/transcriber.py)
- SPEC describes programmatic list dedupe, filename collision suffixes, CPU-only int8, and near-real-time viability, none of which match the code — [SPEC.md:81, 99, 121-122, 374](../../../SPEC.md)

### Inferences
- The strongest robustness gaps relative to tools with model-based diarization are D1-D4 (label ceiling) and D9/D11 (bleed duplication, and remote audio dropped when diarization is off).
- The strongest gaps relative to tools with richer summarization are S5/S6 (silent data loss on failure), S11 (no context), and T8 (no timestamps to anchor notes).
- TinySteno's real strengths for a fair comparison: fully local by default (faster-whisper + Ollama), channel-based You/Others separation that is deterministic and needs no model when the user wears headphones, data-driven personas, and a small codebase (`transcriber.py` 207 lines, `orchestrator.py` 320 lines).

### Gaps
- No empirical data: nothing here quantifies how often each weakness bites (no WER, DER, latency, memory or summary-quality measurements in the repo, and none were run by design).
- Two limitations lean on outside knowledge and are flagged as such above: whether CTranslate2 uses the GPU on Apple Silicon, and the hosted Whisper API upload size limit / Ollama default context window.
- Windows and Linux loopback behaviour was read but not exercised; device-name matching may behave differently across systems.
