# openminutes (dennisklappe/openminutes): diarization and transcript processing/summarization

Scope note. Everything below was read from the default branch `main` at commit `9d64ff7` (2026-11-16, "feat: join Whereby meetings (#48)"), via a shallow clone that was only read, never built or run. Links use the base `https://github.com/dennisklappe/openminutes/blob/9d64ff7/`. Each finding is tagged **[code]** (verified in source), **[docs]** (claimed in the repo's own docs/README, not independently verified in code by me) or **[API]** (GitHub API metadata). Accuracy numbers are the project's own self-reported measurements; I did not reproduce any of them. The repository exists at the given URL and has not moved.

## Diarization: how are speakers separated and labelled?

### Takeaway
openminutes uses two completely different speaker mechanisms depending on capture path. For bot-joined calls (Meet/Teams/Whereby) it does **no audio diarization at all**: the bot scrapes the meeting page's "who is speaking" indicator about 4 times a second and, after transcription, aligns word-level timestamps to those intervals with a lag-corrected Viterbi path, giving real participant names for N speakers. For phone/in-person recordings and imports it runs **post-hoc local diarization** (sherpa-onnx: pyannote segmentation 3.0 + 3D-Speaker ERes2Net embeddings + threshold clustering, plus a custom 1 s-window refinement pass), or takes Deepgram/AssemblyAI labels. On top of both sit optional per-person voice profiles, an LLM "speaker review" pass, per-line confidence, and a human review queue. Nothing is live.

### Cited Findings

**Call path: speaking-indicator scraping, not audio**
- [code] The bot samples the room every 250 ms (`SAMPLE_INTERVAL_MS = 250`), keeps an open interval per speaking display name, and closes it after a gap of more than 1 s (`MERGE_GAP_MS = 1_000`); output is a list of `{name, startMs, endMs}` speaker events plus join/leave times per participant. Names are clipped to 200 chars and the room to 500 people. — [apps/bot/src/room-tracker.ts](https://github.com/dennisklappe/openminutes/blob/9d64ff7/apps/bot/src/room-tracker.ts) (`RoomTracker.observe`, `close`)
- [code] Audio the bot records is a **single mono mixed stream**: ffmpeg reads a PulseAudio source (`-f pulse -i <sink.source>`) and writes `-ac 1 -c:a libopus -b:a 32k -application voip` in fixed-length segments. There is no per-participant audio track, so the indicator is the only per-speaker signal on calls. — [apps/bot/src/recorder.ts](https://github.com/dennisklappe/openminutes/blob/9d64ff7/apps/bot/src/recorder.ts) (`startRecording`), [apps/bot/src/session.ts L235](https://github.com/dennisklappe/openminutes/blob/9d64ff7/apps/bot/src/session.ts#L235)
- [docs] "Speaker names come from the meeting page (who is highlighted as speaking), not from guessing voices." and "Meet, Teams and Whereby calls are not diarized: the speaking indicator decides." — [README.md](https://github.com/dennisklappe/openminutes/blob/9d64ff7/README.md), [docs/speakers.md](https://github.com/dennisklappe/openminutes/blob/9d64ff7/docs/speakers.md)

**Word-to-speaker alignment (the core algorithm)**
- [code] Pipeline in `wordStream`: `estimateLag` -> `removeLag` -> `snapHandovers` -> `assignWords` -> `cohereSentences`, then `groupWords` regroups words into single-speaker lines. — [packages/core/src/speakers.ts L62-106](https://github.com/dennisklappe/openminutes/blob/9d64ff7/packages/core/src/speakers.ts#L62-L106)
- [code] Indicator lag is **measured per meeting**, not assumed: every span edge votes for candidate lags against word onsets/offsets (a word starting/ending a stretch with a 300 ms gap), searched in 50 ms steps up to 2000 ms start lag and 3000 ms end lag, winner refined by median; needs at least 3 samples else lag is 0. — [packages/core/src/speaker-lag.ts](https://github.com/dennisklappe/openminutes/blob/9d64ff7/packages/core/src/speaker-lag.ts) (`lagOf`, `estimateLag`, `removeLag`)
- [code] Per-word assignment is a Viterbi cheapest path: emission cost is `1 - coverage` (share of the word, padded 100 ms each side, covered by a speaker's lag-corrected intervals); switching speaker costs 1, reduced to 0.2 across a gap of 400 ms or more and halved after sentence-ending punctuation. When several speakers cover a word, the one with the **shortest** interval gets a bonus up to 0.8 (an interjection inside a long turn). Words nobody covers stay with the current speaker. — [packages/core/src/speaker-words.ts](https://github.com/dennisklappe/openminutes/blob/9d64ff7/packages/core/src/speaker-words.ts) (`coverage`, `switchCost`, `assignWords`)
- [code] Provider diarization labels (Deepgram/AssemblyAI) are only a hint: the label is mapped to the participant it overlaps most, then adds `DIARIZATION_WEIGHT = 0.5` to that participant's coverage; it never overrides a full indicator overlap. — [packages/core/src/speaker-types.ts](https://github.com/dennisklappe/openminutes/blob/9d64ff7/packages/core/src/speaker-types.ts), [packages/core/src/speakers.ts L39-58](https://github.com/dennisklappe/openminutes/blob/9d64ff7/packages/core/src/speakers.ts#L39-L58)
- [docs] Lines are re-cut from words at every speaker change, at a 2 s pause, and after a sentence once the line is 3.5 s long; nothing merges lines afterwards. Providers without word times fall back to interval-only cuts (speaker lit alone for at least 0.6 s, each part at least 2 words, character-proportional placement). — [docs/speakers.md](https://github.com/dennisklappe/openminutes/blob/9d64ff7/docs/speakers.md) (names `splitAtChanges` in `packages/core/src/speaker-split.ts`)

**Recording/import path: local audio diarization**
- [code] `/diarize` in the bundled Python service uses `sherpa_onnx.OfflineSpeakerDiarization` with the pyannote segmentation 3.0 int8 ONNX model, the `3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx` embedding model (2 threads), `FastClusteringConfig(num_clusters=-1, threshold=0.9 default)`, `min_duration_on=0.3`, `min_duration_off=0.5`. Models download on first use with pinned sha256. — [services/whisper/diarize.py L25-45, L218-233](https://github.com/dennisklappe/openminutes/blob/9d64ff7/services/whisper/diarize.py#L218-L233)
- [docs] Embeddings are L2-normalised 192-dim; each turn is returned with `similarity` to its speaker centroid and a `runnerUp`, so the caller can score uncertainty by the margin. Speakers are ordered by speaking time ("Speaker 1 to N, longest talker first"). — [services/whisper/README.md](https://github.com/dennisklappe/openminutes/blob/9d64ff7/services/whisper/README.md)
- [docs] A custom refinement pass (`refine.py`) re-embeds 1 s windows every 0.5 s: a run of windows fitting another speaker by a 0.12 margin becomes its own turn (rescues 1 to 2 s replies folded into a long turn); loud stretches the segmenter dropped become turns of the nearest voice; a requested `num_speakers` is reached by **splitting/folding after** threshold clustering instead of forcing the clusterer, "because forcing it put a second cluster on a handful of stray turns". — [services/whisper/README.md](https://github.com/dennisklappe/openminutes/blob/9d64ff7/services/whisper/README.md) ("Short turns and quiet voices")
- [docs] Label source precedence for recordings: provider labels (AssemblyAI, Deepgram) if returned, else the local diarizer if transcription is local, else no speakers. "Audio never goes to a third party for speaker detection. The diarizer runs only with the local provider." — [docs/speakers.md](https://github.com/dennisklappe/openminutes/blob/9d64ff7/docs/speakers.md) ("Phone recordings")
- [docs] User can declare the speaker count (Auto, 2, 3, 4, 5+) before recording, and re-run diarization with a fixed count later ("a two person call that shows nine speakers" is named as a real failure mode); a calendar-linked recording defaults the count to the invitee count. — [docs/speakers.md](https://github.com/dennisklappe/openminutes/blob/9d64ff7/docs/speakers.md) ("Fixing the number of people")

**Speaker naming, voice enrollment, embeddings**
- [docs] Voice profiles are opt-in and off by default ("Learn voices", "Check lines against saved voices"). A profile is embeddings only (no audio), per person **per language**, with one vector per "way of sounding" (headset vs room): a new sample joins the nearest vector at cosine 0.6 or more, else starts a new one; scoring uses the best vector, not the mean. Ready = 120 s of clean speech across 5 meetings/5 ways. — [docs/voices.md](https://github.com/dennisklappe/openminutes/blob/9d64ff7/docs/voices.md)
- [docs] Enrollment is implicit: profiles learn from lines the user verifies in a review queue, plus "automatic samples" only when two independent proofs agree (indicator at 90%+ confidence AND voice score at or above 0.6 and 0.1 ahead of every other voice, line 4 s or longer, person already has 30 s user-verified seed). Auto samples weigh 0.4. — [docs/voices.md](https://github.com/dennisklappe/openminutes/blob/9d64ff7/docs/voices.md) ("Automatic samples"; constants said to live in `packages/core/src/voice-auto.ts`)
- [docs] Names are "only ever suggested, never applied": from voice memory (up to 3 matches) or from names spoken aloud. — [docs/speakers.md](https://github.com/dennisklappe/openminutes/blob/9d64ff7/docs/speakers.md)
- [code] The spoken-name suggester prompt (verbatim): "You are given part of a transcript of a recording. Each line is "Speaker N: text". Find first names or full names that people say out loud to address or introduce someone ("Thanks, Anna", "this is Mark"), and decide which Speaker N most likely bears each name. A person addressed by name is usually not the one saying it, and usually speaks right after. Only use names that appear literally in the text. Suggest a name only when you are fairly sure. Reply with JSON only, in exactly this shape: {"names":[{"speaker":"Speaker 1","name":"Anna"}]}." Results are validated against the literal transcript text. — [apps/worker/src/name-suggestions.ts L24-26, L95-103](https://github.com/dennisklappe/openminutes/blob/9d64ff7/apps/worker/src/name-suggestions.ts#L24-L26)
- [docs] A "voice check" for call meetings embeds every line (`POST /embed-lines`), builds temporary per-participant "meeting voices" from anchor lines (3 s+, indicator confidence 90%+, no overlap, max 12 per person, outliers dropped), and flags lines that sound like someone else. It only adjusts confidence and suggests; "No label is changed by this check". — [docs/voices.md](https://github.com/dennisklappe/openminutes/blob/9d64ff7/docs/voices.md) ("The voice check", "Voices of this meeting")
- [docs] Shared-tile handling: a participant whose clean lines cluster into 2+ voice groups (each 3+ lines and 12 s+, max 4) is flagged as "probably several people" (meeting room on one account); user answers "how many?", each voice becomes its own person; the answer is remembered per shared account. — [docs/voices.md](https://github.com/dennisklappe/openminutes/blob/9d64ff7/docs/voices.md) ("Several people in one tile")

**LLM second pass on speaker labels**
- [code] Uncertain lines are batched 10 at a time with 4 context lines each side (`BATCH_SIZE = 10`, `CONTEXT_LINES = 4`) and sent to the summary model. — [apps/worker/src/speaker-review/batches.ts](https://github.com/dennisklappe/openminutes/blob/9d64ff7/apps/worker/src/speaker-review/batches.ts)
- [code] Prompt (verbatim excerpt): "You correct speaker labels in a meeting transcript. Each line is "[index] start-end in seconds, label: text". Lines marked with a question mark after the index, like "[12]?", have a label that may be wrong, with how sure the audio analysis was and which speakers it considered. [...] Keep the label unless the conversation clearly shows it is wrong. [...] Use who answered whom (an answer comes from someone other than the one who asked), names used to address someone ("Thanks, Alice" is said by someone other than Alice), and who owns a topic. Short replies are the most often wrong. Use only the indexes of lines marked with a question mark, and for such a line only a speaker named after "considered" with at least 20%, never anyone else. Reply with JSON only, in exactly this shape: {"changes":[{"index":<number>,"speaker":"<name>"}]}." — [apps/worker/src/speaker-review/prompt.ts L3-7](https://github.com/dennisklappe/openminutes/blob/9d64ff7/apps/worker/src/speaker-review/prompt.ts#L3-L7)
- [code] The reply is defensively validated: JSON extracted by first `{` / last `}`, zod-checked, and the **whole batch rejected** if it names an unknown line index, a speaker not in the meeting, a line twice, or changes more than 60% of the uncertain lines (minimum allowance 2). A failed batch is logged and skipped; the meeting never fails. — [packages/core/src/speaker-review-reply.ts](https://github.com/dennisklappe/openminutes/blob/9d64ff7/packages/core/src/speaker-review-reply.ts) (`parseSpeakerChanges`), [apps/worker/src/speaker-review/model-pass.ts](https://github.com/dennisklappe/openminutes/blob/9d64ff7/apps/worker/src/speaker-review/model-pass.ts)
- [docs] The model's answer is stored as a suggestion (Accept/Reject), not applied. Without a summary provider a text heuristic handles only "answer right after a question". — [docs/speakers.md](https://github.com/dennisklappe/openminutes/blob/9d64ff7/docs/speakers.md) ("Second pass: the context AI")

**Confidence and provenance per line**
- [code] Every line carries a speaker confidence; below 0.7 is "uncertain" (`LOW_SPEAKER_CONFIDENCE = 0.7`); diarized turns under 2.5 s always count as uncertain (`SHORT_UTTERANCE_MS = 2500`); the review may only suggest a candidate the audio scored at 0.2 or more. Lines with `speakerSource === 'user'` are never uncertain and never overwritten. — [packages/core/src/speaker-confidence.ts](https://github.com/dennisklappe/openminutes/blob/9d64ff7/packages/core/src/speaker-confidence.ts)
- [docs] Sources recorded per line: `indicator`, `diarization`, `ai`/`guess`, `user`. User corrections survive re-processing (remembered by name and time range). — [docs/speakers.md](https://github.com/dennisklappe/openminutes/blob/9d64ff7/docs/speakers.md)

**Live vs post-hoc; overlap; echo**
- [code] All labelling runs in the worker after the meeting: merge audio -> transcribe -> `speakers` step -> corrections -> voice check -> review -> save -> summary. — [apps/worker/src/process.ts L212-332](https://github.com/dennisklappe/openminutes/blob/9d64ff7/apps/worker/src/process.ts#L212-L332)
- [docs] "Live captions in the UI" is listed under "Not in version 1". — [docs/roadmap.md](https://github.com/dennisklappe/openminutes/blob/9d64ff7/docs/roadmap.md)
- [code] Echo/bleed between local mic and system audio does not arise on the call path (the bot is a remote participant recording one mixed stream). The phone recorder is a browser `getUserMedia` capture with `echoCancellation`, `noiseSuppression`, `autoGainControl` all true. — [apps/web/ui/recorder/microphone.ts L30-31](https://github.com/dennisklappe/openminutes/blob/9d64ff7/apps/web/ui/recorder/microphone.ts#L30-L31)

**Known limits (self-declared)**
- [docs] "The reading depends on Meet's page structure, which Google changes without notice. If the page cannot be read, the bot keeps recording and the transcript has no names." People are identified by display name ("two people with the same name count as one"); only people in the call grid are seen; "A one-word reply over before its tile lights up can go to the previous speaker. Overlapping speech is only partly recoverable." — [docs/speakers.md](https://github.com/dennisklappe/openminutes/blob/9d64ff7/docs/speakers.md) ("Limits")
- [docs] Self-reported accuracy: local diarizer speaker error 8.0% on synthetic phone conversations, 17.2% on five real AMI meetings; indicator alignment 98.8% of sentences on a synthetic late-indicator meeting and 19/19 on one recorded Teams meeting; on 10 real 2-3 person calls, right majority speaker 89.3-90.0% with diarization vs 96.2-96.3% with a simulated indicator. "Short turns under about 2 s are the weak spot"; only about 28% of 1-3 word replies come out as their own line with diarization, 37% with the indicator. — [docs/speakers.md](https://github.com/dennisklappe/openminutes/blob/9d64ff7/docs/speakers.md) ("Models and accuracy")
- [docs] Diarizer not solved: replies under 0.5 s have no voice print; two similar voices (centroid cosine 0.75) "stay noisy"; a wrong `num_speakers=2` on a one-voice recording invents a speaker; without a count, a quiet speaker merged into the dominant one is not found. The default embedding model is English-trained (VoxCeleb), "works on Dutch speech but was not tuned for it". — [services/whisper/README.md](https://github.com/dennisklappe/openminutes/blob/9d64ff7/services/whisper/README.md), [docs/voices.md](https://github.com/dennisklappe/openminutes/blob/9d64ff7/docs/voices.md)
- [docs] The mixed-line guard (first-half vs second-half embedding cosine under 0.3) is called "a tripwire, not a detector": it flags about 8% of single-speaker lines and catches about 39% of lines across a speaker change. Voice thresholds are described as "starting values that real audio has not tuned". — [docs/voices.md](https://github.com/dennisklappe/openminutes/blob/9d64ff7/docs/voices.md)
- [API] No user-filed issues about diarization exist; the only non-PR issue ever filed is #19 (GitHub export feature request, closed). — [issues list](https://github.com/dennisklappe/openminutes/issues?q=is%3Aissue)

### Inferences
- The call-path approach sidesteps the hardest diarization problems (speaker count, similar voices, naming) by using platform metadata. It is as good as the platform's active-speaker UI and as fragile as its DOM selectors. This rests on the code in `room-tracker.ts`/`speakers.ts` and the docs' own WARNING.
- The design philosophy is "suggest, never silently relabel": LLM and voice evidence only change confidence or add a suggestion. That trades automation for trust and needs a UI/review queue to pay off; a CLI tool could not adopt the review half cheaply.
- The lag estimator plus Viterbi word assignment is independent of where the "who is speaking" intervals come from. The same code path already consumes diarizer turns for recordings (flowchart in docs/speakers.md: "Speaker 1 to N" then "Words are lined up with the speaking intervals"), so it would apply equally to intervals derived from per-channel VAD energy.

### Gaps
- I did not read the DOM selector files (`packages/platforms/src/meet-dom.ts`, `teams-dom.ts`, `whereby-dom.ts`), `speaker-handover.ts`, `speaker-sentences.ts`, `speaker-split.ts`, `refine.py`, `voice-check.ts` or `voice-auto.ts` line by line; claims about them above are tagged [docs].
- How Deepgram/AssemblyAI diarization is requested (`packages/providers/src/transcription/deepgram.ts`, `assemblyai.ts`) was not read.
- No independent benchmark of any accuracy figure exists; all numbers are the author's, several on synthetic (piper TTS) voices.
- `docs/design.md#known-risks` was not read.

## Capture model: local recording, meeting bot, or uploads?

### Takeaway
Primarily a **self-hosted meeting bot**: a headless browser joins Google Meet, Microsoft Teams and (unreleased, merged 2026-11-16) Whereby as a named guest, records one mixed mono Opus stream and logs participant/speaker events. Two secondary paths exist: a phone recorder PWA (single microphone) and a Fireflies export import. There is no desktop app and no local mic+system-audio capture.

### Cited Findings
- [docs] Architecture: "**web** serves the UI and API, **worker** watches calendars and runs jobs, **bot** joins and records. All land in one Postgres and one data volume." Stack: "TypeScript, pnpm monorepo [...] Hono, Drizzle (Postgres), pg-boss, React + Vite, Playwright." — [README.md](https://github.com/dennisklappe/openminutes/blob/9d64ff7/README.md), [docs/design.md L106](https://github.com/dennisklappe/openminutes/blob/9d64ff7/docs/design.md)
- [code] Bot audio: PulseAudio source -> ffmpeg -> mono libopus 32 kbit/s, segmented files; the worker later merges segments into one file (48k mono Opus). — [apps/bot/src/recorder.ts](https://github.com/dennisklappe/openminutes/blob/9d64ff7/apps/bot/src/recorder.ts), [apps/worker/src/merge-audio.ts L15](https://github.com/dennisklappe/openminutes/blob/9d64ff7/apps/worker/src/merge-audio.ts#L15)
- [docs] Joins from Google Calendar/Outlook invites or a pasted link. Meet "refuses signed-out guests from automated browsers" so a dedicated Google account is recommended; Teams works as guest unless the tenant blocks anonymous guests; "Meetings that block guests, Zoom and Webex are not supported yet." — [README.md](https://github.com/dennisklappe/openminutes/blob/9d64ff7/README.md) ("Status and limits")
- [code] Phone recorder: browser `MediaRecorder` over a `getUserMedia` mic stream, uploaded in chunks. — [apps/web/ui/recorder/capture.ts](https://github.com/dennisklappe/openminutes/blob/9d64ff7/apps/web/ui/recorder/capture.ts), [apps/web/ui/recorder/microphone.ts](https://github.com/dennisklappe/openminutes/blob/9d64ff7/apps/web/ui/recorder/microphone.ts)
- [docs] Fireflies data export can be uploaded; recordings are re-transcribed and re-summarised locally. — [README.md](https://github.com/dennisklappe/openminutes/blob/9d64ff7/README.md), [docs/import-fireflies.md](https://github.com/dennisklappe/openminutes/blob/9d64ff7/docs/import-fireflies.md)
- [docs] The bot also scrapes meeting chat, which is fed to the summary. — [docs/providers.md](https://github.com/dennisklappe/openminutes/blob/9d64ff7/docs/providers.md) (`summary.include_chat`)
- [docs] Sizing (design estimates, "not measurements"): about 1 vCPU and 1.5 GB RAM per concurrent meeting plus about 1 GB base; default 1 parallel meeting. — [docs/sizing.md](https://github.com/dennisklappe/openminutes/blob/9d64ff7/docs/sizing.md)

### Inferences
- The bot model gives a diarization signal the baseline cannot get (named active-speaker intervals for every remote participant) but loses the one the baseline has for free (a clean, isolated local-mic channel). In a bot recording the local user is just another tile.
- The bot model has deployment costs a CLI recorder does not: Docker stack, Postgres, a visible bot participant, lobby admission, Google sign-in for Meet, and no Zoom. Evidence: README install and limits sections.

### Gaps
- Whether a general "upload any audio file" path exists beyond the Fireflies import and phone recorder was not confirmed (there is `apps/worker/src/import-audio.ts`, unread).

## Transcription engine and performance

### Takeaway
Batch only, after the meeting. Default is a bundled faster-whisper FastAPI service (`small`, `int8`, CPU) with VAD and word timestamps always on; hosted alternatives are OpenAI, Groq, Deepgram and AssemblyAI. Notable robustness work surrounds the ASR: 10-minute windowing with loudness normalisation for cloud providers, a hallucination detector that re-transcribes suspect windows with the local model, and removal of invented caption-credit lines. No real-time-factor or latency benchmarks are published for transcription.

### Cited Findings
- [code] Local service call: `whisper.transcribe(audio, language=spoken, initial_prompt=PUNCTUATION_PROMPTS.get(spoken), vad_filter=True, word_timestamps=True)`; language is detected first if not given. Defaults `WHISPER_MODEL=small`, `WHISPER_DEVICE=auto`, `WHISPER_COMPUTE_TYPE=int8`. — [services/whisper/app.py L21-23, L89-110](https://github.com/dennisklappe/openminutes/blob/9d64ff7/services/whisper/app.py#L89-L110)
- [code] The local provider posts the whole file in one request with a 6 hour timeout (no client-side chunking). — [packages/providers/src/transcription/local.ts](https://github.com/dennisklappe/openminutes/blob/9d64ff7/packages/providers/src/transcription/local.ts)
- [code] For upload-limited providers: 24 MiB limit, audio prepared into 600 s windows at 16 kHz mono with the filter `highpass=f=80,dynaudnorm=f=200:g=11,alimiter=limit=0.8`; chunks are transcribed **sequentially** and timestamps (segment and word) shifted by the chunk offset. — [packages/providers/src/transcription/chunked.ts](https://github.com/dennisklappe/openminutes/blob/9d64ff7/packages/providers/src/transcription/chunked.ts), [packages/providers/src/media.ts L27-31](https://github.com/dennisklappe/openminutes/blob/9d64ff7/packages/providers/src/media.ts#L27-L31)
- [code] Hallucination defence: `suspectWindows` flags 10-minute windows that look invented (repeated line or metronomic segment spacing) or sparse (under 25% of the recording's median words per minute, when the median is 40+ wpm); `repairWindows` cuts each suspect window out, re-transcribes it with a fallback transcriber, keeps the redo if it is not itself invented and has more words, and drops a window that stays invented. — [packages/core/src/transcript-quality.ts L52-90](https://github.com/dennisklappe/openminutes/blob/9d64ff7/packages/core/src/transcript-quality.ts#L52-L90), [apps/worker/src/repair-windows.ts](https://github.com/dennisklappe/openminutes/blob/9d64ff7/apps/worker/src/repair-windows.ts)
- [code] `dropMadeUpLines` removes subtitle/caption credits by regex ("ondertiteld", "subtitles", "untertitel", "amara", URLs, "TV <name> <year>") and any line of 5 words or fewer that recurs 3+ times while isolated by 8 s of silence on both sides. — [packages/core/src/transcript-quality.ts L92-141](https://github.com/dennisklappe/openminutes/blob/9d64ff7/packages/core/src/transcript-quality.ts#L92-L141)
- [docs] Join/leave chimes that Whisper hears as a word ("you", "woo") at the start of a segment are removed (`tones.py`). — [services/whisper/README.md](https://github.com/dennisklappe/openminutes/blob/9d64ff7/services/whisper/README.md)
- [docs] VAD is faster-whisper's default (chunks cut at 2 s silence, 0.4 s padding); shorter chunks (500 ms/200 ms) "were measured and gave no gain in single-speaker lines, so they were not adopted." — [docs/speakers.md](https://github.com/dennisklappe/openminutes/blob/9d64ff7/docs/speakers.md)
- [docs] Provider table (prices as of October 2026 per the docs): local `small` free; OpenAI `whisper-1` (word timestamps) and `gpt-4o-transcribe`/`gpt-4o-mini-transcribe` (no word timestamps); Groq `whisper-large-v3-turbo` / `whisper-large-v3`; Deepgram `nova-3` (can diarize); AssemblyAI Universal (can diarize). — [docs/providers.md](https://github.com/dennisklappe/openminutes/blob/9d64ff7/docs/providers.md)
- [docs] Performance figures that exist: local diarizer "about 0.5 s CPU per second of audio"; the refinement pass adds "roughly 0.03 to 0.05 s of CPU per second of audio" (16 s for a 10 minute AMI meeting on a busy 20 core host). Local whisper "runs on your CPU after the meeting", "No GPU needed"; `float16` on a GPU is a config option. — [docs/speakers.md](https://github.com/dennisklappe/openminutes/blob/9d64ff7/docs/speakers.md), [services/whisper/README.md](https://github.com/dennisklappe/openminutes/blob/9d64ff7/services/whisper/README.md), [docs/sizing.md](https://github.com/dennisklappe/openminutes/blob/9d64ff7/docs/sizing.md)
- [docs] Languages: one language per recording (chosen or auto-detected by a local "vote"); "a meeting that switches between Dutch and English is tagged with one language today". The project is visibly tuned on Dutch and English. — [docs/voices.md](https://github.com/dennisklappe/openminutes/blob/9d64ff7/docs/voices.md) ("Dutch and other languages"), [docs/providers.md](https://github.com/dennisklappe/openminutes/blob/9d64ff7/docs/providers.md)

### Inferences
- Raw ASR throughput is likely no better than the baseline and plausibly slower by default: both use faster-whisper, but openminutes defaults to `small`/`int8` on CPU with word timestamps on (extra alignment cost) and transcribes cloud chunks sequentially, whereas the baseline auto-detects a GPU. This rests on the defaults in `app.py` and `chunked.ts`; no timing was measured.
- The baseline pays for two full transcription passes (one per channel); openminutes transcribes one mixed stream once, then spends CPU on embeddings instead (about 0.5 s per audio second for recordings, and one embedding per line when the voice check is on).
- The transcript-quality guards (`suspectWindows`, `dropMadeUpLines`) are engine-agnostic and directly relevant to a baseline that transcribes a mostly-silent channel separately, which is exactly where Whisper invents credit lines.

### Gaps
- No real-time-factor, wall-clock or latency numbers for transcription are published anywhere in the repo.
- `PUNCTUATION_PROMPTS`, `language.py` (the language "vote") and `guards.py` were not read.
- Whether `batched` inference (faster-whisper `BatchedInferencePipeline`) is used: the grep of `app.py` shows only a plain `WhisperModel.transcribe` call, so apparently not.

## Transcript processing and summarization

### Takeaway
Summarization is deliberately simple: one system prompt producing **free Markdown with fixed headings** (Summary, Decisions, Action items table, Open questions), no JSON schema and no function calling. Long transcripts go through a sequential map-reduce by character budget (up to 4 condensing rounds). The interesting parts are the context added around the prompt (meeting date for resolving relative deadlines, canonical participant names, a rule that guessed speakers must not own action items) and the product layer after it (numbered summary versions, AI-edit with replay, three minute templates, many export targets). Summaries are off by default.

### Cited Findings

**Between raw transcript and summary**
- [code] Order in `process`: merge audio -> choose language -> transcribe (with window repair) -> speakers -> silence detection -> re-apply earlier user corrections -> voice check -> speaker review -> save transcript -> status `transcribed` + webhook -> auto-learn voices -> archive audio -> `startSummary`. — [apps/worker/src/process.ts L212-332](https://github.com/dennisklappe/openminutes/blob/9d64ff7/apps/worker/src/process.ts#L212-L332)
- [code] There is no LLM "cleanup" pass over transcript text. Text hygiene is rule-based (see `dropMadeUpLines`, tone removal above). The only LLM passes that touch the transcript return indexes and names, never text.
- [code] Transcript rendering for the model: one line per segment, `[hh:mm:ss] Speaker: text`, with `Speaker?:` when the label is doubtful. — [packages/providers/src/prompts.ts L18-25](https://github.com/dennisklappe/openminutes/blob/9d64ff7/packages/providers/src/prompts.ts#L18-L25)

**The default prompt (verbatim, complete)** — [packages/providers/prompts/meeting-summary.md](https://github.com/dennisklappe/openminutes/blob/9d64ff7/packages/providers/prompts/meeting-summary.md)

> You summarise meeting transcripts. The transcript follows as the user message, one line per segment in the form `[hh:mm:ss] Speaker: text`.
>
> Write the summary in the language the meeting was held in. Use only what is in the transcript, never invent names, dates or decisions. If a section has nothing to report, write "None".
>
> Use exactly this structure in Markdown:
>
> ## Summary
> Three to six sentences on what the meeting was about and how it ended.
>
> ## Decisions
> A bullet list of decisions that were made.
>
> ## Action items
> A Markdown table with exactly the columns Owner, Action and Due, one row per task: [...] Write "Not specified" in a cell when the transcript does not say who owns the task or when it is due. If there are no action items, write "None" instead of a table.
>
> ## Open questions
> A bullet list of questions or topics left unresolved.
>
> Never use em dashes or en dashes anywhere in the output. [...]

**Prompt augmentation at run time** — [apps/worker/src/summary-job.ts L99-117](https://github.com/dennisklappe/openminutes/blob/9d64ff7/apps/worker/src/summary-job.ts#L99-L117)
- [code] Date context: "The meeting took place on ${date}. Write every relative day (a weekday, tomorrow, next Monday) with its date resolved from the meeting date, for example "Wednesday (12 Nov)", in the language of the summary. Keep "Not specified" when no day is named." — [summary-job.ts L36-37](https://github.com/dennisklappe/openminutes/blob/9d64ff7/apps/worker/src/summary-job.ts#L36-L37); the output is then post-processed by `addDueDates`.
- [code] Participant list: "The participants were: A, B (joined as X). Refer to people by these names, using first names where they are unambiguous, and never by the name they joined the call as." Generic `Speaker N` names and the bot are excluded. — [apps/worker/src/participant-names.ts](https://github.com/dennisklappe/openminutes/blob/9d64ff7/apps/worker/src/participant-names.ts)
- [code] Only when at least one line is doubtful: "A speaker name followed by a question mark is a guess. Do not assign action items, decisions or quotes to a guessed speaker; write the item without an owner or say the owner is unclear." — [packages/core/src/transcript-text.ts L6-7](https://github.com/dennisklappe/openminutes/blob/9d64ff7/packages/core/src/transcript-text.ts#L6-L7); doubtful = low confidence, pending suggestion, failed voice check, elimination label or guess ([speaker-confidence.ts `isDoubtful`](https://github.com/dennisklappe/openminutes/blob/9d64ff7/packages/core/src/speaker-confidence.ts))
- [code] Chat: transcript gets a trailing `## Chat messages` block and the prompt gets instructions to add a `## Resources` section listing every shared URL, what it is for and who shared it. — [packages/providers/src/prompts.ts L27-29](https://github.com/dennisklappe/openminutes/blob/9d64ff7/packages/providers/src/prompts.ts#L27-L29)
- [code] Output post-processing: `cleanSummaryText` strips em/en dashes; that is the only output repair. — [packages/core/src/summary-text.ts](https://github.com/dennisklappe/openminutes/blob/9d64ff7/packages/core/src/summary-text.ts)

**Long meetings (map-reduce)** — [apps/worker/src/summarize.ts](https://github.com/dennisklappe/openminutes/blob/9d64ff7/apps/worker/src/summarize.ts)
- [code] If text fits `provider.maxInputChars` it is one call. Otherwise `splitText` packs whole lines into parts up to the limit (no overlap between parts; an over-long line is hard-sliced), each part is condensed **sequentially** with: "You are given excerpt ${index} of ${total} of a meeting transcript, or of notes taken on one. Write compact, factual notes in the language of the excerpt: topics discussed, decisions, action items with owner and deadline when named, and open questions. Use only what is in the excerpt and keep speaker names. Do not add an introduction or a conclusion."
- [code] Notes are joined under `### Excerpt N` headings; if they fit, the final call uses the normal prompt plus: "The input is not a transcript. It is a set of notes on consecutive excerpts of one meeting. Treat them together as the whole meeting and follow the instructions above." If they do not fit, the notes are split and condensed again, up to `MAX_ROUNDS = 4`, then it throws a clear error.
- [code] Budgets are in **characters**, defaults 24,000 (OpenAI-compatible) and 300,000 (Anthropic). — [packages/providers/src/summary/openai-compatible.ts L9](https://github.com/dennisklappe/openminutes/blob/9d64ff7/packages/providers/src/summary/openai-compatible.ts#L9), [packages/providers/src/summary/anthropic.ts L10](https://github.com/dennisklappe/openminutes/blob/9d64ff7/packages/providers/src/summary/anthropic.ts#L10)
- [docs] The docs concede the cost: "That costs several requests and loses some detail." — [docs/providers.md](https://github.com/dennisklappe/openminutes/blob/9d64ff7/docs/providers.md)

**Providers, models, structured output**
- [code] OpenAI-compatible: plain `POST /chat/completions` with `{model, temperature?, messages:[system, user]}`; no `response_format`, no tools, no retries in this file; 10 minute timeout; no default model (placeholder `gpt-4o-mini`). — [packages/providers/src/summary/openai-compatible.ts](https://github.com/dennisklappe/openminutes/blob/9d64ff7/packages/providers/src/summary/openai-compatible.ts)
- [code] Anthropic: `POST /v1/messages`, default model string `claude-sonnet-5-5`, `max_tokens` 4096. — [packages/providers/src/summary/anthropic.ts](https://github.com/dennisklappe/openminutes/blob/9d64ff7/packages/providers/src/summary/anthropic.ts)
- [docs] Summary provider default is `none`; Ollama, DeepSeek, OpenRouter are covered by the OpenAI-compatible provider. — [docs/providers.md](https://github.com/dennisklappe/openminutes/blob/9d64ff7/docs/providers.md)
- [code] JSON is used only for the two auxiliary tasks (speaker review, name suggestion), obtained by prompt instruction ("Reply with JSON only, in exactly this shape") and parsed by brace-slicing plus zod with semantic validation. — [packages/core/src/speaker-review-reply.ts](https://github.com/dennisklappe/openminutes/blob/9d64ff7/packages/core/src/speaker-review-reply.ts)

**Title, action items, templates, storage**
- [code] Title: a separate call on the finished summary: "Write a short title for a recording from the summary below. Use at most 6 words, the same language as the summary, no quotes, no full stop at the end and no dashes. Reply with the title only." The reply is sanitised and rejected if over 10 words or 80 chars; only phone recordings are titled, never calendar/link meetings or user-edited titles (re-checked in the SQL `UPDATE` to avoid a race). — [apps/worker/src/auto-title.ts](https://github.com/dennisklappe/openminutes/blob/9d64ff7/apps/worker/src/auto-title.ts)
- [code] After the summary: company detection (`detectCompany`), webhook `meeting.summarized`, optional auto-export and calendar link. No tag generation. — [apps/worker/src/summary-job.ts L146-158](https://github.com/dennisklappe/openminutes/blob/9d64ff7/apps/worker/src/summary-job.ts#L146-L158)
- [docs] Action items are the Markdown table inside the summary text, editable in an action-table editor; "The minutes table of action items is text from the summary, without a stored owner source". — [docs/summaries.md](https://github.com/dennisklappe/openminutes/blob/9d64ff7/docs/summaries.md)
- [docs] Custom prompts: the built-in prompt is read-only; "Copy to edit makes your own and you choose the one for automatic summaries"; Regenerate can use another prompt or provider. — [docs/providers.md](https://github.com/dennisklappe/openminutes/blob/9d64ff7/docs/providers.md)
- [docs] Every summary is a numbered version (AI, edited, restored, AI edit); exactly one active; "Edit with AI" rewrites the active summary from an instruction, and "Regenerate and re-apply my edits" replays earlier AI instructions on a fresh summary. — [docs/summaries.md](https://github.com/dennisklappe/openminutes/blob/9d64ff7/docs/summaries.md); replay in [summary-job.ts L118](https://github.com/dennisklappe/openminutes/blob/9d64ff7/apps/worker/src/summary-job.ts#L118)
- [docs] Minutes: three built-in templates (Classic, Modern, Compact) plus a custom template builder; contents are logo, title, date, location, attendees, absentees, summary, optional transcript and chat; outputs are a read-only share link, Google Doc, Word on OneDrive, HTML/Markdown/Word download, email to attendees, GitHub repo (v0.3.0), webhooks (unreleased 0.4.0). — [docs/minutes.md](https://github.com/dennisklappe/openminutes/blob/9d64ff7/docs/minutes.md), [CHANGELOG.md](https://github.com/dennisklappe/openminutes/blob/9d64ff7/CHANGELOG.md)
- [code] Storage is Postgres (Drizzle, 45 migrations); search is SQL `ILIKE` over title, company, participant name and transcript segment text. No full-text index or vector search. — [apps/web/src/meetings/list-routes.ts L58-63](https://github.com/dennisklappe/openminutes/blob/9d64ff7/apps/web/src/meetings/list-routes.ts#L58-L63), [apps/web/src/library/load.ts L50-71](https://github.com/dennisklappe/openminutes/blob/9d64ff7/apps/web/src/library/load.ts#L50-L71)

### Inferences
- The summary pipeline is structurally the same family as the baseline's (chunk, per-chunk LLM call, combine) and is **less** structured: Markdown instead of a schema-validated JSON object, one fixed minute shape instead of per-persona schemas, character rather than token budgets, no chunk overlap, sequential calls. Evidence: `summarize.ts`, `meeting-summary.md`, `openai-compatible.ts` request body.
- Where it is ahead is input quality and guard rails: timestamps and real names in the transcript, the guessed-speaker rule, date resolution, the canonical-names instruction, and the hierarchical re-condense loop with an explicit failure after 4 rounds.

### Gaps
- `findPrompt`, `addDueDates`, `detectCompany`, `replayEdits`, the AI-edit prompt and the custom template format were not read.
- No evaluation of summary quality exists in the repo that I found (`e2e/eval` covers diarization only: `ami_eval.py`, `chime_eval.py`).

## Better / worse than the baseline (TinySteno)

### Takeaway
All judgements in this section are my inference. openminutes is clearly ahead on speaker attribution for multi-party remote calls and on transcript robustness; roughly equal or behind on summarization structure; behind or not comparable on local-user isolation, deployment weight, platform coverage (no Zoom) and raw performance evidence.

### Cited Findings
- Evidence base for every judgement below is the cited code and docs in the sections above; no new sources.

### Inferences

**Does better (inference, with evidence)**
- More than two speakers, with real names, on remote calls. Baseline caps at `[You]`/`[Others]`; openminutes attributes each word to a named participant via indicator intervals (`room-tracker.ts`, `speaker-words.ts`). Self-reported 96%+ majority-speaker accuracy with an indicator vs about 90% with pure diarization.
- Word-level attribution instead of segment-level. Lines are rebuilt from word timestamps at speaker changes, so a Whisper segment spanning two speakers is split (`speakers.ts` `attributeWords`, docs/speakers.md).
- Explicit uncertainty. Per-line confidence, candidate lists, provenance (`indicator`/`diarization`/`user`), and propagation of doubt into the summary prompt (`Name?:` plus `GUESSED_SPEAKER_RULE`). The baseline has no notion of label confidence.
- In-room / single-mic recordings. A real diarizer (sherpa-onnx pyannote 3.0 + ERes2Net) with a speaker-count hint and a short-turn refinement pass; the baseline cannot separate two people on one channel at all.
- Cross-meeting identity. Opt-in voice profiles per person per language, with multi-prototype "ways of sounding" and two-proof auto-learning.
- ASR hallucination handling. `suspectWindows`/`repairWindows`/`dropMadeUpLines` and chime-word removal; the baseline description mentions none.
- Summary context. Meeting date for relative-deadline resolution, canonical names, chat URLs, summary versioning and AI-edit.
- Test and eval infrastructure for diarization (`e2e/eval/ami_eval.py`, `e2e/transcripts/diarization-score.ts`, recorded fixtures).

**Does worse or not at all (inference, with evidence)**
- No clean local-speaker channel. The bot records one mono mix (`recorder.ts` `-ac 1`); the baseline's mic channel gives a near-perfect "this is me" signal with no model. On calls openminutes depends entirely on the platform UI.
- Fragility. Speaker names depend on scraped DOM selectors ("Google changes without notice"; a dedicated issue template `platform_selector_broken.yml` exists); failure mode is a transcript with no names.
- Platform coverage. No Zoom, no Webex, no meetings that block guests; Meet needs a signed-in bot account. The baseline records anything the OS can play.
- Overlapping speech. Both are weak; openminutes says "only partly recoverable". The baseline's two-channel split actually handles You-vs-Others overlap natively, which openminutes cannot.
- Not live. Same as the baseline; live captions are explicitly out of scope for v1.
- Summarization structure. No JSON schema, no personas/typed fields, no tags; single fixed Markdown shape (custom prompt text only). Char-based budget of 24,000 by default will chunk meetings that fit easily in modern context windows.
- Performance. No published transcription RTF; default CPU `small`/`int8`; sequential chunk transcription and sequential map calls. Diarization adds about 0.5 s CPU per audio second (self-reported).
- Operational weight and maturity. Docker + Postgres + headless browser vs a single CLI; 3 weeks old, 9 stars, single maintainer, thresholds described by the author as untuned on real audio.
- Licence. AGPL-3.0: ideas and algorithms can be re-implemented, but code should not be copied into a differently licensed project without accepting AGPL terms.

**Techniques worth adopting in a local two-channel recorder (inference)**
1. Request word timestamps and rebuild lines at speaker changes rather than trusting Whisper segments. With two channels this means interleaving words from both channel transcripts by time, cutting at channel changes, at 2 s pauses, and after a sentence once a line passes about 3.5 s.
2. Viterbi smoothing with pause-aware switch cost (`assignWords`) to stop bleed-induced flicker: treat per-channel VAD/energy as the "speaking intervals" and assign each word to the cheaper channel, which is a principled way to suppress speaker bleed of system audio into the mic channel.
3. Optional local diarization of only the system-audio channel to split `[Others]` into Speaker 1..N: sherpa-onnx with pyannote-segmentation-3.0 + an ERes2Net embedding, threshold clustering at 0.9, optional user-supplied speaker count applied by split/fold after clustering rather than forced into the clusterer. Runs on CPU, no HF token.
4. Per-turn `similarity` and `runnerUp` margin as a confidence; mark turns under about 2.5 s as uncertain.
5. Carry uncertainty into the LLM: render doubtful labels as `Name?:` and add the one-sentence rule that guessed speakers must not own action items.
6. LLM speaker-review pass constrained to index+name JSON with whole-batch rejection rules (unknown index/name, duplicate, more than 60% changed). Cheap, and cannot corrupt text.
7. Spoken-name suggestion prompt ("Thanks, Anna") to propose names for `Speaker N`, validated against literal transcript text.
8. Hallucination guards for near-silent channels: drop isolated short lines repeated 3+ times and caption-credit regexes; flag windows whose words-per-minute is under 25% of the recording median.
9. Prompt additions: meeting date for resolving relative due dates; `[hh:mm:ss]` prefixes; "write None" for empty sections.
10. Re-condense loop with a bounded number of rounds and an explicit error instead of silently truncating.

### Gaps
- No head-to-head measurement exists; none of the "better/worse" calls is backed by a shared benchmark.
- TinySteno's actual behaviour was taken from the task description, not re-verified in its code for this note.

## Project facts (as of 2026-11-16)

### Takeaway
A three-week-old, fast-moving, single-maintainer TypeScript monorepo (plus a small Python whisper/diarization service), self-hosted via Docker Compose, AGPL-3.0, with a hosted option at openminut.es. Very low adoption so far.

### Cited Findings
- [API] Repo `dennisklappe/openminutes`, description "Self-hosted meeting recorder, transcripts and minutes for Google Meet and Microsoft Teams"; created 2026-10-25; last push 2026-11-16; 9 stars, 0 forks; primary language TypeScript; licence AGPL-3.0; not archived; homepage https://openminut.es. — [GitHub repo](https://github.com/dennisklappe/openminutes)
- [API] 44 commits on `main`; HEAD `9d64ff7` dated 2026-11-16 ("feat: join Whereby meetings (#48)"). Contributors: dennisklappe 27, dependabot 9, github-actions 8. — [commits](https://github.com/dennisklappe/openminutes/commits/main)
- [API] 8 releases in 12 days: v0.1.0 (2026-10-25), v0.2.0 through v0.2.4 (all 2026-10-28), v0.3.0 (2026-11-04), v0.3.1 (2026-11-05). A release-please PR for 0.4.0 (#47, outgoing webhooks, Whereby) is open and unreleased. — [releases](https://github.com/dennisklappe/openminutes/releases), [PR #47](https://github.com/dennisklappe/openminutes/pull/47)
- [API] Issue tracker: 48 numbered items, 47 of them pull requests; one issue ever (#19, feature request, closed by #39). Two open items, both PRs (#47 release, #33 dependabot). — [issue #19](https://github.com/dennisklappe/openminutes/issues/19)
- [docs] Platform: Docker Compose stack (`web`, `worker`, `bot`, Postgres, optional `whisper` under `--profile local`), single admin user ("Single-user first"; "Multiple users" not in v1). "OpenMinutes is early [...] Expect breaking changes before v1." — [README.md](https://github.com/dennisklappe/openminutes/blob/9d64ff7/README.md), [docs/roadmap.md](https://github.com/dennisklappe/openminutes/blob/9d64ff7/docs/roadmap.md), [docs/design.md](https://github.com/dennisklappe/openminutes/blob/9d64ff7/docs/design.md)
- [docs] Privacy stance: no telemetry; with local transcription and no summary provider nothing leaves the server; secrets encrypted with `OPENMINUTES_SECRET`. — [README.md](https://github.com/dennisklappe/openminutes/blob/9d64ff7/README.md)
- Doc inconsistencies observed: the README intro and supported-platforms table list only Meet and Teams while the feature table and install steps mention Whereby (merged the same day); the README says "v0.1.0 is the first release" while v0.3.1 is latest; the roadmap lists "A hosted service" as not in v1 while the README advertises a hosted version. — [README.md](https://github.com/dennisklappe/openminutes/blob/9d64ff7/README.md), [docs/roadmap.md](https://github.com/dennisklappe/openminutes/blob/9d64ff7/docs/roadmap.md)

### Inferences
- The volume and uniformity of code, tests and docs produced by one human in about three weeks (PR #29 alone bundles voice profiles, review, speaker count, re-run and API keys) suggests heavy AI-assisted development. The unusually candid self-measured limits are a point in its favour, but the thresholds are admitted to be untuned and there is essentially no external user feedback yet (9 stars, one outside issue). Treat it as a source of well-documented ideas rather than a proven system.

### Gaps
- No download/pull counts for the published container images were checked.
- No third-party reviews, benchmarks or discussions of the project were searched for; given its age and star count, significant coverage is unlikely but this was not verified.
