# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.10] - 2026-11-20
- build: install mlx-whisper automatically on Apple Silicon
- docs(trail): learning on the mlx backend decisions
- fix: mlx review findings — line length, model-name mapping both ways, doctor check, docs
- feat: wire mlx transcription backend with cross-platform fallback
- feat: MLX (Metal) local transcription backend
- refactor: extract LocalTranscriber base from WhisperTranscriber
- feat: configurable whisper_beam_size for local transcription
- docs: plan for MLX transcription backend and beam-size setting

## [0.2.9] - 2026-11-18
- feat(llm): add enable_thinking setting, off by default

## [0.2.8] - 2026-11-17
- fix(tui): start the multiprocessing helper before the app takes the terminal

## [0.2.7] - 2026-11-17
- fix(tui): a second queued start never begins a second recording
- fix(tui): the quit prompt cannot get stuck when a screen below it is replaced
- test: stricter quit-during-delete check; screenshot script keeps HOME in its temp dir
- fix: the default-persona check without a persona key; remove dead code
- fix(tui): persona listing failures and restores never go unreported
- fix(config): odd config files give errors, not tracebacks; saves keep key order
- fix(tui): a reopened setup guide asks before discarding edits
- fix(tui): a second Enter on a recent recording starts no second run
- fix(cli): config --edit creates the default config when there is none
- fix(tui): the quit prompt considers every open screen
- fix(tui): a queued second activation never pops the screen below
- fix(tui): setup guide saves only what the user chose, plus what the file lacks
- docs: TUI menu, settings, personas, recordings and system check
- test(tui): split the recordings tests
- fix(tui): deleting a recording updates the session ledger and always reports
- fix(tui): personas screen survives odd persona files; home guards persona errors
- feat(tui): recordings screen with re-transcribe and delete
- feat(tui): personas screen
- feat(tui): first-run setup guide and system check
- fix(tui): settings screen hardening
- fix(config): YAML errors never quote the file
- feat(tui): menu and settings screen
- fix(tui): per-run settings snapshot; saves never empty the live config
- fix(doctor): literal output, no side effects, accurate folder checks
- fix(recordings): delete by name inside the opened recordings folder
- feat(tui): dependencies for settings, checks, personas and recordings
- refactor: setup checks and recording file operations as UI-free modules
- test: no module-level path points at the real home
- fix(setup): check the config first and repair invalid settings
- fix(config): refuse read-only files, write through symlinks, fsync
- feat: describe the editable settings in one place
- test: no test can touch the real home directory
- fix(tui): disable screenshot delivery outright
- refactor: move configuration into tinysteno/config.py; saves keep unknown keys
- test(tui): pin the palette commands and prove no route saves a screenshot
- feat(tui): remove the Screenshot command from the app
- docs: plan for TUI setup, settings and management

## [0.2.6] - 2026-11-16
- fix: describe numbered speakers by the labels actually present
- fix: fingerprint the transcription server by digest, never by URL
- test: boundary cases for the speaker-split policy
- docs: timestamps-off layout, TUI notes lifetime, stereo files from elsewhere
- fix: show warnings gathered before a failed stage, and the saved-transcript notice on no speech
- fix: count --notes in the chunk budget and cut notes that cannot fit
- fix: let the summary client rely on its own retry
- fix: guard against non-finite samples and track the API server in the fingerprint
- fix(api): clamp negative times, survive overflowing ones, warn on missing labels
- fix: judge channel levels as recorded, not as normalised
- fix: never label a split microphone or single-channel voice as You
- fix: make every warning one line of at most 300 characters
- style: allow assemble's local count after the split_failed flag
- fix: do not save a transcript whose speaker split failed
- docs: correct the speaker split notes
- fix(api): use server speaker labels on single-channel recordings
- docs: optional speaker split
- feat(api): use a server's speaker labels to split the remote channel
- fix: fingerprint the effective speaker split; guard the whole split step
- feat: optional speaker split of the remote channel, off by default
- fix(diarizer): contain every failure; verify and bound the model downloads
- feat: local speaker diarization through the optional sherpa-onnx extra
- feat: policy that decides when a speaker split is trustworthy
- build: optional diarize extra (sherpa-onnx)
- docs: saved transcripts are matched to the recording and settings (reuse applies to process and TUI)
- docs: saved transcripts are matched to the recording and settings
- test: check the saved-transcript notice precedes the final meeting line
- fix: reuse a saved transcript only when it still matches the recording and settings
- docs: transcript filters, prompt context, saved transcripts
- feat: keep the transcript beside the recording so retries skip transcription
- feat(summary): line-aware chunks sized to the model's context window
- feat(summary): give the model speaker, time, date and note context
- feat: timestamps on every transcript line
- fix(hygiene): measure bleed against the channel a segment came from
- feat: remove speaker bleed transcribed on both channels
- fix(hygiene): never drop real sentences or a listener's short replies
- feat: drop caption credits and repeated lines Whisper invents in silence
- feat(transcribe): voice-activity filtering and a vocabulary prompt
- fix(transcribe): detect CUDA with an API that exists
- fix: no note overwrite, better title source, rca timeline; docs match the code
- feat(tui): show processing warnings on the no-speech screen
- feat: processing warnings are written into the note and shown on exit
- fix(summary): retry once, keep partial results, and say what was lost
- fix(recorder): per-channel normalisation; stereo only when system audio was captured
- fix(api): tolerate malformed segment times from a server
- fix(api): interleave speakers from returned segments; always upload both channels
- fix(transcribe): transcribe both channels of a stereo recording, once each
- feat: transcript segments and a single assembly path for both backends

## [0.2.5] - 2026-11-16
- docs: research and improvements

## [0.2.4] - 2026-11-14
- fix: failed-save notice stays until dismissed; dev tools install by default
- fix(tui): exit messages come from one session ledger, not the screen stack
- fix: a kill or closed terminal saves the recording in headless record too
- fix(recorder): explicit lifecycle state; no leaked mic or cut-short audio
- docs: add TUI screenshots to the README
- docs: document the TUI fixes (log file, meters, safe quit, raw audio)
- fix(cli): TUI logs to a file; reject non-finite timeouts
- fix(tui): never lose a recording to a UI path; review and render fixes
- fix(recorder): keep the raw audio when saving the WAV fails
- chore: add script that regenerates the README TUI screenshots
- docs: document the TUI and the session timeout
- feat(tui): result screen with rendered note and open-in-Obsidian
- fix(tui): quit during processing without waiting for the pipeline
- feat(tui): processing screen with per-stage progress, failure and retry
- fix(tui): report a failed save when quitting during a recording
- feat(tui): recording screen with live meter and abortable session timeout
- feat(tui): Textual app scaffold, home screen, and tui entry points
- feat(recorder): expose live mic and system levels and add discard()
- fix: only treat template errors as note-rendering failures in _process_audio
- refactor: extract run_pipeline with stage reporting from main
- feat: 75-minute default session timeout with per-persona override and --timeout
- feat: add SessionTimer for recording session timeouts
- docs: fix remaining events-log assertion in plan
- docs: plan uses an append-only app.events log for navigation assertions
- docs: add Textual TUI and session timeout implementation plan

## [0.2.3] - 2026-09-10
- feat(personas): ship the 1on1 persona as a built-in
- refactor(personas): normalize system_prompt.md structure

## [0.2.2] - 2026-09-09
- fix: keep uv.lock in sync on bump and push the tag reliably

## [0.2.1] - 2026-09-09
- feat: auto-bump the patch version on pushes to main
- feat(personas): rework executive-summary for decision-ready briefings
- fix(personas): tighten default persona action-item extraction

## [0.2.0] - 2026-09-09
- Baseline release. Entries above this line are added automatically on each push to `main`.
