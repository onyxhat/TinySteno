# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
