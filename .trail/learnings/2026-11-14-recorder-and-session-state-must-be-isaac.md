---
title: Recorder and session state must be explicit; exit messages come from one ledger
author: Isaac
date: 2026-11-14
branch: feat/textual-tui
cost: { minutes: 110, tokens: 1059049 }
id: l-b156f442
scope: repo
status: active
src: agent
touches:
  - README.md
  - SPEC.md
  - tests/test_tui_processing.py
  - tests/test_tui_safety.py
  - tests/test_tui_session.py
  - tinysteno/tui/app.py
  - tinysteno/tui/home.py
  - tinysteno/tui/ledger.py
  - tinysteno/tui/processing.py
  - tinysteno/tui/recording.py
---

## Decided
- `AudioRecorder` has an explicit `RecorderState` (idle / recording / save_failed) that every public method checks. It used to be a bool set before anything could fail plus nullable fields, so a `start()` that failed part-way left the microphone stream open and the recorder claiming to record.
- The stereo mix pads the shorter stream with silence. Cutting to the shorter one threw away the rest of a meeting whenever system-audio capture ended early.
- The TUI's exit message and exit code are read from `SessionLedger` (saved / processed / save_failed), never rebuilt from the screen stack. Screens get popped, and the message then forgot failed saves and unprocessed WAVs.
- SIGHUP/SIGTERM are held back with `deferred_exit_signals()` while a WAV is being written, in headless `record` and in `run_tui`'s crash path, and delivered again afterwards.

## Tried and ruled out
- Deriving "what to tell the user" in `action_quit` by scanning `screen_stack` for a RecordingScreen or ProcessingScreen: it only covers the screens still showing, and the crash path never ran it.
- Tying signal protection to the Textual app's mount/unmount: the safety-net save in `run_tui` runs after unmount, so a kill during that save still ended the process mid-write.

## Watch out
- Tests that call `tinysteno.main.main()` or `cmd_record` must stub `AudioRecorder`; an unstubbed one starts a real recording of the microphone and system audio.
- Run tests with `uv run --extra dev pytest`; plain `uv run pytest` re-syncs without the dev extras and falls through to a system pytest.

In the code: tinysteno/recorder.py (RecorderState, _abandon_start, _mix_stereo), tinysteno/signals.py, tinysteno/tui/ledger.py, tinysteno/tui/app.py (final_result, action_quit), tinysteno/main.py (_capture), tests/test_recorder_lifecycle.py, tests/test_tui_session.py.
