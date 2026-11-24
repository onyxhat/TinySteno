---
title: "Transcript lines are timed from their words, split at pauses, so channels interleave in the right order"
author: Isaac
date: 2026-11-24
branch: main
cost: { minutes: 29, tokens: 369756 }
id: l-d19b804e
scope: repo
status: active
src: agent
touches:
  - CHANGELOG.md
  - pyproject.toml
  - tinysteno/__init__.py
  - uv.lock
  - CLAUDE.md
  - README.md
  - tests/test_doctor.py
  - tests/test_mlx_transcriber.py
  - tinysteno/doctor.py
  - tinysteno/main.py
---

## Decided
- Both local backends ask Whisper for word timestamps and split each segment wherever two words are more than 2.0s apart, timing each piece from its own words. Channels are interleaved by segment start. A Whisper segment can span minutes of silence, so its text was shown before what the other channel said in between.
- Real case (Dynatrace note, 2026-11-20, local `small`, VAD on): VAD found mic speech at 25.4–26.4s, 165.4–170.6s and 181.3–182.7s, and faster-whisper returned one segment, 25.4–182.1s. Its words were "Okay," at 25.4s and the question at 166s. The saved transcript had 16 segments over 20s. After the fix the mic channel has none.
- Bumped the sidecar format to 3 so transcripts saved before the fix are transcribed again on the next process.

## Tried and ruled out
- The mlx backend as the cause: that note was made with `local`. mlx has no VAD, but it still started the same sentence 15s early after silence, and the split fixes that too.

## Watch out
- `tests/test_api_transcriber.py::test_no_labels_warning_needs_server_diarization_and_speech` fails on main, unrelated to this. It downloads the real sherpa speaker models and gets "checksum mismatch for 3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx".

In the code: tinysteno/transcriber.py `split_at_word_gaps`, `WORD_GAP_SECONDS`, `WhisperTranscriber._decode` (`word_timestamps=True`); tinysteno/mlx_transcriber.py `_decode`; tinysteno/transcript.py `SIDECAR_VERSION = 3`.
