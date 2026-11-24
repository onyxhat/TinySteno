---
title: Speaker model downloads broke when upstream re-uploaded the file; re-pin after checking it behaves the same
author: Isaac
date: 2026-11-24
branch: fix/speaker-model-checksum
cost: { minutes: 10, tokens: 46502 }
id: l-67c54733
scope: repo
status: active
src: agent
touches:
  - tests/test_api_transcriber.py
  - tinysteno/sherpa_diarizer.py
---

## Decided
- Re-pinned the speaker-embedding model's SHA-256. On 2026-11-19, k2-fsa re-uploaded `3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx` to the same GitHub release URL, re-exported with PyTorch 2.14.1 (previously 2.4.1). It is 1 byte larger and its digest is new. Every fresh download failed with "checksum mismatch", so the speaker split was broken for new users.
- Re-pinned only after checking the new file behaves the same: on real meeting speech its embeddings match the old file's to cosine 0.995 or more (two clips exactly identical), and the similarity between voices is unchanged. GitHub's own asset digest matched the download, so it was not a corrupted transfer.
- Existing installs are unaffected: `ensure_models` only checks that cached files exist and never re-hashes them.
- `test_no_labels_warning_needs_server_diarization_and_speech` now stands in `_default_turns`. It ran the real local split, so with the `diarize` extra installed it downloaded the models from GitHub. CI doesn't install the extra, so this never showed there.

## Watch out
- GitHub release assets can be replaced. If "checksum mismatch" appears again, compare GitHub's `digest` for the asset (API `releases/tags/<tag>`) and the model's behaviour before re-pinning. Never re-pin from the error alone.

In the code: tinysteno/sherpa_diarizer.py `EMBEDDING_SHA256`, `ensure_models`; tests/test_api_transcriber.py `test_no_labels_warning_needs_server_diarization_and_speech`.
