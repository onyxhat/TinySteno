---
title: "Apple Silicon speed comes from a separate mlx backend that falls back to faster-whisper, not from tuning faster-whisper"
author: Isaac
date: 2026-11-20
branch: feat/mlx-backend
cost: { minutes: 37, tokens: 319890 }
id: l-6c8e2d1c
scope: repo
status: active
src: agent
touches:
  - README.md
  - tests/test_doctor.py
  - tests/test_main.py
  - tests/test_mlx_transcriber.py
  - tinysteno/doctor.py
  - tinysteno/main.py
  - tinysteno/mlx_transcriber.py
  - pyproject.toml
  - tests/test_pipeline.py
  - tests/test_settings.py
---

## Decided
- Local transcription on Apple Silicon is slow because CTranslate2 (faster-whisper) has no Metal backend, so it runs on the CPU. Speed comes from a separate opt-in `mlx` backend (mlx-whisper on the GPU), not from tuning faster-whisper. Measured on 66s of speech with `small`: mlx 1.0s, local beam 5 10.6s, local beam 1 6.6s.
- `mlx` must never break another machine: the extra carries a darwin/arm64 marker, mlx_whisper is imported lazily, and when it is unavailable the factory logs a warning and uses faster-whisper, translating `mlx-community/...` model names back to faster-whisper names (or `small`), so a config synced across machines keeps working.
- Model names map in both directions: bare faster-whisper names (`large`, `turbo`, `small.en`, `distil-large-v3`) go to their mlx-community repos; repo ids with "/" pass through.
- `whisper_beam_size` (default 5) only enters the transcript fingerprint when it isn't 5, and only on the local backend, so transcripts saved before the setting existed stay fresh.

## Tried and ruled out
- Comparing OpenWhispr: it's an Electron dictation app (whisper.cpp with Metal plus Parakeet via sherpa-onnx) with no callable local API, so it's not usable as a backend.

## Watch out
- A plain `uv sync` removes the mlx extra. Use `uv sync --extra mlx`, or the backend silently falls back and becomes about 10x slower. `tinysteno test` shows an optional "MLX backend" item for this.
- The pylint score rounds to 10.00/10 even with a few messages in this repo. Gate on pylint's exit code, not the score.
- mlx-whisper has no VAD and no beam search. It hasn't been tested on long, silence-heavy recordings, where Whisper is most prone to looping.

In the code: tinysteno/mlx_transcriber.py (mlx_model_name, local_model_name), tinysteno/main.py create_transcriber, tinysteno/doctor.py _mlx_backend, pyproject.toml [project.optional-dependencies] mlx.
