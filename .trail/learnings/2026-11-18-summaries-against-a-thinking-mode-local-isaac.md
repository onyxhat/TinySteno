---
title: Summaries against a thinking-mode local model time out; turn thinking off on the server
author: Isaac
date: 2026-11-18
branch: main
cost: { minutes: 18, tokens: 103788 }
id: l-fb0ef298
scope: repo
status: active
src: agent
touches:
  - tinysteno/orchestrator.py
---

## Decided
- Run the self-hosted gufo server (Qwen3.8-27B) with `--think off` (plus `--reasoning-budget 8192` as a cap for clients that opt back in). With thinking on, TinySteno's summary request (temperature 0.3, `response_format: json_object`) sends the model into a reasoning loop: one 3k-token chunk generated 29k+ tokens over 10+ minutes (99% draft acceptance, i.e. repetition). Every `_extract` attempt hit the 120s client timeout, so notes came out "transcript only".
- With thinking off, the same chunk finishes in about 20s with valid JSON. Two full 3-chunk and 4-chunk meetings summarized end to end with no warnings; the slowest request took 24.5s.

## Tried and ruled out
- A retry pile-up as the cause: gufo cancels a request when the client disconnects, streaming or not (`finish=cancelled`), so TinySteno's retry doesn't queue behind the abandoned call. The retry just loops again.
- Raising `_LLM_TIMEOUT_SECONDS`: the loop ran past 10 minutes without finishing, so no reasonable timeout helps.

## Watch out
- "LLM call failed: Request timed out" twice, exactly 120s apart, in `~/.tinysteno/tui.log` is the signature of this loop. Instant "Connection error" lines mean the server was down, not slow: check Komodo's stack history.
- A client-side alternative: send `chat_template_kwargs: {enable_thinking: false}` / `reasoning_effort: "none"` from TinySteno, which would protect against any thinking-default server.

In the code: tinysteno/orchestrator.py `_extract_once` (temperature 0.3, json_object), `_LLM_TIMEOUT_SECONDS = 120.0`, `Orchestrator.__init__` (`max_retries=0`).
