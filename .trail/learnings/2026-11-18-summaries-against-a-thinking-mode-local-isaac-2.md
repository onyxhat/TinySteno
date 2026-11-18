---
title: Summaries against a thinking-mode local model time out; turn thinking off on the server
author: Isaac
date: 2026-11-18
branch: main
cost: { minutes: 9, tokens: 9439 }
id: l-08f0a590
scope: repo
status: active
replaces: l-fb0ef298
src: agent
touches:
  - tinysteno/orchestrator.py
---

## Decided
- Run the self-hosted gufo server (Qwen3.8-27B Q8) with `--think off`, plus `--reasoning-budget 8192` to cap clients that turn thinking back on per request. With thinking on, TinySteno's summary request (temperature 0.3, `response_format: json_object`) times out in two ways:
  - Even a healthy reasoning pass is about 2,700–2,900 tokens. At 17–25 tok/s that's 133–141s per chunk, over the 120s client timeout.
  - Sometimes it loops: one run of the same chunk generated 29k+ tokens over 10+ minutes, with 99% draft acceptance (repetition).
- With thinking off, the same chunk finishes in about 20s with valid JSON. Two full meetings (3 and 4 chunks plus a merge) summarized with no warnings; the slowest request took 24.5s.

## Tried and ruled out
- Sampling penalties with thinking on (one run each): `repeat_penalty 1.1` took 133s and `presence_penalty 1.5` took 212s (it thought longer). Neither brings a chunk under 120s. A repeat penalty also punishes the tokens JSON has to repeat.
- A retry pile-up: gufo cancels a request when the client disconnects, streaming or not (`finish=cancelled`).
- Only raising `_LLM_TIMEOUT_SECONDS`: a loop ran past 10 minutes. It's viable only together with a reasoning budget (e.g. about 300s timeout and a 3–4k token budget), which makes summaries about 5x slower.

## Watch out
- In `~/.tinysteno/tui.log`, "Request timed out" twice, exactly 120s apart, is the sign of this problem. Instant "Connection error" lines mean the server was down; check Komodo's stack history.
- A client-side alternative: send `chat_template_kwargs: {enable_thinking: false}` from TinySteno, which protects against any server that defaults to thinking.

In the code: tinysteno/orchestrator.py `_extract_once` (temperature 0.3, json_object), `_LLM_TIMEOUT_SECONDS = 120.0`, `Orchestrator.__init__` (`max_retries=0`).
