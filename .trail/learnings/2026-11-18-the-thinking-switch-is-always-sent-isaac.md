---
title: "The thinking switch is always sent, and dropped when a service rejects it"
author: Isaac
date: 2026-11-18
branch: main
cost: { minutes: 8, tokens: 67064 }
id: l-b715c1f7
scope: repo
status: active
src: agent
touches:
  - tests/test_settings.py
  - tests/test_pipeline.py
  - tinysteno/orchestrator.py
  - tinysteno/config.py
  - tinysteno/settings.py
  - tinysteno/pipeline.py
  - README.md
  - SPEC.md
---

## Decided
- `enable_thinking` (default false) goes out on every chat call as `chat_template_kwargs.enable_thinking`. If the service answers 400 with a message naming `chat_template_kwargs`, that call is sent again without the field and the orchestrator stops sending it for the rest of the run. That costs one wasted call per run on strict services, and the switch works everywhere else with no setup.
- Why: gufo, vLLM and llama.cpp honour the field (it overrides a server's `--think` default either way), and Ollama ignores it. But api.openai.com, a backend `tinysteno setup` suggests, rejects unknown request fields, as Groq likely does too.

## Tried and ruled out
- Sending it unconditionally: breaks summaries on OpenAI and Groq.
- A blank / "don't send" value as the strict-API escape hatch: existing OpenAI configs would break until the user found the setting.

## Watch out
- A 400 for another reason (one that doesn't name the field) is not retried without the field; it goes through the normal `_extract` retry.

In the code: tinysteno/orchestrator.py `Orchestrator._chat`, `_THINKING_FIELD`; tinysteno/settings.py `enable_thinking`; tinysteno/pipeline.py passes it only when configured.
