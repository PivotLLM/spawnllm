# spawnllm

`github.com/PivotLLM/spawnllm` is the ClawEh ecosystem's **LLM-dispatch core**.
It calls an LLM and drives it to completion:

- **CLI providers** (claude-cli, codex-cli, gemini-cli): run the subprocess, return the result.
- **API providers** (OpenAI-chat/responses, Azure, Anthropic): run the LLM↔tool-call loop, giving the model access to host-injected `toolspec.Tool`s until it is done.

## Boundaries

spawnllm imports only [`toolspec`](https://github.com/PivotLLM/toolspec) and the
standard library (plus provider SDKs). It **never imports a host**. Policy —
model selection, fallback, cooldown, config, results handling — stays in the host
(ClawEh, Maestro). The host resolves a `ProviderSpec`, injects tools, and calls
the worker.

Logging is host-injectable via the `logger` subpackage (`logger.SetBackend`), so
provider/loop logs flow into the host's own logger; spawnllm is silent until a
backend is installed.
