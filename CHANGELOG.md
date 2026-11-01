# Changelog

All notable changes to spawnllm are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.20]

### Added

- Anthropic (`anthropic` and `anthropic-messages`): every Messages request now
  carries the top-level `cache_control: {"type": "ephemeral"}`, which turns on
  automatic prompt caching. Repeated calls in a tool loop read the shared
  prefix (tools, system prompt, earlier messages) from cache instead of paying
  full input price. The `anthropic` provider still honours `SystemParts`
  cache markers and leaves the automatic marker out when four explicit
  markers already fill the request's breakpoint limit.
- OpenRouter (`openai-chat` against `openrouter.ai`): Claude models
  (`anthropic/...` ids) are sent the top-level
  `cache_control: {"type": "ephemeral"}`, without which OpenRouter does not
  cache them. Other models are unchanged.
- OpenRouter: the `prompt_cache_key` option is now also sent as `session_id`
  (cut to OpenRouter's 256-character limit), so a conversation stays on the
  same upstream and keeps hitting its cache. `prompt_cache_key` itself is
  still sent only to OpenAI and Azure.
- `DispatchStatus.CacheCreationTokens` is now filled for Chat Completions
  responses that report `usage.prompt_tokens_details.cache_write_tokens`
  (OpenRouter, OpenAI), and both `CacheReadTokens` and `CacheCreationTokens`
  for Responses API replies that report
  `usage.input_tokens_details.cached_tokens` / `cache_write_tokens`.

- Deferred tools and tool references. `ToolDefinition.Deferred` sends a
  tool's definition without loading it into the model's context;
  `Message.ToolReferences` on a tool result names the tools the host's own
  search found, which loads them. The optional interface
  `DeferredToolsCapable` (`SupportsDeferredTools(model string) bool`) tells
  the host whether a provider does this natively for a model, so it can send
  the same full tool list on every request and keep the prompt cache. Both
  Anthropic providers (and `ClaudeProvider`) support it for Claude 4.5 and
  later: deferred tools are sent with `defer_loading: true`, and references as
  `tool_reference` blocks in the `tool_result` (with a text block before them
  when the result also has text). A reference to a tool not in the request is
  named in the text instead, since the API refuses it.
- Every provider without native support (Chat Completions, Responses, Azure,
  the CLIs, and Anthropic models before 4.5) leaves deferred tools out of the
  request and appends the references to the tool result as text
  ("Tools found: a, b"), so a history that carries them still works after a
  fallback to another provider.

### Changed

- Anthropic: `Usage.PromptTokens` (and so `TotalTokens`) now counts every
  input token, `input_tokens` plus cache reads and cache writes, as the CLI
  providers and the OpenAI-shaped providers already do. With caching on,
  `input_tokens` alone covers only what follows the last cache breakpoint.
  `DispatchStatus` keeps the three counts separate (`InputTokens`,
  `CacheReadTokens`, `CacheCreationTokens`).

[0.1.20]: https://github.com/PivotLLM/spawnllm/compare/v0.1.19...HEAD
