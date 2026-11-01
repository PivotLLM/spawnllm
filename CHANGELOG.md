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
- `ToolDefinition.ToolSearch` marks the host's own tool-search tool. It is
  never sent deferred. On the OpenAI Responses API (`openai-responses`) with
  gpt-5.4 or later (a bare `gpt-<major>.<minor>` id at or above 5.4), when the
  request has deferred tools and a tool-search tool, deferred tools load
  natively through client-executed tool search: the tool-search tool is sent
  as `{"type": "tool_search", "execution": "client"}`, deferred functions
  carry `defer_loading: true`, a `tool_search_call` from the model is
  reported as an ordinary call to the tool-search tool, and its result goes
  back as `tool_search_output` holding the full definitions of the referenced
  tools. That item has no room for text, so the result's own text is not sent
  there. `openai_responses.Provider` implements `DeferredToolsCapable` by the
  same model rule.
- Every provider without native support (Chat Completions, Azure, Responses
  models before gpt-5.4 or requests without a tool-search tool, and Anthropic
  models before 4.5) sends a deferred tool only once a message in the
  request's history names it in `ToolReferences`, as an ordinary tool without
  any defer marker, and leaves the other deferred tools out. Tools are sent
  in a stable order: the non-deferred tools as given, then the referenced
  ones in order of first reference, so the list only grows at its end as the
  conversation goes on. References are appended to the tool result as text
  ("Tools found: a, b"). A host can therefore always send its full tool list
  with the not-yet-found tools marked `Deferred` and always return search
  results as `ToolReferences`, whatever the provider, and a fallback from a
  native provider to another keeps the tools already found. The CLI
  providers take no tool list; they append the references as text.
- OpenRouter (`openai-chat` against `openrouter.ai`): `DispatchStatus.CostUSD`
  is set from the response's `usage.cost` (OpenRouter credits, which are US
  dollars), streaming or not; a reported 0 is kept as $0.00 and a response
  without the field leaves it nil. `usage.cost` from any other host is
  ignored.

### Changed

- **BREAKING** `DispatchStatus.CostUSD` is now `*float64` (JSON `cost_usd`
  is `null` when unset). nil means the provider returned no cost; a non-nil
  value is the cost it reported, $0.00 included. Only the Claude CLI provider
  (`total_cost_usd`) and `openai-chat` against OpenRouter (`usage.cost`)
  report one; every other HTTP provider (`anthropic`, `anthropic-messages`,
  `openai-chat` elsewhere, `openai-responses`, `azure`) and the other CLIs
  leave it nil, where they used to report 0. Migration: callers
  reading `CostUSD` directly test for nil before dereferencing it (for
  example print `cost=unknown` for nil) and add nothing to a running total
  for nil.
- **BREAKING** `DispatchStatus.InputTokens` now means the same on every
  provider: the input tokens neither read from nor written to the prompt
  cache, so `InputTokens + CacheReadTokens + CacheCreationTokens` is the whole
  prompt (`Usage.PromptTokens`). The Anthropic providers and the Claude,
  Cursor and Antigravity CLIs already reported it this way. `openai-chat`,
  `openai-responses`, `azure` and the Codex CLI reported the whole prompt,
  cached tokens included, so a host adding the three counted cached tokens
  twice; they now report the prompt less `cached_tokens` and
  `cache_write_tokens` (never below 0). Migration: a caller that read
  `InputTokens` from those providers as the whole prompt uses
  `Usage.PromptTokens`, or adds `CacheReadTokens` and `CacheCreationTokens`.
- Cursor CLI: `DispatchStatus.CacheCreationTokens` is now filled from
  `cacheWriteTokens`; it was always 0.

- Anthropic: `Usage.PromptTokens` (and so `TotalTokens`) now counts every
  input token, `input_tokens` plus cache reads and cache writes, as the CLI
  providers and the OpenAI-shaped providers already do. With caching on,
  `input_tokens` alone covers only what follows the last cache breakpoint.
  `DispatchStatus` keeps the three counts separate (`InputTokens`,
  `CacheReadTokens`, `CacheCreationTokens`).

### Fixed

- Codex CLI: `Usage.PromptTokens` (and `TotalTokens`) no longer counts
  cached tokens twice. Codex's `input_tokens` already includes
  `cached_input_tokens`, which was added to it again.

[0.1.20]: https://github.com/PivotLLM/spawnllm/compare/v0.1.19...HEAD
