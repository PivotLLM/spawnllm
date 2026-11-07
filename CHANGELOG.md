# Changelog

All notable changes to spawnllm are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.23]

### Fixed

- `openai_compat` and `openai_responses`: the modes given with
  `WithFileModes(dir, file)` were narrowed by the process umask (for example
  0770/0660 became 0750/0640 under the usual umask 022), so a host could not
  give a group write access to the response log. The response log file now
  always gets exactly the configured file mode, set on the open file each time
  it is opened, and each parent directory created for it gets exactly the
  configured directory mode. Directories that already exist are left as they
  are.

## [0.1.22]

### Added

- `WithFileModes(dir, file)` for `openai_compat` and `openai_responses` sets the
  mode of the response log file and of the parent directories created for it.
- The response log file's missing parent directories are now created.

### Changed

- `openai_compat`: the response log file's default mode is now 0600 (was
  0640).

## [0.1.21]

### Fixed

- Anthropic (`anthropic-messages`): a tool result with both text and native
  tool references was sent as a `tool_result` holding a text block and
  `tool_reference` blocks together, which the API rejects with HTTP 400
  ("Tool definitions/code execution functions cannot be mixed with other
  content"), so every conversation whose history held a search result failed
  on Claude 4.5 and later, including history written while the agent ran on
  another provider. The `tool_result` now holds only the `tool_reference`
  blocks and the result's text follows as a separate text block in the same
  user message. A result without native references keeps the plain string
  content.
- OpenAI Responses (`openai-responses`) with client tool search (gpt-5.4 and
  later): a result from a tool other than the search tool that referenced
  deferred tools (for example a tool showing another tool's details) left them
  deferred, so the model could not call them, since only a `tool_search_output`
  loads a deferred tool. Such a result's `function_call_output` is now followed
  by an `additional_tools` item holding the referenced deferred tools that no
  earlier item loaded. A reference to a tool not in the request is still named
  in the result's text.
- OpenAI Responses (`openai-responses`) with client tool search: a replayed
  call to a deferred tool was sent without the `namespace` field the model's
  `function_call` carries (the tool's own name), and the API rejected the
  request ("Missing namespace for function_call"), so the turn after any call
  to a search-loaded tool failed. The replayed call now carries the namespace.
- OpenAI Responses (`openai-responses`) with client tool search: a search
  result's text was dropped, since `tool_search_output` has no place for it,
  so the model never saw an error from the search tool and an empty search
  looked like a silent success. When a search result has text it now follows
  the `tool_search_output` as a developer message ("<search tool> result:
  ..."), with any reference to a tool not in the request named in it.

## [0.1.20]

### Added

- Anthropic (`anthropic-messages`): every Messages request now carries the
  top-level `cache_control: {"type": "ephemeral"}`, which turns on automatic
  prompt caching. Repeated calls in a tool loop read the shared prefix (tools,
  system prompt, earlier messages) from cache instead of paying full input
  price.
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
  the same full tool list on every request and keep the prompt cache. The
  Anthropic provider (`anthropic-messages`) supports it for Claude 4.5 and
  later: deferred tools are sent with `defer_loading: true`, and references as
  `tool_reference` blocks in the `tool_result` (the result's text, when it has
  any, follows as a separate text block; see 0.1.21). A reference to a tool not
  in the request is named in the text instead, since the API refuses it.
- `ToolDefinition.ToolSearch` marks the host's own tool-search tool. It is
  never sent deferred. On the OpenAI Responses API (`openai-responses`) with
  gpt-5.4 or later (a bare `gpt-<major>.<minor>` id at or above 5.4), when the
  request has deferred tools and a tool-search tool, deferred tools load
  natively through client-executed tool search: the tool-search tool is sent
  as `{"type": "tool_search", "execution": "client"}`, deferred functions
  carry `defer_loading: true`, a `tool_search_call` from the model is
  reported as an ordinary call to the tool-search tool, and its result goes
  back as `tool_search_output` holding the full definitions of the referenced
  tools. That item has no room for text, so the result's own text follows it
  as a developer message (see 0.1.21). `openai_responses.Provider` implements `DeferredToolsCapable` by the
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

- **BREAKING** `ProviderSpec.Kind` `"anthropic"` now uses the Anthropic
  Messages client, the same as `"anthropic-messages"`, instead of the
  OpenAI-compatible client, which sent OpenAI-shaped requests to Anthropic.
  Migration: a caller that relied on the old behaviour (for example pointing
  `"anthropic"` at an OpenAI-compatible endpoint such as Anthropic's OpenAI
  SDK compatibility layer) sets `Kind` to `"openai-chat"`.
- **BREAKING** `DispatchStatus.CostUSD` is now `*float64` (JSON `cost_usd`
  is `null` when unset). nil means the provider returned no cost; a non-nil
  value is the cost it reported, $0.00 included. Only the Claude CLI provider
  (`total_cost_usd`) and `openai-chat` against OpenRouter (`usage.cost`)
  report one; every other HTTP provider (`anthropic-messages`,
  `openai-chat` elsewhere, `openai-responses`, `azure`) and the other CLIs
  leave it nil, where they used to report 0. Migration: callers
  reading `CostUSD` directly test for nil before dereferencing it (for
  example print `cost=unknown` for nil) and add nothing to a running total
  for nil.
- **BREAKING** `DispatchStatus.InputTokens` now means the same on every
  provider: the input tokens neither read from nor written to the prompt
  cache, so `InputTokens + CacheReadTokens + CacheCreationTokens` is the whole
  prompt (`Usage.PromptTokens`). The Anthropic provider and the Claude,
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

### Removed

- **BREAKING** The SDK-based Anthropic client is gone: `ClaudeProvider`, its
  constructors (`NewClaudeProvider`, `NewClaudeProviderWithBaseURL`,
  `NewClaudeProviderWithTokenSource`,
  `NewClaudeProviderWithTokenSourceAndBaseURL`) and the `anthropic` package
  (`github.com/PivotLLM/spawnllm/anthropic`). It existed to call the API with
  a Claude subscription login token, which Anthropic's terms do not allow for
  third-party software. spawnllm no longer depends on
  `github.com/anthropics/anthropic-sdk-go`. The `claude-cli` provider is
  unaffected. Migration: use the `anthropic-messages` provider
  (`anthropic_messages.NewProvider` or `NewProviderWithTimeout`) with an
  Anthropic API key.

### Fixed

- Codex CLI: `Usage.PromptTokens` (and `TotalTokens`) no longer counts
  cached tokens twice. Codex's `input_tokens` already includes
  `cached_input_tokens`, which was added to it again.
- Provider kind "cursor-cli" is now built by buildProvider; it was documented but returned "unknown provider kind".

[0.1.20]: https://github.com/PivotLLM/spawnllm/compare/v0.1.19...HEAD
