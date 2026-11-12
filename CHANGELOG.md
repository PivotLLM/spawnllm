# Changelog

All notable changes to spawnllm are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.27]

### Added

- `ToolChoiceOption` (`"tool_choice"`), with `ToolChoiceAuto` (the default)
  and `ToolChoiceNone`: a per-request option that keeps the tools declared
  but stops the model from calling them, so it answers in text. Anthropic
  sends `tool_choice {"type":"none"}`; OpenAI-compatible, Azure and OpenAI
  Responses send `"none"` in place of `"auto"`. It applies only when the
  request carries tools; CLI providers ignore it.

## [0.1.26]

### Added

- `imagegen.ErrNoImages`: a successful image response that carries no image
  (empty `data`, or items with neither `b64_json` nor `url`) is now this
  error instead of an empty result.

### Changed

- `speech`: audio sent without a file extension now gets the extension of
  its `ContentType` (an unnamed `audio/ogg` is sent as `audio.ogg`), since
  the OpenAI-style endpoints choose the decoder by extension and rejected
  the bare name "audio". Audio whose format neither the file name nor the
  content type tells is refused before any request is sent (OpenAI style,
  OpenRouter and ElevenLabs; Gemini still needs only a media type).
- `speech` (OpenRouter transcription): the `input_audio` format falls back
  to the content type when the file name has no extension, and `oga` and
  `opus` are sent as `ogg`, their container (Telegram voice notes are
  `.oga`).
- `speech` (Gemini): a candidate whose finish reason is not `STOP`
  (`SAFETY`, `RECITATION`, `PROHIBITED_CONTENT`, `MAX_TOKENS`, `OTHER`, …)
  is an error naming the reason, for transcription and speech, instead of
  an empty or partial result.
- `speech` (text to speech): a 2xx response that is text, JSON or HTML is an
  error carrying the body, not audio. ElevenLabs audio with no
  `Content-Type` gets the type its `output_format` names (mp3 → audio/mpeg,
  pcm → audio/pcm, ulaw → audio/basic, opus → audio/ogg, wav → audio/wav).
- `common.HandleErrorResponse`: when the error body cannot be read, the
  result is still an `*HTTPStatusError` with the status, Retry-After and
  whatever part of the body was read, so the classifier keeps the status.
- Task calls drain up to 64 KiB of an unread response body before closing,
  so the connection can be reused.
- Base64 request bodies (OpenRouter transcription, Gemini inline audio) are
  encoded straight into the request, without intermediate copies of the
  audio.

### Fixed

- A multipart file name containing CR, LF or other control characters could
  break the part's header; they are now replaced with `_`.

## [0.1.25]

### Added

- Package `speech`: speech-to-text (`Transcriber`) and text-to-speech
  (`Synthesizer`) clients, one request per call, with the base URL, API key
  and model from the caller.
  - `speech.NewOpenAI`: OpenAI-style `POST {base}/audio/transcriptions`
    (multipart: `file`, `model`, `response_format=json`, and `language` and
    `prompt` when set) and `POST {base}/audio/speech` (`model`, `input`,
    `voice`, `response_format`, `instructions`). For an `openrouter.ai` base
    URL, or with `WithTranscriptionAPI(TranscriptionInputAudio)`, a
    transcription is sent as OpenRouter's JSON body
    (`input_audio {data, format}`, the format being the file extension), and
    OpenRouter's `usage.cost` is returned as `CostUSD` and `usage.seconds` as
    the duration. `DefaultSTTModel(baseURL)` returns the usual transcription
    model for Groq, OpenAI and OpenRouter.
  - `speech.NewGemini`: transcription through `generateContent` with the
    audio as `inlineData`, and speech from Gemini TTS models
    (`responseModalities: ["AUDIO"]`, a prebuilt voice name). The audio is
    returned with Gemini's `mimeType` unchanged: raw 16-bit PCM, not a
    playable file.
  - `speech.NewElevenLabs`: Scribe (`POST /v1/speech-to-text`) and
    `POST /v1/text-to-speech/{voice_id}` with `output_format`.
  - Results carry the text or audio as returned, the detected language and
    duration when reported, the usage (`protocoltypes.TaskUsage`: tokens,
    audio seconds, characters synthesized) and the reported cost.
- Package `imagegen`: `imagegen.NewOpenAI` generates images with
  `POST {base}/images/generations` (OpenRouter: `POST {base}/images`) and
  edits them with `POST {base}/images/edits` (multipart, one or more input
  images and an optional mask). Images are returned as `GeneratedImage`
  values, as chat image output is: base64 decoded into `Data`, a URL
  returned as `URL` with `Size` -1 and never fetched. Usage and OpenRouter's
  cost are returned when reported; `/images/edits` is not available on OpenRouter, so
  an edit there returns `ErrEditUnsupported`.
- `protocoltypes.TaskUsage`: the usage a single-shot task call reports.
- Errors from these clients keep the HTTP status (`*common.HTTPStatusError`
  with Retry-After and a body excerpt of at most 1 KiB), so `ClassifyError`
  classifies them as it does chat-provider errors. Each call is logged at
  DEBUG through the `logger` package.

## [0.1.24]

### Added

- `LLMResponse.Images` (`[]GeneratedImage`): the images a model generates are
  returned with the reply instead of being dropped, unfiltered and whatever
  their size; the caller decides what to keep. Each image has exactly one of
  `Data` (decoded from base64, with `ContentType` from the data URL, else
  detected from the bytes) or `URL` (an image given as a URL, returned as
  given and never fetched). `Size` is the decoded length in bytes, so a
  receiver can check it before accepting the image; it is -1 (unknown) for a
  `URL` image. Base64 that does not decode is not dropped: it is
  returned as `URL`, the data URL unchanged (a bare base64 string behind
  `data:;base64,`), with a WARN naming the model. `openai_compat` reads the
  assistant message's `images` array (OpenRouter's format, e.g. Gemini image
  models), streamed (`delta.images`) or not; `openai_responses` reads
  `image_generation_call` output items, streamed or not. The Anthropic and
  CLI providers return no images. A reply may carry images with empty text;
  it is still a normal reply. `Worker.Run` returns the images of the whole
  run in `Result.Images`.

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
