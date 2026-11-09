package protocoltypes

import "encoding/json"

type ToolCall struct {
	ID               string         `json:"id"`
	Type             string         `json:"type,omitempty"`
	Function         *FunctionCall  `json:"function,omitempty"`
	Name             string         `json:"-"`
	Arguments        map[string]any `json:"-"`
	ThoughtSignature string         `json:"-"` // Internal use only
	ExtraContent     *ExtraContent  `json:"extra_content,omitempty"`
}

type ExtraContent struct {
	Google *GoogleExtra `json:"google,omitempty"`
}

type GoogleExtra struct {
	ThoughtSignature string `json:"thought_signature,omitempty"`
}

type FunctionCall struct {
	Name             string `json:"name"`
	Arguments        string `json:"arguments"`
	ThoughtSignature string `json:"thought_signature,omitempty"`
}

type LLMResponse struct {
	Content          string            `json:"content"`
	ReasoningContent string            `json:"reasoning_content,omitempty"`
	ToolCalls        []ToolCall        `json:"tool_calls,omitempty"`
	FinishReason     string            `json:"finish_reason"`
	Normal           bool              `json:"normal"`
	Usage            *UsageInfo        `json:"usage,omitempty"`
	Reasoning        string            `json:"reasoning"`
	ReasoningDetails []ReasoningDetail `json:"reasoning_details"`
	Status           *DispatchStatus   `json:"status,omitempty"`
	// ResponsesReasoning carries opaque OpenAI Responses reasoning items
	// (verbatim, incl. encrypted_content) so they can be replayed before the
	// function_call they produced on the next turn. Reasoning models + tools
	// only; empty otherwise.
	ResponsesReasoning []json.RawMessage `json:"responses_reasoning,omitempty"`
	// AnthropicThinking carries the Anthropic thinking and redacted_thinking
	// content blocks of this response verbatim (incl. signature / data). The
	// API requires them back, unmodified, at the start of the assistant turn
	// when a tool-use loop continues with thinking on. Empty without thinking.
	AnthropicThinking []json.RawMessage `json:"anthropic_thinking,omitempty"`
	// Images holds the images the model generated in this reply, in the order
	// the response gave them, passed through unfiltered whatever their size.
	// Set by openai_compat (the assistant message's images array, OpenRouter's
	// format, streamed or not) and openai_responses (image_generation_call
	// output items); the Anthropic provider and the CLI providers never set
	// it, since those APIs return no images. A reply may carry images and an
	// empty Content; it is still a complete reply.
	Images []GeneratedImage `json:"images,omitempty"`
}

// GeneratedImage is one image a model generated as part of its reply.
// Exactly one of Data or URL is set.
type GeneratedImage struct {
	// ContentType is the image's media type (for example image/png). With
	// Data: the one the response declared, else the one detected from Data.
	// With URL: the media type a data URL declares, else empty, since
	// spawnllm never fetches a URL to learn it.
	ContentType string `json:"content_type,omitempty"`
	// Data is the decoded image, set when the response carried it as base64.
	Data []byte `json:"data,omitempty"`
	// URL is set when the response gave the image as a URL (http, https, …),
	// returned as given and never fetched. It is also set, to a data URL,
	// when the response's base64 does not decode: a data URL the response
	// gave is returned unchanged, and a bare base64 string is returned as
	// "data:;base64," followed by it, so the model's output is never lost.
	URL string `json:"url,omitempty"`
	// Size is the length in bytes of the decoded image, len(Data), so a
	// receiver can check it before accepting the image. It is -1 (unknown)
	// when URL is set, since a URL image is never fetched.
	Size int64 `json:"size"`
}

// DispatchStatus is populated by each provider on every Chat() return
// (success or error) and surfaced in the LLMResponse so the agent loop
// can write a uniform finish event.
type DispatchStatus struct {
	Success  bool   `json:"success"`
	Model    string `json:"model"`
	NumTurns int    `json:"num_turns"`
	// InputTokens counts the input tokens neither read from nor written to
	// the prompt cache, on every provider (Anthropic's meaning of
	// input_tokens). InputTokens + CacheReadTokens + CacheCreationTokens is
	// the whole prompt, which UsageInfo.PromptTokens reports.
	InputTokens         int    `json:"input_tokens"`
	OutputTokens        int    `json:"output_tokens"`
	CacheReadTokens     int    `json:"cache_read_tokens"`
	CacheCreationTokens int    `json:"cache_creation_tokens"`
	StopReason          string `json:"stop_reason"`
	// CostUSD is the request's cost as the provider reported it; nil when the
	// provider returned no cost, which is not the same as a reported $0.00.
	// Set by the Claude CLI (total_cost_usd) and by OpenAI-compatible requests
	// to OpenRouter (usage.cost, in credits: US dollars).
	CostUSD       *float64 `json:"cost_usd"`
	DurationMs    int64    `json:"duration_ms"`
	BytesSent     int64    `json:"bytes_sent"`
	BytesReceived int64    `json:"bytes_received"`
	// DeniedTools lists the tools an agentic CLI wanted to call but was refused
	// by its own permission system during this request (the Claude CLI's
	// permission_denials). The CLI still reports success and answers in prose,
	// so the host needs this to tell a refused turn from a completed one.
	DeniedTools []string `json:"denied_tools,omitempty"`
}

type ReasoningDetail struct {
	Format string `json:"format"`
	Index  int    `json:"index"`
	Type   string `json:"type"`
	Text   string `json:"text"`
}

type UsageInfo struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// CacheControl marks a content block for LLM-side prefix caching.
// Currently only "ephemeral" is supported (used by Anthropic).
type CacheControl struct {
	Type string `json:"type"` // "ephemeral"
}

// ContentBlock represents a structured segment of a system message.
// Adapters that understand SystemParts can use these blocks to set
// per-block cache control (e.g. Anthropic's cache_control: ephemeral).
type ContentBlock struct {
	Type         string        `json:"type"` // "text"
	Text         string        `json:"text"`
	CacheControl *CacheControl `json:"cache_control,omitempty"`
}

// MessageAttachment describes a file attached to a message.
// Stored in the archive alongside the message; NOT sent to the LLM.
type MessageAttachment struct {
	Filename  string `json:"filename"`
	SHA256    string `json:"sha256,omitempty"`
	Size      int64  `json:"size"`
	LocalPath string `json:"local_path,omitempty"`
}

type Message struct {
	Role             string              `json:"role"`
	Content          string              `json:"content"`
	Source           string              `json:"source,omitempty"`
	Type             string              `json:"type,omitempty"`
	Media            []string            `json:"media,omitempty"`
	Attachments      []MessageAttachment `json:"attachments,omitempty"`
	ReasoningContent string              `json:"reasoning_content,omitempty"`
	SystemParts      []ContentBlock      `json:"system_parts,omitempty"` // structured system blocks for cache-aware adapters
	ToolCalls        []ToolCall          `json:"tool_calls,omitempty"`
	ToolCallID       string              `json:"tool_call_id,omitempty"`
	// ResponsesReasoning carries opaque OpenAI Responses reasoning items for this
	// assistant turn, replayed before its function_call items on the next request
	// (Responses API, reasoning models + tools). Persisted with history.
	ResponsesReasoning []json.RawMessage `json:"responses_reasoning,omitempty"`
	// AnthropicThinking carries this assistant turn's Anthropic thinking and
	// redacted_thinking blocks verbatim, sent back before its text and tool_use
	// blocks on the next request when thinking is on. Persisted with history.
	AnthropicThinking []json.RawMessage `json:"anthropic_thinking,omitempty"`
	// ToolReferences names the tools a tool result makes available: the
	// matches of the host's own tool search. On a provider that loads deferred
	// tools natively (see ToolDefinition.Deferred) they are sent as tool
	// references, which load those tools' definitions into the model's context
	// without changing the request's tool list; elsewhere they are appended to
	// Content as text ("Tools found: a, b"). Only meaningful on a tool result.
	// Persisted with history.
	ToolReferences []string `json:"tool_references,omitempty"`
}

type ToolDefinition struct {
	Type     string                 `json:"type"`
	Function ToolFunctionDefinition `json:"function"`
	// Deferred sends the tool's definition with the request but keeps it out
	// of the model's context until a tool result names it in
	// Message.ToolReferences. A provider that does not support deferred tools
	// natively for the model in use sends a deferred tool, as an ordinary
	// tool, only once a message in the request's history names it in
	// ToolReferences, after the other tools in order of first reference; the
	// rest are left out. A host can therefore send the same full tool list to
	// every provider. Never on the wire as is.
	Deferred bool `json:"-"`
	// ToolSearch marks the host's own tool-search tool: the tool whose results
	// carry Message.ToolReferences. It is never deferred (Deferred is ignored
	// on it). On the OpenAI Responses API it is sent as the client-executed
	// tool_search tool when the request has deferred tools; elsewhere it is an
	// ordinary function. Never on the wire as is.
	ToolSearch bool `json:"-"`
}

type ToolFunctionDefinition struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Parameters  map[string]any `json:"parameters"`
}

// TaskUsage is what a single-shot task call (a transcription, a speech
// synthesis, an image generation) reports it used. Each field is zero when
// the call reported nothing for it.
type TaskUsage struct {
	// InputTokens, OutputTokens and TotalTokens are the token counts the
	// response reported (token-billed models: gpt-4o-transcribe, Gemini,
	// GPT image models, OpenRouter).
	InputTokens  int `json:"input_tokens,omitempty"`
	OutputTokens int `json:"output_tokens,omitempty"`
	TotalTokens  int `json:"total_tokens,omitempty"`
	// Seconds is the audio duration the response reported as billed
	// (whisper-1's usage {type: duration}, OpenRouter's usage.seconds).
	Seconds float64 `json:"seconds,omitempty"`
	// Characters is the number of characters (Unicode code points) of text
	// sent for synthesis, counted by spawnllm since no speech API reports it.
	Characters int `json:"characters,omitempty"`
}
