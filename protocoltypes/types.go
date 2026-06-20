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
}

// DispatchStatus is populated by each provider on every Chat() return
// (success or error) and surfaced in the LLMResponse so the agent loop
// can write a uniform finish event.
type DispatchStatus struct {
	Success             bool    `json:"success"`
	Model               string  `json:"model"`
	NumTurns            int     `json:"num_turns"`
	InputTokens         int     `json:"input_tokens"`
	OutputTokens        int     `json:"output_tokens"`
	CacheReadTokens     int     `json:"cache_read_tokens"`
	CacheCreationTokens int     `json:"cache_creation_tokens"`
	StopReason          string  `json:"stop_reason"`
	CostUSD             float64 `json:"cost_usd"`
	DurationMs          int64   `json:"duration_ms"`
	BytesSent           int64   `json:"bytes_sent"`
	BytesReceived       int64   `json:"bytes_received"`
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
}

type ToolDefinition struct {
	Type     string                 `json:"type"`
	Function ToolFunctionDefinition `json:"function"`
}

type ToolFunctionDefinition struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Parameters  map[string]any `json:"parameters"`
}
