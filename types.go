package spawnllm

import (
	"context"
	"fmt"
	"time"

	"github.com/PivotLLM/spawnllm/protocoltypes"
)

type (
	ToolCall               = protocoltypes.ToolCall
	FunctionCall           = protocoltypes.FunctionCall
	LLMResponse            = protocoltypes.LLMResponse
	UsageInfo              = protocoltypes.UsageInfo
	DispatchStatus         = protocoltypes.DispatchStatus
	Message                = protocoltypes.Message
	MessageAttachment      = protocoltypes.MessageAttachment
	ToolDefinition         = protocoltypes.ToolDefinition
	ToolFunctionDefinition = protocoltypes.ToolFunctionDefinition
	ExtraContent           = protocoltypes.ExtraContent
	GoogleExtra            = protocoltypes.GoogleExtra
	ContentBlock           = protocoltypes.ContentBlock
	CacheControl           = protocoltypes.CacheControl
)

type LLMProvider interface {
	Chat(
		ctx context.Context,
		messages []Message,
		tools []ToolDefinition,
		model string,
		options map[string]any,
	) (*LLMResponse, error)
	GetDefaultModel() string
}

type StatefulProvider interface {
	LLMProvider
	Close()
}

// ThinkingCapable is an optional interface for providers that support
// extended thinking (e.g. Anthropic). Used by the agent loop to warn
// when thinking_level is configured but the active provider cannot use it.
type ThinkingCapable interface {
	SupportsThinking() bool
}

// CLIProvider is an optional interface implemented by subprocess-based providers
// (claude-cli, codex-cli, gemini-cli). CLI providers do not accept HTTP request
// parameters such as temperature — options are passed as prompt context only.
type CLIProvider interface {
	IsCLI() bool
}

// FailoverReason classifies why an LLM request failed for fallback decisions.
type FailoverReason string

const (
	FailoverAuth         FailoverReason = "auth"
	FailoverRateLimit    FailoverReason = "rate_limit"
	FailoverBilling      FailoverReason = "billing"
	FailoverTimeout      FailoverReason = "timeout"
	FailoverFormat       FailoverReason = "format"
	FailoverOverloaded   FailoverReason = "overloaded"
	FailoverUnknown      FailoverReason = "unknown"
	FailoverContextLimit FailoverReason = "context_limit"
)

// ReasonText returns a short human phrase for a FailoverReason, for user-facing
// messages. The precise signal is the HTTP status code shown alongside it.
func ReasonText(r FailoverReason) string {
	switch r {
	case FailoverBilling:
		return "out of credits"
	case FailoverAuth:
		return "auth failed"
	case FailoverRateLimit:
		return "rate limited"
	case FailoverTimeout:
		return "timeout"
	case FailoverOverloaded:
		return "overloaded"
	case FailoverContextLimit:
		return "context too large"
	case FailoverFormat:
		return "bad response format"
	default:
		return "error"
	}
}

// CoolsDown reports whether a failure of this kind means the model is unusable
// for a while (so callers should put it in cooldown rather than hammering it):
// billing exhaustion, auth failure, rate limiting, or provider overload. Format,
// context-limit, timeout, and unknown errors do not (they are per-request or
// transient).
func (r FailoverReason) CoolsDown() bool {
	switch r {
	case FailoverBilling, FailoverAuth, FailoverRateLimit, FailoverOverloaded:
		return true
	default:
		return false
	}
}

// FailoverError wraps an LLM provider error with classification metadata.
// RetryAfter, when non-zero, is the server-supplied hint (parsed from the
// Retry-After header) telling callers how long to wait before retrying.
type FailoverError struct {
	Reason     FailoverReason
	Provider   string
	Model      string
	Status     int
	RetryAfter time.Duration
	Wrapped    error
}

func (e *FailoverError) Error() string {
	return fmt.Sprintf("failover(%s): provider=%s model=%s status=%d: %v",
		e.Reason, e.Provider, e.Model, e.Status, e.Wrapped)
}

func (e *FailoverError) Unwrap() error {
	return e.Wrapped
}

// IsRetriable returns true if this error should trigger fallback to next candidate.
// Non-retriable: Format errors (bad request structure, image dimension/size).
func (e *FailoverError) IsRetriable() bool {
	return e.Reason != FailoverFormat
}

// ModelConfig holds an ordered list of models, tried in order (index 0 first).
type ModelConfig struct {
	Models []string
}
