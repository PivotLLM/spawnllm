package anthropicprovider

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync/atomic"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"

	"github.com/PivotLLM/spawnllm/common"
	"github.com/PivotLLM/spawnllm/logger"
	"github.com/PivotLLM/spawnllm/protocoltypes"
)

type (
	ToolCall               = protocoltypes.ToolCall
	FunctionCall           = protocoltypes.FunctionCall
	LLMResponse            = protocoltypes.LLMResponse
	UsageInfo              = protocoltypes.UsageInfo
	DispatchStatus         = protocoltypes.DispatchStatus
	Message                = protocoltypes.Message
	ToolDefinition         = protocoltypes.ToolDefinition
	ToolFunctionDefinition = protocoltypes.ToolFunctionDefinition
)

const (
	defaultBaseURL      = "https://api.anthropic.com"
	anthropicBetaHeader = "oauth-2025-04-20"
)

type Provider struct {
	client      *anthropic.Client
	tokenSource func() (string, error)
	baseURL     string
}

// SupportsThinking implements providers.ThinkingCapable.
func (p *Provider) SupportsThinking() bool { return true }

func NewProvider(token string) *Provider {
	return NewProviderWithBaseURL(token, "")
}

func NewProviderWithBaseURL(token, apiBase string) *Provider {
	baseURL := normalizeBaseURL(apiBase)
	client := anthropic.NewClient(
		option.WithAuthToken(token),
		option.WithBaseURL(baseURL),
	)
	return &Provider{
		client:  &client,
		baseURL: baseURL,
	}
}

func NewProviderWithClient(client *anthropic.Client) *Provider {
	return &Provider{
		client:  client,
		baseURL: defaultBaseURL,
	}
}

func NewProviderWithTokenSource(token string, tokenSource func() (string, error)) *Provider {
	return NewProviderWithTokenSourceAndBaseURL(token, tokenSource, "")
}

func NewProviderWithTokenSourceAndBaseURL(token string, tokenSource func() (string, error), apiBase string) *Provider {
	p := NewProviderWithBaseURL(token, apiBase)
	p.tokenSource = tokenSource
	return p
}

func (p *Provider) Chat(
	ctx context.Context,
	messages []Message,
	tools []ToolDefinition,
	model string,
	options map[string]any,
) (*LLMResponse, error) {
	var opts []option.RequestOption
	if p.tokenSource != nil {
		tok, err := p.tokenSource()
		if err != nil {
			return nil, fmt.Errorf("refreshing token: %w", err)
		}
		opts = append(opts,
			option.WithAuthToken(tok),
			option.WithHeader("anthropic-beta", anthropicBetaHeader),
		)
	}

	params, err := buildParams(messages, tools, model, options)
	if err != nil {
		return nil, err
	}

	counter := &byteCounter{}
	opts = append(opts, option.WithMiddleware(counter.middleware))

	// Token-streaming is opt-in via a non-nil delta callback. The SDK already
	// streams internally on the OAuth/setup-token path; when a callback is
	// present we route through the same streaming path (for API keys too) so
	// text deltas can be surfaced. Absent a callback, API keys keep the
	// unchanged single-shot Messages.New path.
	var streamCB common.TextDeltaFunc
	if cb, ok := options[common.TextDeltaOption].(common.TextDeltaFunc); ok {
		streamCB = cb
	}

	// OAuth/setup-tokens require streaming; API keys use non-streaming unless a
	// delta callback opts them into streaming.
	if p.tokenSource != nil || streamCB != nil {
		return p.chatStreaming(ctx, params, opts, counter, model, streamCB)
	}

	start := time.Now()
	resp, err := p.client.Messages.New(ctx, params, opts...)
	elapsed := time.Since(start).Milliseconds()
	if err != nil {
		return &LLMResponse{
			Status: &DispatchStatus{
				Success:       false,
				Model:         model,
				StopReason:    "error",
				DurationMs:    elapsed,
				BytesSent:     counter.sent(),
				BytesReceived: counter.received(),
			},
		}, fmt.Errorf("claude API call: %w", err)
	}

	out := parseResponse(resp)
	if out.Status != nil {
		out.Status.DurationMs = elapsed
		out.Status.BytesSent = counter.sent()
		out.Status.BytesReceived = counter.received()
	}
	return out, nil
}

func (p *Provider) chatStreaming(
	ctx context.Context,
	params anthropic.MessageNewParams,
	opts []option.RequestOption,
	counter *byteCounter,
	model string,
	cb common.TextDeltaFunc,
) (*LLMResponse, error) {
	start := time.Now()
	stream := p.client.Messages.NewStreaming(ctx, params, opts...)
	defer func() {
		if closeErr := stream.Close(); closeErr != nil {
			logger.DebugCF("anthropic", "closing stream", map[string]any{"error": closeErr.Error()})
		}
	}()

	var msg anthropic.Message
	for stream.Next() {
		event := stream.Current()
		// Surface incremental assistant text to the caller as it arrives. Only
		// text_delta payloads carry model output; thinking_delta/input_json_delta
		// (reasoning, tool-call argument fragments) are intentionally not fired.
		if cb != nil && event.Type == "content_block_delta" {
			delta := event.AsContentBlockDelta().Delta
			if delta.Type == "text_delta" && delta.Text != "" {
				cb(delta.Text)
			}
		}
		if err := msg.Accumulate(event); err != nil {
			return &LLMResponse{
				Status: &DispatchStatus{
					Success:       false,
					Model:         model,
					StopReason:    "error",
					DurationMs:    time.Since(start).Milliseconds(),
					BytesSent:     counter.sent(),
					BytesReceived: counter.received(),
				},
			}, fmt.Errorf("claude streaming accumulate: %w", err)
		}
	}
	if err := stream.Err(); err != nil {
		return &LLMResponse{
			Status: &DispatchStatus{
				Success:       false,
				Model:         model,
				StopReason:    "error",
				DurationMs:    time.Since(start).Milliseconds(),
				BytesSent:     counter.sent(),
				BytesReceived: counter.received(),
			},
		}, fmt.Errorf("claude API call: %w", err)
	}

	out := parseResponse(&msg)
	if out.Status != nil {
		out.Status.DurationMs = time.Since(start).Milliseconds()
		out.Status.BytesSent = counter.sent()
		out.Status.BytesReceived = counter.received()
	}
	return out, nil
}

// byteCounter tracks the bytes written to the request body and read from the
// response body across one Chat() call. It is safe for the SDK's retry path
// which calls the middleware once per attempt.
type byteCounter struct {
	bytesSent     atomic.Int64
	bytesReceived atomic.Int64
}

func (c *byteCounter) sent() int64     { return c.bytesSent.Load() }
func (c *byteCounter) received() int64 { return c.bytesReceived.Load() }

func (c *byteCounter) middleware(req *http.Request, next option.MiddlewareNext) (*http.Response, error) {
	if req.Body != nil && req.GetBody != nil {
		if body, err := req.GetBody(); err == nil {
			if n, copyErr := io.Copy(io.Discard, body); copyErr == nil {
				c.bytesSent.Add(n)
			}
			if closeErr := body.Close(); closeErr != nil {
				logger.DebugCF("anthropic", "closing request body copy", map[string]any{"error": closeErr.Error()})
			}
		}
	} else if req.ContentLength > 0 {
		c.bytesSent.Add(req.ContentLength)
	}
	resp, err := next(req)
	if err != nil || resp == nil {
		return resp, err
	}
	resp.Body = &countingReadCloser{rc: resp.Body, counter: &c.bytesReceived}
	return resp, nil
}

// countingReadCloser wraps an io.ReadCloser and atomically tracks bytes read.
type countingReadCloser struct {
	rc      io.ReadCloser
	counter *atomic.Int64
}

func (c *countingReadCloser) Read(p []byte) (int, error) {
	n, err := c.rc.Read(p)
	if n > 0 {
		c.counter.Add(int64(n))
	}
	return n, err
}

func (c *countingReadCloser) Close() error { return c.rc.Close() }

func (p *Provider) GetDefaultModel() string {
	return "claude-sonnet-4.6"
}

func (p *Provider) BaseURL() string {
	return p.baseURL
}

func buildParams(
	messages []Message,
	tools []ToolDefinition,
	model string,
	options map[string]any,
) (anthropic.MessageNewParams, error) {
	maxTokens := int64(4096)
	if mt, ok := options["max_tokens"].(int); ok {
		maxTokens = int64(mt)
	}

	// Normalize model ID: Anthropic API uses hyphens (claude-sonnet-4-6),
	// but config may use dots (claude-sonnet-4.6).
	apiModel := strings.ReplaceAll(model, ".", "-")

	params := anthropic.MessageNewParams{
		Model:     apiModel,
		MaxTokens: maxTokens,
	}

	if temp, ok := options["temperature"].(float64); ok {
		params.Temperature = anthropic.Float(temp)
	}

	// Adaptive thinking (see common.AnthropicThinking for the mapping).
	// Decided before the messages are built: while thinking is on, each
	// assistant turn's thinking blocks are sent back.
	thinkingOn := false
	if level, ok := options["thinking_level"].(string); ok {
		thinkingOn = applyThinkingConfig(&params, level)
	}

	var system []anthropic.TextBlockParam
	var anthropicMessages []anthropic.MessageParam

	for _, msg := range messages {
		switch msg.Role {
		case "system":
			// Prefer structured SystemParts for per-block cache_control.
			// This enables LLM-side KV cache reuse: the static block's prefix
			// hash stays stable across requests while dynamic parts change freely.
			if len(msg.SystemParts) > 0 {
				for _, part := range msg.SystemParts {
					block := anthropic.TextBlockParam{Text: part.Text}
					if part.CacheControl != nil && part.CacheControl.Type == "ephemeral" {
						block.CacheControl = anthropic.NewCacheControlEphemeralParam()
					}
					system = append(system, block)
				}
			} else {
				system = append(system, anthropic.TextBlockParam{Text: msg.Content})
			}
		case "user":
			if msg.ToolCallID != "" {
				anthropicMessages = append(anthropicMessages,
					anthropic.NewUserMessage(anthropic.NewToolResultBlock(msg.ToolCallID, msg.Content, false)),
				)
			} else {
				anthropicMessages = append(anthropicMessages,
					anthropic.NewUserMessage(anthropic.NewTextBlock(msg.Content)),
				)
			}
		case "assistant":
			var thinking []anthropic.ContentBlockParamUnion
			if thinkingOn {
				thinking = thinkingBlockParams(msg.AnthropicThinking)
			}
			if len(msg.ToolCalls) > 0 {
				blocks := thinking
				if msg.Content != "" {
					blocks = append(blocks, anthropic.NewTextBlock(msg.Content))
				}
				for _, tc := range msg.ToolCalls {
					args := tc.Arguments
					if args == nil && tc.Function != nil && tc.Function.Arguments != "" {
						if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err != nil {
							args = map[string]any{}
						}
					}
					if args == nil {
						args = map[string]any{}
					}
					blocks = append(blocks, anthropic.NewToolUseBlock(tc.ID, args, tc.Name))
				}
				anthropicMessages = append(anthropicMessages, anthropic.NewAssistantMessage(blocks...))
			} else {
				anthropicMessages = append(anthropicMessages,
					anthropic.NewAssistantMessage(append(thinking, anthropic.NewTextBlock(msg.Content))...),
				)
			}
		case "tool":
			anthropicMessages = append(anthropicMessages,
				anthropic.NewUserMessage(anthropic.NewToolResultBlock(msg.ToolCallID, msg.Content, false)),
			)
		}
	}

	params.Messages = anthropicMessages

	if len(system) > 0 {
		params.System = system
	}

	if len(tools) > 0 {
		params.Tools = translateTools(tools)
	}

	return params, nil
}

// thinkingBlock is the wire form of a thinking or redacted_thinking block, as
// kept in LLMResponse/Message.AnthropicThinking.
type thinkingBlock struct {
	Type      string `json:"type"`
	Thinking  string `json:"thinking,omitempty"`
	Signature string `json:"signature,omitempty"`
	Data      string `json:"data,omitempty"`
}

// thinkingBlockParams turns an assistant turn's kept thinking blocks back into
// request blocks, unchanged (thinking text, signature, redacted data).
func thinkingBlockParams(raw []json.RawMessage) []anthropic.ContentBlockParamUnion {
	var blocks []anthropic.ContentBlockParamUnion
	for _, r := range raw {
		var tb thinkingBlock
		if err := json.Unmarshal(r, &tb); err != nil {
			logger.WarnCF("anthropic", "skipping unreadable thinking block from history",
				map[string]any{"error": err.Error()})
			continue
		}
		switch tb.Type {
		case "thinking":
			blocks = append(blocks, anthropic.NewThinkingBlock(tb.Signature, tb.Thinking))
		case "redacted_thinking":
			blocks = append(blocks, anthropic.NewRedactedThinkingBlock(tb.Data))
		default:
			logger.WarnCF("anthropic", "skipping non-thinking block in AnthropicThinking",
				map[string]any{"type": tb.Type})
		}
	}
	return blocks
}

// thinkingBlockJSON is the kept form of a response's thinking or
// redacted_thinking block.
func thinkingBlockJSON(tb thinkingBlock) json.RawMessage {
	data, err := json.Marshal(tb)
	if err != nil {
		// A struct of strings cannot fail to marshal.
		return nil
	}
	return data
}

// applyThinkingConfig sets adaptive thinking and, for an effort level,
// output_config.effort, as common.AnthropicThinking maps level. Nothing is
// set for "", "off" or an unknown level.
//
// Anthropic API constraint: temperature must not be set when thinking is sent.
//
// It reports whether thinking was sent.
func applyThinkingConfig(params *anthropic.MessageNewParams, level string) bool {
	effort, on := common.AnthropicThinking(level)
	if !on {
		return false
	}

	// Anthropic API rejects requests with temperature set alongside thinking.
	// Reset to zero value (omitted from JSON serialization).
	if params.Temperature.Valid() {
		logger.DebugCF("anthropic", "temperature cleared because thinking is enabled",
			map[string]any{"level": level})
	}
	params.Temperature = anthropic.MessageNewParams{}.Temperature

	adaptive := anthropic.ThinkingConfigAdaptiveParam{}
	params.Thinking = anthropic.ThinkingConfigParamUnion{OfAdaptive: &adaptive}
	if effort != "" {
		params.OutputConfig = anthropic.OutputConfigParam{Effort: anthropic.OutputConfigEffort(effort)}
	}
	return true
}

func translateTools(tools []ToolDefinition) []anthropic.ToolUnionParam {
	result := make([]anthropic.ToolUnionParam, 0, len(tools))
	for _, t := range tools {
		tool := anthropic.ToolParam{
			Name: t.Function.Name,
			InputSchema: anthropic.ToolInputSchemaParam{
				Properties: t.Function.Parameters["properties"],
			},
		}
		if desc := t.Function.Description; desc != "" {
			tool.Description = anthropic.String(desc)
		}
		if required := requiredNames(t.Function.Parameters["required"]); required != nil {
			tool.InputSchema.Required = required
		}
		result = append(result, anthropic.ToolUnionParam{OfTool: &tool})
	}
	return result
}

// requiredNames returns the parameter names in a JSON schema "required" value.
// Callers build schemas both as Go literals ([]string) and from decoded JSON
// ([]any), so both are accepted; non-string elements are skipped. It returns
// nil when the value is absent or of another type.
func requiredNames(v any) []string {
	switch req := v.(type) {
	case []string:
		return append([]string{}, req...)
	case []any:
		names := make([]string, 0, len(req))
		for _, r := range req {
			if s, ok := r.(string); ok {
				names = append(names, s)
			}
		}
		return names
	default:
		return nil
	}
}

func parseResponse(resp *anthropic.Message) *LLMResponse {
	var content strings.Builder
	var reasoning strings.Builder
	var toolCalls []ToolCall
	var thinking []json.RawMessage

	for _, block := range resp.Content {
		switch block.Type {
		case "thinking":
			// The union's own fields, which streaming deltas update directly.
			reasoning.WriteString(block.Thinking)
			thinking = append(thinking, thinkingBlockJSON(thinkingBlock{
				Type: "thinking", Thinking: block.Thinking, Signature: block.Signature,
			}))
		case "redacted_thinking":
			thinking = append(thinking, thinkingBlockJSON(thinkingBlock{
				Type: "redacted_thinking", Data: block.Data,
			}))
		case "text":
			tb := block.AsText()
			content.WriteString(tb.Text)
		case "tool_use":
			tu := block.AsToolUse()
			var args map[string]any
			if err := json.Unmarshal(tu.Input, &args); err != nil {
				logger.WarnCF("anthropic", "failed to decode tool call input",
					map[string]any{"name": tu.Name, "error": err.Error()})
				args = map[string]any{"raw": string(tu.Input)}
			}
			toolCalls = append(toolCalls, ToolCall{
				ID:        tu.ID,
				Name:      tu.Name,
				Arguments: args,
			})
		}
	}

	finishReason := "stop"
	switch resp.StopReason {
	case anthropic.StopReasonToolUse:
		finishReason = "tool_calls"
	case anthropic.StopReasonMaxTokens:
		finishReason = "length"
	case anthropic.StopReasonEndTurn:
		finishReason = "stop"
	default:
		// Every other stop reason (stop_sequence, pause_turn, refusal,
		// model_context_window_exceeded, …) keeps the "stop" set above; the raw
		// value is still reported in DispatchStatus.StopReason.
	}

	status := &DispatchStatus{
		Success:             true,
		Model:               resp.Model,
		NumTurns:            1,
		InputTokens:         int(resp.Usage.InputTokens),
		OutputTokens:        int(resp.Usage.OutputTokens),
		CacheReadTokens:     int(resp.Usage.CacheReadInputTokens),
		CacheCreationTokens: int(resp.Usage.CacheCreationInputTokens),
		StopReason:          string(resp.StopReason),
	}

	return &LLMResponse{
		Content:           content.String(),
		Reasoning:         reasoning.String(),
		AnthropicThinking: thinking,
		ToolCalls:         toolCalls,
		FinishReason:      finishReason,
		Normal:            finishReason == "stop" || finishReason == "tool_calls",
		Usage: &UsageInfo{
			PromptTokens:     int(resp.Usage.InputTokens),
			CompletionTokens: int(resp.Usage.OutputTokens),
			TotalTokens:      int(resp.Usage.InputTokens + resp.Usage.OutputTokens),
		},
		Status: status,
	}
}

func normalizeBaseURL(apiBase string) string {
	base := strings.TrimSpace(apiBase)
	if base == "" {
		return defaultBaseURL
	}

	base = strings.TrimRight(base, "/")
	if before, ok := strings.CutSuffix(base, "/v1"); ok {
		base = before
	}
	if base == "" {
		return defaultBaseURL
	}

	return base
}
