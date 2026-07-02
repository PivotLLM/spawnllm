// ClawEh - Personal AI Assistant
// License: MIT
//
// Copyright (c) 2026 PicoClaw contributors

package anthropicmessages

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/PivotLLM/spawnllm/common"
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
	defaultAPIVersion     = "2023-06-01"
	defaultBaseURL        = "https://api.anthropic.com/v1"
	defaultRequestTimeout = 120 * time.Second
)

// Provider implements Anthropic Messages API via HTTP (without SDK).
// It supports custom endpoints that use Anthropic's native message format.
type Provider struct {
	apiKey     string
	apiBase    string
	httpClient *http.Client
}

// NewProvider creates a new Anthropic Messages API provider.
func NewProvider(apiKey, apiBase string) *Provider {
	return NewProviderWithTimeout(apiKey, apiBase, 0)
}

// NewProviderWithTimeout creates a provider with custom request timeout.
func NewProviderWithTimeout(apiKey, apiBase string, timeoutSeconds int) *Provider {
	baseURL := normalizeBaseURL(apiBase)
	timeout := defaultRequestTimeout
	if timeoutSeconds > 0 {
		timeout = time.Duration(timeoutSeconds) * time.Second
	}

	return &Provider{
		apiKey:  apiKey,
		apiBase: baseURL,
		httpClient: &http.Client{
			Timeout: timeout,
		},
	}
}

// Chat sends messages to the Anthropic Messages API and returns the response.
func (p *Provider) Chat(
	ctx context.Context,
	messages []Message,
	tools []ToolDefinition,
	model string,
	options map[string]any,
) (*LLMResponse, error) {
	if p.apiKey == "" {
		return nil, fmt.Errorf("API key not configured")
	}

	// Build request body
	requestBody, err := buildRequestBody(messages, tools, model, options)
	if err != nil {
		return nil, fmt.Errorf("building request body: %w", err)
	}

	// Streaming is opt-in: only when the caller supplied a non-nil delta
	// callback. Absent it, the request body and response handling stay on the
	// unchanged single-shot path.
	streamCB, streaming := options[common.TextDeltaOption].(common.TextDeltaFunc)
	streaming = streaming && streamCB != nil
	if streaming {
		requestBody["stream"] = true
	}

	// Serialize to JSON
	jsonBody, err := json.Marshal(requestBody)
	if err != nil {
		return nil, fmt.Errorf("serializing request body: %w", err)
	}

	// Build request URL
	endpointURL, err := url.JoinPath(p.apiBase, "messages")
	if err != nil {
		return nil, fmt.Errorf("building endpoint URL: %w", err)
	}

	// Create HTTP request
	req, err := http.NewRequestWithContext(ctx, "POST", endpointURL, bytes.NewReader(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("creating HTTP request: %w", err)
	}

	// Set headers
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-API-Key", p.apiKey) //nolint:canonicalheader // Anthropic API requires exact header name
	req.Header.Set("Anthropic-Version", defaultAPIVersion)

	bytesSent := int64(len(jsonBody))

	// Execute request
	start := time.Now()
	resp, err := p.httpClient.Do(req)
	if err != nil {
		elapsed := time.Since(start).Milliseconds()
		return httpErrorResponse(model, "error", elapsed, bytesSent, 0),
			fmt.Errorf("executing HTTP request: %w", err)
	}
	defer resp.Body.Close()

	// Streaming reads the body incrementally; the single-shot path below slurps
	// the whole body. The non-200 error mapping is shared: both paths fall
	// through to the status switch after reading the body.
	if streaming && resp.StatusCode == http.StatusOK {
		return p.readStream(resp, streamCB, model, bytesSent, start)
	}

	// Read response body
	body, err := io.ReadAll(resp.Body)
	durationMs := time.Since(start).Milliseconds()
	bytesReceived := int64(len(body))
	if err != nil {
		return httpErrorResponse(model, "error", durationMs, bytesSent, bytesReceived),
			fmt.Errorf("reading response body: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		errResp := httpErrorResponse(model, "error", durationMs, bytesSent, bytesReceived)
		switch resp.StatusCode {
		case http.StatusUnauthorized:
			return errResp, fmt.Errorf("authentication failed (401): check your API key")
		case http.StatusTooManyRequests:
			return errResp, fmt.Errorf("rate limited (429): %s", string(body))
		case http.StatusBadRequest:
			return errResp, fmt.Errorf("bad request (400): %s", string(body))
		case http.StatusNotFound:
			return errResp, fmt.Errorf("endpoint not found (404): %s", string(body))
		case http.StatusInternalServerError:
			return errResp, fmt.Errorf("internal server error (500): %s", string(body))
		case http.StatusServiceUnavailable:
			return errResp, fmt.Errorf("service unavailable (503): %s", string(body))
		default:
			return errResp, fmt.Errorf("API request failed with status %d: %s", resp.StatusCode, string(body))
		}
	}

	out, err := parseResponseBody(body)
	if err != nil {
		return httpErrorResponse(model, "parse_error", durationMs, bytesSent, bytesReceived), err
	}
	if out.Status != nil {
		out.Status.DurationMs = durationMs
		out.Status.BytesSent = bytesSent
		out.Status.BytesReceived = bytesReceived
	}
	return out, nil
}

// readStream consumes an Anthropic Messages SSE stream, firing cb for each
// text delta as it arrives, then reconstructs the equivalent non-streaming
// Messages-API response body and runs it through the SAME parseResponseBody the
// single-shot path uses so tool/usage/stop mapping is shared verbatim. Status
// fields are set exactly as the single-shot path sets them. An in-band `error`
// event is mapped to an error (partial text may already have been emitted).
func (p *Provider) readStream(
	resp *http.Response,
	cb common.TextDeltaFunc,
	model string,
	bytesSent int64,
	start time.Time,
) (*LLMResponse, error) {
	counter := &countingReader{r: resp.Body}
	body, streamErr := accumulateMessagesStream(counter, cb)
	durationMs := time.Since(start).Milliseconds()
	bytesReceived := counter.n

	if streamErr != nil {
		return httpErrorResponse(model, "error", durationMs, bytesSent, bytesReceived), streamErr
	}

	out, err := parseResponseBody(body)
	if err != nil {
		return httpErrorResponse(model, "parse_error", durationMs, bytesSent, bytesReceived), err
	}
	if out.Status != nil {
		out.Status.DurationMs = durationMs
		out.Status.BytesSent = bytesSent
		out.Status.BytesReceived = bytesReceived
	}
	return out, nil
}

// countingReader tracks bytes read so streaming can report BytesReceived.
type countingReader struct {
	r io.Reader
	n int64
}

func (c *countingReader) Read(p []byte) (int, error) {
	n, err := c.r.Read(p)
	c.n += int64(n)
	return n, err
}

// streamContentBlock accumulates one content block across SSE events, keyed by
// its stream index. text blocks collect text; tool_use blocks collect the id,
// name, and the input_json_delta fragments that reassemble the arguments JSON.
type streamContentBlock struct {
	typ   string
	text  strings.Builder
	id    string
	name  string
	input strings.Builder
}

// accumulateMessagesStream reads an Anthropic Messages SSE stream, firing cb for
// each text_delta, and reconstructs the equivalent non-streaming Messages-API
// JSON body (message_start skeleton + accumulated content blocks + message_delta
// stop_reason/usage). The SSE line framing is reused from common.SSEReader; the
// event/accumulation shape is Anthropic-specific and stays local to this
// package. An `error` event returns a non-nil error.
func accumulateMessagesStream(r io.Reader, cb common.TextDeltaFunc) ([]byte, error) {
	sr := common.NewSSEReader(r)

	var (
		model        string
		stopReason   string
		inputTokens  int64
		outputTokens int64
		cacheRead    int64
		cacheCreate  int64
		order        []int
		byIndex      = map[int]*streamContentBlock{}
	)

	for {
		payload, ok, err := sr.Next()
		if !ok {
			if err != nil && !errors.Is(err, io.EOF) {
				return nil, fmt.Errorf("reading stream: %w", err)
			}
			break
		}

		var evt messagesStreamEvent
		if err := json.Unmarshal([]byte(payload), &evt); err != nil {
			// Skip framing noise / keep-alives we don't understand.
			continue
		}

		switch evt.Type {
		case "error":
			return nil, fmt.Errorf("stream error: %s", streamEventErrorMessage(evt.Error))
		case "message_start":
			if evt.Message != nil {
				model = evt.Message.Model
				inputTokens = evt.Message.Usage.InputTokens
				outputTokens = evt.Message.Usage.OutputTokens
				cacheRead = evt.Message.Usage.CacheReadInputTokens
				cacheCreate = evt.Message.Usage.CacheCreationInputTokens
			}
		case "content_block_start":
			blk := &streamContentBlock{}
			if evt.ContentBlock != nil {
				blk.typ = evt.ContentBlock.Type
				blk.id = evt.ContentBlock.ID
				blk.name = evt.ContentBlock.Name
			}
			byIndex[evt.Index] = blk
			order = append(order, evt.Index)
		case "content_block_delta":
			blk := byIndex[evt.Index]
			if blk == nil {
				blk = &streamContentBlock{}
				byIndex[evt.Index] = blk
				order = append(order, evt.Index)
			}
			switch evt.Delta.Type {
			case "text_delta":
				if evt.Delta.Text != "" {
					blk.text.WriteString(evt.Delta.Text)
					cb(evt.Delta.Text)
				}
			case "input_json_delta":
				// Tool-call argument fragments accumulate but are NOT surfaced
				// via cb; only assistant text is delivered as deltas.
				blk.input.WriteString(evt.Delta.PartialJSON)
			}
		case "message_delta":
			if evt.Delta.StopReason != "" {
				stopReason = evt.Delta.StopReason
			}
			// message_delta carries the running output token count.
			if evt.Usage != nil {
				outputTokens = evt.Usage.OutputTokens
			}
		}
	}

	return reconstructMessagesBody(model, stopReason, inputTokens, outputTokens,
		cacheRead, cacheCreate, order, byIndex), nil
}

// reconstructMessagesBody assembles a non-streaming Messages-API JSON body from
// the accumulated stream state. The shape mirrors exactly what parseResponseBody
// consumes (content[].{text|tool_use}, stop_reason, model, usage), so no parsing
// is duplicated.
func reconstructMessagesBody(
	model, stopReason string,
	inputTokens, outputTokens, cacheRead, cacheCreate int64,
	order []int,
	byIndex map[int]*streamContentBlock,
) []byte {
	content := make([]map[string]any, 0, len(order))
	for _, idx := range order {
		blk := byIndex[idx]
		switch blk.typ {
		case "tool_use":
			var input map[string]any
			raw := blk.input.String()
			if raw == "" {
				input = map[string]any{}
			} else if err := json.Unmarshal([]byte(raw), &input); err != nil {
				input = map[string]any{}
			}
			content = append(content, map[string]any{
				"type":  "tool_use",
				"id":    blk.id,
				"name":  blk.name,
				"input": input,
			})
		default:
			content = append(content, map[string]any{
				"type": "text",
				"text": blk.text.String(),
			})
		}
	}

	body := map[string]any{
		"type":        "message",
		"role":        "assistant",
		"model":       model,
		"stop_reason": stopReason,
		"content":     content,
		"usage": map[string]any{
			"input_tokens":                inputTokens,
			"output_tokens":               outputTokens,
			"cache_read_input_tokens":     cacheRead,
			"cache_creation_input_tokens": cacheCreate,
		},
	}

	out, err := json.Marshal(body)
	if err != nil {
		// A map of plain strings/ints/maps cannot realistically fail to marshal;
		// fall back to a benign empty-content body.
		return []byte(`{"type":"message","content":[]}`)
	}
	return out
}

// messagesStreamEvent is the subset of an Anthropic Messages SSE event we read.
type messagesStreamEvent struct {
	Type    string `json:"type"`
	Index   int    `json:"index"`
	Message *struct {
		Model string    `json:"model"`
		Usage usageInfo `json:"usage"`
	} `json:"message"`
	ContentBlock *struct {
		Type string `json:"type"`
		ID   string `json:"id"`
		Name string `json:"name"`
	} `json:"content_block"`
	Delta struct {
		Type        string `json:"type"`
		Text        string `json:"text"`
		PartialJSON string `json:"partial_json"`
		StopReason  string `json:"stop_reason"`
	} `json:"delta"`
	Usage *struct {
		OutputTokens int64 `json:"output_tokens"`
	} `json:"usage"`
	Error json.RawMessage `json:"error"`
}

// streamEventErrorMessage extracts a human-readable message from an in-band
// error envelope ({"message":"..."} or a bare string); otherwise the raw JSON.
func streamEventErrorMessage(raw json.RawMessage) string {
	if len(raw) == 0 {
		return "messages stream error"
	}
	var obj struct {
		Message string `json:"message"`
	}
	if err := json.Unmarshal(raw, &obj); err == nil && obj.Message != "" {
		return obj.Message
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil && s != "" {
		return s
	}
	return string(raw)
}

// httpErrorResponse builds an LLMResponse with a partial DispatchStatus capturing
// whatever byte counts and elapsed time were known when the error fired.
func httpErrorResponse(model, stopReason string, durationMs, bytesSent, bytesReceived int64) *LLMResponse {
	return &LLMResponse{
		Status: &DispatchStatus{
			Success:       false,
			Model:         model,
			StopReason:    stopReason,
			DurationMs:    durationMs,
			BytesSent:     bytesSent,
			BytesReceived: bytesReceived,
		},
	}
}

// GetDefaultModel returns the default model for this provider.
func (p *Provider) GetDefaultModel() string {
	return "claude-sonnet-4.6"
}

// buildRequestBody converts internal message format to Anthropic Messages API format.
func buildRequestBody(
	messages []Message,
	tools []ToolDefinition,
	model string,
	options map[string]any,
) (map[string]any, error) {
	// max_tokens is required and guaranteed by agent loop
	maxTokens, ok := asInt(options["max_tokens"])
	if !ok {
		return nil, fmt.Errorf("max_tokens is required in options")
	}

	result := map[string]any{
		"model":      model,
		"max_tokens": int64(maxTokens),
		"messages":   []any{},
	}

	// Set temperature from options
	if temp, ok := asFloat(options["temperature"]); ok {
		result["temperature"] = temp
	}

	// Process messages
	var systemPrompt string
	var apiMessages []any

	for _, msg := range messages {
		switch msg.Role {
		case "system":
			// Accumulate system messages
			if systemPrompt != "" {
				systemPrompt += "\n\n" + msg.Content
			} else {
				systemPrompt = msg.Content
			}

		case "user":
			if msg.ToolCallID != "" {
				// Tool result message
				content := []map[string]any{
					{
						"type":        "tool_result",
						"tool_use_id": msg.ToolCallID,
						"content":     msg.Content,
					},
				}
				apiMessages = append(apiMessages, map[string]any{
					"role":    "user",
					"content": content,
				})
			} else {
				// Regular user message
				apiMessages = append(apiMessages, map[string]any{
					"role":    "user",
					"content": msg.Content,
				})
			}

		case "assistant":
			content := []any{}

			// Add text content if present
			if msg.Content != "" {
				content = append(content, map[string]any{
					"type": "text",
					"text": msg.Content,
				})
			}

			// Add tool_use blocks
			for _, tc := range msg.ToolCalls {
				toolUse := map[string]any{
					"type":  "tool_use",
					"id":    tc.ID,
					"name":  tc.Name,
					"input": tc.Arguments,
				}
				content = append(content, toolUse)
			}

			apiMessages = append(apiMessages, map[string]any{
				"role":    "assistant",
				"content": content,
			})

		case "tool":
			// Tool result (alternative format)
			content := []map[string]any{
				{
					"type":        "tool_result",
					"tool_use_id": msg.ToolCallID,
					"content":     msg.Content,
				},
			}
			apiMessages = append(apiMessages, map[string]any{
				"role":    "user",
				"content": content,
			})
		}
	}

	result["messages"] = apiMessages

	// Set system prompt if present
	if systemPrompt != "" {
		result["system"] = systemPrompt
	}

	// Add tools if present
	if len(tools) > 0 {
		result["tools"] = buildTools(tools)
	}

	return result, nil
}

// buildTools converts tool definitions to Anthropic format.
func buildTools(tools []ToolDefinition) []any {
	result := make([]any, len(tools))
	for i, tool := range tools {
		toolDef := map[string]any{
			"name":         tool.Function.Name,
			"description":  tool.Function.Description,
			"input_schema": tool.Function.Parameters,
		}
		result[i] = toolDef
	}
	return result
}

// parseResponseBody parses Anthropic Messages API response.
func parseResponseBody(body []byte) (*LLMResponse, error) {
	var resp anthropicMessageResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, fmt.Errorf("parsing JSON response: %w", err)
	}

	// Extract content and tool calls
	var content strings.Builder
	toolCalls := make([]ToolCall, 0) // Initialize as empty slice (not nil) for consistent JSON serialization

	for _, block := range resp.Content {
		switch block.Type {
		case "text":
			content.WriteString(block.Text)
		case "tool_use":
			argsJSON, _ := json.Marshal(block.Input)
			toolCalls = append(toolCalls, ToolCall{
				ID:        block.ID,
				Name:      block.Name,
				Arguments: block.Input,
				Function: &FunctionCall{
					Name:      block.Name,
					Arguments: string(argsJSON),
				},
			})
		}
	}

	// Map stop_reason
	finishReason := "stop"
	switch resp.StopReason {
	case "tool_use":
		finishReason = "tool_calls"
	case "max_tokens":
		finishReason = "length"
	case "end_turn":
		finishReason = "stop"
	case "stop_sequence":
		finishReason = "stop"
	}

	status := &DispatchStatus{
		Success:             true,
		Model:               resp.Model,
		NumTurns:            1,
		InputTokens:         int(resp.Usage.InputTokens),
		OutputTokens:        int(resp.Usage.OutputTokens),
		CacheReadTokens:     int(resp.Usage.CacheReadInputTokens),
		CacheCreationTokens: int(resp.Usage.CacheCreationInputTokens),
		StopReason:          resp.StopReason,
	}

	return &LLMResponse{
		Content:      content.String(),
		ToolCalls:    toolCalls,
		FinishReason: finishReason,
		Normal:       finishReason == "stop" || finishReason == "tool_calls",
		Usage: &UsageInfo{
			PromptTokens:     int(resp.Usage.InputTokens),
			CompletionTokens: int(resp.Usage.OutputTokens),
			TotalTokens:      int(resp.Usage.InputTokens + resp.Usage.OutputTokens),
		},
		Status: status,
	}, nil
}

// normalizeBaseURL ensures the base URL is properly formatted.
// It removes /v1 suffix if present (to avoid duplication) and always appends /v1.
// This handles edge cases like "https://api.example.com/v1/proxy" correctly.
func normalizeBaseURL(apiBase string) string {
	base := strings.TrimSpace(apiBase)
	if base == "" {
		return defaultBaseURL
	}

	// Remove trailing slashes
	base = strings.TrimRight(base, "/")

	// Remove /v1 suffix if present (will be re-added)
	// This prevents duplication for URLs like "https://api.example.com/v1/proxy"
	if before, ok := strings.CutSuffix(base, "/v1"); ok {
		base = before
	}

	// Ensure we don't have an empty string after cutting
	if base == "" {
		return defaultBaseURL
	}

	// Add /v1 suffix (required by Anthropic Messages API)
	return base + "/v1"
}

// Helper functions for type conversion

func asInt(v any) (int, bool) {
	switch val := v.(type) {
	case int:
		return val, true
	case float64:
		return int(val), true
	case int64:
		return int(val), true
	default:
		return 0, false
	}
}

func asFloat(v any) (float64, bool) {
	switch val := v.(type) {
	case float64:
		return val, true
	case int:
		return float64(val), true
	case int64:
		return float64(val), true
	default:
		return 0, false
	}
}

// Anthropic API response structures

type anthropicMessageResponse struct {
	ID         string         `json:"id"`
	Type       string         `json:"type"`
	Role       string         `json:"role"`
	Content    []contentBlock `json:"content"`
	StopReason string         `json:"stop_reason"`
	Model      string         `json:"model"`
	Usage      usageInfo      `json:"usage"`
}

type contentBlock struct {
	Type  string         `json:"type"`
	Text  string         `json:"text,omitempty"`
	ID    string         `json:"id,omitempty"`
	Name  string         `json:"name,omitempty"`
	Input map[string]any `json:"input,omitempty"`
}

type usageInfo struct {
	InputTokens              int64 `json:"input_tokens"`
	OutputTokens             int64 `json:"output_tokens"`
	CacheCreationInputTokens int64 `json:"cache_creation_input_tokens"`
	CacheReadInputTokens     int64 `json:"cache_read_input_tokens"`
}
