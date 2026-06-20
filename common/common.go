// ClawEh - Personal AI Assistant
// License: MIT
//
// Copyright (c) 2026 PicoClaw contributors

// Package common provides shared utilities used by multiple LLM provider
// implementations (openai_compat, azure, etc.).
package common

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/PivotLLM/spawnllm/logger"
	"github.com/PivotLLM/spawnllm/protocoltypes"
)

// Re-export protocol types used across providers.
type (
	ToolCall               = protocoltypes.ToolCall
	FunctionCall           = protocoltypes.FunctionCall
	LLMResponse            = protocoltypes.LLMResponse
	UsageInfo              = protocoltypes.UsageInfo
	DispatchStatus         = protocoltypes.DispatchStatus
	Message                = protocoltypes.Message
	ToolDefinition         = protocoltypes.ToolDefinition
	ToolFunctionDefinition = protocoltypes.ToolFunctionDefinition
	ExtraContent           = protocoltypes.ExtraContent
	GoogleExtra            = protocoltypes.GoogleExtra
	ReasoningDetail        = protocoltypes.ReasoningDetail
)

const DefaultRequestTimeout = 120 * time.Second

// NewHTTPClient creates an *http.Client with an optional proxy and the default timeout.
func NewHTTPClient(proxy string) *http.Client {
	client := &http.Client{
		Timeout: DefaultRequestTimeout,
	}
	if proxy != "" {
		parsed, err := url.Parse(proxy)
		if err == nil {
			// Preserve http.DefaultTransport settings (TLS, HTTP/2, timeouts, etc.)
			if base, ok := http.DefaultTransport.(*http.Transport); ok {
				tr := base.Clone()
				tr.Proxy = http.ProxyURL(parsed)
				client.Transport = tr
			} else {
				// Fallback: minimal transport if DefaultTransport is not *http.Transport.
				client.Transport = &http.Transport{
					Proxy: http.ProxyURL(parsed),
				}
			}
		} else {
			logger.WarnCF("common", "invalid proxy URL",
				map[string]any{"proxy": proxy, "error": err.Error()})
		}
	}
	return client
}

// --- Message serialization ---

// openaiMessage is the wire-format message for OpenAI-compatible APIs.
// It mirrors protocoltypes.Message but omits SystemParts, which is an
// internal field that would be unknown to third-party endpoints.
type openaiMessage struct {
	Role             string     `json:"role"`
	Content          string     `json:"content"`
	ReasoningContent string     `json:"reasoning_content,omitempty"`
	ToolCalls        []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID       string     `json:"tool_call_id,omitempty"`
}

// SerializeMessages converts internal Message structs to the OpenAI wire format.
//   - Strips SystemParts (unknown to third-party endpoints)
//   - Converts messages with Media to multipart content format (text + image_url parts)
//   - Preserves ToolCallID, ToolCalls, and ReasoningContent for all messages
func SerializeMessages(messages []Message) []any {
	out := make([]any, 0, len(messages))
	for _, m := range messages {
		if len(m.Media) == 0 {
			out = append(out, openaiMessage{
				Role:             m.Role,
				Content:          m.Content,
				ReasoningContent: m.ReasoningContent,
				ToolCalls:        m.ToolCalls,
				ToolCallID:       m.ToolCallID,
			})
			continue
		}

		// Multipart content format for messages with media
		parts := make([]map[string]any, 0, 1+len(m.Media))
		if m.Content != "" {
			parts = append(parts, map[string]any{
				"type": "text",
				"text": m.Content,
			})
		}
		for _, mediaURL := range m.Media {
			if strings.HasPrefix(mediaURL, "data:image/") {
				parts = append(parts, map[string]any{
					"type": "image_url",
					"image_url": map[string]any{
						"url": mediaURL,
					},
				})
			}
		}

		msg := map[string]any{
			"role":    m.Role,
			"content": parts,
		}
		if m.ToolCallID != "" {
			msg["tool_call_id"] = m.ToolCallID
		}
		if len(m.ToolCalls) > 0 {
			msg["tool_calls"] = m.ToolCalls
		}
		if m.ReasoningContent != "" {
			msg["reasoning_content"] = m.ReasoningContent
		}
		out = append(out, msg)
	}
	return out
}

// --- Response parsing ---

// ParseResponse parses a JSON chat completion response body into an LLMResponse.
// The returned LLMResponse carries a partial DispatchStatus (Success and DurationMs
// are intentionally left for the caller, which knows the HTTP outcome and wall-clock).
//
// toolNames is the set of tool names advertised on the request. It is used to
// cross-check candidate tool calls recovered from non-standard response shapes
// (notably content-embedded JSON descriptors emitted by some upstream routes).
// A nil or empty set disables the content-sniff promotion.
func ParseResponse(body io.Reader, toolNames map[string]struct{}) (*LLMResponse, error) {
	var apiResponse struct {
		Model   string `json:"model"`
		Choices []struct {
			Message struct {
				Content          string            `json:"content"`
				ReasoningContent string            `json:"reasoning_content"`
				Reasoning        string            `json:"reasoning"`
				ReasoningDetails []ReasoningDetail `json:"reasoning_details"`
				ToolCalls        []struct {
					ID       string `json:"id"`
					Type     string `json:"type"`
					Function *struct {
						Name      string          `json:"name"`
						Arguments json.RawMessage `json:"arguments"`
					} `json:"function"`
					ExtraContent *struct {
						Google *struct {
							ThoughtSignature string `json:"thought_signature"`
						} `json:"google"`
					} `json:"extra_content"`
				} `json:"tool_calls"`
				// Legacy OpenAI v0 singular function_call. Still emitted by some
				// upstream routes via OpenRouter. Synthesised into tool_calls below.
				FunctionCall *struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				} `json:"function_call"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
		Usage *struct {
			PromptTokens        int `json:"prompt_tokens"`
			CompletionTokens    int `json:"completion_tokens"`
			TotalTokens         int `json:"total_tokens"`
			PromptTokensDetails *struct {
				CachedTokens int `json:"cached_tokens"`
			} `json:"prompt_tokens_details"`
		} `json:"usage"`
	}

	if err := json.NewDecoder(body).Decode(&apiResponse); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	if len(apiResponse.Choices) == 0 {
		return &LLMResponse{
			Content:      "",
			FinishReason: "stop",
			Normal:       false,
			Status: &DispatchStatus{
				Model:    apiResponse.Model,
				NumTurns: 1,
			},
		}, nil
	}

	// Walk every choice. We don't currently request n>1, but accumulating across
	// all choices is cheap defence-in-depth: dropping choices[1..] silently has
	// burned us with OpenRouter routes that emit tool calls there.
	//
	// Content handling across multiple choices: take the first non-empty value
	// per field. This matches the prior single-choice semantics exactly when
	// only choices[0] is populated (the common case) and avoids inventing a
	// concatenation policy that downstream may not expect.
	var aggToolCalls []ToolCall
	var aggContent string
	var aggReasoningContent string
	var aggReasoning string
	var aggReasoningDetails []ReasoningDetail

	for idx, choice := range apiResponse.Choices {
		toolCalls := make([]ToolCall, 0, len(choice.Message.ToolCalls))
		for _, tc := range choice.Message.ToolCalls {
			arguments := make(map[string]any)
			name := ""

			// Extract thought_signature from Gemini/Google-specific extra content
			thoughtSignature := ""
			if tc.ExtraContent != nil && tc.ExtraContent.Google != nil {
				thoughtSignature = tc.ExtraContent.Google.ThoughtSignature
			}

			if tc.Function != nil {
				name = tc.Function.Name
				arguments = DecodeToolCallArguments(tc.Function.Arguments, name)
			}

			toolCall := ToolCall{
				ID:               tc.ID,
				Name:             name,
				Arguments:        arguments,
				ThoughtSignature: thoughtSignature,
			}

			if thoughtSignature != "" {
				toolCall.ExtraContent = &ExtraContent{
					Google: &GoogleExtra{
						ThoughtSignature: thoughtSignature,
					},
				}
			}

			toolCalls = append(toolCalls, toolCall)
		}

		// Fix A: synthesise a tool call from legacy singular function_call.
		// Only fires when tool_calls is empty AND function_call is present.
		// Downstream must not be able to distinguish this from a real
		// tool_calls[0] entry, so Name/Arguments are populated identically.
		if len(toolCalls) == 0 && choice.Message.FunctionCall != nil &&
			choice.Message.FunctionCall.Name != "" {
			fc := choice.Message.FunctionCall
			args := make(map[string]any)
			if strings.TrimSpace(fc.Arguments) != "" {
				if err := json.Unmarshal([]byte(fc.Arguments), &args); err != nil {
					logger.WarnCF("common", "function_call arguments decode failed",
						map[string]any{"name": fc.Name, "error": err.Error()})
					args = map[string]any{"raw": fc.Arguments}
				}
			}
			toolCalls = append(toolCalls, ToolCall{
				ID:        fmt.Sprintf("call_func_%d", idx),
				Name:      fc.Name,
				Arguments: args,
			})
		}

		// Fix B: strict content-sniff fallback. Only fires when both tool_calls
		// and function_call are empty AND content is non-empty. The candidate
		// tool name MUST appear in toolNames; any miss rejects the entire batch.
		// On success, clear content (the inline JSON IS the content).
		content := choice.Message.Content
		if len(toolCalls) == 0 && choice.Message.FunctionCall == nil && content != "" {
			if sniffed, ok := sniffContentToolCalls(content, toolNames, idx); ok {
				toolCalls = sniffed
				content = ""
			}
		}

		aggToolCalls = append(aggToolCalls, toolCalls...)
		if aggContent == "" {
			aggContent = content
		}
		if aggReasoningContent == "" {
			aggReasoningContent = choice.Message.ReasoningContent
		}
		if aggReasoning == "" {
			aggReasoning = choice.Message.Reasoning
		}
		if len(aggReasoningDetails) == 0 {
			aggReasoningDetails = choice.Message.ReasoningDetails
		}
	}

	choice0 := apiResponse.Choices[0]

	var usage *UsageInfo
	status := &DispatchStatus{
		Model:      apiResponse.Model,
		NumTurns:   1,
		StopReason: choice0.FinishReason,
	}
	if apiResponse.Usage != nil {
		usage = &UsageInfo{
			PromptTokens:     apiResponse.Usage.PromptTokens,
			CompletionTokens: apiResponse.Usage.CompletionTokens,
			TotalTokens:      apiResponse.Usage.TotalTokens,
		}
		status.InputTokens = apiResponse.Usage.PromptTokens
		status.OutputTokens = apiResponse.Usage.CompletionTokens
		if apiResponse.Usage.PromptTokensDetails != nil {
			status.CacheReadTokens = apiResponse.Usage.PromptTokensDetails.CachedTokens
		}
	}

	return &LLMResponse{
		Content:          SanitizeModelContent(aggContent),
		ReasoningContent: aggReasoningContent,
		Reasoning:        aggReasoning,
		ReasoningDetails: aggReasoningDetails,
		ToolCalls:        aggToolCalls,
		FinishReason:     choice0.FinishReason,
		Normal:           choice0.FinishReason == "stop" || choice0.FinishReason == "tool_calls",
		Usage:            usage,
		Status:           status,
	}, nil
}

// sniffContentToolCalls attempts to recover tool calls from a response whose
// upstream emitted them as a JSON descriptor inside the assistant message
// content (observed with Llama-derived routes via OpenRouter). The sniff is
// strict: every candidate's name must appear in toolNames; any miss rejects
// the entire batch so partial recoveries cannot promote one call and leak the
// rest as text. A nil/empty toolNames disables the sniff entirely.
//
// Accepted shapes:
//   - {"type":"function","name":"<str>","parameters":<obj>}
//   - {"type":"function","name":"<str>","arguments":<obj-or-string>}
//   - [<obj>, <obj>, ...] of the above
//
// Trailing junk after the JSON value is tolerated (json.Decoder stops at the
// first complete value), because some upstreams append framing tokens like
// `<|header_start|>assistant<|header_end|>`. Trade-off: a stray well-formed
// descriptor inside otherwise-plain text *could* misfire, but the name cross-
// check is the load-bearing guard against that.
func sniffContentToolCalls(content string, toolNames map[string]struct{}, choiceIdx int) ([]ToolCall, bool) {
	if len(toolNames) == 0 {
		return nil, false
	}
	trimmed := strings.TrimLeft(content, " \t\r\n")
	if trimmed == "" {
		return nil, false
	}
	// Require a JSON value at the start; otherwise plain prose with embedded
	// JSON wouldn't be confused for a tool call.
	if trimmed[0] != '{' && trimmed[0] != '[' {
		return nil, false
	}

	dec := json.NewDecoder(strings.NewReader(trimmed))
	var raw any
	if err := dec.Decode(&raw); err != nil {
		return nil, false
	}

	var candidates []map[string]any
	switch v := raw.(type) {
	case map[string]any:
		candidates = []map[string]any{v}
	case []any:
		if len(v) == 0 {
			return nil, false
		}
		candidates = make([]map[string]any, 0, len(v))
		for _, item := range v {
			m, ok := item.(map[string]any)
			if !ok {
				return nil, false
			}
			candidates = append(candidates, m)
		}
	default:
		return nil, false
	}

	out := make([]ToolCall, 0, len(candidates))
	for i, c := range candidates {
		typ, _ := c["type"].(string)
		if typ != "function" {
			return nil, false
		}
		name, _ := c["name"].(string)
		if name == "" {
			return nil, false
		}
		if _, ok := toolNames[name]; !ok {
			return nil, false
		}
		var args map[string]any
		if p, ok := c["parameters"]; ok {
			args = coerceSniffArgs(p)
		} else if a, ok := c["arguments"]; ok {
			args = coerceSniffArgs(a)
		} else {
			args = map[string]any{}
		}
		out = append(out, ToolCall{
			ID:        fmt.Sprintf("call_sniff_%d_%d", choiceIdx, i),
			Name:      name,
			Arguments: args,
		})
	}
	return out, true
}

func coerceSniffArgs(v any) map[string]any {
	switch val := v.(type) {
	case map[string]any:
		return val
	case string:
		m := map[string]any{}
		if strings.TrimSpace(val) == "" {
			return m
		}
		if err := json.Unmarshal([]byte(val), &m); err != nil {
			return map[string]any{"raw": val}
		}
		return m
	default:
		return map[string]any{}
	}
}

// DecodeToolCallArguments decodes a tool call's arguments from raw JSON.
func DecodeToolCallArguments(raw json.RawMessage, name string) map[string]any {
	arguments := make(map[string]any)
	raw = bytes.TrimSpace(raw)
	if len(raw) == 0 || bytes.Equal(raw, []byte("null")) {
		return arguments
	}

	var decoded any
	if err := json.Unmarshal(raw, &decoded); err != nil {
		logger.WarnCF("common", "failed to decode tool call arguments payload",
			map[string]any{"name": name, "error": err.Error()})
		arguments["raw"] = string(raw)
		return arguments
	}

	switch v := decoded.(type) {
	case string:
		if strings.TrimSpace(v) == "" {
			return arguments
		}
		if err := json.Unmarshal([]byte(v), &arguments); err != nil {
			logger.WarnCF("common", "failed to decode tool call arguments",
				map[string]any{"name": name, "error": err.Error()})
			arguments["raw"] = v
		}
		return arguments
	case map[string]any:
		return v
	default:
		logger.WarnCF("common", "unsupported tool call arguments type",
			map[string]any{"name": name, "type": fmt.Sprintf("%T", decoded)})
		arguments["raw"] = string(raw)
		return arguments
	}
}

// --- HTTP response helpers ---

// HTTPStatusError wraps a non-2xx HTTP response so the fallback classifier can
// extract the status code and any Retry-After hint without re-parsing the
// formatted error string. Error() preserves the historical message shape (used
// in logs and existing tests).
type HTTPStatusError struct {
	StatusCode  int
	APIBase     string
	BodyPreview string
	RetryAfter  time.Duration // 0 if absent or unparsable
}

func (e *HTTPStatusError) Error() string {
	return fmt.Sprintf(
		"API request failed:\n  Status: %d\n  Body:   %s",
		e.StatusCode,
		e.BodyPreview,
	)
}

// HandleErrorResponse reads a non-200 response body and returns an appropriate error.
// Status errors are returned as *HTTPStatusError so the classifier can read the
// status code and Retry-After header structurally; HTML responses use the
// existing HTML wrapper.
//
// The read cap is 1024 bytes (was 256) so we can capture the JSON billing
// envelope intact — OpenRouter's billing_url field and OpenAI's
// error.code/error.type fields would otherwise truncate mid-value, leaving
// the classifier unable to distinguish credits-exhausted from rate-limit.
func HandleErrorResponse(resp *http.Response, apiBase string) error {
	contentType := resp.Header.Get("Content-Type")
	body, readErr := io.ReadAll(io.LimitReader(resp.Body, 1024))
	if readErr != nil {
		return fmt.Errorf("failed to read response: %w", readErr)
	}
	if LooksLikeHTML(body, contentType) {
		return WrapHTMLResponseError(resp.StatusCode, body, contentType, apiBase)
	}
	return &HTTPStatusError{
		StatusCode:  resp.StatusCode,
		APIBase:     apiBase,
		BodyPreview: ResponsePreview(body, 1024),
		RetryAfter:  ParseRetryAfter(resp.Header.Get("Retry-After"), time.Now()),
	}
}

// ParseRetryAfter parses an HTTP Retry-After header per RFC 7231:
// either a number of seconds (delta-seconds) or an HTTP-date. Returns 0
// when the value is empty, unparsable, or non-positive.
func ParseRetryAfter(raw string, now time.Time) time.Duration {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return 0
	}
	// delta-seconds (may be a float for some servers).
	if secs, err := strconv.ParseFloat(raw, 64); err == nil {
		if secs <= 0 {
			return 0
		}
		return time.Duration(secs * float64(time.Second))
	}
	// HTTP-date.
	if t, err := http.ParseTime(raw); err == nil {
		if d := t.Sub(now); d > 0 {
			return d
		}
	}
	return 0
}

// ReadAndParseResponse peeks at the response body to detect HTML errors,
// then parses the JSON response into an LLMResponse. toolNames is forwarded
// to ParseResponse for the content-sniff cross-check.
func ReadAndParseResponse(resp *http.Response, apiBase string, toolNames map[string]struct{}) (*LLMResponse, error) {
	out, _, err := ReadParseAndMeasure(resp, apiBase, toolNames)
	return out, err
}

// ReadParseAndMeasure is like ReadAndParseResponse but also returns the raw
// response-body length so callers can fill DispatchStatus.BytesReceived. The
// underlying reader is streamed (mirroring ReadAndParseResponse's prior
// behaviour) so endpoints that close mid-trailer after a complete JSON object
// still parse successfully.
var (
	// deepseekTokenBalanced matches DeepSeek special tokens delimited by the
	// fullwidth vertical bar (U+FF5C), e.g. <｜tool▁calls▁begin｜> or <｜DSML｜…｜>.
	deepseekTokenBalanced = regexp.MustCompile(`<\x{FF5C}[^<>]*?\x{FF5C}>`)
	// deepseekTokenDangling matches an unterminated opener a model leaks as plain
	// text, e.g. a bare "<｜DSML｜function_calls" with no closing "｜>".
	deepseekTokenDangling = regexp.MustCompile(`<\x{FF5C}[^\n<>]*`)
)

// SanitizeModelContent strips provider special tokens — notably DeepSeek's
// fullwidth-bar tool-call delimiters — that some models leak into message
// content instead of emitting them as structured tool calls. Without this they
// surface as literal noise (e.g. "<｜DSML｜function_calls") in the chat. The guard
// keeps the common (no-token) path allocation-free.
func SanitizeModelContent(s string) string {
	if !strings.Contains(s, "<｜") {
		return s
	}
	s = deepseekTokenBalanced.ReplaceAllString(s, "")
	s = deepseekTokenDangling.ReplaceAllString(s, "")
	return strings.TrimSpace(s)
}

func ReadParseAndMeasure(resp *http.Response, apiBase string, toolNames map[string]struct{}) (*LLMResponse, int64, error) {
	contentType := resp.Header.Get("Content-Type")
	counter := &readCounter{r: resp.Body}
	reader := bufio.NewReader(counter)
	prefix, err := reader.Peek(256)
	if err != nil && err != io.EOF && err != bufio.ErrBufferFull {
		return nil, counter.n, fmt.Errorf("failed to inspect response: %w", err)
	}
	if LooksLikeHTML(prefix, contentType) {
		return nil, counter.n, WrapHTMLResponseError(resp.StatusCode, prefix, contentType, apiBase)
	}
	out, err := ParseResponse(reader, toolNames)
	if err != nil {
		return nil, counter.n, fmt.Errorf("failed to parse JSON response: %w", err)
	}
	return out, counter.n, nil
}

// ToolNameSet returns a name set built from a list of tool definitions,
// suitable for passing to ParseResponse/ReadAndParseResponse for the strict
// content-sniff cross-check.
func ToolNameSet(tools []ToolDefinition) map[string]struct{} {
	if len(tools) == 0 {
		return nil
	}
	set := make(map[string]struct{}, len(tools))
	for _, t := range tools {
		if t.Function.Name != "" {
			set[t.Function.Name] = struct{}{}
		}
	}
	return set
}

// readCounter wraps an io.Reader and tracks the cumulative byte count read.
type readCounter struct {
	r io.Reader
	n int64
}

func (c *readCounter) Read(p []byte) (int, error) {
	n, err := c.r.Read(p)
	c.n += int64(n)
	return n, err
}

// LooksLikeHTML checks if the response body appears to be HTML.
func LooksLikeHTML(body []byte, contentType string) bool {
	contentType = strings.ToLower(strings.TrimSpace(contentType))
	if strings.Contains(contentType, "text/html") || strings.Contains(contentType, "application/xhtml+xml") {
		return true
	}
	prefix := bytes.ToLower(leadingTrimmedPrefix(body, 128))
	return bytes.HasPrefix(prefix, []byte("<!doctype html")) ||
		bytes.HasPrefix(prefix, []byte("<html")) ||
		bytes.HasPrefix(prefix, []byte("<head")) ||
		bytes.HasPrefix(prefix, []byte("<body"))
}

// WrapHTMLResponseError creates a descriptive error for HTML responses.
func WrapHTMLResponseError(statusCode int, body []byte, contentType, apiBase string) error {
	respPreview := ResponsePreview(body, 128)
	return fmt.Errorf(
		"API request failed: %s returned HTML instead of JSON (content-type: %s); check api_base or proxy configuration.\n  Status: %d\n  Body:   %s",
		apiBase,
		contentType,
		statusCode,
		respPreview,
	)
}

// ResponsePreview returns a truncated preview of response body for error messages.
func ResponsePreview(body []byte, maxLen int) string {
	trimmed := bytes.TrimSpace(body)
	if len(trimmed) == 0 {
		return "<empty>"
	}
	if len(trimmed) <= maxLen {
		return string(trimmed)
	}
	return string(trimmed[:maxLen]) + "..."
}

func leadingTrimmedPrefix(body []byte, maxLen int) []byte {
	i := 0
	for i < len(body) {
		switch body[i] {
		case ' ', '\t', '\n', '\r', '\f', '\v':
			i++
		default:
			end := i + maxLen
			if end > len(body) {
				end = len(body)
			}
			return body[i:end]
		}
	}
	return nil
}

// --- Numeric helpers ---

// AsInt converts various numeric types to int.
func AsInt(v any) (int, bool) {
	switch val := v.(type) {
	case int:
		return val, true
	case int64:
		return int(val), true
	case float64:
		return int(val), true
	case float32:
		return int(val), true
	default:
		return 0, false
	}
}

// AsFloat converts various numeric types to float64.
func AsFloat(v any) (float64, bool) {
	switch val := v.(type) {
	case float64:
		return val, true
	case float32:
		return float64(val), true
	case int:
		return float64(val), true
	case int64:
		return float64(val), true
	default:
		return 0, false
	}
}
