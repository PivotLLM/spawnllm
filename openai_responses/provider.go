// ClawEh
// License: MIT

// Package openai_responses implements the OpenAI Responses API
// (POST <base>/responses) as a Claw LLMProvider, kept entirely separate from the
// Chat Completions provider (pkg/providers/openai_compat) so work here cannot
// regress the chat path. It targets the canonical OpenAI Responses spec, which
// OpenRouter also exposes (Beta) as a drop-in — only the base_url differs.
package openai_responses

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/PivotLLM/spawnllm/common"
	"github.com/PivotLLM/spawnllm/logger"
	"github.com/PivotLLM/spawnllm/protocoltypes"
)

type (
	ToolCall       = protocoltypes.ToolCall
	FunctionCall   = protocoltypes.FunctionCall
	LLMResponse    = protocoltypes.LLMResponse
	UsageInfo      = protocoltypes.UsageInfo
	DispatchStatus = protocoltypes.DispatchStatus
	Message        = protocoltypes.Message
	ToolDefinition = protocoltypes.ToolDefinition
)

// ResponseFormatJSONObjectOption is the options-map key callers set to force a
// JSON-object response. It must match openai_compat.ResponseFormatJSONObjectOption
// so the summarizer's option flows to either provider unchanged.
const ResponseFormatJSONObjectOption = "response_format_json_object"

// Provider speaks the OpenAI Responses API.
type Provider struct {
	apiKey              string
	apiBase             string
	noParallelToolCalls bool
	reasoningEffort     string         // mapped to reasoning.effort
	extraBody           map[string]any // free-form passthrough merged before marshal
	dropParams          map[string]struct{}
	responseFormatJSON  bool   // may emit text.format=json_object when asked
	protocol            string // for capability gating / logs
	modelLabel          string // user-facing model name for logs
	responseLogFile     string // append-only JSONL diagnostic capture; "" disables
	httpClient          *http.Client

	logMu sync.Mutex
}

type Option func(*Provider)

// NewProvider constructs a Responses provider. apiBase is the provider base URL
// (e.g. https://api.openai.com/v1 or https://openrouter.ai/api/v1); "/responses"
// is appended per request.
func NewProvider(apiKey, apiBase, proxy string, opts ...Option) *Provider {
	p := &Provider{
		apiKey:     apiKey,
		apiBase:    strings.TrimRight(apiBase, "/"),
		httpClient: common.NewHTTPClient(proxy),
	}
	for _, opt := range opts {
		if opt != nil {
			opt(p)
		}
	}
	return p
}

func WithRequestTimeout(timeout time.Duration) Option {
	return func(p *Provider) {
		if timeout > 0 {
			p.httpClient.Timeout = timeout
		}
	}
}

func WithReasoningEffort(level string) Option {
	return func(p *Provider) { p.reasoningEffort = level }
}

func WithNoParallelToolCalls(v bool) Option {
	return func(p *Provider) { p.noParallelToolCalls = v }
}

func WithExtraBody(extra map[string]any) Option {
	return func(p *Provider) { p.extraBody = extra }
}

func WithDropParams(params []string) Option {
	return func(p *Provider) {
		if len(params) == 0 {
			return
		}
		m := make(map[string]struct{}, len(params))
		for _, k := range params {
			if k = strings.TrimSpace(k); k != "" {
				m[k] = struct{}{}
			}
		}
		p.dropParams = m
	}
}

func WithResponseFormatJSONCapable(v bool) Option {
	return func(p *Provider) { p.responseFormatJSON = v }
}

func WithProtocol(protocol string) Option {
	return func(p *Provider) { p.protocol = protocol }
}

func WithModelLabel(label string) Option {
	return func(p *Provider) { p.modelLabel = label }
}

func WithResponseLogFile(path string) Option {
	return func(p *Provider) { p.responseLogFile = path }
}

// GetDefaultModel satisfies LLMProvider; the dispatcher always passes an explicit
// model, so this is only a fallback label.
func (p *Provider) GetDefaultModel() string { return "" }

func (p *Provider) modelLabelOr(model string) string {
	if p.modelLabel != "" {
		return p.modelLabel
	}
	return model
}

// Chat sends one Responses API request and maps the result into an LLMResponse.
func (p *Provider) Chat(
	ctx context.Context,
	messages []Message,
	tools []ToolDefinition,
	model string,
	options map[string]any,
) (*LLMResponse, error) {
	if p.apiBase == "" {
		return nil, fmt.Errorf("API base not configured")
	}

	instructions, input := buildInput(messages)

	requestBody := map[string]any{
		"model": model,
		"input": input,
	}
	if instructions != "" {
		requestBody["instructions"] = instructions
	}

	if len(tools) > 0 {
		requestBody["tools"] = responsesTools(tools)
		requestBody["tool_choice"] = "auto"
		if p.noParallelToolCalls {
			requestBody["parallel_tool_calls"] = false
		}
	}

	if maxTokens, ok := common.AsInt(options["max_tokens"]); ok {
		requestBody["max_output_tokens"] = maxTokens
	}
	// "none" means reasoning is explicitly disabled (sent, but not a reasoning
	// run); a level (low/medium/high) means the model reasons.
	reasoning := p.reasoningEffort != "" && p.reasoningEffort != "none"
	// Reasoning models reject temperature (and top_p); only send it otherwise.
	if !reasoning {
		if temperature, ok := common.AsFloat(options["temperature"]); ok {
			requestBody["temperature"] = temperature
		}
	}
	if p.reasoningEffort != "" {
		requestBody["reasoning"] = map[string]any{"effort": p.reasoningEffort}
	}
	if reasoning {
		// Stateless reasoning preservation: ask for encrypted reasoning so the
		// reasoning item that precedes a function_call can be replayed next turn
		// (store=false → no server-side state to chain via previous_response_id).
		requestBody["include"] = []string{"reasoning.encrypted_content"}
	}
	// Privacy: never persist responses server-side. We rebuild full history each
	// turn, so previous_response_id chaining is intentionally not used.
	requestBody["store"] = false

	// JSON-object output: the Responses API nests this under text.format
	// (vs. chat's top-level response_format). Only emit when capable.
	if wantsJSONObject(options) {
		if p.responseFormatJSON || defaultSupportsResponseFormatJSON(p.protocol) {
			requestBody["text"] = map[string]any{"format": map[string]any{"type": "json_object"}}
		} else {
			logger.DebugCF("openai_responses", "json_object format dropped: protocol not capable",
				map[string]any{"protocol": p.protocol, "model": p.modelLabelOr(model)})
		}
	}

	// Merge extra_body (provider-specific knobs, incl. OpenRouter `provider`/`plugins`),
	// then strip operator drop_params last so they win over everything.
	for k, v := range p.extraBody {
		if _, clash := requestBody[k]; clash {
			logger.WarnCF("openai_responses", "extra_body key collides with managed request field; skipping",
				map[string]any{"model": p.modelLabelOr(model), "key": k})
			continue
		}
		requestBody[k] = v
	}
	for k := range p.dropParams {
		delete(requestBody, k)
	}

	jsonData, err := json.Marshal(requestBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", p.apiBase+"/responses", bytes.NewReader(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if p.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+p.apiKey)
	}

	if p.responseLogFile != "" {
		p.appendLog("request", model, 0, 0, jsonData)
	}

	bytesSent := int64(len(jsonData))
	start := time.Now()
	resp, err := p.httpClient.Do(req)
	if err != nil {
		return errorStatus(model, "error", time.Since(start).Milliseconds(), bytesSent, 0),
			fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	raw, readErr := io.ReadAll(resp.Body)
	durationMs := time.Since(start).Milliseconds()
	if readErr != nil {
		return errorStatus(model, "error", durationMs, bytesSent, int64(len(raw))),
			fmt.Errorf("failed to read response body: %w", readErr)
	}
	bytesReceived := int64(len(raw))
	if p.responseLogFile != "" {
		p.appendLog("response", model, resp.StatusCode, durationMs, raw)
	}

	if resp.StatusCode != http.StatusOK {
		resp.Body = io.NopCloser(bytes.NewReader(raw))
		respErr := common.HandleErrorResponse(resp, p.apiBase)
		return errorStatus(model, "error", durationMs, bytesSent, bytesReceived), respErr
	}

	out, err := parseResponse(raw, model)
	if err != nil {
		return errorStatus(model, "parse_error", durationMs, bytesSent, bytesReceived), err
	}
	if out.Status != nil {
		out.Status.Success = true
		out.Status.DurationMs = durationMs
		out.Status.BytesSent = bytesSent
		out.Status.BytesReceived = bytesReceived
	}
	return out, nil
}

// buildInput converts internal messages into the Responses `input` array and a
// hoisted `instructions` string (system/developer turns). Assistant tool calls
// become function_call items; tool results become function_call_output items.
func buildInput(messages []Message) (instructions string, input []any) {
	var sys strings.Builder
	input = make([]any, 0, len(messages))
	for _, m := range messages {
		switch m.Role {
		case "system", "developer":
			if strings.TrimSpace(m.Content) != "" {
				if sys.Len() > 0 {
					sys.WriteString("\n\n")
				}
				sys.WriteString(m.Content)
			}
		case "tool":
			input = append(input, map[string]any{
				"type":    "function_call_output",
				"call_id": m.ToolCallID,
				"output":  m.Content,
			})
		case "assistant":
			// Replay this turn's reasoning items first (verbatim, incl.
			// encrypted_content): the API requires the reasoning item to precede
			// the function_call it produced. Reasoning models + tools only.
			for _, ri := range m.ResponsesReasoning {
				if len(ri) > 0 {
					input = append(input, json.RawMessage(ri))
				}
			}
			if strings.TrimSpace(m.Content) != "" {
				input = append(input, map[string]any{
					"role":    "assistant",
					"content": m.Content,
				})
			}
			for _, tc := range m.ToolCalls {
				if tc.Function == nil {
					continue
				}
				args := tc.Function.Arguments
				if strings.TrimSpace(args) == "" {
					args = "{}"
				}
				input = append(input, map[string]any{
					"type":      "function_call",
					"call_id":   tc.ID,
					"name":      tc.Function.Name,
					"arguments": args,
				})
			}
		default: // user (and any other) → user input message
			input = append(input, userInputMessage(m))
		}
	}
	return sys.String(), input
}

// userInputMessage maps a user message to a Responses input message, using
// multipart content when media (data: image URLs) is attached.
func userInputMessage(m Message) map[string]any {
	if len(m.Media) == 0 {
		return map[string]any{"role": "user", "content": m.Content}
	}
	parts := make([]map[string]any, 0, 1+len(m.Media))
	if m.Content != "" {
		parts = append(parts, map[string]any{"type": "input_text", "text": m.Content})
	}
	for _, url := range m.Media {
		if strings.HasPrefix(url, "data:image/") {
			parts = append(parts, map[string]any{"type": "input_image", "image_url": url})
		}
	}
	return map[string]any{"role": "user", "content": parts}
}

// responsesTools flattens chat-style tool definitions ({type,function:{...}})
// into the Responses shape ({type:"function", name, description, parameters}).
func responsesTools(tools []ToolDefinition) []map[string]any {
	out := make([]map[string]any, 0, len(tools))
	for _, t := range tools {
		out = append(out, map[string]any{
			"type":        "function",
			"name":        t.Function.Name,
			"description": t.Function.Description,
			"parameters":  t.Function.Parameters,
		})
	}
	return out
}

// responseEnvelope is the subset of the Responses API response we consume.
// Output items are kept raw so reasoning items can be captured verbatim (with
// encrypted_content) for replay; each is then typed via responseOutputItem.
type responseEnvelope struct {
	Output            []json.RawMessage `json:"output"`
	Status            string            `json:"status"`
	IncompleteDetails *struct {
		Reason string `json:"reason"`
	} `json:"incomplete_details"`
	Usage *struct {
		InputTokens  int `json:"input_tokens"`
		OutputTokens int `json:"output_tokens"`
		TotalTokens  int `json:"total_tokens"`
	} `json:"usage"`
}

type responseOutputItem struct {
	Type    string `json:"type"`
	Role    string `json:"role"`
	Content []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	} `json:"content"`
	Summary []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	} `json:"summary"`
	// function_call fields
	CallID    string `json:"call_id"`
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// parseResponse maps a Responses API body into an LLMResponse.
func parseResponse(raw []byte, model string) (*LLMResponse, error) {
	var env responseEnvelope
	if err := json.Unmarshal(raw, &env); err != nil {
		return nil, fmt.Errorf("failed to parse Responses API body: %w", err)
	}

	var content strings.Builder
	var reasoning strings.Builder
	var toolCalls []ToolCall
	var reasoningItems []json.RawMessage
	for _, raw := range env.Output {
		var item responseOutputItem
		if err := json.Unmarshal(raw, &item); err != nil {
			continue
		}
		switch item.Type {
		case "message":
			for _, part := range item.Content {
				if part.Type == "output_text" {
					content.WriteString(part.Text)
				}
			}
		case "function_call":
			args := item.Arguments
			if strings.TrimSpace(args) == "" {
				args = "{}"
			}
			toolCalls = append(toolCalls, ToolCall{
				ID:       item.CallID,
				Type:     "function",
				Function: &FunctionCall{Name: item.Name, Arguments: args},
			})
		case "reasoning":
			for _, s := range item.Summary {
				reasoning.WriteString(s.Text)
			}
			// Capture the whole item verbatim so it can be replayed before its
			// function_call next turn (carries encrypted_content when store=false).
			reasoningItems = append(reasoningItems, append(json.RawMessage(nil), raw...))
		}
	}

	text := content.String()

	finishReason := "stop"
	switch {
	case len(toolCalls) > 0:
		finishReason = "tool_calls"
	case env.Status == "incomplete" && env.IncompleteDetails != nil:
		switch env.IncompleteDetails.Reason {
		case "max_output_tokens":
			finishReason = "length"
		case "content_filter":
			finishReason = "content_filter"
		default:
			finishReason = "incomplete"
		}
	}

	out := &LLMResponse{
		Content:            text,
		ReasoningContent:   reasoning.String(),
		ToolCalls:          toolCalls,
		ResponsesReasoning: reasoningItems,
		FinishReason:       finishReason,
		Normal:             finishReason == "stop" || finishReason == "tool_calls",
		Status: &DispatchStatus{
			Model:      model,
			StopReason: finishReason,
		},
	}
	if env.Usage != nil {
		out.Usage = &UsageInfo{
			PromptTokens:     env.Usage.InputTokens,
			CompletionTokens: env.Usage.OutputTokens,
			TotalTokens:      env.Usage.TotalTokens,
		}
		out.Status.InputTokens = env.Usage.InputTokens
		out.Status.OutputTokens = env.Usage.OutputTokens
	}
	return out, nil
}

func errorStatus(model, stopReason string, durationMs, bytesSent, bytesReceived int64) *LLMResponse {
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

func wantsJSONObject(options map[string]any) bool {
	switch v := options[ResponseFormatJSONObjectOption].(type) {
	case bool:
		return v
	default:
		return false
	}
}

// defaultSupportsResponseFormatJSON mirrors the chat provider: OpenAI-family
// protocols accept JSON-object formatting by default; others must opt in via
// the provider's response_format_json capability flag.
func defaultSupportsResponseFormatJSON(protocol string) bool {
	switch protocol {
	case "openai-chat", "openai-responses", "azure":
		return true
	default:
		return false
	}
}

// appendLog appends one JSONL diagnostic record (request or response).
func (p *Provider) appendLog(dir, model string, status int, durationMs int64, body []byte) {
	rec := map[string]any{
		"ts":     time.Now().Format(time.RFC3339Nano),
		"dir":    dir,
		"model":  model,
		"body":   json.RawMessage(body),
		"status": status,
	}
	if durationMs > 0 {
		rec["duration_ms"] = durationMs
	}
	data, err := json.Marshal(rec)
	if err != nil {
		return
	}
	p.logMu.Lock()
	defer p.logMu.Unlock()
	f, err := os.OpenFile(p.responseLogFile, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o600)
	if err != nil {
		return
	}
	defer f.Close()
	_, _ = f.Write(append(data, '\n'))
}
