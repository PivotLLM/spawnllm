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
	"errors"
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
		return nil, errors.New("API base not configured")
	}

	// Deferred tools load natively through client-executed tool search on
	// models that support it (see newToolSearch); otherwise only those a tool
	// result has referenced are sent, as ordinary functions (common.RequestTools),
	// and tool references in results are sent as text (functionCallOutput).
	search := newToolSearch(tools, model)
	if search == nil {
		tools = common.RequestTools(tools, messages)
	}

	instructions, input := buildInput(messages, search)

	requestBody := map[string]any{
		"model": model,
		"input": input,
	}
	if instructions != "" {
		requestBody["instructions"] = instructions
	}

	// Streaming is opt-in: only when the caller supplied a non-nil delta
	// callback. Absent it, the request flag and the response handling below stay
	// on the unchanged single-shot path.
	streamCB, streaming := options[common.TextDeltaOption].(common.TextDeltaFunc)
	streaming = streaming && streamCB != nil
	if streaming {
		requestBody["stream"] = true
	}

	if len(tools) > 0 {
		requestBody["tools"] = responsesTools(tools, search)
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

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, p.apiBase+"/responses", bytes.NewReader(jsonData))
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
	defer func() {
		if closeErr := resp.Body.Close(); closeErr != nil {
			logger.DebugCF("openai_responses", "closing response body", map[string]any{"error": closeErr.Error()})
		}
	}()

	// Streaming reads the body incrementally; the single-shot path below slurps
	// it whole. A non-200 status is handled the same way in both cases (the
	// error body is small and read in full).
	if streaming && resp.StatusCode == http.StatusOK {
		return p.readStream(resp, streamCB, model, search.toolName(), bytesSent, start)
	}

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

	out, err := parseResponse(raw, model, search.toolName())
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

// readStream consumes a Responses API SSE stream, firing cb on each
// response.output_text.delta event, and captures the final response object from
// the response.completed event. That object is fed through the SAME
// parseResponse used by the single-shot path so output/tool/usage mapping is
// shared verbatim. A response.error event is mapped to the HTTP-error status
// shape (partial text may already have been emitted via cb).
func (p *Provider) readStream(
	resp *http.Response,
	cb common.TextDeltaFunc,
	model, searchTool string,
	bytesSent int64,
	start time.Time,
) (*LLMResponse, error) {
	counter := &countingReader{r: resp.Body}
	sr := common.NewSSEReader(counter)

	var completed json.RawMessage
	for {
		payload, ok, err := sr.Next()
		if !ok {
			// io.EOF is the normal end-of-stream signal; any other error is real.
			if err != nil && !errors.Is(err, io.EOF) {
				durationMs := time.Since(start).Milliseconds()
				return errorStatus(model, "error", durationMs, bytesSent, counter.n),
					fmt.Errorf("failed to read stream: %w", err)
			}
			break
		}
		if payload == "[DONE]" {
			break
		}

		var evt streamEvent
		if err := json.Unmarshal([]byte(payload), &evt); err != nil {
			continue // ignore non-JSON framing
		}

		switch evt.Type {
		case "response.output_text.delta":
			if evt.Delta != "" {
				cb(evt.Delta)
			}
		case "response.completed":
			if len(evt.Response) > 0 {
				completed = evt.Response
			}
		case "response.error", "error":
			durationMs := time.Since(start).Milliseconds()
			return errorStatus(model, "error", durationMs, bytesSent, counter.n),
				&common.HTTPStatusError{
					StatusCode:  http.StatusOK,
					APIBase:     p.apiBase,
					BodyPreview: streamEventErrorMessage(evt),
				}
		}
	}

	durationMs := time.Since(start).Milliseconds()
	if p.responseLogFile != "" && len(completed) > 0 {
		p.appendLog("response", model, resp.StatusCode, durationMs, completed)
	}

	if len(completed) == 0 {
		return errorStatus(model, "parse_error", durationMs, bytesSent, counter.n),
			errors.New("stream ended without response.completed event")
	}

	out, err := parseResponse(completed, model, searchTool)
	if err != nil {
		return errorStatus(model, "parse_error", durationMs, bytesSent, counter.n), err
	}
	if out.Status != nil {
		out.Status.Success = true
		out.Status.DurationMs = durationMs
		out.Status.BytesSent = bytesSent
		out.Status.BytesReceived = counter.n
	}
	return out, nil
}

// streamEvent is the subset of a Responses API SSE event we consume. `response`
// carries the full response object on response.completed (and is fed to the
// shared parseResponse). `delta` carries incremental text on
// response.output_text.delta.
type streamEvent struct {
	Type     string          `json:"type"`
	Delta    string          `json:"delta"`
	Response json.RawMessage `json:"response"`
	Error    *struct {
		Message string `json:"message"`
	} `json:"error"`
	Message string `json:"message"`
}

// streamEventErrorMessage extracts a human-readable message from an error event.
func streamEventErrorMessage(evt streamEvent) string {
	if evt.Error != nil && evt.Error.Message != "" {
		return evt.Error.Message
	}
	if evt.Message != "" {
		return evt.Message
	}
	return "responses stream error"
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

// buildInput converts internal messages into the Responses `input` array and a
// hoisted `instructions` string (system/developer turns). Assistant tool calls
// become function_call items; tool results become function_call_output items.
// With search set, calls to the tool-search tool and their results become
// tool_search_call and tool_search_output items instead (the result's text
// following as a developer message), a call to a deferred
// tool carries its namespace, and the result of any other tool that references
// deferred tools is followed by an additional_tools item that loads them.
func buildInput(messages []Message, search *toolSearch) (instructions string, input []any) {
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
			if search.isCall(m.ToolCallID) {
				input = append(input, search.outputItems(m)...)
				continue
			}
			var additional map[string]any
			if search != nil {
				additional = search.additionalToolsItem(&m)
			}
			input = append(input, map[string]any{
				"type":    "function_call_output",
				"call_id": m.ToolCallID,
				"output":  functionCallOutput(m),
			})
			if additional != nil {
				input = append(input, additional)
			}
		case "assistant":
			// Replay this turn's reasoning items first (verbatim, incl.
			// encrypted_content): the API requires the reasoning item to precede
			// the function_call it produced. Reasoning models + tools only.
			for _, ri := range m.ResponsesReasoning {
				if len(ri) > 0 {
					input = append(input, ri)
				}
			}
			if strings.TrimSpace(m.Content) != "" {
				input = append(input, map[string]any{
					"role":    "assistant",
					"content": m.Content,
				})
			}
			for _, tc := range m.ToolCalls {
				if search.matches(tc) {
					input = append(input, search.callItem(tc))
					continue
				}
				item := map[string]any{
					"type":      "function_call",
					"call_id":   tc.ID,
					"name":      common.ToolCallName(tc),
					"arguments": common.ToolCallArgumentsJSON(tc),
				}
				if ns := search.namespace(common.ToolCallName(tc)); ns != "" {
					item["namespace"] = ns
				}
				input = append(input, item)
			}
		default: // user (and any other) → user input message
			input = append(input, userInputMessage(m))
		}
	}
	return sys.String(), input
}

// functionCallOutput builds the "output" of a Responses function_call_output: a
// plain string when the tool returned only text, or an array of content parts
// (input_text + input_image) when it returned images. The Responses API accepts an
// image/file content array as a function output — this is how a tool (e.g. an MCP
// screenshot) hands an image directly to a vision model via its tool result.
func functionCallOutput(m Message) any {
	m.Content = common.ToolResultText(m)
	if len(m.Media) == 0 {
		return m.Content
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
	if len(parts) == 0 {
		return m.Content
	}
	return parts
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
// With search set, the tool-search tool becomes the client tool_search tool and
// deferred functions carry defer_loading.
func responsesTools(tools []ToolDefinition, search *toolSearch) []map[string]any {
	out := make([]map[string]any, 0, len(tools))
	for _, t := range tools {
		if search != nil && t.ToolSearch && t.Function.Name == search.name {
			out = append(out, map[string]any{
				"type":        "tool_search",
				"execution":   "client",
				"description": t.Function.Description,
				"parameters":  t.Function.Parameters,
			})
			continue
		}
		out = append(out, responsesFunction(t, search != nil))
	}
	return out
}

// responsesFunction is one function tool in the Responses shape; withDeferred
// adds defer_loading to a deferred tool.
func responsesFunction(t ToolDefinition, withDeferred bool) map[string]any {
	fn := map[string]any{
		"type":        "function",
		"name":        t.Function.Name,
		"description": t.Function.Description,
		"parameters":  t.Function.Parameters,
	}
	if withDeferred && common.IsDeferred(t) {
		fn["defer_loading"] = true
	}
	return fn
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
		// input_tokens includes the cached and cache-written tokens.
		InputTokens        int `json:"input_tokens"`
		OutputTokens       int `json:"output_tokens"`
		TotalTokens        int `json:"total_tokens"`
		InputTokensDetails *struct {
			CachedTokens     int `json:"cached_tokens"`
			CacheWriteTokens int `json:"cache_write_tokens"`
		} `json:"input_tokens_details"`
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
	// function_call and tool_search_call fields. arguments is a JSON string on
	// a function_call and an object on a tool_search_call.
	CallID    string          `json:"call_id"`
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
	Execution string          `json:"execution"`
}

// argumentsJSON returns an output item's arguments as a JSON text: a JSON
// string's value, or the raw JSON of any other value; "{}" when absent.
func argumentsJSON(raw json.RawMessage) string {
	if len(raw) == 0 || string(raw) == "null" {
		return "{}"
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		if strings.TrimSpace(s) == "" {
			return "{}"
		}
		return s
	}
	return string(raw)
}

// parseResponse maps a Responses API body into an LLMResponse. searchTool is
// the name of the host's tool-search tool when it was sent as the client
// tool_search tool (else ""): a client tool_search_call is reported as an
// ordinary call to it.
func parseResponse(raw []byte, model, searchTool string) (*LLMResponse, error) {
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
			toolCalls = append(toolCalls, ToolCall{
				ID:       item.CallID,
				Type:     "function",
				Function: &FunctionCall{Name: item.Name, Arguments: argumentsJSON(item.Arguments)},
			})
		case "tool_search_call":
			if searchTool == "" || item.Execution != "client" {
				continue
			}
			toolCalls = append(toolCalls, ToolCall{
				ID:       item.CallID,
				Type:     "function",
				Function: &FunctionCall{Name: searchTool, Arguments: argumentsJSON(item.Arguments)},
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
		out.Status.OutputTokens = env.Usage.OutputTokens
		if d := env.Usage.InputTokensDetails; d != nil {
			out.Status.CacheReadTokens = d.CachedTokens
			out.Status.CacheCreationTokens = d.CacheWriteTokens
		}
		out.Status.InputTokens = common.UncachedInputTokens(env.Usage.InputTokens,
			out.Status.CacheReadTokens, out.Status.CacheCreationTokens)
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
	defer func() {
		if closeErr := f.Close(); closeErr != nil {
			logger.DebugCF("openai_responses", "closing response log", map[string]any{"error": closeErr.Error()})
		}
	}()
	if _, err := f.Write(append(data, '\n')); err != nil {
		logger.DebugCF("openai_responses", "writing response log", map[string]any{"error": err.Error()})
	}
}
