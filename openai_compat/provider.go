package openai_compat

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"

	"github.com/PivotLLM/spawnllm/common"
	"github.com/PivotLLM/spawnllm/logger"
	"github.com/PivotLLM/spawnllm/protocoltypes"
)

// warnFn is the logger hook the provider uses for non-fatal warnings (e.g.
// extra_body merge collisions, diagnostic log open/write failures). Tests
// swap this for a capturing function.
var warnFn = logger.WarnCF

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

type Provider struct {
	apiKey              string
	apiBase             string
	maxTokensField      string              // Field name for max tokens (e.g., "max_completion_tokens" for o1/glm models)
	strictCompat        bool                // Strip non-standard fields for strict OpenAI-compatible endpoints
	noParallelToolCalls bool                // Send parallel_tool_calls=false (required by Groq/some Llama providers)
	responseLogFile     string              // Append JSONL request+response records here when non-empty (diagnostic only)
	reasoningEffort     string              // OpenAI/Grok `reasoning_effort` request field; empty omits it
	extraBody           map[string]any      // Free-form passthrough merged into the request body before marshal
	dropParams          map[string]struct{} // Top-level request-body fields to strip just before marshal
	strictAlternation   bool                // Rewrite messages for chat-only strict user/assistant-alternation models
	modelLabel          string              // User-facing model name used in WarnCF for merge collisions; empty falls back to wire model
	protocol            string              // Protocol prefix (e.g. "openai", "xai", "openrouter"); drives capability gating
	responseFormatJSON  bool                // Whether this provider may emit response_format=json_object when callers ask for it
	httpClient          *http.Client

	logMu      sync.Mutex // serialises appends to responseLogFile across goroutines sharing this Provider
	logFile    *os.File   // lazily opened on first use; held for Provider lifetime so the file is opened at most once
	logOpenErr error      // sticky open error; once set, the file is never re-opened
	logErrOnce sync.Once  // emits at most one WRN per Provider lifetime on log open/write failure
}

type Option func(*Provider)

const defaultRequestTimeout = common.DefaultRequestTimeout

func WithMaxTokensField(maxTokensField string) Option {
	return func(p *Provider) {
		p.maxTokensField = maxTokensField
	}
}

func WithRequestTimeout(timeout time.Duration) Option {
	return func(p *Provider) {
		if timeout > 0 {
			p.httpClient.Timeout = timeout
		}
	}
}

func WithStrictCompat(v bool) Option {
	return func(p *Provider) {
		p.strictCompat = v
	}
}

func WithNoParallelToolCalls(v bool) Option {
	return func(p *Provider) {
		p.noParallelToolCalls = v
	}
}

// WithStrictAlternation enables message normalization for chat-only models that
// require strict user/assistant alternation and reject system/tool roles. See
// normalizeAlternation for the exact rewrite.
func WithStrictAlternation(v bool) Option {
	return func(p *Provider) {
		p.strictAlternation = v
	}
}

// WithResponseLogFile enables append-only JSONL capture of outgoing request
// bodies and incoming response bodies to the given path. Empty disables it
// (the default). Diagnostic feature; see (*Provider).writeLogEntry for the
// record format and pairing semantics.
func WithResponseLogFile(path string) Option {
	return func(p *Provider) {
		p.responseLogFile = path
	}
}

// WithReasoningEffort sets the `reasoning_effort` request field that some
// providers (notably xAI Grok and OpenAI o-series) honour. Empty omits the
// field. Validated upstream in pkg/config; the provider trusts what it gets.
func WithReasoningEffort(level string) Option {
	return func(p *Provider) {
		p.reasoningEffort = level
	}
}

// WithExtraBody supplies a free-form map merged into the JSON request body
// just before marshal. Used as the per-model passthrough for provider-specific
// knobs claw does not model natively. Collisions with claw-managed fields are
// rejected at config load; the merge step here is purely defensive.
func WithExtraBody(extra map[string]any) Option {
	return func(p *Provider) {
		p.extraBody = extra
	}
}

// WithDropParams supplies a set of top-level request-body field names to remove
// just before marshal. Use it to suppress a parameter a particular model or
// upstream rejects (e.g. "temperature" on reasoning models that don't advertise
// it). Applied after extra_body, so it always wins. Empty/blank names are
// ignored; an empty list leaves the request untouched.
func WithDropParams(names []string) Option {
	return func(p *Provider) {
		if len(names) == 0 {
			return
		}
		set := make(map[string]struct{}, len(names))
		for _, n := range names {
			if n != "" {
				set[n] = struct{}{}
			}
		}
		if len(set) > 0 {
			p.dropParams = set
		}
	}
}

// WithModelLabel records the user-facing model name (ModelConfig.ModelName)
// so log lines about this provider can identify the offending entry. Optional;
// when unset, logs fall back to the wire-format model identifier.
func WithModelLabel(label string) Option {
	return func(p *Provider) {
		p.modelLabel = label
	}
}

// WithProtocol records the protocol prefix this provider was built for (e.g.
// "openai", "xai", "openrouter"). Used solely for capability gating decisions
// such as whether response_format=json_object may be emitted. Unset providers
// fall through to the conservative default (no response_format emission).
func WithProtocol(protocol string) Option {
	return func(p *Provider) {
		p.protocol = protocol
	}
}

// WithResponseFormatJSONCapable explicitly enables (or disables) emission of
// response_format=json_object when a caller sets the per-call option. This is
// the configuration override path; the built-in per-protocol defaults are
// applied separately and OR'd together at the call site so that both
// well-known capable protocols (openai, xai) and operator-enabled protocols
// (e.g. openrouter via config) work.
func WithResponseFormatJSONCapable(v bool) Option {
	return func(p *Provider) {
		p.responseFormatJSON = v
	}
}

// ResponseLogFile returns the configured per-provider response log file path.
// Exposed primarily for tests that need to assert per-entry openai_compat
// state was correctly attached to this Provider instance.
func (p *Provider) ResponseLogFile() string { return p.responseLogFile }

// ReasoningEffort returns the configured reasoning_effort. See ResponseLogFile
// for the same caveat about intended use.
func (p *Provider) ReasoningEffort() string { return p.reasoningEffort }

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

func NewProviderWithMaxTokensField(apiKey, apiBase, proxy, maxTokensField string) *Provider {
	return NewProvider(apiKey, apiBase, proxy, WithMaxTokensField(maxTokensField))
}

func NewProviderWithMaxTokensFieldAndTimeout(
	apiKey, apiBase, proxy, maxTokensField string,
	requestTimeoutSeconds int,
) *Provider {
	return NewProvider(
		apiKey,
		apiBase,
		proxy,
		WithMaxTokensField(maxTokensField),
		WithRequestTimeout(time.Duration(requestTimeoutSeconds)*time.Second),
	)
}

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

	model = normalizeModel(model, p.apiBase)

	if p.strictAlternation {
		messages = normalizeAlternation(messages)
	}

	requestBody := map[string]any{
		"model":    model,
		"messages": serializeMessages(messages, p.strictCompat),
	}

	if len(tools) > 0 {
		requestBody["tools"] = tools
		requestBody["tool_choice"] = "auto"
		if p.noParallelToolCalls {
			requestBody["parallel_tool_calls"] = false
		}
		// OpenRouter routing hint: require an upstream that supports the
		// `parameters` field on tool definitions, i.e. one that can actually
		// honour function calling. Without this, OpenRouter silently down-routes
		// tools-enabled requests to upstreams that drop the tools and produce
		// text-only replies. This is an OpenRouter-specific extension — strict
		// OpenAI-compatible endpoints (e.g. Google's /v1beta/openai) reject
		// unknown top-level fields with HTTP 400, so only send it to OpenRouter.
		if isOpenRouterBase(p.apiBase) {
			requestBody["provider"] = map[string]any{
				"require_parameters": true,
			}
		}
	}

	if maxTokens, ok := common.AsInt(options["max_tokens"]); ok {
		// Use configured maxTokensField if specified, otherwise fallback to model-based detection
		fieldName := p.maxTokensField
		if fieldName == "" {
			// Fallback: detect from model name for backward compatibility
			lowerModel := strings.ToLower(model)
			if strings.Contains(lowerModel, "glm") || strings.Contains(lowerModel, "o1") ||
				strings.Contains(lowerModel, "gpt-5") {
				fieldName = "max_completion_tokens"
			} else {
				fieldName = "max_tokens"
			}
		}
		requestBody[fieldName] = maxTokens
	}

	if temperature, ok := common.AsFloat(options["temperature"]); ok {
		lowerModel := strings.ToLower(model)
		// Kimi k2 models only support temperature=1.
		if strings.Contains(lowerModel, "kimi") && strings.Contains(lowerModel, "k2") {
			requestBody["temperature"] = 1.0
		} else {
			requestBody["temperature"] = temperature
		}
	}

	// Prompt caching: pass a stable cache key so OpenAI can bucket requests
	// with the same key and reuse prefix KV cache across calls.
	// The key is typically the agent ID — stable per agent, shared across requests.
	// See: https://platform.openai.com/docs/guides/prompt-caching
	// Prompt caching is only supported by OpenAI-native endpoints.
	// Non-OpenAI providers (Mistral, Gemini, DeepSeek, etc.) reject unknown
	// fields with 422 errors, so only include it for OpenAI APIs.
	if cacheKey, ok := options["prompt_cache_key"].(string); ok && cacheKey != "" {
		if supportsPromptCacheKey(p.apiBase) {
			requestBody["prompt_cache_key"] = cacheKey
		}
	}

	if p.reasoningEffort != "" {
		requestBody["reasoning_effort"] = p.reasoningEffort
	}

	// response_format=json_object: caller (compression code) sets this option
	// to force the worker model to emit machine-parseable JSON. Only emitted
	// for protocols known to support it (or explicitly enabled via config).
	// Silently dropped otherwise so providers that 422 on unknown fields
	// (Mistral, Gemini, …) keep working.
	if wantsJSONObject(options) {
		if p.responseFormatJSON || defaultSupportsResponseFormatJSON(p.protocol) {
			requestBody["response_format"] = map[string]any{"type": "json_object"}
		} else {
			logger.DebugCF("openai_compat",
				"response_format=json_object dropped: protocol not capable",
				map[string]any{
					"protocol": p.protocol,
					"model":    p.modelLabelOr(model),
				})
		}
	}

	// Merge extra_body last so any forgotten claw-managed field still wins.
	// The config-load collision guard ensures this loop should be a no-op for
	// reserved keys; the defensive check below logs and skips if one slips
	// through (e.g. a key claw added after the config validator was last
	// updated).
	for k, v := range p.extraBody {
		if _, clash := requestBody[k]; clash {
			warnFn("openai_compat", "extra_body key collides with claw-managed request field; skipping",
				map[string]any{
					"model": p.modelLabelOr(model),
					"key":   k,
				})
			continue
		}
		requestBody[k] = v
	}

	// Strip operator-configured fields last so drop_params wins over every
	// source, including extra_body and claw-managed defaults.
	for k := range p.dropParams {
		delete(requestBody, k)
	}

	jsonData, err := json.Marshal(requestBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", p.apiBase+"/chat/completions", bytes.NewReader(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	if p.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+p.apiKey)
	}

	// Pair the outgoing request body with its eventual response in the
	// diagnostic log via a single corr_id. The UUID is generated only when
	// logging is enabled so the disabled path stays cheap.
	var corrID string
	if p.responseLogFile != "" {
		corrID = uuid.New().String()
		p.appendRequestLog(corrID, model, jsonData)
	}

	bytesSent := int64(len(jsonData))
	start := time.Now()
	resp, err := p.httpClient.Do(req)
	if err != nil {
		return httpErrorStatus(model, "error", time.Since(start).Milliseconds(), bytesSent, 0),
			fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	// When response logging is enabled, slurp the full body up front so we can
	// append it to the diagnostic file before handing it off to the existing
	// error/parse helpers. The body is then replaced with an in-memory reader
	// so downstream code is unchanged.
	if p.responseLogFile != "" {
		raw, readErr := io.ReadAll(resp.Body)
		if readErr != nil {
			return httpErrorStatus(model, "error", time.Since(start).Milliseconds(), bytesSent, int64(len(raw))),
				fmt.Errorf("failed to read response body: %w", readErr)
		}
		p.appendResponseLog(corrID, model, resp.StatusCode, time.Since(start).Milliseconds(), raw)
		resp.Body = io.NopCloser(bytes.NewReader(raw))
	}

	if resp.StatusCode != http.StatusOK {
		respErr := common.HandleErrorResponse(resp, p.apiBase)
		return httpErrorStatus(model, "error", time.Since(start).Milliseconds(), bytesSent, 0), respErr
	}

	out, bytesReceived, err := common.ReadParseAndMeasure(resp, p.apiBase, common.ToolNameSet(tools))
	durationMs := time.Since(start).Milliseconds()
	if err != nil {
		return httpErrorStatus(model, "parse_error", durationMs, bytesSent, bytesReceived), err
	}
	if out.Status != nil {
		out.Status.Success = true
		out.Status.DurationMs = durationMs
		out.Status.BytesSent = bytesSent
		out.Status.BytesReceived = bytesReceived
		if out.Status.Model == "" {
			out.Status.Model = model
		}
	}
	return out, nil
}

// httpErrorStatus builds an LLMResponse whose Status records a failed HTTP dispatch.
func httpErrorStatus(model, stopReason string, durationMs, bytesSent, bytesReceived int64) *LLMResponse {
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

// openaiMessage is the wire-format message for OpenAI-compatible APIs.
// It mirrors protocoltypes.Message but omits SystemParts, which is an
// internal field that would be unknown to third-party endpoints.
type openaiMessage struct {
	Role             string     `json:"role"`
	Content          *string    `json:"content,omitempty"`
	ReasoningContent string     `json:"reasoning_content,omitempty"`
	ToolCalls        []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID       string     `json:"tool_call_id,omitempty"`
}

// msgContent returns the content pointer for an outbound message.
// When content is empty and tool_calls are present, nil is returned so the
// field is omitted entirely. The OpenAI spec allows content to be absent (or
// null) when tool_calls is set, and some strict providers reject "" in that
// position, causing intermittent failures.
func msgContent(content string, toolCalls []ToolCall) *string {
	if content == "" && len(toolCalls) > 0 {
		return nil
	}
	return &content
}

// normalizeAlternation rewrites a message list for chat-only models that
// require strict user/assistant alternation and reject system/tool roles
// (e.g. Gemma on some Bedrock gateways). It:
//   - folds system/developer content into the first user turn,
//   - converts tool-result messages into user turns,
//   - drops tool_calls/tool_call_id/reasoning (the model can't use them),
//   - merges consecutive same-role messages.
//
// It pairs with no_tools: tool calling cannot work once tool_calls are dropped.
func normalizeAlternation(in []Message) []Message {
	var systemPrefix strings.Builder
	rewritten := make([]Message, 0, len(in))
	for _, msg := range in {
		role := msg.Role
		content := msg.Content
		switch role {
		case "system", "developer":
			if strings.TrimSpace(content) != "" {
				if systemPrefix.Len() > 0 {
					systemPrefix.WriteString("\n\n")
				}
				systemPrefix.WriteString(content)
			}
			continue
		case "tool":
			role = "user"
			if strings.TrimSpace(content) == "" {
				content = "(tool result)"
			}
			content = "Tool result:\n" + content
		case "assistant":
			if strings.TrimSpace(content) == "" && len(msg.ToolCalls) > 0 {
				content = "(requested tool calls)"
			}
		}
		rewritten = append(rewritten, Message{
			Role:    role,
			Content: content,
			Media:   msg.Media,
		})
	}

	// Prepend the folded system content to the first user turn (or create one).
	if systemPrefix.Len() > 0 {
		placed := false
		for i := range rewritten {
			if rewritten[i].Role == "user" {
				rewritten[i].Content = systemPrefix.String() + "\n\n" + rewritten[i].Content
				placed = true
				break
			}
		}
		if !placed {
			rewritten = append([]Message{{Role: "user", Content: systemPrefix.String()}}, rewritten...)
		}
	}

	// Merge consecutive same-role messages.
	merged := make([]Message, 0, len(rewritten))
	for _, msg := range rewritten {
		if n := len(merged); n > 0 && merged[n-1].Role == msg.Role {
			last := &merged[n-1]
			if strings.TrimSpace(msg.Content) != "" {
				if strings.TrimSpace(last.Content) != "" {
					last.Content += "\n\n"
				}
				last.Content += msg.Content
			}
			last.Media = append(last.Media, msg.Media...)
			continue
		}
		merged = append(merged, msg)
	}
	return merged
}

// serializeMessages converts internal Message structs to the OpenAI wire format.
//   - Strips SystemParts (unknown to third-party endpoints)
//   - Converts messages with Media to multipart content format (text + image_url parts)
//   - Preserves ToolCallID, ToolCalls, and ReasoningContent for all messages
//   - When strictCompat is true, strips non-standard fields (reasoning_content, extra_content,
//     thought_signature) that some strict OpenAI-compatible providers reject
func serializeMessages(messages []Message, strictCompat bool) []any {
	out := make([]any, 0, len(messages))
	for _, m := range messages {
		toolCalls := m.ToolCalls
		reasoningContent := m.ReasoningContent

		if strictCompat {
			reasoningContent = ""
			if len(toolCalls) > 0 {
				sanitized := make([]ToolCall, len(toolCalls))
				for i, tc := range toolCalls {
					sanitized[i] = tc
					sanitized[i].ExtraContent = nil
					if tc.Function != nil {
						fnCopy := *tc.Function
						fnCopy.ThoughtSignature = ""
						sanitized[i].Function = &fnCopy
					}
				}
				toolCalls = sanitized
			}
		}

		if len(m.Media) == 0 {
			out = append(out, openaiMessage{
				Role:             m.Role,
				Content:          msgContent(m.Content, toolCalls),
				ReasoningContent: reasoningContent,
				ToolCalls:        toolCalls,
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
		if len(toolCalls) > 0 {
			msg["tool_calls"] = toolCalls
		}
		if reasoningContent != "" {
			msg["reasoning_content"] = reasoningContent
		}
		out = append(out, msg)
	}
	return out
}

// isOpenRouterBase reports whether the configured base URL points at OpenRouter.
// OpenRouter-specific request extensions (the `provider` routing hint, model
// prefixes) are gated on this — other OpenAI-compatible endpoints reject unknown
// fields.
func isOpenRouterBase(apiBase string) bool {
	return strings.Contains(strings.ToLower(apiBase), "openrouter.ai")
}

func normalizeModel(model, apiBase string) string {
	before, after, ok := strings.Cut(model, "/")
	if !ok {
		return model
	}

	if isOpenRouterBase(apiBase) {
		return model
	}

	prefix := strings.ToLower(before)
	switch prefix {
	case "litellm", "moonshot", "nvidia", "groq", "ollama", "deepseek", "google",
		"openrouter", "mistral":
		return after
	default:
		return model
	}
}

// logEntry is one JSONL record written to response_log_file. A request entry
// and its paired response entry share the same CorrID; readers can group them
// by that field. Status and DurationMs are populated only on response entries.
type logEntry struct {
	Ts         string          `json:"ts"`
	CorrID     string          `json:"corr_id"`
	Dir        string          `json:"dir"`
	Model      string          `json:"model"`
	Body       json.RawMessage `json:"body"`
	Status     int             `json:"status,omitempty"`
	DurationMs int64           `json:"duration_ms,omitempty"`
}

// appendRequestLog writes the outgoing request body to the diagnostic log
// before the HTTP round-trip starts. The body is parsed to redact a top-level
// "api_key" field (rare, but defensive — claw never sets one, but extra_body
// passthrough could). Request headers (including Authorization) are never
// logged.
func (p *Provider) appendRequestLog(corrID, wireModel string, body []byte) {
	p.writeLogEntry(logEntry{
		Ts:     time.Now().Format(time.RFC3339Nano),
		CorrID: corrID,
		Dir:    "request",
		Model:  p.modelLabelOr(wireModel),
		Body:   safeRawMessage(redactAPIKey(body)),
	})
}

// appendResponseLog writes one JSONL record describing an HTTP response,
// paired with the matching request entry via corrID. The raw body is embedded
// as json.RawMessage when it parses as JSON; otherwise it is quoted as a
// string so the surrounding entry remains valid JSON (e.g. HTML error pages
// from misconfigured proxies).
func (p *Provider) appendResponseLog(corrID, wireModel string, status int, durationMs int64, body []byte) {
	p.writeLogEntry(logEntry{
		Ts:         time.Now().Format(time.RFC3339Nano),
		CorrID:     corrID,
		Dir:        "response",
		Model:      p.modelLabelOr(wireModel),
		Body:       safeRawMessage(body),
		Status:     status,
		DurationMs: durationMs,
	})
}

// writeLogEntry serialises a logEntry to JSONL and appends it to the configured
// response_log_file. The file handle is opened lazily on first use and reused
// for the Provider's lifetime; mode 0640 keeps the file readable by group but
// not world. Failure to open or write must NOT block or fail the actual
// request — at most one WRN is emitted per Provider lifetime via sync.Once.
func (p *Provider) writeLogEntry(e logEntry) {
	if p.responseLogFile == "" {
		return
	}
	line, err := json.Marshal(e)
	if err != nil {
		return
	}
	line = append(line, '\n')

	p.logMu.Lock()
	defer p.logMu.Unlock()

	if p.logFile == nil && p.logOpenErr == nil {
		f, err := os.OpenFile(p.responseLogFile, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o640)
		if err != nil {
			p.logOpenErr = err
			p.logErrOnce.Do(func() {
				warnFn("openai_compat", "response_log_file open failed; further failures suppressed",
					map[string]any{"path": p.responseLogFile, "err": err.Error()})
			})
			return
		}
		p.logFile = f
	}
	if p.logFile == nil {
		return
	}
	if _, err := p.logFile.Write(line); err != nil {
		p.logErrOnce.Do(func() {
			warnFn("openai_compat", "response_log_file write failed; further failures suppressed",
				map[string]any{"path": p.responseLogFile, "err": err.Error()})
		})
	}
}

// redactAPIKey returns body with any top-level "api_key" field replaced by
// "[REDACTED]". Non-JSON-object bodies are returned unchanged.
func redactAPIKey(body []byte) []byte {
	var m map[string]json.RawMessage
	if err := json.Unmarshal(body, &m); err != nil {
		return body
	}
	if _, has := m["api_key"]; !has {
		return body
	}
	m["api_key"] = json.RawMessage(`"[REDACTED]"`)
	redacted, err := json.Marshal(m)
	if err != nil {
		return body
	}
	return redacted
}

// safeRawMessage returns body as json.RawMessage when it parses as JSON;
// otherwise it returns the body encoded as a JSON string so the surrounding
// entry stays valid JSON even when an upstream returns HTML or plain text.
func safeRawMessage(body []byte) json.RawMessage {
	if len(body) > 0 && json.Valid(body) {
		return json.RawMessage(body)
	}
	quoted, err := json.Marshal(string(body))
	if err != nil {
		return json.RawMessage(`""`)
	}
	return json.RawMessage(quoted)
}

// modelLabelOr returns the configured user-facing model label, or fallback
// when the label is empty. Keeps WarnCF messages identifying the offending
// models entry rather than the bare wire identifier.
func (p *Provider) modelLabelOr(fallback string) string {
	if p.modelLabel != "" {
		return p.modelLabel
	}
	return fallback
}

// ResponseFormatJSONObjectOption is the options-map key callers set to
// request response_format={"type":"json_object"} on the outbound request. The
// provider gates emission on protocol capability; the per-call signal is just
// a "yes please" — the provider decides whether to honour it.
const ResponseFormatJSONObjectOption = "response_format_json_object"

// wantsJSONObject reports whether the caller's options map asks for
// response_format=json_object. Accepts bool true or the string "true" so the
// option can be threaded through config-driven layers without re-typing.
func wantsJSONObject(options map[string]any) bool {
	if options == nil {
		return false
	}
	switch v := options[ResponseFormatJSONObjectOption].(type) {
	case bool:
		return v
	case string:
		return v == "true"
	}
	return false
}

// defaultSupportsResponseFormatJSON returns the built-in capability default
// for a given protocol. The defaults track what each protocol is documented
// to accept: OpenAI and xAI honour response_format natively; everything else
// either rejects the field outright or routes downstream in a way that
// silently drops it. Operators can flip a protocol on via configuration
// (Config.OpenAICompatResponseFormat) — that override is applied at the
// factory layer via WithResponseFormatJSONCapable and OR'd with this default.
func defaultSupportsResponseFormatJSON(protocol string) bool {
	switch protocol {
	case "openai-chat", "xai":
		return true
	default:
		return false
	}
}

// supportsPromptCacheKey reports whether the given API base is known to
// support the prompt_cache_key request field. Currently only OpenAI's own
// API and Azure OpenAI support this. All other OpenAI-compatible providers
// (Mistral, Gemini, DeepSeek, Groq, etc.) reject unknown fields with 422 errors.
func supportsPromptCacheKey(apiBase string) bool {
	u, err := url.Parse(apiBase)
	if err != nil {
		return false
	}
	host := u.Hostname()
	return host == "api.openai.com" || strings.HasSuffix(host, ".openai.azure.com")
}
