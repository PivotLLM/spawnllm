package openai_compat

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/PivotLLM/spawnllm/protocoltypes"
)

func TestProviderChat_UsesMaxCompletionTokensForGLM(t *testing.T) {
	var requestBody map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/chat/completions" {
			http.Error(w, "not found", http.StatusNotFound)
			return
		}
		if err := json.NewDecoder(r.Body).Decode(&requestBody); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		resp := map[string]any{
			"choices": []map[string]any{
				{
					"message":       map[string]any{"content": "ok"},
					"finish_reason": "stop",
				},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	p := NewProvider("key", server.URL, "")
	_, err := p.Chat(
		t.Context(),
		[]Message{{Role: "user", Content: "hi"}},
		nil,
		"glm-4.7",
		map[string]any{"max_tokens": 1234},
	)
	if err != nil {
		t.Fatalf("Chat() error = %v", err)
	}

	if _, ok := requestBody["max_completion_tokens"]; !ok {
		t.Fatalf("expected max_completion_tokens in request body")
	}
	if _, ok := requestBody["max_tokens"]; ok {
		t.Fatalf("did not expect max_tokens key for glm model")
	}
}

func TestProviderChat_ParsesToolCalls(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := map[string]any{
			"choices": []map[string]any{
				{
					"message": map[string]any{
						"content": "",
						"tool_calls": []map[string]any{
							{
								"id":   "call_1",
								"type": "function",
								"function": map[string]any{
									"name":      "get_weather",
									"arguments": "{\"city\":\"SF\"}",
								},
							},
						},
					},
					"finish_reason": "tool_calls",
				},
			},
			"usage": map[string]any{
				"prompt_tokens":     10,
				"completion_tokens": 5,
				"total_tokens":      15,
			},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	p := NewProvider("key", server.URL, "")
	out, err := p.Chat(t.Context(), []Message{{Role: "user", Content: "hi"}}, nil, "gpt-4o", nil)
	if err != nil {
		t.Fatalf("Chat() error = %v", err)
	}
	if len(out.ToolCalls) != 1 {
		t.Fatalf("len(ToolCalls) = %d, want 1", len(out.ToolCalls))
	}
	if out.ToolCalls[0].Name != "get_weather" {
		t.Fatalf("ToolCalls[0].Name = %q, want %q", out.ToolCalls[0].Name, "get_weather")
	}
	if out.ToolCalls[0].Arguments["city"] != "SF" {
		t.Fatalf("ToolCalls[0].Arguments[city] = %v, want SF", out.ToolCalls[0].Arguments["city"])
	}
}

func TestProviderChat_ParsesToolCallsWithObjectArguments(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := map[string]any{
			"choices": []map[string]any{
				{
					"message": map[string]any{
						"content": "",
						"tool_calls": []map[string]any{
							{
								"id":   "call_1",
								"type": "function",
								"function": map[string]any{
									"name": "get_weather",
									"arguments": map[string]any{
										"city":   "SF",
										"metric": true,
									},
								},
							},
						},
					},
					"finish_reason": "tool_calls",
				},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	p := NewProvider("key", server.URL, "")
	out, err := p.Chat(t.Context(), []Message{{Role: "user", Content: "hi"}}, nil, "gpt-4o", nil)
	if err != nil {
		t.Fatalf("Chat() error = %v", err)
	}
	if len(out.ToolCalls) != 1 {
		t.Fatalf("len(ToolCalls) = %d, want 1", len(out.ToolCalls))
	}
	if out.ToolCalls[0].Name != "get_weather" {
		t.Fatalf("ToolCalls[0].Name = %q, want %q", out.ToolCalls[0].Name, "get_weather")
	}
	if out.ToolCalls[0].Arguments["city"] != "SF" {
		t.Fatalf("ToolCalls[0].Arguments[city] = %v, want SF", out.ToolCalls[0].Arguments["city"])
	}
	if out.ToolCalls[0].Arguments["metric"] != true {
		t.Fatalf("ToolCalls[0].Arguments[metric] = %v, want true", out.ToolCalls[0].Arguments["metric"])
	}
}

func TestProviderChat_ParsesReasoningContent(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := map[string]any{
			"choices": []map[string]any{
				{
					"message": map[string]any{
						"content":           "The answer is 2",
						"reasoning_content": "Let me think step by step... 1+1=2",
						"tool_calls": []map[string]any{
							{
								"id":   "call_1",
								"type": "function",
								"function": map[string]any{
									"name":      "calculator",
									"arguments": "{\"expr\":\"1+1\"}",
								},
							},
						},
					},
					"finish_reason": "tool_calls",
				},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	p := NewProvider("key", server.URL, "")
	out, err := p.Chat(t.Context(), []Message{{Role: "user", Content: "1+1=?"}}, nil, "kimi-k2.5", nil)
	if err != nil {
		t.Fatalf("Chat() error = %v", err)
	}
	if out.ReasoningContent != "Let me think step by step... 1+1=2" {
		t.Fatalf("ReasoningContent = %q, want %q", out.ReasoningContent, "Let me think step by step... 1+1=2")
	}
	if out.Content != "The answer is 2" {
		t.Fatalf("Content = %q, want %q", out.Content, "The answer is 2")
	}
	if len(out.ToolCalls) != 1 {
		t.Fatalf("len(ToolCalls) = %d, want 1", len(out.ToolCalls))
	}
}

func TestProviderChat_PreservesReasoningContentInHistory(t *testing.T) {
	var requestBody map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&requestBody); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		resp := map[string]any{
			"choices": []map[string]any{
				{
					"message":       map[string]any{"content": "ok"},
					"finish_reason": "stop",
				},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	p := NewProvider("key", server.URL, "")

	// Simulate a multi-turn conversation where the assistant's previous
	// reply included reasoning_content (e.g. from kimi-k2.5).
	messages := []Message{
		{Role: "user", Content: "What is 1+1?"},
		{Role: "assistant", Content: "2", ReasoningContent: "Let me think... 1+1=2"},
		{Role: "user", Content: "What about 2+2?"},
	}

	_, err := p.Chat(t.Context(), messages, nil, "kimi-k2.5", nil)
	if err != nil {
		t.Fatalf("Chat() error = %v", err)
	}

	// Verify reasoning_content is preserved in the serialized request.
	reqMessages, ok := requestBody["messages"].([]any)
	if !ok {
		t.Fatalf("messages is not []any: %T", requestBody["messages"])
	}
	assistantMsg, ok := reqMessages[1].(map[string]any)
	if !ok {
		t.Fatalf("assistant message is not map[string]any: %T", reqMessages[1])
	}
	if assistantMsg["reasoning_content"] != "Let me think... 1+1=2" {
		t.Errorf("reasoning_content not preserved in request, got %v", assistantMsg["reasoning_content"])
	}
}

func TestProviderChat_HTTPError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "bad request", http.StatusBadRequest)
	}))
	defer server.Close()

	p := NewProvider("key", server.URL, "")
	_, err := p.Chat(t.Context(), []Message{{Role: "user", Content: "hi"}}, nil, "gpt-4o", nil)
	if err == nil {
		t.Fatal("expected error, got nil")
	}
}

func TestProviderChat_JSONHTTPErrorDoesNotReportHTML(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error":"bad request"}`))
	}))
	defer server.Close()

	p := NewProvider("key", server.URL, "")
	_, err := p.Chat(t.Context(), []Message{{Role: "user", Content: "hi"}}, nil, "gpt-4o", nil)
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !strings.Contains(err.Error(), "Status: 400") {
		t.Fatalf("expected status code in error, got %v", err)
	}
	if strings.Contains(err.Error(), "returned HTML instead of JSON") {
		t.Fatalf("expected non-HTML http error, got %v", err)
	}
}

func TestProviderChat_HTMLResponsesReturnHelpfulError(t *testing.T) {
	tests := []struct {
		name        string
		contentType string
		statusCode  int
		body        string
	}{
		{
			name:        "html success response",
			contentType: "text/html; charset=utf-8",
			statusCode:  http.StatusOK,
			body:        "<!DOCTYPE html><html><body>gateway login</body></html>",
		},
		{
			name:        "html error response",
			contentType: "text/html; charset=utf-8",
			statusCode:  http.StatusBadGateway,
			body:        "<!DOCTYPE html><html><body>bad gateway</body></html>",
		},
		{
			name:        "mislabeled html success response",
			contentType: "application/json",
			statusCode:  http.StatusOK,
			body:        "   \r\n\t<!DOCTYPE html><html><body>gateway login</body></html>",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", tt.contentType)
				w.WriteHeader(tt.statusCode)
				_, _ = w.Write([]byte(tt.body))
			}))
			defer server.Close()

			p := NewProvider("key", server.URL, "")
			_, err := p.Chat(t.Context(), []Message{{Role: "user", Content: "hi"}}, nil, "gpt-4o", nil)
			if err == nil {
				t.Fatal("expected error, got nil")
			}
			if !strings.Contains(err.Error(), fmt.Sprintf("Status: %d", tt.statusCode)) {
				t.Fatalf("expected status code in error, got %v", err)
			}
			if !strings.Contains(err.Error(), "returned HTML instead of JSON") {
				t.Fatalf("expected helpful HTML error, got %v", err)
			}
			if !strings.Contains(err.Error(), "check api_base or proxy configuration") {
				t.Fatalf("expected configuration hint, got %v", err)
			}
		})
	}
}

func TestProviderChat_SuccessResponseUsesStreamingDecoder(t *testing.T) {
	content := strings.Repeat("a", 1024)
	body := `{"choices":[{"message":{"content":"` + content + `"},"finish_reason":"stop"}]}`

	p := NewProvider("key", "https://example.com/v1", "")
	p.httpClient = &http.Client{
		Transport: roundTripperFunc(func(r *http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusOK,
				Header:     http.Header{"Content-Type": []string{"application/json"}},
				Body: &errAfterDataReadCloser{
					data:      []byte(body),
					chunkSize: 64,
				},
			}, nil
		}),
	}

	out, err := p.Chat(t.Context(), []Message{{Role: "user", Content: "hi"}}, nil, "gpt-4o", nil)
	if err != nil {
		t.Fatalf("Chat() error = %v", err)
	}
	if out.Content != content {
		t.Fatalf("Content = %q, want %q", out.Content, content)
	}
}

func TestProviderChat_LargeHTMLResponsePreviewIsTruncated(t *testing.T) {
	body := append([]byte("<!DOCTYPE html><html><body>"), bytes.Repeat([]byte("A"), 2048)...)
	body = append(body, []byte("</body></html>")...)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.WriteHeader(http.StatusBadGateway)
		_, _ = w.Write(body)
	}))
	defer server.Close()

	p := NewProvider("key", server.URL, "")
	_, err := p.Chat(t.Context(), []Message{{Role: "user", Content: "hi"}}, nil, "gpt-4o", nil)
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !strings.Contains(err.Error(), "Body:   <!DOCTYPE html><html><body>") {
		t.Fatalf("expected html preview in error, got %v", err)
	}
	if !strings.Contains(err.Error(), "...") {
		t.Fatalf("expected truncated preview, got %v", err)
	}
}

func TestProviderChat_StripsMoonshotPrefixAndNormalizesKimiTemperature(t *testing.T) {
	var requestBody map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&requestBody); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		resp := map[string]any{
			"choices": []map[string]any{
				{
					"message":       map[string]any{"content": "ok"},
					"finish_reason": "stop",
				},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	p := NewProvider("key", server.URL, "")
	_, err := p.Chat(
		t.Context(),
		[]Message{{Role: "user", Content: "hi"}},
		nil,
		"moonshot/kimi-k2.5",
		map[string]any{"temperature": 0.3},
	)
	if err != nil {
		t.Fatalf("Chat() error = %v", err)
	}

	if requestBody["model"] != "kimi-k2.5" {
		t.Fatalf("model = %v, want kimi-k2.5", requestBody["model"])
	}
	if requestBody["temperature"] != 1.0 {
		t.Fatalf("temperature = %v, want 1.0", requestBody["temperature"])
	}
}

// TestProviderChat_DropParamsStripsFields verifies that WithDropParams removes
// the named top-level fields from the outgoing request body, even when claw
// would normally set them (temperature here), while leaving other fields intact.
// This is the generic, model-agnostic escape hatch for upstreams that reject a
// parameter (e.g. OpenRouter reasoning models that 404 under
// provider.require_parameters when temperature is present).
func TestProviderChat_DropParamsStripsFields(t *testing.T) {
	var requestBody map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&requestBody); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		resp := map[string]any{
			"choices": []map[string]any{
				{"message": map[string]any{"content": "ok"}, "finish_reason": "stop"},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	p := NewProvider("key", server.URL, "", WithDropParams([]string{"temperature"}))
	_, err := p.Chat(
		t.Context(),
		[]Message{{Role: "user", Content: "hi"}},
		nil,
		"openai/gpt-5.4",
		map[string]any{"temperature": 0.2, "max_tokens": 100},
	)
	if err != nil {
		t.Fatalf("Chat() error = %v", err)
	}

	if _, present := requestBody["temperature"]; present {
		t.Fatalf("temperature should be stripped by drop_params, got %v", requestBody["temperature"])
	}
	// Other fields must survive — drop_params is a targeted filter, not a wipe.
	if requestBody["model"] != "openai/gpt-5.4" {
		t.Fatalf("model = %v, want openai/gpt-5.4", requestBody["model"])
	}
	if _, ok := requestBody["messages"]; !ok {
		t.Fatalf("messages should be present, body = %v", requestBody)
	}
}

// TestProviderChat_NoDropParamsKeepsTemperature is the off-path: without
// WithDropParams the temperature claw sets is sent as usual.
func TestProviderChat_NoDropParamsKeepsTemperature(t *testing.T) {
	var requestBody map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&requestBody); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		resp := map[string]any{
			"choices": []map[string]any{
				{"message": map[string]any{"content": "ok"}, "finish_reason": "stop"},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	p := NewProvider("key", server.URL, "")
	_, err := p.Chat(
		t.Context(),
		[]Message{{Role: "user", Content: "hi"}},
		nil,
		"openai/gpt-4o",
		map[string]any{"temperature": 0.2},
	)
	if err != nil {
		t.Fatalf("Chat() error = %v", err)
	}
	if requestBody["temperature"] != 0.2 {
		t.Fatalf("temperature = %v, want 0.2", requestBody["temperature"])
	}
}

func TestProviderChat_StripsProtocolPrefixes(t *testing.T) {
	tests := []struct {
		name      string
		input     string
		wantModel string
	}{
		{
			name:      "strips litellm prefix and preserves proxy model name",
			input:     "litellm/my-proxy-alias",
			wantModel: "my-proxy-alias",
		},
		{
			name:      "strips groq prefix and keeps nested model",
			input:     "groq/openai/gpt-oss-120b",
			wantModel: "openai/gpt-oss-120b",
		},
		{
			name:      "strips ollama prefix",
			input:     "ollama/qwen2.5:14b",
			wantModel: "qwen2.5:14b",
		},
		{
			name:      "strips deepseek prefix",
			input:     "deepseek/deepseek-chat",
			wantModel: "deepseek-chat",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var requestBody map[string]any

			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if err := json.NewDecoder(r.Body).Decode(&requestBody); err != nil {
					http.Error(w, err.Error(), http.StatusBadRequest)
					return
				}
				resp := map[string]any{
					"choices": []map[string]any{
						{
							"message":       map[string]any{"content": "ok"},
							"finish_reason": "stop",
						},
					},
				}
				w.Header().Set("Content-Type", "application/json")
				json.NewEncoder(w).Encode(resp)
			}))
			defer server.Close()

			p := NewProvider("key", server.URL, "")
			_, err := p.Chat(t.Context(), []Message{{Role: "user", Content: "hi"}}, nil, tt.input, nil)
			if err != nil {
				t.Fatalf("Chat() error = %v", err)
			}

			if requestBody["model"] != tt.wantModel {
				t.Fatalf("model = %v, want %s", requestBody["model"], tt.wantModel)
			}
		})
	}
}

func TestProvider_ProxyConfigured(t *testing.T) {
	proxyURL := "http://127.0.0.1:8080"
	p := NewProvider("key", "https://example.com", proxyURL)

	transport, ok := p.httpClient.Transport.(*http.Transport)
	if !ok || transport == nil {
		t.Fatalf("expected http transport with proxy, got %T", p.httpClient.Transport)
	}

	req := &http.Request{URL: &url.URL{Scheme: "https", Host: "api.example.com"}}
	gotProxy, err := transport.Proxy(req)
	if err != nil {
		t.Fatalf("proxy function returned error: %v", err)
	}
	if gotProxy == nil || gotProxy.String() != proxyURL {
		t.Fatalf("proxy = %v, want %s", gotProxy, proxyURL)
	}
}

func TestProviderChat_AcceptsNumericOptionTypes(t *testing.T) {
	var requestBody map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&requestBody); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		resp := map[string]any{
			"choices": []map[string]any{
				{
					"message":       map[string]any{"content": "ok"},
					"finish_reason": "stop",
				},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	p := NewProvider("key", server.URL, "")
	_, err := p.Chat(
		t.Context(),
		[]Message{{Role: "user", Content: "hi"}},
		nil,
		"gpt-4o",
		map[string]any{"max_tokens": float64(512), "temperature": 1},
	)
	if err != nil {
		t.Fatalf("Chat() error = %v", err)
	}

	if requestBody["max_tokens"] != float64(512) {
		t.Fatalf("max_tokens = %v, want 512", requestBody["max_tokens"])
	}
	if requestBody["temperature"] != float64(1) {
		t.Fatalf("temperature = %v, want 1", requestBody["temperature"])
	}
}

func TestNormalizeModel_UsesAPIBase(t *testing.T) {
	if got := normalizeModel("deepseek/deepseek-chat", "https://api.deepseek.com/v1"); got != "deepseek-chat" {
		t.Fatalf("normalizeModel(deepseek) = %q, want %q", got, "deepseek-chat")
	}
	if got := normalizeModel("openrouter/auto", "https://openrouter.ai/api/v1"); got != "openrouter/auto" {
		t.Fatalf("normalizeModel(openrouter) = %q, want %q", got, "openrouter/auto")
	}
}

func TestProvider_RequestTimeoutDefault(t *testing.T) {
	p := NewProviderWithMaxTokensFieldAndTimeout("key", "https://example.com/v1", "", "", 0)
	if p.httpClient.Timeout != defaultRequestTimeout {
		t.Fatalf("http timeout = %v, want %v", p.httpClient.Timeout, defaultRequestTimeout)
	}
}

func TestProvider_RequestTimeoutOverride(t *testing.T) {
	p := NewProviderWithMaxTokensFieldAndTimeout("key", "https://example.com/v1", "", "", 300)
	if p.httpClient.Timeout != 300*time.Second {
		t.Fatalf("http timeout = %v, want %v", p.httpClient.Timeout, 300*time.Second)
	}
}

type roundTripperFunc func(*http.Request) (*http.Response, error)

func (f roundTripperFunc) RoundTrip(r *http.Request) (*http.Response, error) {
	return f(r)
}

type errAfterDataReadCloser struct {
	data      []byte
	chunkSize int
	offset    int
}

func (r *errAfterDataReadCloser) Read(p []byte) (int, error) {
	if r.offset >= len(r.data) {
		return 0, io.ErrUnexpectedEOF
	}

	n := r.chunkSize
	if n <= 0 || n > len(p) {
		n = len(p)
	}
	remaining := len(r.data) - r.offset
	if n > remaining {
		n = remaining
	}
	copy(p, r.data[r.offset:r.offset+n])
	r.offset += n
	return n, nil
}

func (r *errAfterDataReadCloser) Close() error {
	return nil
}

func TestProvider_FunctionalOptionMaxTokensField(t *testing.T) {
	p := NewProvider("key", "https://example.com/v1", "", WithMaxTokensField("max_completion_tokens"))
	if p.maxTokensField != "max_completion_tokens" {
		t.Fatalf("maxTokensField = %q, want %q", p.maxTokensField, "max_completion_tokens")
	}
}

func TestProvider_FunctionalOptionRequestTimeout(t *testing.T) {
	p := NewProvider("key", "https://example.com/v1", "", WithRequestTimeout(45*time.Second))
	if p.httpClient.Timeout != 45*time.Second {
		t.Fatalf("http timeout = %v, want %v", p.httpClient.Timeout, 45*time.Second)
	}
}

func TestProvider_FunctionalOptionRequestTimeoutNonPositive(t *testing.T) {
	p := NewProvider("key", "https://example.com/v1", "", WithRequestTimeout(-1*time.Second))
	if p.httpClient.Timeout != defaultRequestTimeout {
		t.Fatalf("http timeout = %v, want %v", p.httpClient.Timeout, defaultRequestTimeout)
	}
}

func TestSerializeMessages_PlainText(t *testing.T) {
	messages := []protocoltypes.Message{
		{Role: "user", Content: "hello"},
		{Role: "assistant", Content: "hi", ReasoningContent: "thinking..."},
	}
	result := serializeMessages(messages, false)

	data, err := json.Marshal(result)
	if err != nil {
		t.Fatal(err)
	}

	var msgs []map[string]any
	json.Unmarshal(data, &msgs)

	if msgs[0]["content"] != "hello" {
		t.Fatalf("expected plain string content, got %v", msgs[0]["content"])
	}
	if msgs[1]["reasoning_content"] != "thinking..." {
		t.Fatalf("reasoning_content not preserved, got %v", msgs[1]["reasoning_content"])
	}
}

func TestSerializeMessages_WithMedia(t *testing.T) {
	messages := []protocoltypes.Message{
		{Role: "user", Content: "describe this", Media: []string{"data:image/png;base64,abc123"}},
	}
	result := serializeMessages(messages, false)

	data, _ := json.Marshal(result)
	var msgs []map[string]any
	json.Unmarshal(data, &msgs)

	content, ok := msgs[0]["content"].([]any)
	if !ok {
		t.Fatalf("expected array content for media message, got %T", msgs[0]["content"])
	}
	if len(content) != 2 {
		t.Fatalf("expected 2 content parts, got %d", len(content))
	}

	textPart := content[0].(map[string]any)
	if textPart["type"] != "text" || textPart["text"] != "describe this" {
		t.Fatalf("text part mismatch: %v", textPart)
	}

	imgPart := content[1].(map[string]any)
	if imgPart["type"] != "image_url" {
		t.Fatalf("expected image_url type, got %v", imgPart["type"])
	}
	imgURL := imgPart["image_url"].(map[string]any)
	if imgURL["url"] != "data:image/png;base64,abc123" {
		t.Fatalf("image url mismatch: %v", imgURL["url"])
	}
}

func TestSerializeMessages_MediaWithToolCallID(t *testing.T) {
	messages := []protocoltypes.Message{
		{Role: "tool", Content: "image result", Media: []string{"data:image/png;base64,xyz"}, ToolCallID: "call_1"},
	}
	result := serializeMessages(messages, false)

	data, _ := json.Marshal(result)
	var msgs []map[string]any
	json.Unmarshal(data, &msgs)

	if msgs[0]["tool_call_id"] != "call_1" {
		t.Fatalf("tool_call_id not preserved with media, got %v", msgs[0]["tool_call_id"])
	}
	// Content should be multipart array
	if _, ok := msgs[0]["content"].([]any); !ok {
		t.Fatalf("expected array content, got %T", msgs[0]["content"])
	}
}

// chatWithCacheKey sets up a test server, sends a Chat request with prompt_cache_key,
// and returns the decoded request body for assertion.
func chatWithCacheKey(t *testing.T, apiBase string) map[string]any {
	t.Helper()
	var requestBody map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&requestBody); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		resp := map[string]any{
			"choices": []map[string]any{
				{
					"message":       map[string]any{"content": "ok"},
					"finish_reason": "stop",
				},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	p := NewProvider("key", server.URL, "")
	p.apiBase = apiBase
	p.httpClient = &http.Client{
		Transport: roundTripperFunc(func(r *http.Request) (*http.Response, error) {
			r.URL, _ = url.Parse(server.URL + r.URL.Path)
			return http.DefaultTransport.RoundTrip(r)
		}),
	}

	_, err := p.Chat(
		t.Context(),
		[]Message{{Role: "user", Content: "hi"}},
		nil,
		"test-model",
		map[string]any{"prompt_cache_key": "agent-main"},
	)
	if err != nil {
		t.Fatalf("Chat() error = %v", err)
	}
	return requestBody
}

func TestProviderChat_PromptCacheKeySentToOpenAI(t *testing.T) {
	body := chatWithCacheKey(t, "https://api.openai.com/v1")
	if body["prompt_cache_key"] != "agent-main" {
		t.Fatalf("prompt_cache_key = %v, want %q", body["prompt_cache_key"], "agent-main")
	}
}

func TestProviderChat_PromptCacheKeyOmittedForNonOpenAI(t *testing.T) {
	tests := []struct {
		name    string
		apiBase string
	}{
		{"mistral", "https://api.mistral.ai/v1"},
		{"gemini", "https://generativelanguage.googleapis.com/v1beta"},
		{"deepseek", "https://api.deepseek.com/v1"},
		{"groq", "https://api.groq.com/openai/v1"},
		{"ollama_local", "http://localhost:11434/v1"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			body := chatWithCacheKey(t, tt.apiBase)
			if _, exists := body["prompt_cache_key"]; exists {
				t.Fatalf("prompt_cache_key should NOT be sent to %s, but was included in request", tt.name)
			}
		})
	}
}

func TestSupportsPromptCacheKey(t *testing.T) {
	tests := []struct {
		apiBase string
		want    bool
	}{
		{"https://api.openai.com/v1", true},
		{"https://api.openai.com/v1/", true},
		{"https://myresource.openai.azure.com/openai/deployments/gpt-4", true},
		{"https://eastus.openai.azure.com/v1", true},
		{"https://api.mistral.ai/v1", false},
		{"https://generativelanguage.googleapis.com/v1beta", false},
		{"https://api.deepseek.com/v1", false},
		{"https://api.groq.com/openai/v1", false},
		{"http://localhost:11434/v1", false},
		{"https://openrouter.ai/api/v1", false},
		// Edge cases: proxy URLs with openai.com in path should NOT match
		{"https://my-proxy.com/api.openai.com/v1", false},
		{"https://proxy.example.com/openai.azure.com/v1", false},
		// Malformed or empty
		{"", false},
		{"not-a-url", false},
	}
	for _, tt := range tests {
		if got := supportsPromptCacheKey(tt.apiBase); got != tt.want {
			t.Errorf("supportsPromptCacheKey(%q) = %v, want %v", tt.apiBase, got, tt.want)
		}
	}
}

func TestSerializeMessages_OmitsContentWhenEmptyAndToolCallsPresent(t *testing.T) {
	messages := []protocoltypes.Message{
		{
			Role:    "assistant",
			Content: "",
			ToolCalls: []protocoltypes.ToolCall{
				{ID: "call_1", Type: "function", Function: &protocoltypes.FunctionCall{Name: "fn", Arguments: "{}"}},
			},
		},
	}
	result := serializeMessages(messages, false)

	data, _ := json.Marshal(result)
	var msgs []map[string]any
	json.Unmarshal(data, &msgs)

	if _, ok := msgs[0]["content"]; ok {
		t.Fatalf("content should be omitted when empty and tool_calls present, got %v", msgs[0]["content"])
	}
	if msgs[0]["tool_calls"] == nil {
		t.Fatal("tool_calls should be present")
	}
}

func TestSerializeMessages_IncludesContentWhenNonEmptyWithToolCalls(t *testing.T) {
	messages := []protocoltypes.Message{
		{
			Role:    "assistant",
			Content: "thinking...",
			ToolCalls: []protocoltypes.ToolCall{
				{ID: "call_1", Type: "function", Function: &protocoltypes.FunctionCall{Name: "fn", Arguments: "{}"}},
			},
		},
	}
	result := serializeMessages(messages, false)

	data, _ := json.Marshal(result)
	var msgs []map[string]any
	json.Unmarshal(data, &msgs)

	if msgs[0]["content"] != "thinking..." {
		t.Fatalf("content should be preserved when non-empty, got %v", msgs[0]["content"])
	}
}

func TestSerializeMessages_StripsSystemParts(t *testing.T) {
	messages := []protocoltypes.Message{
		{
			Role:    "system",
			Content: "you are helpful",
			SystemParts: []protocoltypes.ContentBlock{
				{Type: "text", Text: "you are helpful"},
			},
		},
	}
	result := serializeMessages(messages, false)

	data, _ := json.Marshal(result)
	raw := string(data)
	if strings.Contains(raw, "system_parts") {
		t.Fatal("system_parts should not appear in serialized output")
	}
}

func TestSerializeMessages_StrictCompat_StripsReasoningContent(t *testing.T) {
	messages := []protocoltypes.Message{
		{Role: "user", Content: "What is 1+1?"},
		{Role: "assistant", Content: "2", ReasoningContent: "Let me think... 1+1=2"},
	}
	result := serializeMessages(messages, true)

	data, _ := json.Marshal(result)
	var msgs []map[string]any
	json.Unmarshal(data, &msgs)

	if _, ok := msgs[1]["reasoning_content"]; ok {
		t.Fatalf("reasoning_content should be stripped when strictCompat=true, got %v", msgs[1]["reasoning_content"])
	}
	if msgs[1]["content"] != "2" {
		t.Fatalf("content should be preserved, got %v", msgs[1]["content"])
	}
}

func TestSerializeMessages_StrictCompat_StripsExtraContent(t *testing.T) {
	messages := []protocoltypes.Message{
		{
			Role:    "assistant",
			Content: "",
			ToolCalls: []protocoltypes.ToolCall{
				{
					ID:   "call_1",
					Type: "function",
					Function: &protocoltypes.FunctionCall{
						Name:      "get_weather",
						Arguments: `{"city":"SF"}`,
					},
					ExtraContent: &protocoltypes.ExtraContent{
						Google: &protocoltypes.GoogleExtra{
							ThoughtSignature: "sig123",
						},
					},
				},
			},
		},
	}
	result := serializeMessages(messages, true)

	data, _ := json.Marshal(result)
	raw := string(data)
	if strings.Contains(raw, "extra_content") {
		t.Fatalf("extra_content should be stripped when strictCompat=true, got: %s", raw)
	}
	if strings.Contains(raw, "sig123") {
		t.Fatalf("thought_signature value should be stripped when strictCompat=true, got: %s", raw)
	}
}

func TestSerializeMessages_StrictCompat_StripsThoughtSignature(t *testing.T) {
	messages := []protocoltypes.Message{
		{
			Role:    "assistant",
			Content: "",
			ToolCalls: []protocoltypes.ToolCall{
				{
					ID:   "call_1",
					Type: "function",
					Function: &protocoltypes.FunctionCall{
						Name:             "search",
						Arguments:        `{"query":"test"}`,
						ThoughtSignature: "thought-sig-abc",
					},
				},
			},
		},
	}
	result := serializeMessages(messages, true)

	data, _ := json.Marshal(result)
	raw := string(data)
	if strings.Contains(raw, "thought_signature") {
		t.Fatalf("thought_signature should be stripped when strictCompat=true, got: %s", raw)
	}
	if strings.Contains(raw, "thought-sig-abc") {
		t.Fatalf("thought_signature value should be stripped when strictCompat=true, got: %s", raw)
	}
}

func TestSerializeMessages_NoStrictCompat_PreservesFields(t *testing.T) {
	messages := []protocoltypes.Message{
		{
			Role:             "assistant",
			Content:          "result",
			ReasoningContent: "my reasoning",
			ToolCalls: []protocoltypes.ToolCall{
				{
					ID:   "call_1",
					Type: "function",
					Function: &protocoltypes.FunctionCall{
						Name:             "get_weather",
						Arguments:        `{"city":"SF"}`,
						ThoughtSignature: "thought-sig-xyz",
					},
					ExtraContent: &protocoltypes.ExtraContent{
						Google: &protocoltypes.GoogleExtra{
							ThoughtSignature: "sig456",
						},
					},
				},
			},
		},
	}
	result := serializeMessages(messages, false)

	data, _ := json.Marshal(result)
	raw := string(data)
	if !strings.Contains(raw, "my reasoning") {
		t.Fatalf("reasoning_content should be preserved when strictCompat=false, got: %s", raw)
	}
	if !strings.Contains(raw, "extra_content") {
		t.Fatalf("extra_content should be preserved when strictCompat=false, got: %s", raw)
	}
	if !strings.Contains(raw, "sig456") {
		t.Fatalf("thought_signature value in extra_content should be preserved when strictCompat=false, got: %s", raw)
	}
}

func TestProviderChat_PopulatesDispatchStatus(t *testing.T) {
	respBody := `{
		"model": "gpt-4o-2024-11-20",
		"choices": [
			{
				"message": {"content": "hi"},
				"finish_reason": "stop"
			}
		],
		"usage": {
			"prompt_tokens": 12,
			"completion_tokens": 4,
			"total_tokens": 16,
			"prompt_tokens_details": {"cached_tokens": 8}
		}
	}`

	var receivedRequestBody []byte
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedRequestBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(respBody))
	}))
	defer server.Close()

	p := NewProvider("key", server.URL, "")
	out, err := p.Chat(
		t.Context(),
		[]Message{{Role: "user", Content: "hi from Alice"}},
		nil,
		"gpt-4o",
		map[string]any{"max_tokens": 64},
	)
	if err != nil {
		t.Fatalf("Chat() error = %v", err)
	}
	if out.Status == nil {
		t.Fatal("expected non-nil Status")
	}
	if !out.Status.Success {
		t.Fatalf("expected Success=true, got %+v", out.Status)
	}
	if out.Status.Model != "gpt-4o-2024-11-20" {
		t.Fatalf("Model = %q, want %q", out.Status.Model, "gpt-4o-2024-11-20")
	}
	if out.Status.InputTokens != 12 || out.Status.OutputTokens != 4 {
		t.Fatalf("tokens = (%d,%d), want (12,4)", out.Status.InputTokens, out.Status.OutputTokens)
	}
	if out.Status.CacheReadTokens != 8 {
		t.Fatalf("CacheReadTokens = %d, want 8", out.Status.CacheReadTokens)
	}
	if out.Status.StopReason != "stop" {
		t.Fatalf("StopReason = %q, want stop", out.Status.StopReason)
	}
	if out.Status.BytesSent != int64(len(receivedRequestBody)) {
		t.Fatalf("BytesSent = %d, want %d", out.Status.BytesSent, len(receivedRequestBody))
	}
	if out.Status.BytesReceived != int64(len(respBody)) {
		t.Fatalf("BytesReceived = %d, want %d", out.Status.BytesReceived, len(respBody))
	}
	if out.Status.DurationMs < 0 {
		t.Fatalf("DurationMs must be non-negative, got %d", out.Status.DurationMs)
	}
}

func TestProviderChat_HTTPErrorPopulatesStatus(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "bad request", http.StatusBadRequest)
	}))
	defer server.Close()

	p := NewProvider("key", server.URL, "")
	out, err := p.Chat(
		t.Context(),
		[]Message{{Role: "user", Content: "hi from Bob"}},
		nil,
		"gpt-4o",
		map[string]any{"max_tokens": 64},
	)
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if out == nil || out.Status == nil {
		t.Fatalf("expected non-nil response+status on HTTP error, got out=%v", out)
	}
	if out.Status.Success {
		t.Fatalf("expected Success=false")
	}
	if out.Status.BytesSent == 0 {
		t.Fatal("BytesSent should be non-zero even on HTTP error")
	}
}

// TestProviderChat_OmitsRequireParametersForNonOpenRouter verifies the
// OpenRouter-only `provider.require_parameters` hint is NOT sent to other
// OpenAI-compatible endpoints (which reject unknown fields, e.g. Google → 400),
// even when tools are advertised. The httptest server is a non-OpenRouter base.
func TestProviderChat_OmitsRequireParametersForNonOpenRouter(t *testing.T) {
	var requestBody map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&requestBody); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		resp := map[string]any{
			"choices": []map[string]any{
				{
					"message":       map[string]any{"content": "ok"},
					"finish_reason": "stop",
				},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	tools := []protocoltypes.ToolDefinition{
		{
			Type: "function",
			Function: protocoltypes.ToolFunctionDefinition{
				Name:        "get_weather",
				Description: "Get weather",
				Parameters:  map[string]any{"type": "object"},
			},
		},
	}

	p := NewProvider("key", server.URL, "")
	_, err := p.Chat(t.Context(), []Message{{Role: "user", Content: "hi"}}, tools, "gpt-4o", nil)
	if err != nil {
		t.Fatalf("Chat() error = %v", err)
	}

	if _, ok := requestBody["provider"]; ok {
		t.Fatalf("provider key must be omitted for non-OpenRouter endpoints, got %v", requestBody["provider"])
	}
}

func TestIsOpenRouterBase(t *testing.T) {
	cases := []struct {
		base string
		want bool
	}{
		{"https://openrouter.ai/api/v1", true},
		{"https://OpenRouter.ai/api/v1", true},
		{"https://api.openai.com/v1", false},
		{"https://generativelanguage.googleapis.com/v1beta/openai", false},
		{"https://api.groq.com/openai/v1", false},
		{"", false},
	}
	for _, tc := range cases {
		if got := isOpenRouterBase(tc.base); got != tc.want {
			t.Errorf("isOpenRouterBase(%q) = %v, want %v", tc.base, got, tc.want)
		}
	}
}

func TestProviderChat_OmitsProviderObjectWhenNoTools(t *testing.T) {
	var requestBody map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&requestBody); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		resp := map[string]any{
			"choices": []map[string]any{
				{
					"message":       map[string]any{"content": "ok"},
					"finish_reason": "stop",
				},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	p := NewProvider("key", server.URL, "")
	_, err := p.Chat(t.Context(), []Message{{Role: "user", Content: "hi"}}, nil, "gpt-4o", nil)
	if err != nil {
		t.Fatalf("Chat() error = %v", err)
	}

	if _, ok := requestBody["provider"]; ok {
		t.Fatalf("provider key should be omitted when no tools advertised")
	}
}

// ensure protocoltypes import is used in this file
var _ = protocoltypes.DispatchStatus{}

// logRecord mirrors the JSONL record written by (*Provider).writeLogEntry.
// Body stays as json.RawMessage so tests can decode it either as a JSON
// object (claw's request body) or as a JSON string (non-JSON upstream body).
type logRecord struct {
	Ts         string          `json:"ts"`
	CorrID     string          `json:"corr_id"`
	Dir        string          `json:"dir"`
	Model      string          `json:"model"`
	Body       json.RawMessage `json:"body"`
	Status     int             `json:"status,omitempty"`
	DurationMs int64           `json:"duration_ms,omitempty"`
}

func parseLogFile(t *testing.T, path string) []logRecord {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read log file: %v", err)
	}
	var out []logRecord
	for i, line := range bytes.Split(bytes.TrimRight(data, "\n"), []byte("\n")) {
		if len(line) == 0 {
			continue
		}
		var rec logRecord
		if err := json.Unmarshal(line, &rec); err != nil {
			t.Fatalf("line %d not valid JSON: %v; line=%s", i, err, line)
		}
		out = append(out, rec)
	}
	return out
}

func TestRequestBodyLogging_PairsWithResponse(t *testing.T) {
	respBody := `{"choices":[{"message":{"content":"hi"},"finish_reason":"stop"}]}`
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, respBody)
	}))
	defer server.Close()

	dir := t.TempDir()
	logPath := filepath.Join(dir, "rrlog.jsonl")

	p := NewProvider("key", server.URL, "",
		WithResponseLogFile(logPath),
		WithModelLabel("test-alias"),
	)
	if _, err := p.Chat(t.Context(), []Message{{Role: "user", Content: "hi"}}, nil, "test-model", nil); err != nil {
		t.Fatalf("Chat() error = %v", err)
	}

	recs := parseLogFile(t, logPath)
	if len(recs) != 2 {
		t.Fatalf("expected 2 JSONL records (request+response), got %d: %+v", len(recs), recs)
	}
	if recs[0].Dir != "request" {
		t.Errorf("recs[0].Dir = %q, want \"request\"", recs[0].Dir)
	}
	if recs[1].Dir != "response" {
		t.Errorf("recs[1].Dir = %q, want \"response\"", recs[1].Dir)
	}
	if recs[0].CorrID == "" {
		t.Error("request corr_id must be non-empty")
	}
	if recs[0].CorrID != recs[1].CorrID {
		t.Errorf("corr_id mismatch: request=%q response=%q", recs[0].CorrID, recs[1].CorrID)
	}
	if recs[0].Model != "test-alias" || recs[1].Model != "test-alias" {
		t.Errorf("expected model=test-alias on both records; got req=%q resp=%q", recs[0].Model, recs[1].Model)
	}
	if recs[1].Status != http.StatusOK {
		t.Errorf("response status = %d, want 200", recs[1].Status)
	}
	if recs[1].DurationMs < 0 {
		t.Errorf("response duration_ms must be non-negative, got %d", recs[1].DurationMs)
	}

	var reqBody map[string]any
	if err := json.Unmarshal(recs[0].Body, &reqBody); err != nil {
		t.Fatalf("request body not valid JSON object: %v; body=%s", err, recs[0].Body)
	}
	if reqBody["model"] != "test-model" {
		t.Errorf("logged request model = %v, want test-model", reqBody["model"])
	}

	var respBodyParsed map[string]any
	if err := json.Unmarshal(recs[1].Body, &respBodyParsed); err != nil {
		t.Fatalf("response body not valid JSON object: %v; body=%s", err, recs[1].Body)
	}
}

func TestRequestBodyLogging_RedactsAPIKey(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"choices":[{"message":{"content":"hi"},"finish_reason":"stop"}]}`)
	}))
	defer server.Close()

	dir := t.TempDir()
	logPath := filepath.Join(dir, "rrlog.jsonl")

	// Smuggle an api_key field into the request body via extra_body to
	// exercise the redaction path. claw's config validator would normally
	// reject this, but the provider must still defend the diagnostic log.
	p := NewProvider("super-secret-key", server.URL, "",
		WithResponseLogFile(logPath),
		WithExtraBody(map[string]any{"api_key": "leak-me-not"}),
	)
	if _, err := p.Chat(t.Context(), []Message{{Role: "user", Content: "hi"}}, nil, "test-model", nil); err != nil {
		t.Fatalf("Chat() error = %v", err)
	}

	recs := parseLogFile(t, logPath)
	if len(recs) < 1 {
		t.Fatalf("expected at least 1 record; got %d", len(recs))
	}
	reqRec := recs[0]
	if reqRec.Dir != "request" {
		t.Fatalf("first record dir = %q, want request", reqRec.Dir)
	}

	if bytes.Contains(reqRec.Body, []byte("leak-me-not")) {
		t.Errorf("api_key value should be redacted, but body contains it: %s", reqRec.Body)
	}

	var reqBody map[string]any
	if err := json.Unmarshal(reqRec.Body, &reqBody); err != nil {
		t.Fatalf("request body not valid JSON: %v; body=%s", err, reqRec.Body)
	}
	if reqBody["api_key"] != "[REDACTED]" {
		t.Errorf("api_key = %v, want [REDACTED]", reqBody["api_key"])
	}

	// Bearer Authorization header must never appear in the log either.
	raw, _ := os.ReadFile(logPath)
	if bytes.Contains(raw, []byte("super-secret-key")) {
		t.Errorf("bearer key should never appear in log; got:\n%s", raw)
	}
	if bytes.Contains(raw, []byte("Authorization")) {
		t.Errorf("Authorization header should never be logged; got:\n%s", raw)
	}
}

func TestRequestBodyLogging_NoOpWhenUnset(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"choices":[{"message":{"content":"hi"},"finish_reason":"stop"}]}`)
	}))
	defer server.Close()

	dir := t.TempDir()
	logPath := filepath.Join(dir, "rrlog.jsonl")

	p := NewProvider("key", server.URL, "") // no WithResponseLogFile
	if _, err := p.Chat(t.Context(), []Message{{Role: "user", Content: "hi"}}, nil, "test-model", nil); err != nil {
		t.Fatalf("Chat() error = %v", err)
	}

	if _, err := os.Stat(logPath); !os.IsNotExist(err) {
		t.Errorf("log file must not be created when response_log_file is unset; stat err=%v", err)
	}
	if p.logFile != nil {
		t.Errorf("provider must not open a log handle when response_log_file is unset; got %v", p.logFile)
	}
}

func TestRequestBodyLogging_NonBlockingOnWriteError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]}`)
	}))
	defer server.Close()

	dir := t.TempDir()
	logPath := filepath.Join(dir, "rrlog.jsonl")

	type warnRecord struct {
		component string
		message   string
		fields    map[string]any
	}
	var warns []warnRecord
	prev := warnFn
	warnFn = func(component, message string, fields map[string]any) {
		copied := make(map[string]any, len(fields))
		for k, v := range fields {
			copied[k] = v
		}
		warns = append(warns, warnRecord{component, message, copied})
	}
	t.Cleanup(func() { warnFn = prev })

	p := NewProvider("key", server.URL, "", WithResponseLogFile(logPath))

	// First call lazily opens the file handle.
	if _, err := p.Chat(t.Context(), []Message{{Role: "user", Content: "first"}}, nil, "test-model", nil); err != nil {
		t.Fatalf("Chat #1 error = %v", err)
	}
	if p.logFile == nil {
		t.Fatal("expected logFile to be opened after first request")
	}

	// Force write failures on subsequent calls by closing the handle out
	// from under the provider. The provider must not reopen it (sticky
	// state) and must not propagate the write error.
	if err := p.logFile.Close(); err != nil {
		t.Fatalf("close logFile: %v", err)
	}

	// Two more calls; both should succeed, and the provider should emit
	// at most one WRN total.
	for i := 0; i < 2; i++ {
		if _, err := p.Chat(t.Context(), []Message{{Role: "user", Content: "after-close"}}, nil, "test-model", nil); err != nil {
			t.Fatalf("Chat #%d after close: error = %v", i+2, err)
		}
	}

	// Filter for log-write warnings (extra_body collision uses a different
	// message; this test does not trigger it but stay defensive).
	var logWarns []warnRecord
	for _, w := range warns {
		if strings.Contains(w.message, "response_log_file") {
			logWarns = append(logWarns, w)
		}
	}
	if len(logWarns) != 1 {
		t.Fatalf("expected exactly 1 response_log_file WRN, got %d: %+v", len(logWarns), logWarns)
	}
	if logWarns[0].component != "openai_compat" {
		t.Errorf("WRN component = %q, want openai_compat", logWarns[0].component)
	}
	if logWarns[0].fields["path"] != logPath {
		t.Errorf("WRN fields.path = %v, want %q", logWarns[0].fields["path"], logPath)
	}
}

func TestResponseLogFile_LogsErrorResponses(t *testing.T) {
	errBody := `{"error":{"message":"boom"}}`
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		io.WriteString(w, errBody)
	}))
	defer server.Close()

	dir := t.TempDir()
	logPath := filepath.Join(dir, "rrlog.jsonl")

	p := NewProvider("key", server.URL, "", WithResponseLogFile(logPath))
	if _, err := p.Chat(t.Context(), []Message{{Role: "user", Content: "hi"}}, nil, "test-model", nil); err == nil {
		t.Fatalf("expected error from 500 response")
	}

	recs := parseLogFile(t, logPath)
	if len(recs) != 2 {
		t.Fatalf("expected 2 records (request+response), got %d: %+v", len(recs), recs)
	}
	if recs[1].Status != http.StatusInternalServerError {
		t.Errorf("response status = %d, want 500", recs[1].Status)
	}
	if !bytes.Contains(recs[1].Body, []byte("boom")) {
		t.Errorf("response body missing upstream error text; got: %s", recs[1].Body)
	}
}

func TestReasoningEffort_AddedToRequestBody(t *testing.T) {
	var captured map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&captured); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{"message": map[string]any{"content": "ok"}, "finish_reason": "stop"}},
		})
	}))
	defer server.Close()

	p := NewProvider("key", server.URL, "", WithReasoningEffort("high"))
	if _, err := p.Chat(t.Context(), []Message{{Role: "user", Content: "hi"}}, nil, "grok-3", nil); err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if got, _ := captured["reasoning_effort"].(string); got != "high" {
		t.Errorf("reasoning_effort = %v, want high; body=%v", captured["reasoning_effort"], captured)
	}
}

func TestReasoningEffort_OmittedWhenEmpty(t *testing.T) {
	var captured map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&captured)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{"message": map[string]any{"content": "ok"}, "finish_reason": "stop"}},
		})
	}))
	defer server.Close()

	p := NewProvider("key", server.URL, "")
	if _, err := p.Chat(t.Context(), []Message{{Role: "user", Content: "hi"}}, nil, "gpt-4o", nil); err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if _, ok := captured["reasoning_effort"]; ok {
		t.Errorf("reasoning_effort should be omitted; body=%v", captured)
	}
}

func TestExtraBody_MergedIntoRequest(t *testing.T) {
	var captured map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&captured)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{"message": map[string]any{"content": "ok"}, "finish_reason": "stop"}},
		})
	}))
	defer server.Close()

	extra := map[string]any{
		"search_parameters": map[string]any{"mode": "auto"},
		"custom_flag":       true,
	}
	p := NewProvider("key", server.URL, "", WithExtraBody(extra))
	if _, err := p.Chat(t.Context(), []Message{{Role: "user", Content: "hi"}}, nil, "gpt-4o", nil); err != nil {
		t.Fatalf("Chat: %v", err)
	}
	sp, ok := captured["search_parameters"].(map[string]any)
	if !ok {
		t.Fatalf("search_parameters missing or wrong type: %v", captured)
	}
	if sp["mode"] != "auto" {
		t.Errorf("search_parameters.mode = %v, want auto", sp["mode"])
	}
	if captured["custom_flag"] != true {
		t.Errorf("custom_flag = %v, want true", captured["custom_flag"])
	}
}

// TestExtraBody_CollisionLogsAndDoesNotOverwrite exercises the defensive merge
// guard. We force a clash by stuffing a reserved key into extra_body — the
// config validator normally rejects this at load, but bypassing the validator
// here lets us verify the provider's defensive log+skip path.
func TestExtraBody_CollisionLogsAndDoesNotOverwrite(t *testing.T) {
	var captured map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&captured)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{"message": map[string]any{"content": "ok"}, "finish_reason": "stop"}},
		})
	}))
	defer server.Close()

	type warnRecord struct {
		component string
		message   string
		fields    map[string]any
	}
	var warns []warnRecord
	prev := warnFn
	warnFn = func(component, message string, fields map[string]any) {
		copied := make(map[string]any, len(fields))
		for k, v := range fields {
			copied[k] = v
		}
		warns = append(warns, warnRecord{component, message, copied})
	}
	t.Cleanup(func() { warnFn = prev })

	extra := map[string]any{
		"model":       "should-not-overwrite", // collides with the wire model
		"extra_field": "kept",
	}
	p := NewProvider(
		"key", server.URL, "",
		WithExtraBody(extra),
		WithModelLabel("grok-3-config-label"),
	)
	if _, err := p.Chat(t.Context(), []Message{{Role: "user", Content: "hi"}}, nil, "grok-real-wire", nil); err != nil {
		t.Fatalf("Chat: %v", err)
	}

	if got := captured["model"]; got != "grok-real-wire" {
		t.Errorf("model overwritten: got %v, want grok-real-wire", got)
	}
	if got := captured["extra_field"]; got != "kept" {
		t.Errorf("extra_field lost: got %v", got)
	}
	if len(warns) != 1 {
		t.Fatalf("expected 1 warn call, got %d: %+v", len(warns), warns)
	}
	w := warns[0]
	if w.component != "openai_compat" {
		t.Errorf("warn component = %q, want openai_compat", w.component)
	}
	if w.fields["key"] != "model" {
		t.Errorf("warn fields.key = %v, want %q", w.fields["key"], "model")
	}
	if w.fields["model"] != "grok-3-config-label" {
		t.Errorf("warn fields.model = %v, want %q (label fallback)", w.fields["model"], "grok-3-config-label")
	}
}

func TestResponseLogFile_OpenFailureDoesNotPropagate(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]}`)
	}))
	defer server.Close()

	dir := t.TempDir()
	// Point response_log_file at a directory — os.OpenFile will fail.
	p := NewProvider("key", server.URL, "", WithResponseLogFile(dir))
	out, err := p.Chat(t.Context(), []Message{{Role: "user", Content: "hi"}}, nil, "test-model", nil)
	if err != nil {
		t.Fatalf("Chat() must not propagate log-file errors, got: %v", err)
	}
	if out == nil || len(out.Content) == 0 {
		t.Fatalf("expected parsed response despite log failure, got: %+v", out)
	}
}
