package openai_compat

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"github.com/PivotLLM/spawnllm/common"
)

// chatVia sends one Chat request to apiBase (rerouted to a local test server)
// and returns the decoded request body and the parsed response. reply is the
// JSON the server answers with.
func chatVia(t *testing.T, apiBase, model string, options map[string]any, reply string) (map[string]any, *LLMResponse) {
	t.Helper()
	var requestBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&requestBody); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		if _, err := w.Write([]byte(reply)); err != nil {
			t.Errorf("write response: %v", err)
		}
	}))
	t.Cleanup(server.Close)

	p := NewProvider("key", apiBase, "")
	p.httpClient = &http.Client{
		Transport: roundTripperFunc(func(r *http.Request) (*http.Response, error) {
			u, err := url.Parse(server.URL + r.URL.Path)
			if err != nil {
				return nil, err
			}
			r.URL = u
			return http.DefaultTransport.RoundTrip(r)
		}),
	}
	out, err := p.Chat(t.Context(), []Message{{Role: "user", Content: "hi"}}, nil, model, options)
	if err != nil {
		t.Fatalf("Chat() error = %v", err)
	}
	return requestBody, out
}

const okReply = `{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]}`

func TestChat_OpenRouterClaudeSendsCacheControl(t *testing.T) {
	tests := []struct {
		name    string
		apiBase string
		model   string
		want    bool
	}{
		{"openrouter claude", "https://openrouter.ai/api/v1", "anthropic/claude-sonnet-4.6", true},
		{"openrouter claude alias", "https://openrouter.ai/api/v1", "~anthropic/claude-sonnet-latest", true},
		{"openrouter claude variant", "https://openrouter.ai/api/v1", "anthropic/claude-opus-4.8:thinking", true},
		{"openrouter other model", "https://openrouter.ai/api/v1", "openai/gpt-5.2", false},
		{"openrouter bare claude-like name", "https://openrouter.ai/api/v1", "claude-sonnet-4.6", false},
		{"not openrouter", "https://api.example.com/v1", "anthropic/claude-sonnet-4.6", false},
		{"openrouter only in path", "https://proxy.example.com/openrouter.ai/v1", "anthropic/claude-sonnet-4.6", false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			body, _ := chatVia(t, tt.apiBase, tt.model, nil, okReply)
			cc, present := body["cache_control"]
			if present != tt.want {
				t.Fatalf("cache_control present = %v, want %v (body %v)", present, tt.want, body)
			}
			if tt.want {
				m, ok := cc.(map[string]any)
				if !ok || m["type"] != "ephemeral" {
					t.Errorf("cache_control = %v, want {type: ephemeral}", cc)
				}
			}
		})
	}
}

func TestChat_OpenRouterSessionID(t *testing.T) {
	long := strings.Repeat("a", 255) + "é" // 257 bytes; é straddles byte 256
	tests := []struct {
		name    string
		apiBase string
		options map[string]any
		want    string // "" = absent
	}{
		{"openrouter with key", "https://openrouter.ai/api/v1", map[string]any{"prompt_cache_key": "agent-main"}, "agent-main"},
		{"openrouter long key truncated", "https://openrouter.ai/api/v1", map[string]any{"prompt_cache_key": long}, strings.Repeat("a", 255)},
		{"openrouter without key", "https://openrouter.ai/api/v1", nil, ""},
		{"openai with key", "https://api.openai.com/v1", map[string]any{"prompt_cache_key": "agent-main"}, ""},
		{"other with key", "https://api.deepseek.com/v1", map[string]any{"prompt_cache_key": "agent-main"}, ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			body, _ := chatVia(t, tt.apiBase, "openai/gpt-5.2", tt.options, okReply)
			got, present := body["session_id"]
			if tt.want == "" {
				if present {
					t.Fatalf("session_id = %v, want absent", got)
				}
				return
			}
			if got != tt.want {
				t.Fatalf("session_id = %v, want %q", got, tt.want)
			}
			if _, sent := body["prompt_cache_key"]; sent {
				t.Error("prompt_cache_key must not be sent to OpenRouter")
			}
		})
	}
}

func TestTruncateUTF8(t *testing.T) {
	tests := []struct {
		in   string
		max  int
		want string
	}{
		{"abc", 5, "abc"},
		{"abcdef", 3, "abc"},
		{"aé", 2, "a"},
		{"aé", 3, "aé"},
		{"", 3, ""},
	}
	for _, tt := range tests {
		if got := truncateUTF8(tt.in, tt.max); got != tt.want {
			t.Errorf("truncateUTF8(%q, %d) = %q, want %q", tt.in, tt.max, got, tt.want)
		}
	}
}

func TestChat_CacheWriteTokensParsed(t *testing.T) {
	reply := `{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],` +
		`"usage":{"prompt_tokens":1000,"completion_tokens":10,"total_tokens":1010,` +
		`"prompt_tokens_details":{"cached_tokens":600,"cache_write_tokens":300}}}`
	_, out := chatVia(t, "https://openrouter.ai/api/v1", "anthropic/claude-sonnet-4.6", nil, reply)
	if out.Status.CacheReadTokens != 600 || out.Status.CacheCreationTokens != 300 {
		t.Errorf("cache read/creation = %d/%d, want 600/300",
			out.Status.CacheReadTokens, out.Status.CacheCreationTokens)
	}
	if out.Status.CostUSD != nil {
		t.Errorf("CostUSD = %v, want nil (no cost reported)", *out.Status.CostUSD)
	}
	// prompt_tokens already counts cached and written tokens: not added again.
	if out.Usage.PromptTokens != 1000 || out.Usage.TotalTokens != 1010 {
		t.Errorf("usage = %+v, want prompt 1000 total 1010", out.Usage)
	}
	// InputTokens is only the uncached part; the three add up to the prompt.
	if out.Status.InputTokens != 100 {
		t.Errorf("InputTokens = %d, want 100 (1000 less 600 read, 300 written)", out.Status.InputTokens)
	}
	if sum := out.Status.InputTokens + out.Status.CacheReadTokens + out.Status.CacheCreationTokens; sum != out.Usage.PromptTokens {
		t.Errorf("input+read+write = %d, want PromptTokens %d", sum, out.Usage.PromptTokens)
	}

	_, plain := chatVia(t, "https://openrouter.ai/api/v1", "openai/gpt-5.2", nil,
		`{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":1,"total_tokens":6}}`)
	if plain.Status.CacheReadTokens != 0 || plain.Status.CacheCreationTokens != 0 {
		t.Errorf("no details: cache read/creation = %d/%d, want 0/0",
			plain.Status.CacheReadTokens, plain.Status.CacheCreationTokens)
	}
	if plain.Status.InputTokens != 5 {
		t.Errorf("no details: InputTokens = %d, want 5", plain.Status.InputTokens)
	}
}

func TestChat_InputTokensExcludeCacheStreaming(t *testing.T) {
	body := "data: " + `{"model":"m","choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}` + "\n\n" +
		"data: " + `{"model":"m","choices":[],"usage":{"prompt_tokens":1000,"completion_tokens":1,"total_tokens":1001,` +
		`"prompt_tokens_details":{"cached_tokens":600,"cache_write_tokens":300}}}` + "\n\n" +
		"data: [DONE]\n\n"
	server := sseServer(t, body, nil)
	defer server.Close()
	p := NewProvider("key", server.URL, "")
	out, err := p.Chat(t.Context(), []Message{{Role: "user", Content: "hi"}}, nil, "m",
		map[string]any{common.TextDeltaOption: func(string) {}})
	if err != nil {
		t.Fatalf("Chat() error = %v", err)
	}
	s := out.Status
	if s.InputTokens != 100 || s.CacheReadTokens != 600 || s.CacheCreationTokens != 300 {
		t.Errorf("input/read/write = %d/%d/%d, want 100/600/300", s.InputTokens, s.CacheReadTokens, s.CacheCreationTokens)
	}
	if sum := s.InputTokens + s.CacheReadTokens + s.CacheCreationTokens; sum != out.Usage.PromptTokens {
		t.Errorf("input+read+write = %d, want PromptTokens %d", sum, out.Usage.PromptTokens)
	}
}
