package openai_compat

import (
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"

	"github.com/PivotLLM/spawnllm/common"
)

func TestChat_OpenRouterUsageCost(t *testing.T) {
	const openRouter = "https://openrouter.ai/api/v1"
	usage := func(extra string) string {
		return `{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],` +
			`"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12` + extra + `}}`
	}
	tests := []struct {
		name    string
		apiBase string
		reply   string
		want    *float64
	}{
		{"openrouter cost", openRouter, usage(`,"cost":0.00123`), new(0.00123)},
		{"openrouter zero cost kept", openRouter, usage(`,"cost":0`), new(0.0)},
		{"openrouter no cost", openRouter, usage(""), nil},
		{"openrouter no usage", openRouter, okReply, nil},
		{"other host ignored", "https://api.example.com/v1", usage(`,"cost":0.5`), nil},
		{"openrouter only in path ignored", "https://proxy.example.com/openrouter.ai/v1", usage(`,"cost":0.5`), nil},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, out := chatVia(t, tt.apiBase, "openai/gpt-5.2", nil, tt.reply)
			assertCost(t, out.Status.CostUSD, tt.want)
		})
	}
}

func TestChat_OpenRouterUsageCostStreaming(t *testing.T) {
	sse := func(cost string) string {
		return "data: " + `{"model":"m","choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}` + "\n\n" +
			"data: " + `{"model":"m","choices":[],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3` + cost + `}}` + "\n\n" +
			"data: [DONE]\n\n"
	}
	tests := []struct {
		name    string
		apiBase string
		body    string
		want    *float64
	}{
		{"openrouter cost", "https://openrouter.ai/api/v1", sse(`,"cost":0.25`), new(0.25)},
		{"openrouter zero cost kept", "https://openrouter.ai/api/v1", sse(`,"cost":0`), new(0.0)},
		{"openrouter no cost", "https://openrouter.ai/api/v1", sse(""), nil},
		{"other host ignored", "https://api.example.com/v1", sse(`,"cost":0.25`), nil},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := sseServer(t, tt.body, nil)
			defer server.Close()
			p := NewProvider("key", tt.apiBase, "")
			p.httpClient = &http.Client{Transport: rerouteTo(server)}
			out, err := p.Chat(t.Context(), []Message{{Role: "user", Content: "hi"}}, nil, "m",
				map[string]any{common.TextDeltaOption: func(string) {}})
			if err != nil {
				t.Fatalf("Chat() error = %v", err)
			}
			assertCost(t, out.Status.CostUSD, tt.want)
		})
	}
}

// rerouteTo sends every request to server, keeping its path.
func rerouteTo(server *httptest.Server) http.RoundTripper {
	return roundTripperFunc(func(r *http.Request) (*http.Response, error) {
		u, err := url.Parse(server.URL + r.URL.Path)
		if err != nil {
			return nil, err
		}
		r.URL = u
		return http.DefaultTransport.RoundTrip(r)
	})
}

func assertCost(t *testing.T, got, want *float64) {
	t.Helper()
	switch {
	case want == nil && got != nil:
		t.Fatalf("CostUSD = %v, want nil", *got)
	case want != nil && got == nil:
		t.Fatalf("CostUSD = nil, want %v", *want)
	case want != nil && *got != *want:
		t.Fatalf("CostUSD = %v, want %v", *got, *want)
	}
}
