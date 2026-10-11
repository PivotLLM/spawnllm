package openai_compat

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"reflect"
	"testing"

	"github.com/PivotLLM/spawnllm/protocoltypes"
)

// redirectTransport sends every request to target while keeping the
// provider's configured (OpenRouter) base, so the OpenRouter-only request
// shaping is exercised against a local server.
type redirectTransport struct{ target *url.URL }

func (rt redirectTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	r = r.Clone(r.Context())
	r.URL.Scheme = rt.target.Scheme
	r.URL.Host = rt.target.Host
	r.Host = rt.target.Host
	return http.DefaultTransport.RoundTrip(r)
}

// captureOpenRouterRequest sends one Chat call through a provider whose base
// is OpenRouter and returns the decoded request body.
func captureOpenRouterRequest(t *testing.T, extra map[string]any, withTools bool) map[string]any {
	t.Helper()
	var body map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{
				"message":       map[string]any{"content": "ok"},
				"finish_reason": "stop",
			}},
		}); err != nil {
			t.Errorf("encode response: %v", err)
		}
	}))
	t.Cleanup(server.Close)
	target, err := url.Parse(server.URL)
	if err != nil {
		t.Fatalf("parse server URL: %v", err)
	}

	var opts []Option
	if extra != nil {
		opts = append(opts, WithExtraBody(extra))
	}
	p := NewProvider("key", "https://openrouter.ai/api/v1", "", opts...)
	p.httpClient.Transport = redirectTransport{target: target}

	var tools []protocoltypes.ToolDefinition
	if withTools {
		tools = []protocoltypes.ToolDefinition{{
			Type: "function",
			Function: protocoltypes.ToolFunctionDefinition{
				Name:       "get_weather",
				Parameters: map[string]any{"type": "object"},
			},
		}}
	}
	if _, err := p.Chat(t.Context(), []Message{{Role: "user", Content: "hi"}}, tools, "openai/gpt-4o", nil); err != nil {
		t.Fatalf("Chat() error = %v", err)
	}
	return body
}

// TestOpenRouterExtraBodyProvider pins the rule for the OpenRouter `provider`
// object: the operator's extra_body.provider is always sent, and spawnllm's
// own routing keys are added only where the operator did not set that key.
func TestOpenRouterExtraBodyProvider(t *testing.T) {
	tests := []struct {
		name      string
		extra     map[string]any
		withTools bool
		want      any // expected "provider" value; nil = key absent
	}{
		{
			name:      "no tools, no user provider",
			withTools: false,
			want:      nil,
		},
		{
			name:      "tools, no user provider",
			withTools: true,
			want:      map[string]any{"require_parameters": true},
		},
		{
			name: "no tools, user provider sent as is",
			extra: map[string]any{"provider": map[string]any{
				"order": []any{"Alice", "Bob"}, "allow_fallbacks": false,
			}},
			withTools: false,
			want: map[string]any{
				"order": []any{"Alice", "Bob"}, "allow_fallbacks": false,
			},
		},
		{
			name: "tools, user provider merged with require_parameters",
			extra: map[string]any{"provider": map[string]any{
				"order": []any{"Alice", "Bob"}, "allow_fallbacks": false,
			}},
			withTools: true,
			want: map[string]any{
				"order": []any{"Alice", "Bob"}, "allow_fallbacks": false,
				"require_parameters": true,
			},
		},
		{
			name: "tools, user require_parameters wins",
			extra: map[string]any{"provider": map[string]any{
				"order": []any{"Alice"}, "require_parameters": false,
			}},
			withTools: true,
			want: map[string]any{
				"order": []any{"Alice"}, "require_parameters": false,
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			body := captureOpenRouterRequest(t, tc.extra, tc.withTools)
			got, ok := body["provider"]
			if tc.want == nil {
				if ok {
					t.Fatalf("provider = %v, want absent", got)
				}
				return
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("provider = %#v, want %#v", got, tc.want)
			}
		})
	}
}

// TestOpenRouterExtraBodyProvider_UserMapUnchanged checks that merging
// spawnllm's keys never writes into the operator's configured map, which is
// shared by every request the provider sends.
func TestOpenRouterExtraBodyProvider_UserMapUnchanged(t *testing.T) {
	userProvider := map[string]any{"order": []any{"Alice"}}
	captureOpenRouterRequest(t, map[string]any{"provider": userProvider}, true)
	if _, ok := userProvider["require_parameters"]; ok {
		t.Fatalf("configured extra_body.provider was modified: %v", userProvider)
	}
	if len(userProvider) != 1 {
		t.Fatalf("configured extra_body.provider = %v, want only order", userProvider)
	}
}
