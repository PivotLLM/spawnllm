package azure

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/PivotLLM/spawnllm/protocoltypes"
)

func TestProviderChat_DeferredToolsLeftOutAndReferencesAsText(t *testing.T) {
	var body map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode: %v", err)
		}
		writeValidResponse(t, w)
	}))
	defer server.Close()

	tools := []ToolDefinition{
		{Type: "function", Function: protocoltypes.ToolFunctionDefinition{Name: "search_tools"}},
		{Type: "function", Function: protocoltypes.ToolFunctionDefinition{Name: "get_weather"}, Deferred: true},
	}
	msgs := []Message{
		{Role: "user", Content: "weather?"},
		{Role: "tool", ToolCallID: "c1", ToolReferences: []string{"get_weather"}},
	}
	p := NewProvider("k", server.URL, "")
	if _, err := p.Chat(t.Context(), msgs, tools, "dep", nil); err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if sent := list(t, body["tools"]); len(sent) != 1 {
		t.Errorf("tools = %v, want only search_tools", sent)
	}
	if got := dig(t, body, "messages", 1, "content"); got != "Tools found: get_weather" {
		t.Errorf("tool result content = %v", got)
	}
}

// dig walks decoded JSON by object keys (string) and list indices (int; a
// negative one counts from the end), failing the test when a step does not fit.
func dig(t *testing.T, v any, path ...any) any {
	t.Helper()
	for _, step := range path {
		switch k := step.(type) {
		case string:
			m, ok := v.(map[string]any)
			if !ok {
				t.Fatalf("at %q: %T is not an object", k, v)
			}
			v = m[k]
		case int:
			l, ok := v.([]any)
			if ok && k < 0 {
				k += len(l)
			}
			if !ok || k < 0 || k >= len(l) {
				t.Fatalf("at [%d]: %T is not a list that long", k, v)
			}
			v = l[k]
		}
	}
	return v
}

// list returns v as a decoded JSON list.
func list(t *testing.T, v any) []any {
	t.Helper()
	l, ok := v.([]any)
	if !ok {
		t.Fatalf("%T is not a list", v)
	}
	return l
}
