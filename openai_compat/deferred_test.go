package openai_compat

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestChat_ReferencedDeferredToolsSentAndReferencesAsText(t *testing.T) {
	var body map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		if _, err := w.Write([]byte(okReply)); err != nil {
			t.Errorf("write: %v", err)
		}
	}))
	defer server.Close()

	tools := []ToolDefinition{
		{Type: "function", Function: ToolFunctionDefinition{Name: "search_tools"}},
		{Type: "function", Function: ToolFunctionDefinition{Name: "read_file"}, Deferred: true},
		{Type: "function", Function: ToolFunctionDefinition{Name: "get_weather"}, Deferred: true},
	}
	msgs := []Message{
		{Role: "user", Content: "weather?"},
		{Role: "assistant", ToolCalls: []ToolCall{{ID: "c1", Name: "search_tools", Arguments: map[string]any{}}}},
		{Role: "tool", ToolCallID: "c1", ToolReferences: []string{"get_weather"}},
	}
	p := NewProvider("k", server.URL, "")
	if _, err := p.Chat(t.Context(), msgs, tools, "gpt-5.2", nil); err != nil {
		t.Fatalf("Chat: %v", err)
	}

	// read_file was never referenced: left out. get_weather was: an ordinary
	// tool after the loaded ones.
	sent := list(t, body["tools"])
	if len(sent) != 2 {
		t.Fatalf("tools = %v, want search_tools, get_weather", sent)
	}
	if name := dig(t, sent, 0, "function", "name"); name != "search_tools" {
		t.Errorf("tools[0] = %v, want search_tools", name)
	}
	if name := dig(t, sent, 1, "function", "name"); name != "get_weather" {
		t.Errorf("tools[1] = %v, want get_weather", name)
	}
	for i := range sent {
		if dig(t, sent, i, "deferred") != nil || dig(t, sent, i, "Deferred") != nil ||
			dig(t, sent, i, "function", "defer_loading") != nil {
			t.Errorf("tools[%d]: no defer marker may reach the wire", i)
		}
	}
	if got := dig(t, body, "messages", 2, "content"); got != "Tools found: get_weather" {
		t.Errorf("tool result content = %v, want the references as text", got)
	}
}

func TestNormalizeAlternation_ToolReferencesAsText(t *testing.T) {
	out := normalizeAlternation([]Message{
		{Role: "user", Content: "q"},
		{Role: "tool", ToolCallID: "c1", Content: "hits", ToolReferences: []string{"a"}},
	})
	if out[0].Content != "q\n\nTool result:\nhits\n\nTools found: a" {
		t.Errorf("content = %q", out[0].Content)
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
