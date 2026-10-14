package openai_compat

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestProviderChat_ToolCallsInEitherForm(t *testing.T) {
	tests := []struct {
		name   string
		call   ToolCall
		strict bool
	}{
		{"direct name and arguments", ToolCall{ID: "c1", Name: "get_weather", Arguments: map[string]any{"city": "Paris"}}, false},
		{"direct name and arguments, strict", ToolCall{ID: "c1", Name: "get_weather", Arguments: map[string]any{"city": "Paris"}}, true},
		{"function only", ToolCall{ID: "c1", Type: "function", Function: &FunctionCall{Name: "get_weather", Arguments: `{"city":"Paris"}`}}, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var requestBody map[string]any
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if err := json.NewDecoder(r.Body).Decode(&requestBody); err != nil {
					t.Errorf("decode request: %v", err)
				}
				w.Header().Set("Content-Type", "application/json")
				if _, err := w.Write([]byte(`{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]}`)); err != nil {
					t.Errorf("write response: %v", err)
				}
			}))
			defer server.Close()

			p := NewProvider("key", server.URL, "", WithStrictCompat(tt.strict))
			msgs := []Message{
				{Role: "user", Content: "Weather?"},
				{Role: "assistant", ToolCalls: []ToolCall{tt.call}},
				{Role: "tool", ToolCallID: "c1", Content: "sunny"},
			}
			if _, err := p.Chat(t.Context(), msgs, nil, "gpt-4o", nil); err != nil {
				t.Fatalf("Chat() error = %v", err)
			}

			sent := asType[[]any](t, requestBody["messages"], "sent")
			assistant := asType[map[string]any](t, sent[1], "assistant")
			calls := asType[[]any](t, assistant["tool_calls"], "calls")
			if len(calls) != 1 {
				t.Fatalf("tool_calls = %v, want one", assistant["tool_calls"])
			}
			call := asType[map[string]any](t, calls[0], "call")
			fn := asType[map[string]any](t, call["function"], "fn")
			if call["id"] != "c1" || call["type"] != "function" || fn["name"] != "get_weather" ||
				fn["arguments"] != `{"city":"Paris"}` {
				t.Errorf("tool call = %v", call)
			}
		})
	}
}

// asType returns v as a T, failing the test when it has another type. what
// names the value in the failure message.
func asType[T any](t *testing.T, v any, what string) T {
	t.Helper()
	got, ok := v.(T)
	if !ok {
		t.Fatalf("%s is %T, want %T", what, v, got)
	}
	return got
}
