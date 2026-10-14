package anthropicprovider

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"
)

// roundTripMessages passes history through JSON, as a host that persists it
// does: a tool call keeps only ID, Type and Function.
func roundTripMessages(t *testing.T, msgs []Message) []Message {
	t.Helper()
	b, err := json.Marshal(msgs)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var out []Message
	if err := json.Unmarshal(b, &out); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	return out
}

func TestChat_ToolUseFromHistory(t *testing.T) {
	tests := []struct {
		name      string
		call      ToolCall
		roundTrip bool
		wantInput map[string]any
	}{
		{
			name:      "reloaded from history",
			call:      ToolCall{ID: "toolu_1", Type: "function", Function: &FunctionCall{Name: "get_weather", Arguments: `{"city":"Paris"}`}},
			roundTrip: true,
			wantInput: map[string]any{"city": "Paris"},
		},
		{
			name:      "reloaded with empty arguments",
			call:      ToolCall{ID: "toolu_1", Type: "function", Function: &FunctionCall{Name: "get_weather", Arguments: ""}},
			roundTrip: true,
			wantInput: map[string]any{},
		},
		{
			name:      "direct name and arguments",
			call:      ToolCall{ID: "toolu_1", Name: "get_weather", Arguments: map[string]any{"city": "Paris"}},
			wantInput: map[string]any{"city": "Paris"},
		},
		{
			name:      "direct name without arguments",
			call:      ToolCall{ID: "toolu_1", Name: "get_weather"},
			wantInput: map[string]any{},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var captured map[string]any
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if err := json.NewDecoder(r.Body).Decode(&captured); err != nil {
					t.Errorf("decode request: %v", err)
				}
				w.Header().Set("Content-Type", "application/json")
				if err := json.NewEncoder(w).Encode(map[string]any{
					"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-sonnet-4-6",
					"stop_reason": "end_turn",
					"content":     []map[string]any{{"type": "text", "text": "ok"}},
					"usage":       map[string]any{"input_tokens": 1, "output_tokens": 1},
				}); err != nil {
					t.Errorf("encode response: %v", err)
				}
			}))
			defer server.Close()

			msgs := []Message{
				{Role: "user", Content: "Weather?"},
				{Role: "assistant", ToolCalls: []ToolCall{tt.call}},
				{Role: "tool", ToolCallID: "toolu_1", Content: "sunny"},
			}
			if tt.roundTrip {
				msgs = roundTripMessages(t, msgs)
			}
			p := NewProviderWithClient(createAnthropicTestClient(server.URL, "test-token"))
			if _, err := p.Chat(t.Context(), msgs, nil, "claude-sonnet-4.6", map[string]any{"max_tokens": 64}); err != nil {
				t.Fatalf("Chat() error: %v", err)
			}

			sent := asType[[]any](t, captured["messages"], "sent")
			if len(sent) != 3 {
				t.Fatalf("sent %d messages, want 3: %v", len(sent), captured["messages"])
			}
			assistant := asType[map[string]any](t, sent[1], "assistant")
			blocks := asType[[]any](t, assistant["content"], "blocks")
			if len(blocks) != 1 {
				t.Fatalf("assistant content = %v, want one tool_use block", assistant["content"])
			}
			toolUse := asType[map[string]any](t, blocks[0], "toolUse")
			if toolUse["type"] != "tool_use" || toolUse["id"] != "toolu_1" || toolUse["name"] != "get_weather" {
				t.Errorf("tool_use block = %v", toolUse)
			}
			input, ok := toolUse["input"].(map[string]any)
			if !ok {
				t.Fatalf("tool_use input = %#v, want an object", toolUse["input"])
			}
			if !reflect.DeepEqual(input, tt.wantInput) {
				t.Errorf("tool_use input = %v, want %v", input, tt.wantInput)
			}

			user := asType[map[string]any](t, sent[2], "user")
			results := asType[[]any](t, user["content"], "results")
			if len(results) != 1 {
				t.Fatalf("tool result content = %v, want one block", user["content"])
			}
			result := asType[map[string]any](t, results[0], "result")
			if result["type"] != "tool_result" || result["tool_use_id"] != "toolu_1" {
				t.Errorf("tool_result block = %v", result)
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
