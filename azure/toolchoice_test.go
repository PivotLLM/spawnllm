package azure

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/PivotLLM/spawnllm/common"
	"github.com/PivotLLM/spawnllm/protocoltypes"
)

func TestProviderChat_ToolChoice(t *testing.T) {
	tools := []ToolDefinition{{
		Type:     "function",
		Function: protocoltypes.ToolFunctionDefinition{Name: "get_weather", Parameters: map[string]any{"type": "object"}},
	}}

	tests := []struct {
		name   string
		tools  []ToolDefinition
		choice any // nil leaves the key out
		want   any // nil means tool_choice absent
	}{
		{name: "default is auto", tools: tools, want: "auto"},
		{name: "explicit auto", tools: tools, choice: common.ToolChoiceAuto, want: "auto"},
		{name: "none", tools: tools, choice: common.ToolChoiceNone, want: "none"},
		{name: "unknown value is auto", tools: tools, choice: "required", want: "auto"},
		{name: "none without tools not sent", choice: common.ToolChoiceNone},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var body map[string]any
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
					t.Errorf("decode request: %v", err)
				}
				w.Header().Set("Content-Type", "application/json")
				resp := map[string]any{"choices": []map[string]any{{
					"message":       map[string]any{"content": "ok"},
					"finish_reason": "stop",
				}}}
				if err := json.NewEncoder(w).Encode(resp); err != nil {
					t.Errorf("encode response: %v", err)
				}
			}))
			defer server.Close()

			options := map[string]any{}
			if tt.choice != nil {
				options[common.ToolChoiceOption] = tt.choice
			}
			p := NewProvider("key", server.URL, "")
			if _, err := p.Chat(t.Context(), []Message{{Role: "user", Content: "hi"}}, tt.tools, "deployment", options); err != nil {
				t.Fatalf("Chat() error = %v", err)
			}
			got, set := body["tool_choice"]
			if tt.want == nil {
				if set {
					t.Errorf("tool_choice = %v, want absent", got)
				}
				return
			}
			if got != tt.want {
				t.Errorf("tool_choice = %v, want %v", got, tt.want)
			}
			if body["tools"] == nil {
				t.Error("tools must stay declared")
			}
		})
	}
}
