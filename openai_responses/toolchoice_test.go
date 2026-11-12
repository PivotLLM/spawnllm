package openai_responses

import (
	"context"
	"testing"

	"github.com/PivotLLM/spawnllm/common"
	"github.com/PivotLLM/spawnllm/protocoltypes"
)

func TestChat_ToolChoice(t *testing.T) {
	reply := `{"status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]}]}`
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
			p, body := captureServer(t, reply)
			options := map[string]any{}
			if tt.choice != nil {
				options[common.ToolChoiceOption] = tt.choice
			}
			if _, err := p.Chat(context.Background(), []Message{{Role: "user", Content: "hi"}}, tt.tools, "gpt-5", options); err != nil {
				t.Fatalf("Chat error: %v", err)
			}
			got, set := (*body)["tool_choice"]
			if tt.want == nil {
				if set {
					t.Errorf("tool_choice = %v, want absent", got)
				}
				return
			}
			if got != tt.want {
				t.Errorf("tool_choice = %v, want %v", got, tt.want)
			}
			if (*body)["tools"] == nil {
				t.Error("tools must stay declared")
			}
		})
	}
}
