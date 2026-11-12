package anthropicmessages

import (
	"reflect"
	"testing"

	"github.com/PivotLLM/spawnllm/common"
)

func TestBuildRequestBody_ToolChoice(t *testing.T) {
	tools := []ToolDefinition{{
		Type:     "function",
		Function: ToolFunctionDefinition{Name: "get_weather", Parameters: map[string]any{"type": "object"}},
	}}
	none := map[string]any{"type": "none"}

	tests := []struct {
		name    string
		tools   []ToolDefinition
		choice  any // value of options[common.ToolChoiceOption]; nil leaves the key out
		want    any // expected tool_choice; nil means absent
		wantSet bool
	}{
		{name: "default is auto, not sent", tools: tools},
		{name: "explicit auto, not sent", tools: tools, choice: common.ToolChoiceAuto},
		{name: "none sent", tools: tools, choice: common.ToolChoiceNone, want: none, wantSet: true},
		{name: "unknown value is auto", tools: tools, choice: "required"},
		{name: "non-string value is auto", tools: tools, choice: true},
		{name: "none without tools not sent", choice: common.ToolChoiceNone},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			options := map[string]any{"max_tokens": 1024}
			if tt.choice != nil {
				options[common.ToolChoiceOption] = tt.choice
			}
			body, err := buildRequestBody([]Message{{Role: "user", Content: "hi"}}, tt.tools, "claude-test", options)
			if err != nil {
				t.Fatalf("buildRequestBody() error = %v", err)
			}
			got, set := body["tool_choice"]
			if set != tt.wantSet {
				t.Fatalf("tool_choice present = %v, want %v (got %v)", set, tt.wantSet, got)
			}
			if set && !reflect.DeepEqual(got, tt.want) {
				t.Errorf("tool_choice = %v, want %v", got, tt.want)
			}
			if len(tt.tools) > 0 && body["tools"] == nil {
				t.Error("tools must stay declared")
			}
		})
	}
}
