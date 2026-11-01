package common

import (
	"reflect"
	"testing"
)

func tool(name string, deferred bool) ToolDefinition {
	return ToolDefinition{Type: "function", Function: ToolFunctionDefinition{Name: name}, Deferred: deferred}
}

func TestLoadedTools(t *testing.T) {
	none := []ToolDefinition{tool("a", false), tool("b", false)}
	if got := LoadedTools(none); &got[0] != &none[0] || len(got) != 2 {
		t.Errorf("no deferred tools: want the input slice unchanged, got %v", got)
	}
	mixed := []ToolDefinition{tool("a", false), tool("b", true), tool("c", false)}
	got := LoadedTools(mixed)
	if len(got) != 2 || got[0].Function.Name != "a" || got[1].Function.Name != "c" {
		t.Errorf("LoadedTools = %v, want a, c", got)
	}
	if len(mixed) != 3 {
		t.Error("input slice modified")
	}
	if got := LoadedTools(nil); got != nil {
		t.Errorf("LoadedTools(nil) = %v, want nil", got)
	}
}

func TestToolResultText(t *testing.T) {
	tests := []struct {
		name string
		msg  Message
		want string
	}{
		{"no references", Message{Content: "result"}, "result"},
		{"references only", Message{ToolReferences: []string{"a", "b"}}, "Tools found: a, b"},
		{"text and references", Message{Content: "2 matches", ToolReferences: []string{"a", "b"}}, "2 matches\n\nTools found: a, b"},
	}
	for _, tt := range tests {
		if got := ToolResultText(tt.msg); got != tt.want {
			t.Errorf("%s: ToolResultText = %q, want %q", tt.name, got, tt.want)
		}
	}
}

func TestAnthropicSupportsDeferredTools(t *testing.T) {
	tests := map[string]bool{
		"claude-opus-5-5":              true,
		"claude-fable-5-1":             true,
		"claude-sonnet-4-6":            true,
		"claude-sonnet-4.6":            true,
		"claude-opus-4-5-20251101":     true,
		"claude-haiku-4-5-20251001":    true,
		"claude-sonnet-4-5-20250929":   true,
		"anthropic/claude-sonnet-4.6":  true,
		"us.anthropic.claude-opus-4-8": true,
		"CLAUDE-OPUS-4-7":              true,
		"claude-opus-4-1-20250805":     false,
		"claude-opus-4-20250514":       false,
		"claude-sonnet-4-20250514":     false,
		"claude-3-7-sonnet-20250219":   false,
		"claude-3-5-haiku-latest":      false,
		"gpt-5.4":                      false,
		"claude-":                      false,
		"claude-sonnet":                false,
		"":                             false,
	}
	for model, want := range tests {
		if got := AnthropicSupportsDeferredTools(model); got != want {
			t.Errorf("AnthropicSupportsDeferredTools(%q) = %v, want %v", model, got, want)
		}
	}
}

func TestSerializeMessages_ToolReferencesAsText(t *testing.T) {
	out := SerializeMessages([]Message{{Role: "tool", ToolCallID: "c1", Content: "found", ToolReferences: []string{"x"}}})
	got, ok := out[0].(openaiMessage)
	if !ok {
		t.Fatalf("out[0] = %T", out[0])
	}
	if !reflect.DeepEqual(got.Content, "found\n\nTools found: x") {
		t.Errorf("content = %q", got.Content)
	}
}
