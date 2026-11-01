package common

import (
	"reflect"
	"testing"
)

func tool(name string, deferred bool) ToolDefinition {
	return ToolDefinition{Type: "function", Function: ToolFunctionDefinition{Name: name}, Deferred: deferred}
}

func names(tools []ToolDefinition) []string {
	out := make([]string, len(tools))
	for i, t := range tools {
		out[i] = t.Function.Name
	}
	return out
}

func refs(names ...string) Message {
	return Message{Role: "tool", ToolCallID: "c", ToolReferences: names}
}

func TestRequestTools(t *testing.T) {
	tools := []ToolDefinition{tool("a", false), tool("x", true), tool("b", false), tool("y", true), tool("z", true)}
	tests := []struct {
		name     string
		messages []Message
		want     []string
	}{
		{"no references drops every deferred tool", []Message{{Role: "user", Content: "hi"}}, []string{"a", "b"}},
		{"referenced deferred tool included", []Message{refs("y")}, []string{"a", "b", "y"}},
		{"order of first reference, not tool order", []Message{refs("z"), refs("x", "z")}, []string{"a", "b", "z", "x"}},
		{"order within one result", []Message{refs("y", "x")}, []string{"a", "b", "y", "x"}},
		{"reference to a loaded or unknown tool adds nothing", []Message{refs("a", "nope")}, []string{"a", "b"}},
		{"references on any message count", []Message{{Role: "assistant", ToolReferences: []string{"x"}}}, []string{"a", "b", "x"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := RequestTools(tools, tt.messages)
			if !reflect.DeepEqual(names(got), tt.want) {
				t.Errorf("RequestTools = %v, want %v", names(got), tt.want)
			}
			for _, g := range got {
				if g.Deferred {
					t.Errorf("tool %q still Deferred", g.Function.Name)
				}
			}
		})
	}
	if !tools[1].Deferred || len(tools) != 5 {
		t.Error("input slice modified")
	}
}

// A conversation's tool list only grows at its end as references are added,
// so the earlier request's list is a prefix of the later one's.
func TestRequestTools_StableAcrossRequests(t *testing.T) {
	tools := []ToolDefinition{tool("a", false), tool("x", true), tool("y", true), tool("b", false)}
	first := names(RequestTools(tools, []Message{{Role: "user", Content: "hi"}, refs("y")}))
	second := names(RequestTools(tools, []Message{
		{Role: "user", Content: "hi"},
		refs("y"),
		{Role: "assistant", Content: "ok"},
		refs("x", "y"),
	}))
	if want := []string{"a", "b", "y"}; !reflect.DeepEqual(first, want) {
		t.Fatalf("first = %v, want %v", first, want)
	}
	if want := []string{"a", "b", "y", "x"}; !reflect.DeepEqual(second, want) {
		t.Fatalf("second = %v, want %v", second, want)
	}
}

func TestRequestTools_NothingDeferred(t *testing.T) {
	none := []ToolDefinition{tool("a", false), tool("b", false)}
	if got := RequestTools(none, []Message{refs("a")}); &got[0] != &none[0] || len(got) != 2 {
		t.Errorf("no deferred tools: want the input slice unchanged, got %v", got)
	}
	if got := RequestTools(nil, nil); got != nil {
		t.Errorf("RequestTools(nil) = %v, want nil", got)
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

func TestRequestTools_KeepsSearchTool(t *testing.T) {
	search := tool("search_tools", true)
	search.ToolSearch = true
	got := RequestTools([]ToolDefinition{search, tool("b", true)}, nil)
	if len(got) != 1 || got[0].Function.Name != "search_tools" {
		t.Errorf("RequestTools = %v, want the search tool only", names(got))
	}
}
