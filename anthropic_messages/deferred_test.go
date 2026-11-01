package anthropicmessages

import (
	"encoding/json"
	"reflect"
	"testing"
)

// requestJSON builds a request body and returns it decoded from JSON, as the
// API would see it.
func requestJSON(t *testing.T, messages []Message, tools []ToolDefinition, model string) map[string]any {
	t.Helper()
	body, err := buildRequestBody(messages, tools, model, map[string]any{"max_tokens": 100})
	if err != nil {
		t.Fatalf("buildRequestBody: %v", err)
	}
	data, err := json.Marshal(body)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var out map[string]any
	if err := json.Unmarshal(data, &out); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	return out
}

func deferredTools() []ToolDefinition {
	return []ToolDefinition{
		{Type: "function", Function: ToolFunctionDefinition{Name: "search_tools", Parameters: map[string]any{"type": "object"}}},
		{Type: "function", Function: ToolFunctionDefinition{Name: "get_weather", Parameters: map[string]any{"type": "object"}}, Deferred: true},
	}
}

func searchConversation(refs []string, content string) []Message {
	return []Message{
		{Role: "user", Content: "weather?"},
		{Role: "assistant", ToolCalls: []ToolCall{{ID: "c1", Name: "search_tools", Arguments: map[string]any{}}}},
		{Role: "tool", ToolCallID: "c1", Content: content, ToolReferences: refs},
	}
}

// toolResultContent returns the content of the tool_result block of the
// request's last message.
func toolResultContent(t *testing.T, body map[string]any) any {
	t.Helper()
	return dig(t, body, "messages", -1, "content", 0, "content")
}

func TestBuildRequestBody_DeferredToolsNative(t *testing.T) {
	body := requestJSON(t, searchConversation([]string{"get_weather"}, ""), deferredTools(), "claude-sonnet-4-6")
	tools := list(t, body["tools"])
	if len(tools) != 2 {
		t.Fatalf("tools = %d, want 2 (deferred tools are sent)", len(tools))
	}
	if dig(t, tools, 0, "defer_loading") != nil {
		t.Error("search_tools must not carry defer_loading")
	}
	if got := dig(t, tools, 1, "defer_loading"); got != true {
		t.Errorf("get_weather defer_loading = %v, want true", got)
	}
	want := []any{map[string]any{"type": "tool_reference", "tool_name": "get_weather"}}
	if got := toolResultContent(t, body); !reflect.DeepEqual(got, want) {
		t.Errorf("tool_result content = %v, want %v", got, want)
	}
}

func TestBuildRequestBody_ToolReferencesWithText(t *testing.T) {
	body := requestJSON(t, searchConversation([]string{"get_weather", "gone"}, "2 matches"), deferredTools(), "claude-opus-5-5")
	want := []any{
		// A reference to a tool not in the request is named in the text.
		map[string]any{"type": "text", "text": "2 matches\n\nTools found: gone"},
		map[string]any{"type": "tool_reference", "tool_name": "get_weather"},
	}
	if got := toolResultContent(t, body); !reflect.DeepEqual(got, want) {
		t.Errorf("tool_result content = %v, want %v", got, want)
	}
}

func TestBuildRequestBody_DeferredToolsUnsupportedModel(t *testing.T) {
	body := requestJSON(t, searchConversation([]string{"get_weather"}, ""), deferredTools(), "claude-3-7-sonnet-20250219")
	tools := list(t, body["tools"])
	if len(tools) != 1 || dig(t, tools, 0, "name") != "search_tools" {
		t.Fatalf("tools = %v, want only search_tools", tools)
	}
	if got := toolResultContent(t, body); got != "Tools found: get_weather" {
		t.Errorf("tool_result content = %v, want the references as text", got)
	}
}

func TestBuildRequestBody_PlainToolResultUnchanged(t *testing.T) {
	body := requestJSON(t, searchConversation(nil, "sunny"), deferredTools(), "claude-sonnet-4-6")
	if got := toolResultContent(t, body); got != "sunny" {
		t.Errorf("tool_result content = %v, want plain string", got)
	}
}

func TestProvider_SupportsDeferredTools(t *testing.T) {
	p := NewProvider("k", "")
	if !p.SupportsDeferredTools("claude-sonnet-4-6") || p.SupportsDeferredTools("claude-opus-4-1") {
		t.Error("SupportsDeferredTools does not follow the model rule")
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

func TestBuildRequestBody_SearchToolNeverDeferred(t *testing.T) {
	tools := deferredTools()
	tools[0].Deferred, tools[0].ToolSearch = true, true
	body := requestJSON(t, []Message{{Role: "user", Content: "hi"}}, tools, "claude-sonnet-4-6")
	if dig(t, body, "tools", 0, "defer_loading") != nil {
		t.Error("the tool-search tool must not be sent deferred")
	}
	if dig(t, body, "tools", 1, "defer_loading") != true {
		t.Error("get_weather must stay deferred")
	}
}
