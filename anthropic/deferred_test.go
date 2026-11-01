package anthropicprovider

import (
	"reflect"
	"testing"
)

func deferredTools() []ToolDefinition {
	return []ToolDefinition{
		{Type: "function", Function: ToolFunctionDefinition{Name: "search_tools", Parameters: map[string]any{"type": "object"}}},
		{Type: "function", Function: ToolFunctionDefinition{Name: "get_weather", Parameters: map[string]any{"type": "object"}}, Deferred: true},
	}
}

// withUnreferencedTool returns deferredTools plus read_file, a deferred tool
// no conversation here references.
func withUnreferencedTool() []ToolDefinition {
	readFile := ToolDefinition{
		Type:     "function",
		Function: ToolFunctionDefinition{Name: "read_file", Parameters: map[string]any{"type": "object"}},
		Deferred: true,
	}
	return append(deferredTools(), readFile)
}

func searchConversation(refs []string, content string) []Message {
	return []Message{
		{Role: "user", Content: "weather?"},
		{Role: "assistant", ToolCalls: []ToolCall{{ID: "c1", Name: "search_tools", Arguments: map[string]any{}}}},
		{Role: "tool", ToolCallID: "c1", Content: content, ToolReferences: refs},
	}
}

func buildJSON(t *testing.T, messages []Message, model string) map[string]any {
	t.Helper()
	params, err := buildParams(messages, deferredTools(), model, map[string]any{"max_tokens": 100})
	if err != nil {
		t.Fatalf("buildParams: %v", err)
	}
	return paramsJSON(t, params)
}

func lastToolResultContent(t *testing.T, body map[string]any) any {
	t.Helper()
	return dig(t, body, "messages", -1, "content", 0, "content")
}

func TestBuildParams_DeferredToolsNative(t *testing.T) {
	body := buildJSON(t, searchConversation([]string{"get_weather", "gone"}, "2 matches"), "claude-sonnet-4.6")
	tools := list(t, body["tools"])
	if len(tools) != 2 {
		t.Fatalf("tools = %d, want 2", len(tools))
	}
	if dig(t, tools, 0, "defer_loading") != nil {
		t.Error("search_tools must not carry defer_loading")
	}
	if got := dig(t, tools, 1, "defer_loading"); got != true {
		t.Errorf("get_weather defer_loading = %v, want true", got)
	}
	want := []any{
		map[string]any{"type": "text", "text": "2 matches\n\nTools found: gone"},
		map[string]any{"type": "tool_reference", "tool_name": "get_weather"},
	}
	if got := lastToolResultContent(t, body); !reflect.DeepEqual(got, want) {
		t.Errorf("tool_result content = %v, want %v", got, want)
	}
}

func TestBuildParams_DeferredToolsUnsupportedModel(t *testing.T) {
	// read_file is deferred and never referenced: left out. get_weather was
	// referenced: sent after the loaded tools without defer_loading.
	tools := withUnreferencedTool()
	params, err := buildParams(searchConversation([]string{"get_weather"}, ""), tools,
		"claude-opus-4-1-20250805", map[string]any{"max_tokens": 100})
	if err != nil {
		t.Fatalf("buildParams: %v", err)
	}
	body := paramsJSON(t, params)
	sent := list(t, body["tools"])
	if len(sent) != 2 || dig(t, sent, 0, "name") != "search_tools" || dig(t, sent, 1, "name") != "get_weather" {
		t.Fatalf("tools = %v, want search_tools, get_weather", sent)
	}
	if dig(t, sent, 1, "defer_loading") != nil {
		t.Error("get_weather must not carry defer_loading")
	}
	want := []any{map[string]any{"type": "text", "text": "Tools found: get_weather"}}
	if got := lastToolResultContent(t, body); !reflect.DeepEqual(got, want) {
		t.Errorf("tool_result content = %v, want %v", got, want)
	}
}

func TestProvider_SupportsDeferredTools(t *testing.T) {
	p := NewProvider("k")
	if !p.SupportsDeferredTools("claude-opus-5-5") || p.SupportsDeferredTools("claude-3-5-haiku-latest") {
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

// On a native model every deferred tool is sent, referenced or not, in the
// caller's order and with defer_loading: the history does not change the list.
func TestBuildParams_NativeSendsUnreferencedDeferredTools(t *testing.T) {
	tools := withUnreferencedTool()
	params, err := buildParams(searchConversation([]string{"get_weather"}, ""), tools,
		"claude-sonnet-4.6", map[string]any{"max_tokens": 100})
	if err != nil {
		t.Fatalf("buildParams: %v", err)
	}
	sent := list(t, paramsJSON(t, params)["tools"])
	if len(sent) != 3 || dig(t, sent, 1, "name") != "get_weather" || dig(t, sent, 2, "name") != "read_file" {
		t.Fatalf("tools = %v, want search_tools, get_weather, read_file", sent)
	}
	if dig(t, sent, 1, "defer_loading") != true || dig(t, sent, 2, "defer_loading") != true {
		t.Error("deferred tools must carry defer_loading on a native model")
	}
}
