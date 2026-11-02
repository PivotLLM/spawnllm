package openai_responses

import (
	"context"
	"reflect"
	"testing"

	"github.com/PivotLLM/spawnllm/protocoltypes"
)

func TestSupportsToolSearch(t *testing.T) {
	tests := map[string]bool{
		"gpt-5.4":            true,
		"gpt-5.5-mini":       true,
		"GPT-5.10":           true,
		"gpt-6":              true,
		"gpt-5.4-2026-03-05": true,
		"gpt-5.3":            false,
		"gpt-5":              false,
		"gpt-5-mini":         false,
		"gpt-4o":             false,
		"gpt-4.1":            false,
		"o3":                 false,
		"openai/gpt-5.4":     false,
		"gpt-x":              false,
		"":                   false,
	}
	for model, want := range tests {
		if got := supportsToolSearch(model); got != want {
			t.Errorf("supportsToolSearch(%q) = %v, want %v", model, got, want)
		}
	}
	if !NewProvider("k", "https://api.openai.com/v1", "").SupportsDeferredTools("gpt-5.4") {
		t.Error("SupportsDeferredTools must follow supportsToolSearch")
	}
}

var schema = map[string]any{"type": "object"}

func searchTools(withSearchFlag bool) []ToolDefinition {
	return []ToolDefinition{
		{Type: "function", Function: protocoltypes.ToolFunctionDefinition{Name: "search_tools", Description: "Find tools", Parameters: schema}, ToolSearch: withSearchFlag},
		{Type: "function", Function: protocoltypes.ToolFunctionDefinition{Name: "read_file", Description: "Read", Parameters: schema}},
		{Type: "function", Function: protocoltypes.ToolFunctionDefinition{Name: "get_weather", Description: "Weather", Parameters: schema}, Deferred: true},
	}
}

func searchHistory() []Message {
	return []Message{
		{Role: "user", Content: "weather?"},
		{Role: "assistant", ToolCalls: []ToolCall{{ID: "call_1", Name: "search_tools", Arguments: map[string]any{"goal": "weather"}}}},
		{Role: "tool", ToolCallID: "call_1", Content: "1 match", ToolReferences: []string{"get_weather", "unknown"}},
	}
}

func TestChat_ClientToolSearch(t *testing.T) {
	p, body := captureServer(t, `{"status":"completed","output":[]}`)
	if _, err := p.Chat(context.Background(), searchHistory(), searchTools(true), "gpt-5.4", nil); err != nil {
		t.Fatalf("Chat error: %v", err)
	}

	wantTools := []any{
		map[string]any{"type": "tool_search", "execution": "client", "description": "Find tools", "parameters": schema},
		map[string]any{"type": "function", "name": "read_file", "description": "Read", "parameters": schema},
		map[string]any{"type": "function", "name": "get_weather", "description": "Weather", "parameters": schema, "defer_loading": true},
	}
	if got := (*body)["tools"]; !reflect.DeepEqual(got, wantTools) {
		t.Errorf("tools =\n%v\nwant\n%v", got, wantTools)
	}

	input := asType[[]any](t, (*body)["input"], "input")
	if len(input) != 4 {
		t.Fatalf("input = %v, want 4 items", input)
	}
	wantCall := map[string]any{
		"type": "tool_search_call", "execution": "client", "call_id": "call_1",
		"status": "completed", "arguments": map[string]any{"goal": "weather"},
	}
	if !reflect.DeepEqual(input[1], wantCall) {
		t.Errorf("input[1] = %v, want %v", input[1], wantCall)
	}
	wantOutput := map[string]any{
		"type": "tool_search_output", "execution": "client", "call_id": "call_1", "status": "completed",
		"tools": []any{
			map[string]any{"type": "function", "name": "get_weather", "description": "Weather", "parameters": schema, "defer_loading": true},
		},
	}
	if !reflect.DeepEqual(input[2], wantOutput) {
		t.Errorf("input[2] = %v, want %v", input[2], wantOutput)
	}
	// The result's text follows as a developer message, with the reference to
	// the tool not in the request named in it.
	wantText := map[string]any{"role": "developer", "content": "search_tools result: 1 match\n\nTools found: unknown"}
	if !reflect.DeepEqual(input[3], wantText) {
		t.Errorf("input[3] = %v, want %v", input[3], wantText)
	}
}

// A search result without text whose references are all in the request is
// the tool_search_output alone.
func TestChat_SearchResultWithoutTextHasNoDeveloperMessage(t *testing.T) {
	history := searchHistory()
	history[2].Content, history[2].ToolReferences = "", []string{"get_weather"}
	p, body := captureServer(t, `{"status":"completed","output":[]}`)
	if _, err := p.Chat(context.Background(), history, searchTools(true), "gpt-5.4", nil); err != nil {
		t.Fatalf("Chat error: %v", err)
	}
	input := asType[[]any](t, (*body)["input"], "input")
	if len(input) != 3 {
		t.Fatalf("input = %v, want 3 items (no developer message)", input)
	}
	if got := asType[map[string]any](t, input[2], "input[2]")["type"]; got != "tool_search_output" {
		t.Errorf("input[2] type = %v, want tool_search_output", got)
	}
}

// An empty search with an error text: the tool_search_output has no tools and
// the error reaches the model as the developer message.
func TestChat_SearchErrorTextReachesModel(t *testing.T) {
	history := searchHistory()
	history[2].Content, history[2].ToolReferences = "Error: invalid regex", nil
	p, body := captureServer(t, `{"status":"completed","output":[]}`)
	if _, err := p.Chat(context.Background(), history, searchTools(true), "gpt-5.4", nil); err != nil {
		t.Fatalf("Chat error: %v", err)
	}
	input := asType[[]any](t, (*body)["input"], "input")
	if len(input) != 4 {
		t.Fatalf("input = %v, want 4 items", input)
	}
	if got := asType[map[string]any](t, input[2], "input[2]")["tools"]; !reflect.DeepEqual(got, []any{}) {
		t.Errorf("tool_search_output tools = %v, want empty", got)
	}
	wantText := map[string]any{"role": "developer", "content": "search_tools result: Error: invalid regex"}
	if !reflect.DeepEqual(input[3], wantText) {
		t.Errorf("input[3] = %v, want %v", input[3], wantText)
	}
}

func TestChat_ToolSearchCallReportedAsToolCall(t *testing.T) {
	reply := `{"status":"completed","output":[{"type":"tool_search_call","execution":"client","call_id":"call_9","status":"completed","arguments":{"goal":"weather"}}]}`
	p, _ := captureServer(t, reply)
	out, err := p.Chat(context.Background(), []Message{{Role: "user", Content: "weather?"}}, searchTools(true), "gpt-5.4", nil)
	if err != nil {
		t.Fatalf("Chat error: %v", err)
	}
	if len(out.ToolCalls) != 1 {
		t.Fatalf("ToolCalls = %v, want 1", out.ToolCalls)
	}
	tc := out.ToolCalls[0]
	if tc.ID != "call_9" || tc.Function == nil || tc.Function.Name != "search_tools" || tc.Function.Arguments != `{"goal":"weather"}` {
		t.Errorf("tool call = %+v / %+v", tc, tc.Function)
	}
	if out.FinishReason != "tool_calls" {
		t.Errorf("FinishReason = %q, want tool_calls", out.FinishReason)
	}
}

func TestChat_ToolSearchNotUsed(t *testing.T) {
	tests := []struct {
		name      string
		model     string
		tools     []ToolDefinition
		wantNames []string
	}{
		// Older model: the referenced deferred tool is sent as an ordinary
		// function after the others; the search tool is a function.
		{"model before gpt-5.4", "gpt-5", searchTools(true), []string{"search_tools", "read_file", "get_weather"}},
		// No tool marked as the search tool: nothing to send as tool_search.
		{"no search tool", "gpt-5.4", searchTools(false), []string{"search_tools", "read_file", "get_weather"}},
		// Nothing deferred: plain functions.
		{"nothing deferred", "gpt-5.4", searchTools(true)[:2], []string{"search_tools", "read_file"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p, body := captureServer(t, `{"status":"completed","output":[]}`)
			if _, err := p.Chat(context.Background(), searchHistory(), tt.tools, tt.model, nil); err != nil {
				t.Fatalf("Chat error: %v", err)
			}
			sent := asType[[]any](t, (*body)["tools"], "tools")
			names := make([]string, 0, len(sent))
			for i, s := range sent {
				tool := asType[map[string]any](t, s, "tool")
				if tool["type"] != "function" {
					t.Errorf("tools[%d] type = %v, want function", i, tool["type"])
				}
				if _, set := tool["defer_loading"]; set {
					t.Errorf("tools[%d] carries defer_loading", i)
				}
				names = append(names, asType[string](t, tool["name"], "name"))
			}
			if !reflect.DeepEqual(names, tt.wantNames) {
				t.Errorf("tools = %v, want %v", names, tt.wantNames)
			}
			input := asType[[]any](t, (*body)["input"], "input")
			call := asType[map[string]any](t, input[1], "input[1]")
			result := asType[map[string]any](t, input[2], "input[2]")
			if call["type"] != "function_call" || result["type"] != "function_call_output" {
				t.Errorf("history items = %v / %v, want function_call / function_call_output", call["type"], result["type"])
			}
			if result["output"] != "1 match\n\nTools found: get_weather, unknown" {
				t.Errorf("output = %v", result["output"])
			}
		})
	}
}

// A result from a tool other than the search tool (here get_tool_details)
// that references deferred tools is followed by an additional_tools item
// loading the ones no earlier item loaded; a reference to a tool the request
// lacks stays in the result's text.
func TestChat_ReferencesOutsideSearchResultLoadAdditionalTools(t *testing.T) {
	tools := append(searchTools(true),
		ToolDefinition{Type: "function", Function: protocoltypes.ToolFunctionDefinition{Name: "get_tool_details", Description: "Details", Parameters: schema}},
		ToolDefinition{Type: "function", Function: protocoltypes.ToolFunctionDefinition{Name: "get_forecast", Description: "Forecast", Parameters: schema}, Deferred: true},
	)
	history := append(searchHistory(),
		Message{Role: "assistant", ToolCalls: []ToolCall{{ID: "call_2", Name: "get_tool_details", Arguments: map[string]any{"name": "get_forecast"}}}},
		// get_weather was loaded by the search result, read_file is not
		// deferred, unknown is not in the request: only get_forecast loads.
		Message{Role: "tool", ToolCallID: "call_2", Content: "schema", ToolReferences: []string{"get_forecast", "get_weather", "read_file", "unknown"}},
	)
	p, body := captureServer(t, `{"status":"completed","output":[]}`)
	if _, err := p.Chat(context.Background(), history, tools, "gpt-5.4", nil); err != nil {
		t.Fatalf("Chat error: %v", err)
	}
	input := asType[[]any](t, (*body)["input"], "input")
	if len(input) != 7 {
		t.Fatalf("input = %v, want 7 items", input)
	}
	wantOutput := map[string]any{"type": "function_call_output", "call_id": "call_2", "output": "schema\n\nTools found: unknown"}
	if !reflect.DeepEqual(input[5], wantOutput) {
		t.Errorf("input[5] = %v, want %v", input[5], wantOutput)
	}
	wantAdditional := map[string]any{
		"type": "additional_tools", "role": "developer",
		"tools": []any{
			map[string]any{"type": "function", "name": "get_forecast", "description": "Forecast", "parameters": schema, "defer_loading": true},
		},
	}
	if !reflect.DeepEqual(input[6], wantAdditional) {
		t.Errorf("input[6] = %v, want %v", input[6], wantAdditional)
	}
}

// Without deferred references there is no additional_tools item.
func TestChat_NoAdditionalToolsWithoutDeferredReferences(t *testing.T) {
	history := append(searchHistory(),
		Message{Role: "assistant", ToolCalls: []ToolCall{{ID: "call_2", Name: "read_file", Arguments: map[string]any{}}}},
		Message{Role: "tool", ToolCallID: "call_2", Content: "text", ToolReferences: []string{"get_weather"}},
	)
	p, body := captureServer(t, `{"status":"completed","output":[]}`)
	if _, err := p.Chat(context.Background(), history, searchTools(true), "gpt-5.4", nil); err != nil {
		t.Fatalf("Chat error: %v", err)
	}
	input := asType[[]any](t, (*body)["input"], "input")
	if len(input) != 6 {
		t.Fatalf("input = %v, want 6 items", input)
	}
	if got := asType[map[string]any](t, input[5], "input[5]")["output"]; got != "text" {
		t.Errorf("output = %v, want the plain text (get_weather is already loaded)", got)
	}
}

// A replayed call to a deferred tool carries the tool's namespace, which is
// its own name; a call to any other tool has none.
func TestChat_DeferredToolCallCarriesNamespace(t *testing.T) {
	history := append(searchHistory(),
		Message{Role: "assistant", ToolCalls: []ToolCall{
			{ID: "call_2", Name: "get_weather", Arguments: map[string]any{"city": "Ottawa"}},
			{ID: "call_3", Name: "read_file", Arguments: map[string]any{}},
		}},
		Message{Role: "tool", ToolCallID: "call_2", Content: "4C"},
		Message{Role: "tool", ToolCallID: "call_3", Content: "text"},
	)
	p, body := captureServer(t, `{"status":"completed","output":[]}`)
	if _, err := p.Chat(context.Background(), history, searchTools(true), "gpt-5.4", nil); err != nil {
		t.Fatalf("Chat error: %v", err)
	}
	input := asType[[]any](t, (*body)["input"], "input")
	if len(input) != 8 {
		t.Fatalf("input = %v, want 8 items", input)
	}
	wantDeferred := map[string]any{"type": "function_call", "call_id": "call_2", "name": "get_weather", "namespace": "get_weather", "arguments": `{"city":"Ottawa"}`}
	if !reflect.DeepEqual(input[4], wantDeferred) {
		t.Errorf("input[4] = %v, want %v", input[4], wantDeferred)
	}
	wantPlain := map[string]any{"type": "function_call", "call_id": "call_3", "name": "read_file", "arguments": `{}`}
	if !reflect.DeepEqual(input[5], wantPlain) {
		t.Errorf("input[5] = %v, want %v", input[5], wantPlain)
	}
}

// Without tool search (older model) a call to a deferred tool has no namespace.
func TestChat_NoNamespaceWithoutToolSearch(t *testing.T) {
	history := append(searchHistory(),
		Message{Role: "assistant", ToolCalls: []ToolCall{{ID: "call_2", Name: "get_weather", Arguments: map[string]any{}}}},
		Message{Role: "tool", ToolCallID: "call_2", Content: "4C"},
	)
	p, body := captureServer(t, `{"status":"completed","output":[]}`)
	if _, err := p.Chat(context.Background(), history, searchTools(true), "gpt-5.3", nil); err != nil {
		t.Fatalf("Chat error: %v", err)
	}
	input := asType[[]any](t, (*body)["input"], "input")
	for _, it := range input {
		if m, ok := it.(map[string]any); ok && m["name"] == "get_weather" {
			if _, has := m["namespace"]; has {
				t.Errorf("function_call carries a namespace without tool search: %v", m)
			}
			return
		}
	}
	t.Fatal("no function_call to get_weather in input")
}
