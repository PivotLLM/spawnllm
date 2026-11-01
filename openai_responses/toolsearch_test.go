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
	if len(input) != 3 {
		t.Fatalf("input = %v, want 3 items", input)
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
		// Older model: deferred tools left out, the search tool is a function.
		{"model before gpt-5.4", "gpt-5", searchTools(true), []string{"search_tools", "read_file"}},
		// No tool marked as the search tool: nothing to send as tool_search.
		{"no search tool", "gpt-5.4", searchTools(false), []string{"search_tools", "read_file"}},
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
