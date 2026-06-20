package spawnllm

import (
	"encoding/json"
	"testing"
)

func TestNormalizeToolCall_NameFromFunction(t *testing.T) {
	tc := ToolCall{
		ID:   "call_1",
		Type: "function",
		Function: &FunctionCall{
			Name:      "get_weather",
			Arguments: `{"location":"NYC"}`,
		},
	}
	result := NormalizeToolCall(tc)
	if result.Name != "get_weather" {
		t.Errorf("Name = %q, want %q", result.Name, "get_weather")
	}
	if result.Function.Name != "get_weather" {
		t.Errorf("Function.Name = %q, want %q", result.Function.Name, "get_weather")
	}
	if result.Arguments["location"] != "NYC" {
		t.Errorf("Arguments[location] = %v, want NYC", result.Arguments["location"])
	}
}

func TestNormalizeToolCall_ArgumentsFromFunctionString(t *testing.T) {
	tc := ToolCall{
		ID:   "call_2",
		Type: "function",
		Name: "list_files",
		Function: &FunctionCall{
			Name:      "list_files",
			Arguments: `{"path":"/tmp"}`,
		},
	}
	result := NormalizeToolCall(tc)
	if result.Arguments["path"] != "/tmp" {
		t.Errorf("Arguments[path] = %v, want /tmp", result.Arguments["path"])
	}
}

func TestNormalizeToolCall_NilFunctionCreated(t *testing.T) {
	tc := ToolCall{
		ID:        "call_3",
		Type:      "function",
		Name:      "read_file",
		Arguments: map[string]any{"path": "foo.txt"},
	}
	result := NormalizeToolCall(tc)
	if result.Function == nil {
		t.Fatal("Function should not be nil after normalization")
	}
	if result.Function.Name != "read_file" {
		t.Errorf("Function.Name = %q, want %q", result.Function.Name, "read_file")
	}
	// Arguments should be encoded back into Function.Arguments
	var parsed map[string]any
	if err := json.Unmarshal([]byte(result.Function.Arguments), &parsed); err != nil {
		t.Fatalf("Function.Arguments parse error: %v", err)
	}
	if parsed["path"] != "foo.txt" {
		t.Errorf("Function.Arguments path = %v, want foo.txt", parsed["path"])
	}
}

func TestNormalizeToolCall_NilArgumentsInitialized(t *testing.T) {
	tc := ToolCall{
		ID:   "call_4",
		Name: "no_args_tool",
	}
	result := NormalizeToolCall(tc)
	if result.Arguments == nil {
		t.Error("Arguments should not be nil after normalization")
	}
}

func TestNormalizeToolCall_ExistingArgumentsPreserved(t *testing.T) {
	tc := ToolCall{
		ID:        "call_5",
		Name:      "search",
		Arguments: map[string]any{"query": "golang"},
		Function: &FunctionCall{
			Name:      "search",
			Arguments: `{"query":"old"}`, // should not override existing Arguments
		},
	}
	result := NormalizeToolCall(tc)
	// Since Arguments is already set, it should not be overwritten from Function.Arguments
	if result.Arguments["query"] != "golang" {
		t.Errorf("Arguments[query] = %v, want golang (pre-set arguments should be preserved)", result.Arguments["query"])
	}
}

func TestNormalizeToolCall_FunctionNameFallback(t *testing.T) {
	// Name is empty, Function.Name should be copied to top-level Name
	tc := ToolCall{
		ID:   "call_6",
		Type: "function",
		Function: &FunctionCall{
			Name:      "calculate",
			Arguments: `{}`,
		},
	}
	result := NormalizeToolCall(tc)
	if result.Name != "calculate" {
		t.Errorf("Name = %q, want calculate", result.Name)
	}
}

func TestNormalizeToolCall_InvalidFunctionArguments(t *testing.T) {
	// Invalid JSON in Function.Arguments should not crash
	tc := ToolCall{
		ID:   "call_7",
		Name: "tool",
		Function: &FunctionCall{
			Name:      "tool",
			Arguments: `not-valid-json`,
		},
	}
	result := NormalizeToolCall(tc)
	// Arguments should be initialized to empty map, not panic
	if result.Arguments == nil {
		t.Error("Arguments should not be nil")
	}
}
