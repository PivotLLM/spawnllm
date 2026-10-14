package common

import (
	"encoding/json"
	"reflect"
	"testing"
)

// roundTrip passes calls through JSON, as a host that persists history does:
// Name and Arguments are not serialised, so only Function.* survives.
func roundTrip(t *testing.T, calls []ToolCall) []ToolCall {
	t.Helper()
	b, err := json.Marshal(calls)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var out []ToolCall
	if err := json.Unmarshal(b, &out); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	return out
}

func TestToolCallNameAndInput(t *testing.T) {
	tests := []struct {
		name      string
		call      ToolCall
		wantName  string
		wantInput map[string]any
	}{
		{
			name:      "direct name and arguments",
			call:      ToolCall{ID: "c1", Name: "get_weather", Arguments: map[string]any{"city": "Paris"}},
			wantName:  "get_weather",
			wantInput: map[string]any{"city": "Paris"},
		},
		{
			name:      "function only",
			call:      ToolCall{ID: "c1", Function: &FunctionCall{Name: "get_weather", Arguments: `{"city":"Paris"}`}},
			wantName:  "get_weather",
			wantInput: map[string]any{"city": "Paris"},
		},
		{
			name:      "direct fields win over function",
			call:      ToolCall{Name: "a", Arguments: map[string]any{"x": "1"}, Function: &FunctionCall{Name: "b", Arguments: `{"x":"2"}`}},
			wantName:  "a",
			wantInput: map[string]any{"x": "1"},
		},
		{
			name:      "empty argument string",
			call:      ToolCall{Function: &FunctionCall{Name: "ping", Arguments: ""}},
			wantName:  "ping",
			wantInput: map[string]any{},
		},
		{
			name:      "invalid argument string",
			call:      ToolCall{Function: &FunctionCall{Name: "ping", Arguments: `{not json`}},
			wantName:  "ping",
			wantInput: map[string]any{},
		},
		{
			name:      "argument string is not an object",
			call:      ToolCall{Function: &FunctionCall{Name: "ping", Arguments: `[1,2]`}},
			wantName:  "ping",
			wantInput: map[string]any{},
		},
		{
			name:      "null argument string",
			call:      ToolCall{Function: &FunctionCall{Name: "ping", Arguments: `null`}},
			wantName:  "ping",
			wantInput: map[string]any{},
		},
		{
			name:      "nothing set",
			call:      ToolCall{ID: "c1"},
			wantName:  "",
			wantInput: map[string]any{},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := ToolCallName(tt.call); got != tt.wantName {
				t.Errorf("ToolCallName = %q, want %q", got, tt.wantName)
			}
			got := ToolCallInput(tt.call)
			if got == nil {
				t.Fatal("ToolCallInput returned nil, want an object")
			}
			if !reflect.DeepEqual(got, tt.wantInput) {
				t.Errorf("ToolCallInput = %v, want %v", got, tt.wantInput)
			}
		})
	}
}

func TestToolCallArgumentsJSON(t *testing.T) {
	tests := []struct {
		name string
		call ToolCall
		want string
	}{
		{"function arguments verbatim", ToolCall{Function: &FunctionCall{Arguments: `{"b":2,"a":1}`}}, `{"b":2,"a":1}`},
		{"encoded from arguments", ToolCall{Arguments: map[string]any{"a": 1}}, `{"a":1}`},
		{"blank function arguments fall back", ToolCall{Arguments: map[string]any{"a": 1}, Function: &FunctionCall{Arguments: " "}}, `{"a":1}`},
		{"nothing set", ToolCall{}, `{}`},
		{"unencodable arguments", ToolCall{Arguments: map[string]any{"f": func() {}}}, `{}`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := ToolCallArgumentsJSON(tt.call); got != tt.want {
				t.Errorf("ToolCallArgumentsJSON = %s, want %s", got, tt.want)
			}
		})
	}
}

func TestWireToolCalls_FillsFunctionWithoutMutatingInput(t *testing.T) {
	in := []ToolCall{
		{ID: "c1", Name: "get_weather", Arguments: map[string]any{"city": "Paris"}},
		{ID: "c2", Type: "function", Function: &FunctionCall{Name: "ping", Arguments: `{}`, ThoughtSignature: "sig"}},
	}
	out := WireToolCalls(in)

	if in[0].Function != nil {
		t.Error("input call was modified")
	}
	if out[0].Type != "function" || out[0].Function == nil ||
		out[0].Function.Name != "get_weather" || out[0].Function.Arguments != `{"city":"Paris"}` {
		t.Errorf("out[0] = %+v (function %+v)", out[0], out[0].Function)
	}
	if out[1].Function == in[1].Function {
		t.Error("output shares the input's Function pointer")
	}
	if out[1].Function.Name != "ping" || out[1].Function.ThoughtSignature != "sig" {
		t.Errorf("out[1].Function = %+v", out[1].Function)
	}
	if WireToolCalls(nil) != nil {
		t.Error("WireToolCalls(nil) != nil")
	}
}

func TestSerializeMessages_ToolCallsInEitherForm(t *testing.T) {
	direct := []ToolCall{{ID: "c1", Name: "get_weather", Arguments: map[string]any{"city": "Paris"}}}
	tests := []struct {
		name  string
		calls []ToolCall
	}{
		{"direct name and arguments", direct},
		{"reloaded from history", roundTrip(t, WireToolCalls(direct))},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			b, err := json.Marshal(SerializeMessages([]Message{{Role: "assistant", ToolCalls: tt.calls}}))
			if err != nil {
				t.Fatalf("marshal: %v", err)
			}
			var got []struct {
				ToolCalls []struct {
					ID       string `json:"id"`
					Type     string `json:"type"`
					Function struct {
						Name      string `json:"name"`
						Arguments string `json:"arguments"`
					} `json:"function"`
				} `json:"tool_calls"`
			}
			if err := json.Unmarshal(b, &got); err != nil {
				t.Fatalf("unmarshal: %v", err)
			}
			tc := got[0].ToolCalls[0]
			if tc.ID != "c1" || tc.Type != "function" || tc.Function.Name != "get_weather" ||
				tc.Function.Arguments != `{"city":"Paris"}` {
				t.Errorf("wire tool call = %+v, body %s", tc, b)
			}
		})
	}
}
