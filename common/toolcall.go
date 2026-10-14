package common

import (
	"encoding/json"
	"strings"

	"github.com/PivotLLM/spawnllm/logger"
)

// A ToolCall carries its name and arguments in two places: Name/Arguments
// (set by the providers that parse a response into them) and
// Function.Name/Function.Arguments (the OpenAI wire form, and the only part
// that survives JSON, since Name and Arguments are not serialised). A call
// reloaded from persisted history therefore has only Function.*, and a call
// fresh from a response may have only Name/Arguments. Every request builder
// reads a call through the helpers below so either form works.

// ToolCallName returns the call's name: Name, else Function.Name.
func ToolCallName(tc ToolCall) string {
	if tc.Name != "" {
		return tc.Name
	}
	if tc.Function != nil {
		return tc.Function.Name
	}
	return ""
}

// ToolCallInput returns the call's arguments as an object: Arguments, else
// Function.Arguments parsed as a JSON object. It never returns nil: an absent,
// empty or unparsable argument string gives an empty object (Anthropic
// requires tool_use input to be an object); an unparsable one is logged at
// DEBUG.
func ToolCallInput(tc ToolCall) map[string]any {
	if len(tc.Arguments) > 0 {
		return tc.Arguments
	}
	if tc.Function != nil && strings.TrimSpace(tc.Function.Arguments) != "" {
		var parsed map[string]any
		if err := json.Unmarshal([]byte(tc.Function.Arguments), &parsed); err != nil {
			logger.DebugCF("common", "tool call arguments are not a JSON object; sending {}",
				map[string]any{"tool": ToolCallName(tc), "id": tc.ID, "error": err.Error()})
		} else if parsed != nil {
			return parsed
		}
	}
	return map[string]any{}
}

// ToolCallArgumentsJSON returns the call's arguments as a JSON object string
// for the OpenAI wire: Function.Arguments verbatim when set, else Arguments
// encoded, else "{}".
func ToolCallArgumentsJSON(tc ToolCall) string {
	if tc.Function != nil && strings.TrimSpace(tc.Function.Arguments) != "" {
		return tc.Function.Arguments
	}
	if len(tc.Arguments) > 0 {
		b, err := json.Marshal(tc.Arguments)
		if err == nil {
			return string(b)
		}
		logger.DebugCF("common", "tool call arguments are not JSON-encodable; sending {}",
			map[string]any{"tool": ToolCallName(tc), "id": tc.ID, "error": err.Error()})
	}
	return "{}"
}

// WireToolCalls returns copies of calls with Function filled in from
// Name/Arguments where it is missing, so a call serialises to the OpenAI wire
// form whichever form it was built in. The input slice is not modified.
func WireToolCalls(calls []ToolCall) []ToolCall {
	if len(calls) == 0 {
		return calls
	}
	out := make([]ToolCall, len(calls))
	for i, tc := range calls {
		out[i] = tc
		fn := FunctionCall{}
		if tc.Function != nil {
			fn = *tc.Function
		}
		fn.Name = ToolCallName(tc)
		fn.Arguments = ToolCallArgumentsJSON(tc)
		out[i].Function = &fn
		if out[i].Type == "" {
			out[i].Type = "function"
		}
	}
	return out
}
