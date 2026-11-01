package openai_responses

import (
	"encoding/json"
	"strconv"
	"strings"

	"github.com/PivotLLM/spawnllm/common"
)

// SupportsDeferredTools implements spawnllm.DeferredToolsCapable. Deferred
// tools load natively through client-executed tool search, which the
// Responses API offers on gpt-5.4 and later (supportsToolSearch). The request
// must also carry the host's tool-search tool (ToolDefinition.ToolSearch);
// without one, deferred tools are left out as on any other provider.
func (p *Provider) SupportsDeferredTools(model string) bool {
	return supportsToolSearch(model)
}

// supportsToolSearch reports whether a model id is gpt-5.4 or later: "gpt-"
// then a major version and an optional ".minor" ("gpt-5.4", "gpt-5.5-mini",
// "gpt-6"), compared numerically. Only a bare OpenAI id counts: a namespaced
// one ("openai/gpt-5.4" on OpenRouter) and every other model do not.
func supportsToolSearch(model string) bool {
	rest, ok := strings.CutPrefix(strings.ToLower(strings.TrimSpace(model)), "gpt-")
	if !ok {
		return false
	}
	if i := strings.Index(rest, "-"); i >= 0 {
		rest = rest[:i]
	}
	majorStr, minorStr, hasMinor := strings.Cut(rest, ".")
	major, err := strconv.Atoi(majorStr)
	if err != nil {
		return false
	}
	minor := 0
	if hasMinor {
		if minor, err = strconv.Atoi(minorStr); err != nil {
			return false
		}
	}
	return major > 5 || (major == 5 && minor >= 4)
}

// toolSearch is the client-executed tool search of one request: the host's
// tool-search tool (sent as the tool_search tool), the request's tools by name
// (to return full definitions in tool_search_output) and the call ids of
// search calls seen in the history.
type toolSearch struct {
	name  string
	defs  map[string]ToolDefinition
	calls map[string]bool
}

// newToolSearch returns the tool search for a request, or nil when it is not
// used: the model does not support it, no tool is deferred, or there is no
// tool-search tool to send as the tool_search tool.
func newToolSearch(tools []ToolDefinition, model string) *toolSearch {
	if !supportsToolSearch(model) {
		return nil
	}
	name, deferred := "", false
	for _, t := range tools {
		if t.ToolSearch && name == "" {
			name = t.Function.Name
		}
		if common.IsDeferred(t) {
			deferred = true
		}
	}
	if name == "" || !deferred {
		return nil
	}
	defs := make(map[string]ToolDefinition, len(tools))
	for _, t := range tools {
		defs[t.Function.Name] = t
	}
	return &toolSearch{name: name, defs: defs, calls: map[string]bool{}}
}

// toolName is the tool-search tool's name, "" when search is not used.
func (s *toolSearch) toolName() string {
	if s == nil {
		return ""
	}
	return s.name
}

// matches reports whether tc calls the tool-search tool.
func (s *toolSearch) matches(tc ToolCall) bool {
	return s != nil && common.ToolCallName(tc) == s.name
}

// isCall reports whether callID is a tool-search call seen in the history.
func (s *toolSearch) isCall(callID string) bool {
	return s != nil && callID != "" && s.calls[callID]
}

// callItem is the tool_search_call input item replaying a search call, and
// records its id so its result becomes a tool_search_output.
func (s *toolSearch) callItem(tc ToolCall) map[string]any {
	s.calls[tc.ID] = true
	args := common.ToolCallArgumentsJSON(tc)
	if !json.Valid([]byte(args)) {
		args = "{}"
	}
	return map[string]any{
		"type":      "tool_search_call",
		"execution": "client",
		"call_id":   tc.ID,
		"status":    "completed",
		"arguments": json.RawMessage(args),
	}
}

// outputItem is the tool_search_output input item for a search result: the
// full definitions of the referenced tools that are in the request. The item
// has no place for text, so the result's content and any reference to a tool
// not in the request are not sent.
func (s *toolSearch) outputItem(m Message) map[string]any {
	tools := make([]map[string]any, 0, len(m.ToolReferences))
	for _, name := range m.ToolReferences {
		if def, ok := s.defs[name]; ok && !def.ToolSearch {
			tools = append(tools, responsesFunction(def, true))
		}
	}
	return map[string]any{
		"type":      "tool_search_output",
		"execution": "client",
		"call_id":   m.ToolCallID,
		"status":    "completed",
		"tools":     tools,
	}
}
