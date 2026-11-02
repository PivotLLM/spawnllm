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
// without one, deferred tools are handled as on any other provider
// (common.RequestTools).
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
// (to return full definitions in tool_search_output), the call ids of search
// calls seen in the history and the names of the deferred tools an earlier
// item in the input has loaded.
type toolSearch struct {
	name   string
	defs   map[string]ToolDefinition
	calls  map[string]bool
	loaded map[string]bool
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
	return &toolSearch{name: name, defs: defs, calls: map[string]bool{}, loaded: map[string]bool{}}
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

// namespace is the namespace of a replayed call to the named tool: a tool
// loaded by search is called in a namespace of its own name, which the model's
// function_call carries and the API requires back on replay ("Missing
// namespace for function_call"); any other tool is in the default namespace,
// and the result is "".
func (s *toolSearch) namespace(name string) string {
	if s == nil {
		return ""
	}
	if def, ok := s.defs[name]; ok && common.IsDeferred(def) {
		return name
	}
	return ""
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

// outputItems are the input items for a search result: the tool_search_output
// holding the full definitions of the referenced tools that are in the request
// and, when the result has text, a developer message after it carrying that
// text, named as the search tool's result. The tool_search_output has no place
// for text, and the model would otherwise never see an error from the search
// tool. A reference to a tool not in the request is named in the text.
func (s *toolSearch) outputItems(m Message) []any {
	tools := make([]map[string]any, 0, len(m.ToolReferences))
	var rest []string
	for _, name := range m.ToolReferences {
		def, ok := s.defs[name]
		if !ok || def.ToolSearch {
			rest = append(rest, name)
			continue
		}
		tools = append(tools, responsesFunction(def, true))
		s.loaded[name] = true
	}
	items := []any{map[string]any{
		"type":      "tool_search_output",
		"execution": "client",
		"call_id":   m.ToolCallID,
		"status":    "completed",
		"tools":     tools,
	}}
	if text := common.WithToolsFound(m.Content, rest); text != "" {
		items = append(items, map[string]any{
			"role":    "developer",
			"content": s.name + " result: " + text,
		})
	}
	return items
}

// additionalToolsItem is the additional_tools input item that follows the
// function_call_output of a result from a tool other than the search tool
// (e.g. one showing a tool's details) when the result references deferred
// tools: only a tool_search_output or additional_tools item makes a deferred
// tool callable. It holds the referenced deferred tools no earlier item has
// loaded, and is nil when there are none. The names of referenced tools that
// are in the request are removed from m.ToolReferences, so that only a
// reference to a tool the request lacks is named in the result's text.
func (s *toolSearch) additionalToolsItem(m *Message) map[string]any {
	var tools []map[string]any
	var rest []string
	for _, name := range m.ToolReferences {
		def, ok := s.defs[name]
		if !ok {
			rest = append(rest, name)
			continue
		}
		if common.IsDeferred(def) && !s.loaded[name] {
			tools = append(tools, responsesFunction(def, true))
			s.loaded[name] = true
		}
	}
	m.ToolReferences = rest
	if len(tools) == 0 {
		return nil
	}
	return map[string]any{
		"type":  "additional_tools",
		"role":  "developer",
		"tools": tools,
	}
}
