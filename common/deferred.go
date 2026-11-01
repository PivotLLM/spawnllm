package common

import (
	"strconv"
	"strings"
)

// IsDeferred reports whether a tool is sent deferred: Deferred is set and it
// is not the host's tool-search tool, which is never deferred.
func IsDeferred(t ToolDefinition) bool {
	return t.Deferred && !t.ToolSearch
}

// RequestTools returns the request tool list for a provider that cannot load
// deferred tools natively: the tools that are not deferred, in the order
// given, then each deferred tool whose name appears in the ToolReferences of a
// message in messages, in order of first reference, sent as an ordinary tool
// (Deferred cleared). Every other deferred tool is left out. A tool the host's
// search found therefore stays callable, and the list only grows at its end as
// the history gains references, which keeps a cached prompt prefix valid. The
// input slice is returned unchanged when nothing is deferred.
func RequestTools(tools []ToolDefinition, messages []Message) []ToolDefinition {
	deferred := make(map[string]int)
	for i, t := range tools {
		if IsDeferred(t) {
			if _, seen := deferred[t.Function.Name]; !seen {
				deferred[t.Function.Name] = i
			}
		}
	}
	if len(deferred) == 0 {
		return tools
	}
	out := make([]ToolDefinition, 0, len(tools)-len(deferred))
	for _, t := range tools {
		if !IsDeferred(t) {
			out = append(out, t)
		}
	}
	for _, m := range messages {
		for _, name := range m.ToolReferences {
			i, ok := deferred[name]
			if !ok {
				continue
			}
			delete(deferred, name)
			t := tools[i]
			t.Deferred = false
			out = append(out, t)
		}
	}
	return out
}

// ToolResultText returns a message's content with its tool references, if any,
// appended as text: the form of a reference-carrying tool result for a
// provider that cannot load deferred tools natively.
func ToolResultText(m Message) string {
	return WithToolsFound(m.Content, m.ToolReferences)
}

// WithToolsFound appends a "Tools found: a, b" line naming tools to content.
// It returns content unchanged when tools is empty.
func WithToolsFound(content string, tools []string) string {
	if len(tools) == 0 {
		return content
	}
	line := "Tools found: " + strings.Join(tools, ", ")
	if content == "" {
		return line
	}
	return content + "\n\n" + line
}

// AnthropicSupportsDeferredTools reports whether an Anthropic model id names a
// model that accepts defer_loading tools and tool_reference blocks: Claude 4.5
// and later (Opus, Sonnet, Haiku 4.5; every 5.x model). Claude 4.1 and earlier
// do not.
//
// The id is read from its "claude-" part, so prefixed forms
// ("anthropic/claude-...", "us.anthropic.claude-...") work, and dots count as
// hyphens ("claude-sonnet-4.6"). After an optional family word come the major
// and minor version ("claude-opus-4-5-20251101" is 4.5); a date in the minor
// position means no minor ("claude-sonnet-4-20250514" is 4.0), and the older
// version-first form ("claude-3-5-sonnet-...") is read the same way (3.5). An
// id that does not parse is reported as unsupported.
func AnthropicSupportsDeferredTools(model string) bool {
	m := strings.ToLower(strings.ReplaceAll(model, ".", "-"))
	_, rest, ok := strings.Cut(m, "claude-")
	if !ok {
		return false
	}
	parts := strings.Split(rest, "-")
	if len(parts) > 0 {
		if _, err := strconv.Atoi(parts[0]); err != nil {
			parts = parts[1:] // family word
		}
	}
	if len(parts) == 0 {
		return false
	}
	major, err := strconv.Atoi(parts[0])
	if err != nil {
		return false
	}
	minor := 0
	if len(parts) > 1 && len(parts[1]) <= 2 {
		if n, err := strconv.Atoi(parts[1]); err == nil {
			minor = n
		}
	}
	return major > 4 || (major == 4 && minor >= 5)
}
