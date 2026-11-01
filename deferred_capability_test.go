package spawnllm

import (
	"strings"
	"testing"

	anthropicmessages "github.com/PivotLLM/spawnllm/anthropic_messages"
)

func TestDeferredToolsCapability(t *testing.T) {
	var p LLMProvider = anthropicmessages.NewProvider("k", "")
	c, ok := p.(DeferredToolsCapable)
	if !ok || !c.SupportsDeferredTools("claude-sonnet-4-6") || c.SupportsDeferredTools("claude-3-5-sonnet-20241022") {
		t.Error("anthropic_messages must report native support by model")
	}
	p = NewHTTPProviderWithOptions("k", "https://openrouter.ai/api/v1", "")
	if _, ok := p.(DeferredToolsCapable); ok {
		t.Error("openai_compat must not report native deferred tools")
	}
}

func TestCLIPrompt_ToolReferencesAsText(t *testing.T) {
	p := NewClaudeCliProvider("claude", "", nil, nil)
	prompt := p.messagesToPrompt([]Message{
		{Role: "user", Content: "q"},
		{Role: "tool", ToolCallID: "c1", Content: "hits", ToolReferences: []string{"a", "b"}},
	})
	if !strings.Contains(prompt, "[Tool Result for c1]: hits\n\nTools found: a, b") {
		t.Errorf("prompt = %q", prompt)
	}
}
