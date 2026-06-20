package spawnllm

import (
	"testing"

	anthropicprovider "github.com/PivotLLM/spawnllm/anthropic"
)

// Tests for anthropic sub-package functions not covered.

func TestAnthropicProvider_SupportsThinking(t *testing.T) {
	p := anthropicprovider.NewProvider("test-token")
	if !p.SupportsThinking() {
		t.Error("SupportsThinking() should return true")
	}
}

func TestAnthropicProvider_NewProviderWithTokenSource(t *testing.T) {
	tokenSource := func() (string, error) { return "refreshed", nil }
	p := anthropicprovider.NewProviderWithTokenSource("initial", tokenSource)
	if p == nil {
		t.Fatal("NewProviderWithTokenSource() returned nil")
	}
	model := p.GetDefaultModel()
	if model == "" {
		t.Error("GetDefaultModel() returned empty string")
	}
}
