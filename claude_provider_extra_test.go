package spawnllm

import (
	"testing"
)

// Tests for NewClaudeProviderWithBaseURL and NewClaudeProviderWithTokenSourceAndBaseURL.

func TestNewClaudeProviderWithBaseURL(t *testing.T) {
	p := NewClaudeProviderWithBaseURL("test-token", "https://example.com/v1")
	if p == nil {
		t.Fatal("NewClaudeProviderWithBaseURL() returned nil")
	}
	if p.delegate == nil {
		t.Fatal("delegate should not be nil")
	}
}

func TestNewClaudeProviderWithTokenSourceAndBaseURL(t *testing.T) {
	tokenSource := func() (string, error) { return "refreshed-token", nil }
	p := NewClaudeProviderWithTokenSourceAndBaseURL("initial-token", tokenSource, "https://example.com/v1")
	if p == nil {
		t.Fatal("NewClaudeProviderWithTokenSourceAndBaseURL() returned nil")
	}
	if p.delegate == nil {
		t.Fatal("delegate should not be nil")
	}
}

func TestClaudeProvider_GetDefaultModel_NonEmpty(t *testing.T) {
	p := NewClaudeProvider("test-token")
	model := p.GetDefaultModel()
	// Should return a non-empty default model string.
	if model == "" {
		t.Error("GetDefaultModel() returned empty string")
	}
}
