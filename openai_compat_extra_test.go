package spawnllm

import (
	"testing"

	"github.com/PivotLLM/spawnllm/openai_compat"
)

// Tests for openai_compat options and constructor not covered.

func TestOpenAICompat_WithStrictCompat(t *testing.T) {
	// WithStrictCompat(true) should be accepted without panic.
	p := openai_compat.NewProvider("test-key", "https://api.openai.com/v1", "",
		openai_compat.WithStrictCompat(true),
	)
	if p == nil {
		t.Fatal("NewProvider() with WithStrictCompat returned nil")
	}
}

func TestOpenAICompat_WithStrictCompat_False(t *testing.T) {
	p := openai_compat.NewProvider("test-key", "https://api.openai.com/v1", "",
		openai_compat.WithStrictCompat(false),
	)
	if p == nil {
		t.Fatal("NewProvider() with WithStrictCompat(false) returned nil")
	}
}

func TestOpenAICompat_WithNoParallelToolCalls(t *testing.T) {
	p := openai_compat.NewProvider("test-key", "https://api.groq.com/openai/v1", "",
		openai_compat.WithNoParallelToolCalls(true),
	)
	if p == nil {
		t.Fatal("NewProvider() with WithNoParallelToolCalls returned nil")
	}
}

func TestOpenAICompat_NewProviderWithMaxTokensField(t *testing.T) {
	p := openai_compat.NewProviderWithMaxTokensField(
		"test-key",
		"https://api.openai.com/v1",
		"",
		"max_completion_tokens",
	)
	if p == nil {
		t.Fatal("NewProviderWithMaxTokensField() returned nil")
	}
}
