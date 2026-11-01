package spawnllm

import (
	"testing"
	"time"

	anthropicmessages "github.com/PivotLLM/spawnllm/anthropic_messages"
)

func TestBuildProvider_KindSelectsClient(t *testing.T) {
	tests := []struct {
		kind  string
		check func(LLMProvider) bool
		want  string
	}{
		{"anthropic", isAnthropicMessages, "*anthropicmessages.Provider"},
		{"anthropic-messages", isAnthropicMessages, "*anthropicmessages.Provider"},
		{"openai-chat", isHTTPProvider, "*spawnllm.HTTPProvider"},
		{"cursor-cli", isCursorCli, "*spawnllm.CursorCliProvider"},
	}
	for _, tt := range tests {
		t.Run(tt.kind, func(t *testing.T) {
			p, model, err := buildProvider(ProviderSpec{
				Kind:    tt.kind,
				BaseURL: "https://example.invalid",
				APIKey:  "test-key",
				Model:   "test-model",
				Timeout: 30 * time.Second,
			})
			if err != nil {
				t.Fatalf("buildProvider(%q) error: %v", tt.kind, err)
			}
			if model != "test-model" {
				t.Errorf("buildProvider(%q) model = %q, want %q", tt.kind, model, "test-model")
			}
			if !tt.check(p) {
				t.Errorf("buildProvider(%q) = %T, want %s", tt.kind, p, tt.want)
			}
		})
	}
}

func isAnthropicMessages(p LLMProvider) bool {
	_, ok := p.(*anthropicmessages.Provider)
	return ok
}

func isHTTPProvider(p LLMProvider) bool {
	_, ok := p.(*HTTPProvider)
	return ok
}

func isCursorCli(p LLMProvider) bool {
	_, ok := p.(*CursorCliProvider)
	return ok
}
