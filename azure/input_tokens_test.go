package azure

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestProviderChat_AzureInputTokensExcludeCache(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if _, err := w.Write([]byte(`{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],` +
			`"usage":{"prompt_tokens":1000,"completion_tokens":5,"total_tokens":1005,` +
			`"prompt_tokens_details":{"cached_tokens":768}}}`)); err != nil {
			t.Errorf("write response: %v", err)
		}
	}))
	defer server.Close()

	p := NewProvider("test-key", server.URL, "")
	out, err := p.Chat(t.Context(), []Message{{Role: "user", Content: "hi"}}, nil, "dep", nil)
	if err != nil {
		t.Fatalf("Chat() error = %v", err)
	}
	s := out.Status
	if s.InputTokens != 232 || s.CacheReadTokens != 768 || s.CacheCreationTokens != 0 {
		t.Errorf("input/read/write = %d/%d/%d, want 232/768/0", s.InputTokens, s.CacheReadTokens, s.CacheCreationTokens)
	}
	if out.Usage.PromptTokens != 1000 {
		t.Errorf("PromptTokens = %d, want 1000", out.Usage.PromptTokens)
	}
	if sum := s.InputTokens + s.CacheReadTokens + s.CacheCreationTokens; sum != out.Usage.PromptTokens {
		t.Errorf("input+read+write = %d, want PromptTokens %d", sum, out.Usage.PromptTokens)
	}
}
