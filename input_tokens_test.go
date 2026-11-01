package spawnllm

import "testing"

// TestCLIInputTokensExcludeCache checks every CLI provider reports
// DispatchStatus.InputTokens as the input not read from or written to the
// cache, so InputTokens + CacheReadTokens + CacheCreationTokens is the whole
// prompt (Usage.PromptTokens).
func TestCLIInputTokensExcludeCache(t *testing.T) {
	tests := []struct {
		name       string
		parse      func() (*LLMResponse, error)
		wantInput  int
		wantRead   int
		wantWrite  int
		wantPrompt int
	}{
		{
			name: "claude-cli",
			parse: func() (*LLMResponse, error) {
				return NewClaudeCliProvider("", "/workspace", nil, nil).parseClaudeCliResponse(
					`{"type":"result","subtype":"success","is_error":false,"result":"ok",` +
						`"usage":{"input_tokens":10,"cache_creation_input_tokens":300,"cache_read_input_tokens":600,"output_tokens":5}}`)
			},
			wantInput: 10, wantRead: 600, wantWrite: 300, wantPrompt: 910,
		},
		{
			// Codex's input_tokens is the whole prompt; cached_input_tokens is
			// the part of it read from cache.
			name: "codex-cli",
			parse: func() (*LLMResponse, error) {
				return (&CodexCliProvider{}).parseJSONLEvents(
					`{"type":"item.completed","item":{"id":"i","type":"agent_message","text":"ok"}}`+"\n"+
						`{"type":"turn.completed","usage":{"input_tokens":1000,"cached_input_tokens":700,"output_tokens":5}}`,
					"gpt-5-codex", 1)
			},
			wantInput: 300, wantRead: 700, wantWrite: 0, wantPrompt: 1000,
		},
		{
			name: "cursor-cli",
			parse: func() (*LLMResponse, error) {
				return (&CursorCliProvider{}).parseCursorCliResponse(
					`{"type":"result","subtype":"success","is_error":false,"result":"ok",` +
						`"usage":{"inputTokens":40,"outputTokens":5,"cacheReadTokens":800,"cacheWriteTokens":20}}`)
			},
			wantInput: 40, wantRead: 800, wantWrite: 20, wantPrompt: 860,
		},
		{
			name: "antigravity-cli",
			parse: func() (*LLMResponse, error) {
				return (&AntigravityCliProvider{}).parseAntigravityCliResponse(
					`{"status":"SUCCESS","response":"ok","num_turns":1,` +
						`"usage":{"input_tokens":50,"output_tokens":5,"cache_read_tokens":900}}`)
			},
			wantInput: 50, wantRead: 900, wantWrite: 0, wantPrompt: 950,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resp, err := tt.parse()
			if err != nil {
				t.Fatalf("parse error: %v", err)
			}
			s := resp.Status
			if s.InputTokens != tt.wantInput || s.CacheReadTokens != tt.wantRead || s.CacheCreationTokens != tt.wantWrite {
				t.Errorf("input/read/write = %d/%d/%d, want %d/%d/%d", s.InputTokens, s.CacheReadTokens,
					s.CacheCreationTokens, tt.wantInput, tt.wantRead, tt.wantWrite)
			}
			if resp.Usage == nil || resp.Usage.PromptTokens != tt.wantPrompt {
				t.Fatalf("Usage = %+v, want PromptTokens %d", resp.Usage, tt.wantPrompt)
			}
			if sum := s.InputTokens + s.CacheReadTokens + s.CacheCreationTokens; sum != resp.Usage.PromptTokens {
				t.Errorf("input+read+write = %d, want PromptTokens %d", sum, resp.Usage.PromptTokens)
			}
		})
	}
}
