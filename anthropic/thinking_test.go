package anthropicprovider

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
)

// TestChat_ThinkingLevelRequestBody checks the request body the SDK sends
// for every thinking_level.
func TestChat_ThinkingLevelRequestBody(t *testing.T) {
	adaptive := map[string]any{"type": "adaptive"}
	tests := []struct {
		name         string
		level        any // nil = option absent
		wantThinking any // nil = no thinking key
		wantOutput   any // nil = no output_config key
	}{
		{name: "absent", level: nil},
		{name: "off", level: "off"},
		{name: "empty", level: ""},
		{name: "unknown", level: "bogus"},
		{name: "non-string", level: 3},
		{name: "adaptive", level: "adaptive", wantThinking: adaptive},
		{name: "low", level: "low", wantThinking: adaptive, wantOutput: map[string]any{"effort": "low"}},
		{name: "medium", level: "medium", wantThinking: adaptive, wantOutput: map[string]any{"effort": "medium"}},
		{name: "high", level: "high", wantThinking: adaptive, wantOutput: map[string]any{"effort": "high"}},
		{name: "xhigh", level: "xhigh", wantThinking: adaptive, wantOutput: map[string]any{"effort": "xhigh"}},
		{name: "max", level: "max", wantThinking: adaptive, wantOutput: map[string]any{"effort": "max"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var captured []map[string]any
			srv := sdkMessagesServer(t, []string{sdkPlainText}, &captured)
			p := NewProviderWithClient(createAnthropicTestClient(srv.URL, "test-token"))
			opts := map[string]any{"max_tokens": 1000, "temperature": 0.7}
			if tt.level != nil {
				opts["thinking_level"] = tt.level
			}
			if _, err := p.Chat(t.Context(), []Message{{Role: "user", Content: "hi"}}, nil, "claude-sonnet-4-6", opts); err != nil {
				t.Fatalf("Chat() error = %v", err)
			}
			if len(captured) != 1 {
				t.Fatalf("captured %d requests, want 1", len(captured))
			}
			body := captured[0]
			if got := body["thinking"]; !reflect.DeepEqual(got, tt.wantThinking) {
				t.Errorf("thinking = %#v, want %#v", got, tt.wantThinking)
			}
			if got := body["output_config"]; !reflect.DeepEqual(got, tt.wantOutput) {
				t.Errorf("output_config = %#v, want %#v", got, tt.wantOutput)
			}
			if got, set := body["temperature"]; set {
				t.Errorf("temperature = %v, want absent", got)
			}
			if got := body["max_tokens"]; got != float64(1000) {
				t.Errorf("max_tokens = %v, want 1000", got)
			}
		})
	}
}

// unmarshalBlocks constructs []ContentBlockUnion via JSON round-trip so that
// the internal JSON.raw field is populated (required by AsText/AsThinking).
func unmarshalBlocks(t *testing.T, jsonStr string) []anthropic.ContentBlockUnion {
	t.Helper()
	var blocks []anthropic.ContentBlockUnion
	if err := json.Unmarshal([]byte(jsonStr), &blocks); err != nil {
		t.Fatalf("unmarshalBlocks: %v", err)
	}
	return blocks
}

func TestParseResponse_ThinkingBlock(t *testing.T) {
	resp := &anthropic.Message{
		Content: unmarshalBlocks(t, `[
			{"type":"thinking","thinking":"Let me reason step by step...","signature":"sig"},
			{"type":"text","text":"The answer is 42."}
		]`),
		StopReason: anthropic.StopReasonEndTurn,
	}

	result := parseResponse(resp)

	if result.Reasoning != "Let me reason step by step..." {
		t.Errorf("Reasoning = %q, want thinking content", result.Reasoning)
	}
	if result.Content != "The answer is 42." {
		t.Errorf("Content = %q, want text content", result.Content)
	}
	if result.FinishReason != "stop" {
		t.Errorf("FinishReason = %q, want stop", result.FinishReason)
	}
}

func TestParseResponse_NoThinkingBlock(t *testing.T) {
	resp := &anthropic.Message{
		Content: unmarshalBlocks(t, `[
			{"type":"text","text":"Just a normal response."}
		]`),
		StopReason: anthropic.StopReasonEndTurn,
	}

	result := parseResponse(resp)

	if result.Reasoning != "" {
		t.Errorf("Reasoning = %q, want empty", result.Reasoning)
	}
	if result.Content != "Just a normal response." {
		t.Errorf("Content = %q, want text content", result.Content)
	}
}
