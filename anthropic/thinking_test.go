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
		wantTemp     bool
	}{
		{name: "absent", level: nil, wantTemp: true},
		{name: "off", level: "off", wantTemp: true},
		{name: "empty", level: "", wantTemp: true},
		{name: "unknown", level: "bogus", wantTemp: true},
		{name: "non-string", level: 3, wantTemp: true},
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
			if _, got := body["temperature"]; got != tt.wantTemp {
				t.Errorf("temperature present = %v, want %v", got, tt.wantTemp)
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

func TestBuildParams_NoThinkingKeepsTemperature(t *testing.T) {
	msgs := []Message{{Role: "user", Content: "hello"}}
	opts := map[string]any{
		"temperature": 0.8,
	}

	params, err := buildParams(msgs, nil, "claude-sonnet-4-6", opts)
	if err != nil {
		t.Fatal(err)
	}

	if !params.Temperature.Valid() {
		t.Error("temperature should be preserved when thinking is not set")
	}
	if params.Temperature.Value != 0.8 {
		t.Errorf("temperature = %f, want 0.8", params.Temperature.Value)
	}
}
