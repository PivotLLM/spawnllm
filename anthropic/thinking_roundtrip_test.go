package anthropicprovider

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/PivotLLM/spawnllm/common"
)

// sdkMessagesServer replies with each of responses in turn (the last one
// repeated) and records every decoded request body.
func sdkMessagesServer(t *testing.T, responses []string, captured *[]map[string]any) *httptest.Server {
	t.Helper()
	call := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request: %v", err)
		}
		*captured = append(*captured, body)
		resp := responses[min(call, len(responses)-1)]
		call++
		w.Header().Set("Content-Type", "application/json")
		if _, err := w.Write([]byte(resp)); err != nil {
			t.Errorf("write response: %v", err)
		}
	}))
	t.Cleanup(srv.Close)
	return srv
}

const sdkToolUseWithThinking = `{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-4-6",` +
	`"stop_reason":"tool_use","content":[` +
	`{"type":"thinking","thinking":"Alice wants the weather.","signature":"sig-abc"},` +
	`{"type":"redacted_thinking","data":"opaque-xyz"},` +
	`{"type":"text","text":"Checking."},` +
	`{"type":"tool_use","id":"toolu_1","name":"get_weather","input":{"city":"NYC"}}],` +
	`"usage":{"input_tokens":5,"output_tokens":9}}`

const sdkPlainText = `{"id":"msg_2","type":"message","role":"assistant","model":"claude-sonnet-4-6",` +
	`"stop_reason":"end_turn","content":[{"type":"text","text":"ok"}],` +
	`"usage":{"input_tokens":1,"output_tokens":1}}`

var sdkThinkingBlocks = []map[string]any{
	{"type": "thinking", "thinking": "Alice wants the weather.", "signature": "sig-abc"},
	{"type": "redacted_thinking", "data": "opaque-xyz"},
}

// sdkToolLoop sends a first request, builds the follow-up history the way a
// tool loop does, and sends it with secondOpts.
func sdkToolLoop(t *testing.T, firstOpts, secondOpts map[string]any) (*LLMResponse, []map[string]any) {
	t.Helper()
	var captured []map[string]any
	srv := sdkMessagesServer(t, []string{sdkToolUseWithThinking, sdkPlainText}, &captured)
	p := NewProviderWithClient(createAnthropicTestClient(srv.URL, "test-token"))
	tools := []ToolDefinition{{
		Type:     "function",
		Function: ToolFunctionDefinition{Name: "get_weather", Parameters: map[string]any{"type": "object"}},
	}}
	msgs := make([]Message, 0, 3)
	msgs = append(msgs, Message{Role: "user", Content: "Weather in NYC?"})
	resp, err := p.Chat(t.Context(), msgs, tools, "claude-sonnet-4-6", firstOpts)
	if err != nil {
		t.Fatalf("first Chat() error = %v", err)
	}
	msgs = append(msgs,
		Message{Role: "assistant", Content: resp.Content, ToolCalls: resp.ToolCalls, AnthropicThinking: resp.AnthropicThinking},
		Message{Role: "tool", ToolCallID: resp.ToolCalls[0].ID, Content: "sunny"},
	)
	if _, err := p.Chat(t.Context(), msgs, tools, "claude-sonnet-4-6", secondOpts); err != nil {
		t.Fatalf("second Chat() error = %v", err)
	}
	return resp, captured
}

func sdkAssistantContent(t *testing.T, body map[string]any) []any {
	t.Helper()
	msgs, ok := body["messages"].([]any)
	if !ok {
		t.Fatalf("messages = %T, want array", body["messages"])
	}
	for _, m := range msgs {
		mm, ok := m.(map[string]any)
		if !ok || mm["role"] != "assistant" {
			continue
		}
		content, ok := mm["content"].([]any)
		if !ok {
			t.Fatalf("assistant content = %T, want array", mm["content"])
		}
		return content
	}
	t.Fatal("no assistant message in request")
	return nil
}

func decodeBlocks(t *testing.T, raw []json.RawMessage) []map[string]any {
	t.Helper()
	out := make([]map[string]any, 0, len(raw))
	for _, r := range raw {
		var m map[string]any
		if err := json.Unmarshal(r, &m); err != nil {
			t.Fatalf("decode block %s: %v", r, err)
		}
		out = append(out, m)
	}
	return out
}

func TestChat_SDKThinkingBlocksRoundTripAcrossToolCall(t *testing.T) {
	opts := map[string]any{"max_tokens": 16000, "thinking_level": "low"}
	resp, captured := sdkToolLoop(t, opts, opts)

	if got := decodeBlocks(t, resp.AnthropicThinking); !reflect.DeepEqual(got, sdkThinkingBlocks) {
		t.Fatalf("AnthropicThinking = %v, want %v", got, sdkThinkingBlocks)
	}

	content := sdkAssistantContent(t, captured[1])
	if len(content) != 4 {
		t.Fatalf("assistant content has %d blocks, want 4: %v", len(content), content)
	}
	for i, want := range sdkThinkingBlocks {
		if !reflect.DeepEqual(content[i], want) {
			t.Errorf("assistant block %d = %#v, want %#v", i, content[i], want)
		}
	}
	for i, want := range map[int]string{2: "text", 3: "tool_use"} {
		if blk, ok := content[i].(map[string]any); !ok || blk["type"] != want {
			t.Errorf("assistant block %d = %v, want %s", i, content[i], want)
		}
	}
}

func TestChat_SDKThinkingBlocksNotSentWhenThinkingOff(t *testing.T) {
	_, captured := sdkToolLoop(t,
		map[string]any{"max_tokens": 16000, "thinking_level": "low"},
		map[string]any{"max_tokens": 16000})

	for _, blk := range sdkAssistantContent(t, captured[1]) {
		m, ok := blk.(map[string]any)
		if !ok {
			t.Fatalf("assistant block = %T, want object", blk)
		}
		if typ, ok := m["type"].(string); ok && strings.Contains(typ, "thinking") {
			t.Fatalf("thinking block sent with thinking off: %v", blk)
		}
	}
}

func TestChat_SDKStreamingKeepsThinkingBlocks(t *testing.T) {
	events := []string{
		`{"type":"message_start","message":{"id":"msg_s","type":"message","role":"assistant","content":[],"model":"claude-sonnet-4-6","stop_reason":null,"usage":{"input_tokens":3,"output_tokens":0}}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Bob asked "}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"for a sum."}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig-123"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"content_block_start","index":1,"content_block":{"type":"redacted_thinking","data":"opaque-xyz"}}`,
		`{"type":"content_block_stop","index":1}`,
		`{"type":"content_block_start","index":2,"content_block":{"type":"text","text":""}}`,
		`{"type":"content_block_delta","index":2,"delta":{"type":"text_delta","text":"It is 4."}}`,
		`{"type":"content_block_stop","index":2}`,
		`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":12}}`,
		`{"type":"message_stop"}`,
	}
	framed := make([]string, len(events))
	for i, e := range events {
		var typ struct {
			Type string `json:"type"`
		}
		if err := json.Unmarshal([]byte(e), &typ); err != nil {
			t.Fatal(err)
		}
		framed[i] = "event: " + typ.Type + "\ndata: " + e + "\n\n"
	}
	srv := anthropicSSEServer(t, framed)
	p := NewProviderWithClient(createAnthropicTestClient(srv.URL, "test-token"))

	var deltas []string
	resp, err := p.Chat(t.Context(), []Message{{Role: "user", Content: "2+2?"}}, nil, "claude-sonnet-4-6",
		map[string]any{
			"max_tokens":           64000,
			"thinking_level":       "low",
			common.TextDeltaOption: func(d string) { deltas = append(deltas, d) },
		})
	if err != nil {
		t.Fatalf("Chat() error = %v", err)
	}
	if !reflect.DeepEqual(deltas, []string{"It is 4."}) {
		t.Errorf("text deltas = %q, want only the answer text", deltas)
	}
	if resp.Reasoning != "Bob asked for a sum." {
		t.Errorf("Reasoning = %q", resp.Reasoning)
	}
	want := []map[string]any{
		{"type": "thinking", "thinking": "Bob asked for a sum.", "signature": "sig-123"},
		{"type": "redacted_thinking", "data": "opaque-xyz"},
	}
	if got := decodeBlocks(t, resp.AnthropicThinking); !reflect.DeepEqual(got, want) {
		t.Fatalf("AnthropicThinking = %v, want %v", got, want)
	}
}
