package anthropicmessages

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/PivotLLM/spawnllm/common"
)

// jsonMessagesServer replies with each of responses in turn (the last one
// repeated) and records every decoded request body.
func jsonMessagesServer(t *testing.T, responses []string, captured *[]map[string]any) *httptest.Server {
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

const plainTextResponse = `{"type":"message","role":"assistant","model":"claude-sonnet-4-6",` +
	`"stop_reason":"end_turn","content":[{"type":"text","text":"ok"}],` +
	`"usage":{"input_tokens":1,"output_tokens":1}}`

func TestProvider_SupportsThinking(t *testing.T) {
	var p any = NewProvider("key", "")
	tc, ok := p.(interface{ SupportsThinking() bool })
	if !ok || !tc.SupportsThinking() {
		t.Fatal("anthropic_messages.Provider must report thinking support")
	}
}

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
			srv := jsonMessagesServer(t, []string{plainTextResponse}, &captured)
			p := NewProvider("key", srv.URL)
			opts := map[string]any{"max_tokens": 1000, "temperature": 0.7}
			if tt.level != nil {
				opts["thinking_level"] = tt.level
			}
			if _, err := p.Chat(t.Context(), []Message{{Role: "user", Content: "hi"}}, nil, "claude-sonnet-4-6", opts); err != nil {
				t.Fatalf("Chat() error = %v", err)
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

// toolUseWithThinking is a response whose assistant turn starts with a
// thinking and a redacted_thinking block, then calls a tool.
const toolUseWithThinking = `{"type":"message","role":"assistant","model":"claude-sonnet-4-6",` +
	`"stop_reason":"tool_use","content":[` +
	`{"type":"thinking","thinking":"Alice wants the weather.","signature":"sig-abc"},` +
	`{"type":"redacted_thinking","data":"opaque-xyz"},` +
	`{"type":"text","text":"Checking."},` +
	`{"type":"tool_use","id":"toolu_1","name":"get_weather","input":{"city":"NYC"}}],` +
	`"usage":{"input_tokens":5,"output_tokens":9}}`

var thinkingBlocksJSON = []string{
	`{"type":"thinking","thinking":"Alice wants the weather.","signature":"sig-abc"}`,
	`{"type":"redacted_thinking","data":"opaque-xyz"}`,
}

// runToolLoop sends a first request, builds the follow-up history the way a
// tool loop does, and sends it with secondOpts. It returns the first response
// and both captured request bodies.
func runToolLoop(t *testing.T, firstOpts, secondOpts map[string]any) (*LLMResponse, []map[string]any) {
	t.Helper()
	var captured []map[string]any
	srv := jsonMessagesServer(t, []string{toolUseWithThinking, plainTextResponse}, &captured)
	p := NewProvider("key", srv.URL)
	tools := []ToolDefinition{{
		Type: "function",
		Function: ToolFunctionDefinition{
			Name:       "get_weather",
			Parameters: map[string]any{"type": "object"},
		},
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

func assistantContent(t *testing.T, body map[string]any) []any {
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

func TestChat_ThinkingBlocksRoundTripAcrossToolCall(t *testing.T) {
	opts := map[string]any{"max_tokens": 64000, "thinking_level": "medium"}
	resp, captured := runToolLoop(t, opts, opts)

	if resp.Reasoning != "Alice wants the weather." {
		t.Errorf("Reasoning = %q", resp.Reasoning)
	}
	if len(resp.AnthropicThinking) != len(thinkingBlocksJSON) {
		t.Fatalf("AnthropicThinking has %d blocks, want %d", len(resp.AnthropicThinking), len(thinkingBlocksJSON))
	}
	for i, want := range thinkingBlocksJSON {
		if string(resp.AnthropicThinking[i]) != want {
			t.Errorf("AnthropicThinking[%d] = %s, want %s", i, resp.AnthropicThinking[i], want)
		}
	}

	content := assistantContent(t, captured[1])
	if len(content) != 4 {
		t.Fatalf("assistant content has %d blocks, want 4: %v", len(content), content)
	}
	for i, want := range thinkingBlocksJSON {
		var wantBlock map[string]any
		if err := json.Unmarshal([]byte(want), &wantBlock); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(content[i], wantBlock) {
			t.Errorf("assistant block %d = %#v, want %#v", i, content[i], wantBlock)
		}
	}
	for i, want := range map[int]string{2: "text", 3: "tool_use"} {
		if blk, ok := content[i].(map[string]any); !ok || blk["type"] != want {
			t.Errorf("assistant block %d = %v, want %s", i, content[i], want)
		}
	}
}

func TestChat_ThinkingBlocksNotSentWhenThinkingOff(t *testing.T) {
	_, captured := runToolLoop(t,
		map[string]any{"max_tokens": 64000, "thinking_level": "medium"},
		map[string]any{"max_tokens": 64000})

	for _, blk := range assistantContent(t, captured[1]) {
		m, ok := blk.(map[string]any)
		if !ok {
			t.Fatalf("assistant block = %T, want object", blk)
		}
		if typ, ok := m["type"].(string); ok && strings.Contains(typ, "thinking") {
			t.Fatalf("thinking block sent with thinking off: %v", blk)
		}
	}
}

func TestChat_StreamingThinking(t *testing.T) {
	events := []string{
		`data: {"type":"message_start","message":{"model":"claude-sonnet-4-6","usage":{"input_tokens":3,"output_tokens":0}}}`,
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Bob asked "}}`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"for a sum."}}`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig-123"}}`,
		`data: {"type":"content_block_stop","index":0}`,
		`data: {"type":"content_block_start","index":1,"content_block":{"type":"redacted_thinking","data":"opaque-xyz"}}`,
		`data: {"type":"content_block_stop","index":1}`,
		`data: {"type":"content_block_start","index":2,"content_block":{"type":"text","text":""}}`,
		`data: {"type":"content_block_delta","index":2,"delta":{"type":"text_delta","text":"It is 4."}}`,
		`data: {"type":"content_block_stop","index":2}`,
		`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":12}}`,
		`data: {"type":"message_stop"}`,
	}
	var reqBody map[string]any
	srv := sseMessagesServer(t, strings.Join(events, "\n\n")+"\n\n", &reqBody)
	p := NewProvider("key", srv.URL)

	var deltas []string
	opts := map[string]any{
		"max_tokens":           64000,
		"thinking_level":       "low",
		common.TextDeltaOption: func(s string) { deltas = append(deltas, s) },
	}
	resp, err := p.Chat(t.Context(), []Message{{Role: "user", Content: "2+2?"}}, nil, "claude-sonnet-4-6", opts)
	if err != nil {
		t.Fatalf("Chat() error = %v", err)
	}

	if reqBody["thinking"] == nil {
		t.Error("streaming request carries no thinking parameter")
	}
	if !reflect.DeepEqual(deltas, []string{"It is 4."}) {
		t.Errorf("text deltas = %q, want only the answer text", deltas)
	}
	if resp.Content != "It is 4." {
		t.Errorf("Content = %q", resp.Content)
	}
	if resp.Reasoning != "Bob asked for a sum." {
		t.Errorf("Reasoning = %q", resp.Reasoning)
	}
	want := []map[string]any{
		{"type": "thinking", "thinking": "Bob asked for a sum.", "signature": "sig-123"},
		{"type": "redacted_thinking", "data": "opaque-xyz"},
	}
	if len(resp.AnthropicThinking) != len(want) {
		t.Fatalf("AnthropicThinking has %d blocks, want %d", len(resp.AnthropicThinking), len(want))
	}
	for i, w := range want {
		var got map[string]any
		if err := json.Unmarshal(resp.AnthropicThinking[i], &got); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(got, w) {
			t.Errorf("AnthropicThinking[%d] = %v, want %v", i, got, w)
		}
	}
}
