// ClawEh
// License: MIT

package anthropicprovider

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/PivotLLM/spawnllm/common"
)

// anthropicSSEServer serves a canned Messages SSE stream. It is used to drive
// the SDK's NewStreaming loop so the delta callback can be exercised.
func anthropicSSEServer(t *testing.T, events []string) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/messages" {
			http.Error(w, "not found", http.StatusNotFound)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		for _, e := range events {
			_, _ = w.Write([]byte(e))
			if flusher != nil {
				flusher.Flush()
			}
		}
	}))
	t.Cleanup(srv.Close)
	return srv
}

func textStreamEvents() []string {
	return []string{
		"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_stream\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"model\":\"claude-sonnet-4-6\",\"stop_reason\":null,\"usage\":{\"input_tokens\":12,\"output_tokens\":0}}}\n\n",
		"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}\n\n",
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\" world\"}}\n\n",
		"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
		"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":5}}\n\n",
		"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
	}
}

// TestChat_StreamingFiresTextDeltas verifies the API-key path opts into
// streaming when a delta callback is present, fires cb per text delta in order,
// and returns the same accumulated LLMResponse as the non-streaming path.
func TestChat_StreamingFiresTextDeltas(t *testing.T) {
	srv := anthropicSSEServer(t, textStreamEvents())
	p := NewProviderWithClient(createAnthropicTestClient(srv.URL, "test-token"))

	var deltas []string
	resp, err := p.Chat(
		t.Context(),
		[]Message{{Role: "user", Content: "Hello"}},
		nil,
		"claude-sonnet-4.6",
		map[string]any{
			"max_tokens":           1024,
			common.TextDeltaOption: common.TextDeltaFunc(func(d string) { deltas = append(deltas, d) }),
		},
	)
	if err != nil {
		t.Fatalf("Chat() error: %v", err)
	}
	if strings.Join(deltas, "") != "Hello world" {
		t.Fatalf("deltas joined = %q, want 'Hello world'", strings.Join(deltas, ""))
	}
	if len(deltas) != 2 {
		t.Fatalf("len(deltas) = %d, want 2", len(deltas))
	}
	// Final response identical to what the non-streaming path would return.
	if resp.Content != "Hello world" {
		t.Fatalf("Content = %q, want 'Hello world'", resp.Content)
	}
	if resp.FinishReason != "stop" {
		t.Fatalf("FinishReason = %q, want stop", resp.FinishReason)
	}
	if resp.Usage == nil || resp.Usage.CompletionTokens != 5 {
		t.Fatalf("Usage = %+v, want CompletionTokens=5", resp.Usage)
	}
}

// TestChat_StreamingSkipsNonTextDeltas verifies the callback fires only for
// text_delta payloads, not thinking or tool-input fragments.
func TestChat_StreamingSkipsNonTextDeltas(t *testing.T) {
	events := []string{
		"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"m\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"model\":\"claude-sonnet-4-6\",\"stop_reason\":null,\"usage\":{\"input_tokens\":1,\"output_tokens\":0}}}\n\n",
		"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"get_weather\",\"input\":{}}}\n\n",
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"city\\\":\\\"NYC\\\"}\"}}\n\n",
		"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
		"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":3}}\n\n",
		"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
	}
	srv := anthropicSSEServer(t, events)
	p := NewProviderWithClient(createAnthropicTestClient(srv.URL, "test-token"))

	var deltas []string
	resp, err := p.Chat(
		t.Context(),
		[]Message{{Role: "user", Content: "weather?"}},
		nil,
		"claude-sonnet-4.6",
		map[string]any{
			"max_tokens":           1024,
			common.TextDeltaOption: common.TextDeltaFunc(func(d string) { deltas = append(deltas, d) }),
		},
	)
	if err != nil {
		t.Fatalf("Chat() error: %v", err)
	}
	if len(deltas) != 0 {
		t.Fatalf("deltas = %v, want none (tool-input fragments must not fire cb)", deltas)
	}
	if len(resp.ToolCalls) != 1 || resp.ToolCalls[0].Name != "get_weather" {
		t.Fatalf("ToolCalls = %+v, want get_weather", resp.ToolCalls)
	}
	if resp.FinishReason != "tool_calls" {
		t.Fatalf("FinishReason = %q, want tool_calls", resp.FinishReason)
	}
}

// TestChat_NilCallbackKeepsSingleShot verifies that with an API key and a nil
// callback the provider uses the non-streaming Messages.New path unchanged.
func TestChat_NilCallbackKeepsSingleShot(t *testing.T) {
	var streamed bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/messages" {
			http.Error(w, "not found", http.StatusNotFound)
			return
		}
		// Non-streaming request: Accept header is application/json (the SDK sets
		// text/event-stream only for NewStreaming).
		if strings.Contains(r.Header.Get("Accept"), "text/event-stream") {
			streamed = true
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"m","type":"message","role":"assistant","model":"claude-sonnet-4-6","stop_reason":"end_turn","content":[{"type":"text","text":"ok"}],"usage":{"input_tokens":1,"output_tokens":1}}`))
	}))
	t.Cleanup(srv.Close)

	p := NewProviderWithClient(createAnthropicTestClient(srv.URL, "test-token"))
	resp, err := p.Chat(
		t.Context(),
		[]Message{{Role: "user", Content: "hi"}},
		nil,
		"claude-sonnet-4.6",
		map[string]any{
			"max_tokens":           1024,
			common.TextDeltaOption: common.TextDeltaFunc(nil),
		},
	)
	if err != nil {
		t.Fatalf("Chat() error: %v", err)
	}
	if streamed {
		t.Fatalf("provider streamed despite nil callback")
	}
	if resp.Content != "ok" {
		t.Fatalf("Content = %q, want ok", resp.Content)
	}
}
