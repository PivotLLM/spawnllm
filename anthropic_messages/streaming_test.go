// ClawEh
// License: MIT

package anthropicmessages

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/PivotLLM/spawnllm/common"
)

// sseMessagesServer replies with a raw Anthropic Messages SSE body and captures
// the request body.
func sseMessagesServer(t *testing.T, body string, captured *map[string]any) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if captured != nil {
			_ = json.NewDecoder(r.Body).Decode(captured)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(srv.Close)
	return srv
}

// messagesSSE builds a canonical Messages SSE stream: message_start with input
// usage, a text block streamed in two deltas, one tool_use block whose input
// arrives as two input_json_delta fragments, then message_delta (stop_reason +
// output usage) and message_stop.
func messagesSSE() string {
	events := []string{
		`event: message_start` + "\n" +
			`data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-4-6","content":[],"stop_reason":null,"usage":{"input_tokens":11,"output_tokens":0,"cache_read_input_tokens":2,"cache_creation_input_tokens":3}}}`,
		`event: content_block_start` + "\n" +
			`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
		`event: content_block_delta` + "\n" +
			`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello "}}`,
		`event: content_block_delta` + "\n" +
			`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"world"}}`,
		`event: content_block_stop` + "\n" +
			`data: {"type":"content_block_stop","index":0}`,
		`event: content_block_start` + "\n" +
			`data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_1","name":"get_weather","input":{}}}`,
		`event: content_block_delta` + "\n" +
			`data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"city\":"}}`,
		`event: content_block_delta` + "\n" +
			`data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"\"NYC\"}"}}`,
		`event: content_block_stop` + "\n" +
			`data: {"type":"content_block_stop","index":1}`,
		`event: message_delta` + "\n" +
			`data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":7}}`,
		`event: message_stop` + "\n" +
			`data: {"type":"message_stop"}`,
	}
	return strings.Join(events, "\n\n") + "\n\n"
}

// equivalentWholeBody is the non-streaming Messages-API body equivalent to
// messagesSSE, used to assert the reconstructed stream matches a direct parse.
func equivalentWholeBody() string {
	return `{"type":"message","role":"assistant","model":"claude-sonnet-4-6",` +
		`"stop_reason":"tool_use",` +
		`"content":[{"type":"text","text":"Hello world"},` +
		`{"type":"tool_use","id":"toolu_1","name":"get_weather","input":{"city":"NYC"}}],` +
		`"usage":{"input_tokens":11,"output_tokens":7,"cache_read_input_tokens":2,"cache_creation_input_tokens":3}}`
}

func TestChat_StreamingReconstructsWholeBody(t *testing.T) {
	var reqBody map[string]any
	srv := sseMessagesServer(t, messagesSSE(), &reqBody)

	var deltas []string
	p := NewProvider("k", srv.URL)
	out, err := p.Chat(context.Background(),
		[]Message{{Role: "user", Content: "weather?"}}, nil, "claude-sonnet-4.6",
		map[string]any{
			"max_tokens":           1024,
			common.TextDeltaOption: common.TextDeltaFunc(func(d string) { deltas = append(deltas, d) }),
		})
	if err != nil {
		t.Fatalf("Chat error: %v", err)
	}

	if reqBody["stream"] != true {
		t.Fatalf("request stream = %v, want true", reqBody["stream"])
	}

	// cb must have received exactly the text deltas, in order, and nothing from
	// the tool-call input_json fragments.
	if strings.Join(deltas, "") != "Hello world" {
		t.Fatalf("deltas joined = %q, want 'Hello world'", strings.Join(deltas, ""))
	}
	if len(deltas) != 2 {
		t.Fatalf("len(deltas) = %d, want 2", len(deltas))
	}

	// Reconstructed response must equal a non-streaming parse of the whole body.
	want, err := parseResponseBody([]byte(equivalentWholeBody()))
	if err != nil {
		t.Fatalf("parse equivalent body: %v", err)
	}
	if out.Content != want.Content {
		t.Fatalf("Content = %q, want %q", out.Content, want.Content)
	}
	if out.FinishReason != want.FinishReason {
		t.Fatalf("FinishReason = %q, want %q", out.FinishReason, want.FinishReason)
	}
	if len(out.ToolCalls) != len(want.ToolCalls) || len(out.ToolCalls) != 1 {
		t.Fatalf("ToolCalls = %+v, want 1 matching %+v", out.ToolCalls, want.ToolCalls)
	}
	if out.ToolCalls[0].Name != "get_weather" {
		t.Fatalf("tool name = %q, want get_weather", out.ToolCalls[0].Name)
	}
	if out.ToolCalls[0].Function == nil || out.ToolCalls[0].Function.Arguments != want.ToolCalls[0].Function.Arguments {
		t.Fatalf("tool args = %+v, want %+v", out.ToolCalls[0].Function, want.ToolCalls[0].Function)
	}
	if out.Usage == nil || out.Usage.PromptTokens != 11 || out.Usage.CompletionTokens != 7 {
		t.Fatalf("Usage = %+v, want prompt=11 completion=7", out.Usage)
	}
	if out.Status == nil || out.Status.CacheReadTokens != 2 || out.Status.CacheCreationTokens != 3 {
		t.Fatalf("Status cache tokens = %+v, want read=2 create=3", out.Status)
	}
	if out.Status.BytesReceived == 0 {
		t.Fatalf("BytesReceived = 0, want > 0")
	}
}

func TestChat_StreamingErrorEvent(t *testing.T) {
	body := `event: message_start` + "\n" +
		`data: {"type":"message_start","message":{"id":"m","type":"message","role":"assistant","model":"claude-sonnet-4-6","content":[],"usage":{"input_tokens":1,"output_tokens":0}}}` + "\n\n" +
		`event: content_block_delta` + "\n" +
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"partial"}}` + "\n\n" +
		`event: error` + "\n" +
		`data: {"type":"error","error":{"type":"overloaded_error","message":"model overloaded"}}` + "\n\n"

	srv := sseMessagesServer(t, body, nil)
	var deltas []string
	p := NewProvider("k", srv.URL)
	out, err := p.Chat(context.Background(),
		[]Message{{Role: "user", Content: "hi"}}, nil, "claude-sonnet-4.6",
		map[string]any{
			"max_tokens":           1024,
			common.TextDeltaOption: common.TextDeltaFunc(func(d string) { deltas = append(deltas, d) }),
		})
	if err == nil {
		t.Fatalf("expected error from in-band error event")
	}
	if !strings.Contains(err.Error(), "model overloaded") {
		t.Fatalf("error = %v, want to contain 'model overloaded'", err)
	}
	if out == nil || out.Status == nil || out.Status.Success {
		t.Fatalf("expected failed status, got %+v", out.Status)
	}
	if strings.Join(deltas, "") != "partial" {
		t.Fatalf("deltas joined = %q, want partial", strings.Join(deltas, ""))
	}
}

func TestChat_NonStreamingWhenNoCallback(t *testing.T) {
	// Nil callback must keep the single-shot JSON path (no stream flag).
	var reqBody map[string]any
	reply := `{"type":"message","role":"assistant","model":"claude-sonnet-4-6","stop_reason":"end_turn","content":[{"type":"text","text":"ok"}],"usage":{"input_tokens":1,"output_tokens":1}}`
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&reqBody)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(reply))
	}))
	t.Cleanup(srv.Close)

	p := NewProvider("k", srv.URL)
	out, err := p.Chat(context.Background(),
		[]Message{{Role: "user", Content: "hi"}}, nil, "claude-sonnet-4.6",
		map[string]any{
			"max_tokens":           1024,
			common.TextDeltaOption: common.TextDeltaFunc(nil),
		})
	if err != nil {
		t.Fatalf("Chat error: %v", err)
	}
	if _, ok := reqBody["stream"]; ok {
		t.Fatalf("stream flag set despite nil callback")
	}
	if out.Content != "ok" {
		t.Fatalf("Content = %q, want ok", out.Content)
	}
}
