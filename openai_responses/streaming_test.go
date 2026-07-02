// ClawEh
// License: MIT

package openai_responses

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/PivotLLM/spawnllm/common"
)

// sseResponsesServer replies with a raw SSE body and captures the request body.
func sseResponsesServer(t *testing.T, body string, captured *map[string]any) *httptest.Server {
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

func TestChat_StreamingText(t *testing.T) {
	// Two output_text deltas, then the completed event carrying the full
	// response object (same shape the non-streaming parser consumes).
	final := `{"status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"hello world"}]}],"usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15}}`
	body := "data: " + `{"type":"response.output_text.delta","delta":"hello "}` + "\n\n" +
		"data: " + `{"type":"response.output_text.delta","delta":"world"}` + "\n\n" +
		"data: " + `{"type":"response.completed","response":` + final + `}` + "\n\n"

	var reqBody map[string]any
	srv := sseResponsesServer(t, body, &reqBody)

	var deltas []string
	p := NewProvider("k", srv.URL, "")
	out, err := p.Chat(context.Background(),
		[]Message{{Role: "user", Content: "hi"}}, nil, "gpt-5",
		map[string]any{common.TextDeltaOption: common.TextDeltaFunc(func(d string) { deltas = append(deltas, d) })})
	if err != nil {
		t.Fatalf("Chat error: %v", err)
	}

	if reqBody["stream"] != true {
		t.Fatalf("request stream = %v, want true", reqBody["stream"])
	}
	if out.Content != "hello world" {
		t.Fatalf("Content = %q, want 'hello world'", out.Content)
	}
	if strings.Join(deltas, "") != "hello world" {
		t.Fatalf("deltas joined = %q, want 'hello world'", strings.Join(deltas, ""))
	}
	if out.Usage == nil || out.Usage.TotalTokens != 15 {
		t.Fatalf("Usage = %+v, want TotalTokens=15", out.Usage)
	}
	if out.Status == nil || !out.Status.Success || out.Status.BytesReceived == 0 {
		t.Fatalf("Status = %+v", out.Status)
	}
}

func TestChat_StreamingToolCall(t *testing.T) {
	final := `{"status":"completed","output":[{"type":"function_call","call_id":"call_1","name":"get_weather","arguments":"{\"city\":\"NYC\"}"}]}`
	body := "data: " + `{"type":"response.output_item.added","item":{"type":"function_call","name":"get_weather"}}` + "\n\n" +
		"data: " + `{"type":"response.completed","response":` + final + `}` + "\n\n"

	srv := sseResponsesServer(t, body, nil)
	p := NewProvider("k", srv.URL, "")
	out, err := p.Chat(context.Background(),
		[]Message{{Role: "user", Content: "weather?"}}, nil, "gpt-5",
		map[string]any{common.TextDeltaOption: common.TextDeltaFunc(func(string) {})})
	if err != nil {
		t.Fatalf("Chat error: %v", err)
	}
	if len(out.ToolCalls) != 1 {
		t.Fatalf("len(ToolCalls) = %d, want 1", len(out.ToolCalls))
	}
	if out.ToolCalls[0].Function == nil || out.ToolCalls[0].Function.Name != "get_weather" {
		t.Fatalf("ToolCalls[0] = %+v, want get_weather", out.ToolCalls[0])
	}
	if out.FinishReason != "tool_calls" {
		t.Fatalf("FinishReason = %q, want tool_calls", out.FinishReason)
	}
}

func TestChat_StreamingError(t *testing.T) {
	body := "data: " + `{"type":"response.output_text.delta","delta":"partial"}` + "\n\n" +
		"data: " + `{"type":"response.error","error":{"message":"model overloaded"}}` + "\n\n"

	srv := sseResponsesServer(t, body, nil)
	var deltas []string
	p := NewProvider("k", srv.URL, "")
	out, err := p.Chat(context.Background(),
		[]Message{{Role: "user", Content: "hi"}}, nil, "gpt-5",
		map[string]any{common.TextDeltaOption: common.TextDeltaFunc(func(d string) { deltas = append(deltas, d) })})
	if err == nil {
		t.Fatalf("expected error from response.error event")
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
	reply := `{"status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]}]}`
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&reqBody)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(reply))
	}))
	t.Cleanup(srv.Close)

	p := NewProvider("k", srv.URL, "")
	out, err := p.Chat(context.Background(),
		[]Message{{Role: "user", Content: "hi"}}, nil, "gpt-5",
		map[string]any{common.TextDeltaOption: common.TextDeltaFunc(nil)})
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
