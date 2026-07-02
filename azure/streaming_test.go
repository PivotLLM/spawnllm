// ClawEh
// License: MIT

package azure

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/PivotLLM/spawnllm/common"
)

// sseAzureServer replies with a raw SSE body and captures the request body.
func sseAzureServer(t *testing.T, body string, captured *map[string]any) *httptest.Server {
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
	// Two content deltas then a usage-only final chunk (include_usage), the same
	// chat/completions streaming shape Azure emits, then [DONE].
	body := "data: " + `{"model":"gpt-4o","choices":[{"delta":{"role":"assistant","content":"hello "}}]}` + "\n\n" +
		"data: " + `{"model":"gpt-4o","choices":[{"delta":{"content":"world"},"finish_reason":"stop"}]}` + "\n\n" +
		"data: " + `{"model":"gpt-4o","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}` + "\n\n" +
		"data: [DONE]\n\n"

	var reqBody map[string]any
	srv := sseAzureServer(t, body, &reqBody)

	var deltas []string
	p := NewProvider("k", srv.URL, "")
	out, err := p.Chat(context.Background(),
		[]Message{{Role: "user", Content: "hi"}}, nil, "gpt-4o-deployment",
		map[string]any{common.TextDeltaOption: common.TextDeltaFunc(func(d string) { deltas = append(deltas, d) })})
	if err != nil {
		t.Fatalf("Chat error: %v", err)
	}

	if reqBody["stream"] != true {
		t.Fatalf("request stream = %v, want true", reqBody["stream"])
	}
	if opts, ok := reqBody["stream_options"].(map[string]any); !ok || opts["include_usage"] != true {
		t.Fatalf("stream_options = %v, want include_usage=true", reqBody["stream_options"])
	}

	// The reconstructed response must match a non-streaming parse of the
	// equivalent whole body.
	if out.Content != "hello world" {
		t.Fatalf("Content = %q, want 'hello world'", out.Content)
	}
	if strings.Join(deltas, "") != "hello world" {
		t.Fatalf("deltas joined = %q, want 'hello world'", strings.Join(deltas, ""))
	}
	if out.Usage == nil || out.Usage.TotalTokens != 15 {
		t.Fatalf("Usage = %+v, want TotalTokens=15", out.Usage)
	}
	if out.FinishReason != "stop" {
		t.Fatalf("FinishReason = %q, want stop", out.FinishReason)
	}
	if out.Status == nil || !out.Status.Success || out.Status.BytesReceived == 0 {
		t.Fatalf("Status = %+v", out.Status)
	}
	if out.Status.Model != "gpt-4o" {
		t.Fatalf("Status.Model = %q, want gpt-4o", out.Status.Model)
	}
}

func TestChat_StreamingToolCall(t *testing.T) {
	body := "data: " + `{"model":"gpt-4o","choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":"}}]}}]}` + "\n\n" +
		"data: " + `{"model":"gpt-4o","choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"NYC\"}"}}]},"finish_reason":"tool_calls"}]}` + "\n\n" +
		"data: [DONE]\n\n"

	srv := sseAzureServer(t, body, nil)
	p := NewProvider("k", srv.URL, "")
	out, err := p.Chat(context.Background(),
		[]Message{{Role: "user", Content: "weather?"}}, nil, "gpt-4o-deployment",
		map[string]any{common.TextDeltaOption: common.TextDeltaFunc(func(string) {})})
	if err != nil {
		t.Fatalf("Chat error: %v", err)
	}
	if len(out.ToolCalls) != 1 {
		t.Fatalf("len(ToolCalls) = %d, want 1", len(out.ToolCalls))
	}
	if out.ToolCalls[0].Name != "get_weather" {
		t.Fatalf("ToolCalls[0].Name = %q, want get_weather", out.ToolCalls[0].Name)
	}
	// The split arguments fragments must reassemble into the parsed object.
	if out.ToolCalls[0].Arguments["city"] != "NYC" {
		t.Fatalf("ToolCalls[0].Arguments[city] = %v, want NYC", out.ToolCalls[0].Arguments["city"])
	}
	if out.FinishReason != "tool_calls" {
		t.Fatalf("FinishReason = %q, want tool_calls", out.FinishReason)
	}
}

func TestChat_StreamingError(t *testing.T) {
	body := "data: " + `{"model":"gpt-4o","choices":[{"delta":{"content":"partial"}}]}` + "\n\n" +
		"data: " + `{"error":{"message":"model overloaded"}}` + "\n\n"

	srv := sseAzureServer(t, body, nil)
	var deltas []string
	p := NewProvider("k", srv.URL, "")
	out, err := p.Chat(context.Background(),
		[]Message{{Role: "user", Content: "hi"}}, nil, "gpt-4o-deployment",
		map[string]any{common.TextDeltaOption: common.TextDeltaFunc(func(d string) { deltas = append(deltas, d) })})
	if err == nil {
		t.Fatalf("expected error from in-band error chunk")
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
	reply := `{"model":"gpt-4o","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&reqBody)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(reply))
	}))
	t.Cleanup(srv.Close)

	p := NewProvider("k", srv.URL, "")
	out, err := p.Chat(context.Background(),
		[]Message{{Role: "user", Content: "hi"}}, nil, "gpt-4o-deployment",
		map[string]any{common.TextDeltaOption: common.TextDeltaFunc(nil)})
	if err != nil {
		t.Fatalf("Chat error: %v", err)
	}
	if _, ok := reqBody["stream"]; ok {
		t.Fatalf("stream flag set despite nil callback")
	}
	if _, ok := reqBody["stream_options"]; ok {
		t.Fatalf("stream_options set despite nil callback")
	}
	if out.Content != "ok" {
		t.Fatalf("Content = %q, want ok", out.Content)
	}
}
