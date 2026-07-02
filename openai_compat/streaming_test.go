package openai_compat

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/PivotLLM/spawnllm/common"
)

// sseServer returns an httptest server that writes the given raw SSE body for
// /chat/completions requests, capturing the decoded request body.
func sseServer(t *testing.T, body string, captured *map[string]any) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if captured != nil {
			_ = json.NewDecoder(r.Body).Decode(captured)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(body))
	}))
}

func TestProviderChat_StreamingPlainText(t *testing.T) {
	body := "data: " + `{"model":"gpt-4o","choices":[{"delta":{"content":"Hel"}}]}` + "\n\n" +
		"data: " + `{"model":"gpt-4o","choices":[{"delta":{"content":"lo"},"finish_reason":"stop"}]}` + "\n\n" +
		"data: " + `{"model":"gpt-4o","choices":[],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}` + "\n\n" +
		"data: [DONE]\n\n"

	var reqBody map[string]any
	server := sseServer(t, body, &reqBody)
	defer server.Close()

	var deltas []string
	p := NewProvider("key", server.URL, "")
	out, err := p.Chat(t.Context(), []Message{{Role: "user", Content: "hi"}}, nil, "gpt-4o",
		map[string]any{common.TextDeltaOption: common.TextDeltaFunc(func(d string) { deltas = append(deltas, d) })})
	if err != nil {
		t.Fatalf("Chat() error = %v", err)
	}

	if reqBody["stream"] != true {
		t.Fatalf("request stream = %v, want true", reqBody["stream"])
	}
	if _, ok := reqBody["stream_options"]; !ok {
		t.Fatalf("expected stream_options with include_usage in request")
	}
	if out.Content != "Hello" {
		t.Fatalf("Content = %q, want Hello", out.Content)
	}
	if strings.Join(deltas, "") != "Hello" {
		t.Fatalf("deltas joined = %q, want Hello", strings.Join(deltas, ""))
	}
	if out.Status == nil || !out.Status.Success {
		t.Fatalf("Status.Success not set: %+v", out.Status)
	}
	if out.Status.BytesReceived == 0 {
		t.Fatalf("BytesReceived not measured")
	}
	if out.Usage == nil || out.Usage.TotalTokens != 3 {
		t.Fatalf("Usage = %+v, want TotalTokens=3", out.Usage)
	}
}

func TestProviderChat_NonStreamingWhenNoCallback(t *testing.T) {
	// Nil callback in the option must NOT trigger streaming; the request must
	// omit the stream flag and use the single-shot JSON handler.
	var reqBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&reqBody)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"model":"gpt-4o","choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]}`))
	}))
	defer server.Close()

	p := NewProvider("key", server.URL, "")
	out, err := p.Chat(t.Context(), []Message{{Role: "user", Content: "hi"}}, nil, "gpt-4o",
		map[string]any{common.TextDeltaOption: common.TextDeltaFunc(nil)})
	if err != nil {
		t.Fatalf("Chat() error = %v", err)
	}
	if _, ok := reqBody["stream"]; ok {
		t.Fatalf("stream flag set despite nil callback")
	}
	if out.Content != "ok" {
		t.Fatalf("Content = %q, want ok", out.Content)
	}
}

func TestProviderChat_StreamingInBandError(t *testing.T) {
	body := "data: " + `{"model":"gpt-4o","choices":[{"delta":{"content":"partial"}}]}` + "\n\n" +
		"data: " + `{"error":{"message":"boom"}}` + "\n\n"

	server := sseServer(t, body, nil)
	defer server.Close()

	var deltas []string
	p := NewProvider("key", server.URL, "")
	out, err := p.Chat(t.Context(), []Message{{Role: "user", Content: "hi"}}, nil, "gpt-4o",
		map[string]any{common.TextDeltaOption: common.TextDeltaFunc(func(d string) { deltas = append(deltas, d) })})
	if err == nil {
		t.Fatalf("expected error from in-band error chunk")
	}
	if !strings.Contains(err.Error(), "boom") {
		t.Fatalf("error = %v, want to contain 'boom'", err)
	}
	if out == nil || out.Status == nil || out.Status.Success {
		t.Fatalf("expected failed status, got %+v", out.Status)
	}
	// Partial content was delivered before the error surfaced.
	if strings.Join(deltas, "") != "partial" {
		t.Fatalf("deltas joined = %q, want partial", strings.Join(deltas, ""))
	}
}
