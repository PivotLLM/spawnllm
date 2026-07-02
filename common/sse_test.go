package common

import (
	"errors"
	"strings"
	"testing"
)

// buildSSE joins raw data payloads into an SSE stream, one `data:` line each
// followed by the dispatching blank line, matching what OpenAI emits.
func buildSSE(chunks ...string) string {
	var b strings.Builder
	for _, c := range chunks {
		b.WriteString("data: ")
		b.WriteString(c)
		b.WriteString("\n\n")
	}
	return b.String()
}

func TestAccumulateChatStream_PlainText(t *testing.T) {
	stream := buildSSE(
		`{"model":"gpt-4o","choices":[{"delta":{"role":"assistant","content":"Hello"}}]}`,
		`{"model":"gpt-4o","choices":[{"delta":{"content":", "}}]}`,
		`{"model":"gpt-4o","choices":[{"delta":{"content":"world"},"finish_reason":"stop"}]}`,
		`{"model":"gpt-4o","choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}`,
		"[DONE]",
	)

	var deltas []string
	cb := func(d string) { deltas = append(deltas, d) }

	out, _, err := AccumulateChatStream(strings.NewReader(stream), cb, nil)
	if err != nil {
		t.Fatalf("AccumulateChatStream() error = %v", err)
	}
	if out.Content != "Hello, world" {
		t.Fatalf("Content = %q, want %q", out.Content, "Hello, world")
	}
	if out.FinishReason != "stop" {
		t.Fatalf("FinishReason = %q, want stop", out.FinishReason)
	}
	if !out.Normal {
		t.Fatalf("Normal = false, want true")
	}
	if out.Usage == nil || out.Usage.TotalTokens != 5 {
		t.Fatalf("Usage = %+v, want TotalTokens=5", out.Usage)
	}
	if want := []string{"Hello", ", ", "world"}; !equalStrings(deltas, want) {
		t.Fatalf("deltas = %v, want %v", deltas, want)
	}
}

func TestAccumulateChatStream_MatchesNonStreamingParser(t *testing.T) {
	// The reconstructed body fed to ParseResponse must produce the same
	// LLMResponse as parsing the equivalent whole non-streaming body directly.
	stream := buildSSE(
		`{"model":"gpt-4o","choices":[{"delta":{"content":"Hi there"},"finish_reason":"stop"}]}`,
		`{"model":"gpt-4o","choices":[],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}`,
		"[DONE]",
	)
	streamed, _, err := AccumulateChatStream(strings.NewReader(stream), func(string) {}, nil)
	if err != nil {
		t.Fatalf("AccumulateChatStream() error = %v", err)
	}

	whole := `{"model":"gpt-4o","choices":[{"message":{"role":"assistant","content":"Hi there"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}`
	direct, err := ParseResponse(strings.NewReader(whole), nil)
	if err != nil {
		t.Fatalf("ParseResponse() error = %v", err)
	}

	if streamed.Content != direct.Content {
		t.Fatalf("Content mismatch: streamed=%q direct=%q", streamed.Content, direct.Content)
	}
	if streamed.FinishReason != direct.FinishReason {
		t.Fatalf("FinishReason mismatch: streamed=%q direct=%q", streamed.FinishReason, direct.FinishReason)
	}
	if streamed.Usage.TotalTokens != direct.Usage.TotalTokens {
		t.Fatalf("Usage mismatch: streamed=%+v direct=%+v", streamed.Usage, direct.Usage)
	}
}

func TestAccumulateChatStream_ToolCallSplitAcrossChunks(t *testing.T) {
	// id/name/type arrive in the first fragment; arguments split across three.
	stream := buildSSE(
		`{"model":"gpt-4o","choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"ci"}}]}}]}`,
		`{"model":"gpt-4o","choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"ty\":\"S"}}]}}]}`,
		`{"model":"gpt-4o","choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"F\"}"}}]},"finish_reason":"tool_calls"}]}`,
		`{"model":"gpt-4o","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`,
		"[DONE]",
	)

	out, _, err := AccumulateChatStream(strings.NewReader(stream), func(string) {}, nil)
	if err != nil {
		t.Fatalf("AccumulateChatStream() error = %v", err)
	}
	if len(out.ToolCalls) != 1 {
		t.Fatalf("len(ToolCalls) = %d, want 1", len(out.ToolCalls))
	}
	tc := out.ToolCalls[0]
	if tc.ID != "call_1" {
		t.Fatalf("ToolCalls[0].ID = %q, want call_1", tc.ID)
	}
	if tc.Name != "get_weather" {
		t.Fatalf("ToolCalls[0].Name = %q, want get_weather", tc.Name)
	}
	if tc.Arguments["city"] != "SF" {
		t.Fatalf("ToolCalls[0].Arguments[city] = %v, want SF", tc.Arguments["city"])
	}
	if out.FinishReason != "tool_calls" {
		t.Fatalf("FinishReason = %q, want tool_calls", out.FinishReason)
	}
	if out.Usage == nil || out.Usage.TotalTokens != 15 {
		t.Fatalf("Usage = %+v, want TotalTokens=15", out.Usage)
	}
}

func TestAccumulateChatStream_InBandError(t *testing.T) {
	// A content delta arrives, then an in-band error chunk aborts the stream.
	stream := buildSSE(
		`{"model":"gpt-4o","choices":[{"delta":{"content":"partial"}}]}`,
		`{"error":{"message":"rate limit exceeded","type":"rate_limit_error"}}`,
	)

	var deltas []string
	out, _, err := AccumulateChatStream(strings.NewReader(stream), func(d string) { deltas = append(deltas, d) }, nil)
	if out != nil {
		t.Fatalf("expected nil response on in-band error, got %+v", out)
	}
	var streamErr *StreamChatError
	if !errors.As(err, &streamErr) {
		t.Fatalf("error = %v (%T), want *StreamChatError", err, err)
	}
	if streamErr.Message != "rate limit exceeded" {
		t.Fatalf("error message = %q, want %q", streamErr.Message, "rate limit exceeded")
	}
	// Partial content was still delivered via the callback before the error.
	if want := []string{"partial"}; !equalStrings(deltas, want) {
		t.Fatalf("deltas = %v, want %v", deltas, want)
	}
}

func TestAccumulateChatStream_IgnoresCommentsAndBlanks(t *testing.T) {
	stream := ": keep-alive comment\n\n" +
		"data: " + `{"model":"gpt-4o","choices":[{"delta":{"content":"x"},"finish_reason":"stop"}]}` + "\n\n" +
		"\n" +
		"data: [DONE]\n\n"

	var deltas []string
	out, _, err := AccumulateChatStream(strings.NewReader(stream), func(d string) { deltas = append(deltas, d) }, nil)
	if err != nil {
		t.Fatalf("AccumulateChatStream() error = %v", err)
	}
	if out.Content != "x" {
		t.Fatalf("Content = %q, want x", out.Content)
	}
	if want := []string{"x"}; !equalStrings(deltas, want) {
		t.Fatalf("deltas = %v, want %v", deltas, want)
	}
}

func equalStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
