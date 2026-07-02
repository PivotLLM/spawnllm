package common

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
)

// --- Server-Sent Events (SSE) parsing ---

// sseData reads one SSE `data:` payload at a time from r. It skips blank lines
// and comment lines (those beginning with ':'), and strips the single optional
// space after "data:". Multi-line data fields are joined with '\n' per the SSE
// spec, flushed at the dispatching blank line. It returns io.EOF when the
// stream ends. Non-data fields (event:, id:, retry:) are ignored — the OpenAI
// stream carries its payload entirely in data lines.
type SSEReader struct {
	sc      *bufio.Scanner
	pending strings.Builder
	have    bool
}

// NewSSEReader wraps r as a Server-Sent Events reader that yields one `data:`
// payload per Next() call. Shared by the chat/completions and Responses API
// streaming paths.
func NewSSEReader(r io.Reader) *SSEReader {
	sc := bufio.NewScanner(r)
	// SSE chunks are small, but a single JSON payload can exceed bufio's default
	// 64 KiB token cap (large tool-call arguments). Raise the max line size.
	sc.Buffer(make([]byte, 0, 64*1024), 8*1024*1024)
	return &SSEReader{sc: sc}
}

// Next returns the next complete SSE data payload. The bool is false at EOF.
func (s *SSEReader) Next() (string, bool, error) {
	for s.sc.Scan() {
		line := s.sc.Text()
		// Blank line dispatches the buffered event.
		if line == "" {
			if s.have {
				out := s.pending.String()
				s.pending.Reset()
				s.have = false
				return out, true, nil
			}
			continue
		}
		// Comment line.
		if strings.HasPrefix(line, ":") {
			continue
		}
		if strings.HasPrefix(line, "data:") {
			payload := strings.TrimPrefix(line, "data:")
			payload = strings.TrimPrefix(payload, " ")
			if s.have {
				s.pending.WriteByte('\n')
			}
			s.pending.WriteString(payload)
			s.have = true
		}
		// Other fields (event:, id:, retry:) are ignored.
	}
	if err := s.sc.Err(); err != nil {
		return "", false, err
	}
	// Flush any final event not terminated by a trailing blank line.
	if s.have {
		out := s.pending.String()
		s.pending.Reset()
		s.have = false
		return out, true, nil
	}
	return "", false, io.EOF
}

// --- Chat Completions streaming accumulation ---

// chatStreamChunk is the subset of a chat/completions streaming chunk we read.
type chatStreamChunk struct {
	Model   string `json:"model"`
	Choices []struct {
		Delta struct {
			Role             string           `json:"role"`
			Content          string           `json:"content"`
			ReasoningContent string           `json:"reasoning_content"`
			Reasoning        string           `json:"reasoning"`
			ToolCalls        []streamToolCall `json:"tool_calls"`
		} `json:"delta"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage *json.RawMessage `json:"usage"`
	// In-band error envelope: some upstreams emit {"error":{...}} as a data
	// chunk mid-stream instead of an HTTP status.
	Error *json.RawMessage `json:"error"`
}

type streamToolCall struct {
	Index    int    `json:"index"`
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function *struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

// accToolCall accumulates one tool call across chunks, keyed by its index.
type accToolCall struct {
	id   string
	typ  string
	name string
	args strings.Builder
}

// StreamChatError reports an in-band `{"error":...}` chunk observed in the SSE
// stream. Content emitted before the error may already have been delivered via
// the delta callback. The provider maps this to its usual HTTP-error path.
type StreamChatError struct {
	Message string
}

func (e *StreamChatError) Error() string { return e.Message }

// AccumulateChatStream consumes a chat/completions SSE stream from r, firing cb
// with each non-empty content delta (in order), and reconstructs the equivalent
// NON-streaming chat/completions JSON body. That reconstructed body is then fed
// through ParseResponse so all existing post-processing (content-sniffing,
// legacy function_call synthesis, tool-name filtering, sanitization) is reused
// verbatim. toolNames is forwarded to ParseResponse for the content-sniff
// cross-check.
//
// An in-band error chunk aborts accumulation and returns *StreamChatError.
// The reconstructed body bytes are returned (best-effort, possibly partial) so
// callers can log what was assembled.
func AccumulateChatStream(r io.Reader, cb TextDeltaFunc, toolNames map[string]struct{}) (*LLMResponse, []byte, error) {
	sr := NewSSEReader(r)

	var (
		model        string
		content      strings.Builder
		reasoningC   strings.Builder
		reasoning    strings.Builder
		finishReason string
		usageRaw     json.RawMessage
		toolOrder    []int
		toolByIndex  = map[int]*accToolCall{}
	)

	for {
		payload, ok, err := sr.Next()
		if !ok {
			// io.EOF is the normal end-of-stream signal; any other error is real.
			if err != nil && !errors.Is(err, io.EOF) {
				return nil, nil, fmt.Errorf("failed to read stream: %w", err)
			}
			break
		}
		if payload == "[DONE]" {
			break
		}

		var chunk chatStreamChunk
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			// Skip anything that isn't a JSON object we understand (keep-alives,
			// framing noise). ParseResponse handles the aggregate later.
			continue
		}

		if chunk.Error != nil {
			return nil, nil, &StreamChatError{Message: streamErrorMessage(*chunk.Error)}
		}

		if chunk.Model != "" {
			model = chunk.Model
		}
		if chunk.Usage != nil {
			usageRaw = *chunk.Usage
		}

		for _, ch := range chunk.Choices {
			if ch.Delta.Content != "" {
				content.WriteString(ch.Delta.Content)
				cb(ch.Delta.Content)
			}
			if ch.Delta.ReasoningContent != "" {
				reasoningC.WriteString(ch.Delta.ReasoningContent)
			}
			if ch.Delta.Reasoning != "" {
				reasoning.WriteString(ch.Delta.Reasoning)
			}
			for _, tc := range ch.Delta.ToolCalls {
				acc, seen := toolByIndex[tc.Index]
				if !seen {
					acc = &accToolCall{}
					toolByIndex[tc.Index] = acc
					toolOrder = append(toolOrder, tc.Index)
				}
				if tc.ID != "" {
					acc.id = tc.ID
				}
				if tc.Type != "" {
					acc.typ = tc.Type
				}
				if tc.Function != nil {
					if tc.Function.Name != "" {
						acc.name = tc.Function.Name
					}
					acc.args.WriteString(tc.Function.Arguments)
				}
			}
			// Take the last non-empty finish_reason we observe.
			if ch.FinishReason != "" {
				finishReason = ch.FinishReason
			}
		}
	}

	body := reconstructChatBody(model, content.String(), reasoningC.String(),
		reasoning.String(), finishReason, usageRaw, toolOrder, toolByIndex)

	out, err := ParseResponse(strings.NewReader(string(body)), toolNames)
	if err != nil {
		return nil, body, fmt.Errorf("failed to parse reconstructed stream body: %w", err)
	}
	return out, body, nil
}

// reconstructChatBody assembles a non-streaming chat/completions JSON body from
// the accumulated stream state. The shape mirrors exactly what ParseResponse
// expects (choices[0].message.{content,tool_calls}, finish_reason, usage,
// model), so no parsing logic is duplicated.
func reconstructChatBody(
	model, content, reasoningContent, reasoning, finishReason string,
	usageRaw json.RawMessage,
	toolOrder []int,
	toolByIndex map[int]*accToolCall,
) []byte {
	message := map[string]any{
		"role":    "assistant",
		"content": content,
	}
	if reasoningContent != "" {
		message["reasoning_content"] = reasoningContent
	}
	if reasoning != "" {
		message["reasoning"] = reasoning
	}
	if len(toolOrder) > 0 {
		calls := make([]map[string]any, 0, len(toolOrder))
		for _, idx := range toolOrder {
			acc := toolByIndex[idx]
			typ := acc.typ
			if typ == "" {
				typ = "function"
			}
			calls = append(calls, map[string]any{
				"id":   acc.id,
				"type": typ,
				"function": map[string]any{
					"name":      acc.name,
					"arguments": acc.args.String(),
				},
			})
		}
		message["tool_calls"] = calls
	}

	choice := map[string]any{
		"message":       message,
		"finish_reason": finishReason,
	}
	body := map[string]any{
		"model":   model,
		"choices": []any{choice},
	}
	if len(usageRaw) > 0 {
		body["usage"] = usageRaw
	}

	out, err := json.Marshal(body)
	if err != nil {
		// Marshaling a map of plain strings/maps cannot realistically fail; fall
		// back to an empty choices body so ParseResponse yields a benign result.
		return []byte(`{"choices":[]}`)
	}
	return out
}

// streamErrorMessage extracts a human-readable message from an in-band error
// envelope. Accepts {"message":"..."} or a bare string; otherwise returns the
// raw JSON.
func streamErrorMessage(raw json.RawMessage) string {
	var obj struct {
		Message string `json:"message"`
	}
	if err := json.Unmarshal(raw, &obj); err == nil && obj.Message != "" {
		return obj.Message
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil && s != "" {
		return s
	}
	return string(raw)
}
