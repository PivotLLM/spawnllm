package anthropicmessages

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/PivotLLM/spawnllm/common"
	"github.com/PivotLLM/spawnllm/protocoltypes"
)

var ephemeral = map[string]any{"type": "ephemeral"}

// countCacheControl counts every cache_control key anywhere in a decoded
// JSON request body, the top-level one included.
func countCacheControl(v any) int {
	n := 0
	switch x := v.(type) {
	case map[string]any:
		for k, child := range x {
			if k == "cache_control" {
				n++
			}
			n += countCacheControl(child)
		}
	case []any:
		for _, child := range x {
			n += countCacheControl(child)
		}
	}
	return n
}

func TestChat_SendsAutomaticCacheControl(t *testing.T) {
	for _, streaming := range []bool{false, true} {
		name := "single shot"
		if streaming {
			name = "streaming"
		}
		t.Run(name, func(t *testing.T) {
			var reqBody map[string]any
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				raw, err := io.ReadAll(r.Body)
				if err != nil {
					t.Errorf("read request: %v", err)
				}
				if err := json.Unmarshal(raw, &reqBody); err != nil {
					t.Errorf("decode request: %v", err)
				}
				if streaming {
					w.Header().Set("Content-Type", "text/event-stream")
					if _, err := io.WriteString(w, "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"model\":\"m\",\"usage\":{\"input_tokens\":1}}}\n\n"+
						"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":1}}\n\n"); err != nil {
						t.Errorf("write response: %v", err)
					}
					return
				}
				w.Header().Set("Content-Type", "application/json")
				if _, err := io.WriteString(w, `{"type":"message","role":"assistant","model":"m","stop_reason":"end_turn",
					"content":[{"type":"text","text":"ok"}],"usage":{"input_tokens":1,"output_tokens":1}}`); err != nil {
					t.Errorf("write response: %v", err)
				}
			}))
			defer server.Close()

			p := NewProvider("test-key", server.URL)
			messages := []Message{
				{Role: "system", Content: "You are Alice."},
				{Role: "user", Content: "Hello"},
			}
			opts := map[string]any{"max_tokens": 64}
			if streaming {
				opts[common.TextDeltaOption] = func(string) {}
			}
			if _, err := p.Chat(t.Context(), messages, nil, "m", opts); err != nil {
				t.Fatalf("Chat() error: %v", err)
			}
			if got := reqBody["cache_control"]; !reflect.DeepEqual(got, ephemeral) {
				t.Errorf("top-level cache_control = %v, want %v", got, ephemeral)
			}
			if streaming && reqBody["stream"] != true {
				t.Errorf("stream = %v, want true", reqBody["stream"])
			}
		})
	}
}

func TestBuildRequestBody_OnlyTheAutomaticBreakpoint(t *testing.T) {
	// This path sets no explicit markers (SystemParts are not sent), so the
	// request carries exactly one breakpoint, at the default TTL, whatever the
	// caller marked.
	messages := []Message{
		{Role: "system", Content: "static\n\ndynamic", SystemParts: []protocoltypes.ContentBlock{
			{Type: "text", Text: "static", CacheControl: &protocoltypes.CacheControl{Type: "ephemeral"}},
			{Type: "text", Text: "dynamic"},
		}},
		{Role: "user", Content: "Hello"},
		{Role: "assistant", Content: "Hi"},
		{Role: "user", Content: "Again"},
	}
	body, err := buildRequestBody(messages, nil, "m", map[string]any{"max_tokens": 64})
	if err != nil {
		t.Fatalf("buildRequestBody() error: %v", err)
	}
	raw, err := json.Marshal(body)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var decoded map[string]any
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if got := decoded["cache_control"]; !reflect.DeepEqual(got, ephemeral) {
		t.Errorf("top-level cache_control = %v, want %v", got, ephemeral)
	}
	if got := countCacheControl(decoded); got != 1 {
		t.Errorf("cache_control markers = %d, want 1", got)
	}
}

func TestParseResponseBody_CacheTokensInUsage(t *testing.T) {
	body := []byte(`{"type":"message","role":"assistant","model":"m","stop_reason":"end_turn",
		"content":[{"type":"text","text":"ok"}],
		"usage":{"input_tokens":10,"output_tokens":5,
			"cache_read_input_tokens":90000,"cache_creation_input_tokens":2000}}`)
	got, err := parseResponseBody(body)
	if err != nil {
		t.Fatalf("parseResponseBody() error: %v", err)
	}
	if got.Status.CacheReadTokens != 90000 || got.Status.CacheCreationTokens != 2000 {
		t.Errorf("Status cache tokens = (read %d, create %d), want (90000, 2000)",
			got.Status.CacheReadTokens, got.Status.CacheCreationTokens)
	}
	if got.Status.InputTokens != 10 {
		t.Errorf("Status.InputTokens = %d, want 10 (uncached input only)", got.Status.InputTokens)
	}
	if got.Status.CostUSD != nil {
		t.Errorf("CostUSD = %v, want nil (no cost reported)", *got.Status.CostUSD)
	}
	// The prompt's size is every input token, cached or not.
	if got.Usage.PromptTokens != 92010 {
		t.Errorf("Usage.PromptTokens = %d, want 92010", got.Usage.PromptTokens)
	}
	if got.Usage.TotalTokens != 92015 {
		t.Errorf("Usage.TotalTokens = %d, want 92015", got.Usage.TotalTokens)
	}
	if sum := got.Status.InputTokens + got.Status.CacheReadTokens + got.Status.CacheCreationTokens; sum != got.Usage.PromptTokens {
		t.Errorf("input+read+write = %d, want PromptTokens %d", sum, got.Usage.PromptTokens)
	}
}
