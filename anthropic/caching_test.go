package anthropicprovider

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"

	"github.com/PivotLLM/spawnllm/protocoltypes"
)

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

// paramsJSON serializes params exactly as the SDK sends them.
func paramsJSON(t *testing.T, params anthropic.MessageNewParams) map[string]any {
	t.Helper()
	data, err := json.Marshal(params)
	if err != nil {
		t.Fatalf("marshal params: %v", err)
	}
	var body map[string]any
	if err := json.Unmarshal(data, &body); err != nil {
		t.Fatalf("unmarshal params: %v", err)
	}
	return body
}

// systemBlocks returns the request's system blocks.
func systemBlocks(t *testing.T, body map[string]any) []map[string]any {
	t.Helper()
	list, ok := body["system"].([]any)
	if !ok {
		t.Fatalf("system = %T, want a list of blocks", body["system"])
	}
	blocks := make([]map[string]any, 0, len(list))
	for i, item := range list {
		blk, ok := item.(map[string]any)
		if !ok {
			t.Fatalf("system[%d] = %T, want an object", i, item)
		}
		blocks = append(blocks, blk)
	}
	return blocks
}

var ephemeral = map[string]any{"type": "ephemeral"}

func TestChat_SendsAutomaticCacheControl(t *testing.T) {
	var reqBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read request: %v", err)
		}
		if err := json.Unmarshal(raw, &reqBody); err != nil {
			t.Errorf("decode request: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		if _, err := io.WriteString(w, `{"id":"m","type":"message","role":"assistant","model":"claude-sonnet-4-6",
			"stop_reason":"end_turn","content":[{"type":"text","text":"ok"}],
			"usage":{"input_tokens":1,"output_tokens":1}}`); err != nil {
			t.Errorf("write response: %v", err)
		}
	}))
	defer server.Close()

	provider := NewProviderWithClient(createAnthropicTestClient(server.URL, "test-token"))
	messages := []Message{
		{Role: "system", Content: "You are Alice."},
		{Role: "user", Content: "Hello"},
	}
	if _, err := provider.Chat(t.Context(), messages, nil, "claude-sonnet-4.6", map[string]any{"max_tokens": 64}); err != nil {
		t.Fatalf("Chat() error: %v", err)
	}
	if got := reqBody["cache_control"]; !reflect.DeepEqual(got, ephemeral) {
		t.Errorf("top-level cache_control = %v, want %v", got, ephemeral)
	}
}

func TestBuildParams_CacheBreakpointsWithSystemParts(t *testing.T) {
	tests := []struct {
		name         string
		explicit     int
		wantTopLevel bool
		wantMarkers  int
	}{
		{name: "no explicit markers", explicit: 0, wantTopLevel: true, wantMarkers: 1},
		{name: "one explicit marker", explicit: 1, wantTopLevel: true, wantMarkers: 2},
		{name: "three explicit markers", explicit: 3, wantTopLevel: true, wantMarkers: 4},
		{name: "four explicit markers fill every slot", explicit: 4, wantTopLevel: false, wantMarkers: 4},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// One unmarked part plus tt.explicit marked ones.
			parts := []protocoltypes.ContentBlock{{Type: "text", Text: "dynamic"}}
			for range tt.explicit {
				parts = append(parts, protocoltypes.ContentBlock{
					Type: "text", Text: "static", CacheControl: &protocoltypes.CacheControl{Type: "ephemeral"},
				})
			}
			messages := []Message{
				{Role: "system", SystemParts: parts},
				{Role: "user", Content: "Hello"},
			}
			params, err := buildParams(messages, nil, "claude-sonnet-4.6", map[string]any{"max_tokens": 64})
			if err != nil {
				t.Fatalf("buildParams() error: %v", err)
			}
			body := paramsJSON(t, params)

			top, hasTop := body["cache_control"]
			if hasTop != tt.wantTopLevel {
				t.Fatalf("top-level cache_control present = %v, want %v", hasTop, tt.wantTopLevel)
			}
			if hasTop && !reflect.DeepEqual(top, ephemeral) {
				t.Errorf("top-level cache_control = %v, want %v (default TTL)", top, ephemeral)
			}
			if got := countCacheControl(body); got != tt.wantMarkers {
				t.Errorf("cache_control markers = %d, want %d", got, tt.wantMarkers)
			}
			// Every explicit marker has the default TTL, the same as the
			// top-level one, so a marker on the last block cannot conflict.
			for i, blk := range systemBlocks(t, body) {
				cc, ok := blk["cache_control"]
				if ok && !reflect.DeepEqual(cc, ephemeral) {
					t.Errorf("system[%d].cache_control = %v, want %v", i, cc, ephemeral)
				}
			}
		})
	}
}

func TestBuildParams_SystemOnlyMarkerOnLastBlockKeepsDefaultTTL(t *testing.T) {
	// With no messages after it, the marked system block is the last block,
	// where automatic caching lands; both must have the same (default) TTL.
	messages := []Message{{Role: "system", SystemParts: []protocoltypes.ContentBlock{
		{Type: "text", Text: "static", CacheControl: &protocoltypes.CacheControl{Type: "ephemeral"}},
	}}}
	params, err := buildParams(messages, nil, "claude-sonnet-4.6", map[string]any{"max_tokens": 64})
	if err != nil {
		t.Fatalf("buildParams() error: %v", err)
	}
	body := paramsJSON(t, params)
	if !reflect.DeepEqual(body["cache_control"], ephemeral) {
		t.Errorf("top-level cache_control = %v, want %v", body["cache_control"], ephemeral)
	}
	last := systemBlocks(t, body)[0]["cache_control"]
	if !reflect.DeepEqual(last, ephemeral) {
		t.Errorf("last block cache_control = %v, want %v", last, ephemeral)
	}
}

func TestParseResponse_CacheTokens(t *testing.T) {
	resp := &anthropic.Message{
		Usage: anthropic.Usage{
			InputTokens:              10,
			OutputTokens:             5,
			CacheReadInputTokens:     90000,
			CacheCreationInputTokens: 2000,
		},
	}
	got := parseResponse(resp)
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
