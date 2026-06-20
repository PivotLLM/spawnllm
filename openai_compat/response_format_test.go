// ClawEh
// License: MIT

package openai_compat

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

// captureChatRequestBody runs a single Chat call against an httptest server
// configured with the given protocol+capability options and returns the JSON
// request body the server received. Centralised so each response_format test
// stays focused on the assertion.
func captureChatRequestBody(
	t *testing.T,
	opts []Option,
	options map[string]any,
	model string,
) map[string]any {
	t.Helper()
	var captured map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&captured); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{
				{"message": map[string]any{"content": "ok"}, "finish_reason": "stop"},
			},
		})
	}))
	t.Cleanup(server.Close)

	p := NewProvider("key", server.URL, "", opts...)
	if _, err := p.Chat(t.Context(), []Message{{Role: "user", Content: "hi"}}, nil, model, options); err != nil {
		t.Fatalf("Chat() error = %v", err)
	}
	return captured
}

func wantResponseFormatJSON(t *testing.T, body map[string]any) {
	t.Helper()
	rf, ok := body["response_format"].(map[string]any)
	if !ok {
		t.Fatalf("response_format missing or wrong type; body=%v", body)
	}
	if rf["type"] != "json_object" {
		t.Fatalf("response_format.type = %v, want json_object", rf["type"])
	}
}

func wantNoResponseFormat(t *testing.T, body map[string]any) {
	t.Helper()
	if _, has := body["response_format"]; has {
		t.Fatalf("response_format must be omitted; body=%v", body)
	}
}

// TestCompressionRequest_SetsResponseFormat_XAI — xAI is a capable protocol
// by default, so the per-call option turns on response_format on the wire.
func TestCompressionRequest_SetsResponseFormat_XAI(t *testing.T) {
	body := captureChatRequestBody(t,
		[]Option{WithProtocol("xai")},
		map[string]any{ResponseFormatJSONObjectOption: true},
		"grok-4",
	)
	wantResponseFormatJSON(t, body)
}

// TestCompressionRequest_SetsResponseFormat_OpenAI — same as xAI but for
// the openai protocol.
func TestCompressionRequest_SetsResponseFormat_OpenAI(t *testing.T) {
	body := captureChatRequestBody(t,
		[]Option{WithProtocol("openai-chat")},
		map[string]any{ResponseFormatJSONObjectOption: true},
		"gpt-4o",
	)
	wantResponseFormatJSON(t, body)
}

// TestCompressionRequest_OmitsResponseFormat_Openrouter — openrouter is
// configurable; default is off, so the option is silently dropped.
func TestCompressionRequest_OmitsResponseFormat_Openrouter(t *testing.T) {
	body := captureChatRequestBody(t,
		[]Option{WithProtocol("openrouter")},
		map[string]any{ResponseFormatJSONObjectOption: true},
		"openrouter/auto",
	)
	wantNoResponseFormat(t, body)
}

// TestCompressionRequest_OmitsResponseFormat_AdHocCompatProtocol — protocols
// added via openai_compat_protocols default to off, matching the
// conservative gate for unknown vendors.
func TestCompressionRequest_OmitsResponseFormat_AdHocCompatProtocol(t *testing.T) {
	body := captureChatRequestBody(t,
		[]Option{WithProtocol("vendor-x")},
		map[string]any{ResponseFormatJSONObjectOption: true},
		"vendor-x/model-1",
	)
	wantNoResponseFormat(t, body)
}

// TestCompressionRequest_OmitsResponseFormat_Anthropic — when an Anthropic
// model routes through the openai-compat HTTP path (api_key auth), the
// "anthropic" protocol is not on the capable list and the request stays
// quiet. Anthropic native (anthropic-messages, OAuth Claude) bypasses
// this provider entirely; the option is a no-op there by construction.
func TestCompressionRequest_OmitsResponseFormat_Anthropic(t *testing.T) {
	body := captureChatRequestBody(t,
		[]Option{WithProtocol("anthropic")},
		map[string]any{ResponseFormatJSONObjectOption: true},
		"anthropic/claude-sonnet-4.6",
	)
	wantNoResponseFormat(t, body)
}

// TestCompressionRequest_RespectsConfigOverride — a per-protocol override
// flips openrouter on and the field is emitted.
func TestCompressionRequest_RespectsConfigOverride(t *testing.T) {
	body := captureChatRequestBody(t,
		[]Option{WithProtocol("openrouter"), WithResponseFormatJSONCapable(true)},
		map[string]any{ResponseFormatJSONObjectOption: true},
		"openrouter/auto",
	)
	wantResponseFormatJSON(t, body)
}

// TestCompressionRequest_OmitsWhenOptionUnset — without the per-call
// option, even capable protocols don't emit response_format.
func TestCompressionRequest_OmitsWhenOptionUnset(t *testing.T) {
	body := captureChatRequestBody(t,
		[]Option{WithProtocol("xai")},
		nil,
		"grok-4",
	)
	wantNoResponseFormat(t, body)
}

// TestDefaultSupportsResponseFormatJSON — unit-tests the protocol allowlist
// directly so a config-driven override is the only way the answer changes
// for protocols outside it.
func TestDefaultSupportsResponseFormatJSON(t *testing.T) {
	tests := []struct {
		protocol string
		want     bool
	}{
		{"openai-chat", true},
		{"xai", true},
		{"openrouter", false},
		{"groq", false},
		{"litellm", false},
		{"anthropic", false},
		{"", false},
		{"vendor-x", false},
	}
	for _, tc := range tests {
		t.Run(tc.protocol, func(t *testing.T) {
			if got := defaultSupportsResponseFormatJSON(tc.protocol); got != tc.want {
				t.Errorf("defaultSupportsResponseFormatJSON(%q) = %v, want %v", tc.protocol, got, tc.want)
			}
		})
	}
}

// TestWantsJSONObject — option type tolerance: bool true, string "true",
// other values report false.
func TestWantsJSONObject(t *testing.T) {
	tests := []struct {
		name string
		in   map[string]any
		want bool
	}{
		{"bool true", map[string]any{ResponseFormatJSONObjectOption: true}, true},
		{"bool false", map[string]any{ResponseFormatJSONObjectOption: false}, false},
		{"string true", map[string]any{ResponseFormatJSONObjectOption: "true"}, true},
		{"string other", map[string]any{ResponseFormatJSONObjectOption: "yes"}, false},
		{"missing", map[string]any{}, false},
		{"nil map", nil, false},
		{"wrong type", map[string]any{ResponseFormatJSONObjectOption: 1}, false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := wantsJSONObject(tc.in); got != tc.want {
				t.Errorf("wantsJSONObject(%v) = %v, want %v", tc.in, got, tc.want)
			}
		})
	}
}
