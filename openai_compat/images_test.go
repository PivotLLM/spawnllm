package openai_compat

import (
	"bytes"
	"encoding/base64"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/PivotLLM/spawnllm/common"
)

var pngBytes = []byte("\x89PNG\r\n\x1a\nnot-really-a-png")

func pngDataURL() string {
	return "data:image/png;base64," + base64.StdEncoding.EncodeToString(pngBytes)
}

// TestProviderChat_GeneratedImages checks the OpenRouter image-output shape
// (message.images, delta.images when streaming) reaches LLMResponse.Images,
// alongside the text, in both modes.
func TestProviderChat_GeneratedImages(t *testing.T) {
	nonStreaming := `{"id":"gen-1","model":"google/gemini-2.5-flash-image","choices":[{"index":0,"finish_reason":"stop",` +
		`"message":{"role":"assistant","content":"A cat.","images":[{"type":"image_url","image_url":{"url":"` + pngDataURL() + `"}}]}}],` +
		`"usage":{"prompt_tokens":5,"completion_tokens":1290,"total_tokens":1295}}`
	streaming := "data: " + `{"model":"google/gemini-2.5-flash-image","choices":[{"index":0,"delta":{"role":"assistant","content":"A cat."}}]}` + "\n\n" +
		"data: " + `{"model":"google/gemini-2.5-flash-image","choices":[{"index":0,"delta":{"role":"assistant","content":"","images":[{"type":"image_url","index":0,"image_url":{"url":"` + pngDataURL() + `"}}]},"finish_reason":"stop"}]}` + "\n\n" +
		"data: " + `{"model":"google/gemini-2.5-flash-image","choices":[],"usage":{"prompt_tokens":5,"completion_tokens":1290,"total_tokens":1295}}` + "\n\n" +
		"data: [DONE]\n\n"

	tests := []struct {
		name   string
		body   string
		stream bool
	}{
		{name: "non-streaming", body: nonStreaming},
		{name: "streaming", body: streaming, stream: true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				if tc.stream {
					w.Header().Set("Content-Type", "text/event-stream")
				} else {
					w.Header().Set("Content-Type", "application/json")
				}
				if _, err := w.Write([]byte(tc.body)); err != nil {
					t.Errorf("write response: %v", err)
				}
			}))
			defer srv.Close()

			opts := map[string]any{}
			if tc.stream {
				opts[common.TextDeltaOption] = func(string) {}
			}
			p := NewProvider("key", srv.URL, "")
			out, err := p.Chat(t.Context(), []Message{{Role: "user", Content: "draw a cat"}}, nil,
				"google/gemini-2.5-flash-image", opts)
			if err != nil {
				t.Fatalf("Chat() error = %v", err)
			}
			if out.Content != "A cat." {
				t.Errorf("Content = %q, want %q", out.Content, "A cat.")
			}
			if len(out.Images) != 1 {
				t.Fatalf("Images = %d, want 1", len(out.Images))
			}
			if out.Images[0].ContentType != "image/png" || !bytes.Equal(out.Images[0].Data, pngBytes) {
				t.Errorf("Images[0] = %q %q, want image/png and the PNG bytes", out.Images[0].ContentType, out.Images[0].Data)
			}
			if !out.Normal {
				t.Errorf("Normal = false, want true")
			}
		})
	}
}
