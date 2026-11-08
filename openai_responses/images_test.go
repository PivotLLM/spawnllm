package openai_responses

import (
	"bytes"
	"context"
	"encoding/base64"
	"testing"

	"github.com/PivotLLM/spawnllm/common"
)

var pngBytes = []byte("\x89PNG\r\n\x1a\nnot-really-a-png")

// imageResponse is a completed Responses API body with an image_generation_call
// output item (result is base64) beside the assistant message.
func imageResponse(result string) string {
	return `{"id":"resp_1","status":"completed","output":[` +
		`{"id":"ig_1","type":"image_generation_call","status":"completed","revised_prompt":"a cat","result":"` + result + `"},` +
		`{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"output_text","text":"A cat."}]}],` +
		`"usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15}}`
}

func TestChat_GeneratedImages(t *testing.T) {
	good := base64.StdEncoding.EncodeToString(pngBytes)
	tests := []struct {
		name       string
		result     string
		stream     bool
		wantImages int
	}{
		{name: "non-streaming", result: good, wantImages: 1},
		{name: "streaming", result: good, stream: true, wantImages: 1},
		{name: "malformed base64 is skipped", result: "@@not base64@@", wantImages: 0},
		{name: "malformed base64 is skipped when streaming", result: "@@not base64@@", stream: true, wantImages: 0},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			final := imageResponse(tc.result)
			var p *Provider
			opts := map[string]any{}
			if tc.stream {
				body := "data: " + `{"type":"response.output_text.delta","delta":"A cat."}` + "\n\n" +
					"data: " + `{"type":"response.completed","response":` + final + `}` + "\n\n"
				p = NewProvider("k", sseResponsesServer(t, body, nil).URL, "")
				opts[common.TextDeltaOption] = func(string) {}
			} else {
				p, _ = captureServer(t, final)
			}
			out, err := p.Chat(context.Background(), []Message{{Role: "user", Content: "draw a cat"}}, nil, "gpt-5", opts)
			if err != nil {
				t.Fatalf("Chat: %v", err)
			}
			if out.Content != "A cat." {
				t.Errorf("Content = %q, want %q", out.Content, "A cat.")
			}
			if !out.Normal {
				t.Errorf("Normal = false, want true")
			}
			if len(out.Images) != tc.wantImages {
				t.Fatalf("Images = %d, want %d", len(out.Images), tc.wantImages)
			}
			if tc.wantImages == 1 && (out.Images[0].ContentType != "image/png" || !bytes.Equal(out.Images[0].Data, pngBytes)) {
				t.Errorf("Images[0] = %q %q, want image/png and the PNG bytes", out.Images[0].ContentType, out.Images[0].Data)
			}
		})
	}
}
