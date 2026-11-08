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
		name    string
		result  string
		stream  bool
		wantURL string // empty: the decoded PNG is expected
	}{
		{name: "non-streaming", result: good},
		{name: "streaming", result: good, stream: true},
		{name: "malformed base64 is passed through", result: "@@not base64@@", wantURL: "data:;base64,@@not base64@@"},
		{name: "malformed base64 is passed through when streaming", result: "@@not base64@@", stream: true, wantURL: "data:;base64,@@not base64@@"},
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
			if len(out.Images) != 1 {
				t.Fatalf("Images = %d, want 1", len(out.Images))
			}
			img := out.Images[0]
			if tc.wantURL != "" {
				if img.URL != tc.wantURL || img.Data != nil || img.Size != -1 {
					t.Errorf("Images[0] = %+v, want URL %q only and Size -1", img, tc.wantURL)
				}
				return
			}
			if img.ContentType != "image/png" || !bytes.Equal(img.Data, pngBytes) || img.Size != int64(len(pngBytes)) || img.URL != "" {
				t.Errorf("Images[0] = %+v, want image/png and the PNG bytes", img)
			}
		})
	}
}
