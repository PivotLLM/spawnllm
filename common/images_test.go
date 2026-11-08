package common

import (
	"bytes"
	"encoding/base64"
	"strings"
	"testing"
)

// pngBytes starts with the PNG signature, so http.DetectContentType sniffs it
// as image/png.
var pngBytes = []byte("\x89PNG\r\n\x1a\nnot-really-a-png")

func pngDataURL() string {
	return "data:image/png;base64," + base64.StdEncoding.EncodeToString(pngBytes)
}

// captureImageWarns swaps the image WARN hook for one that records each
// skipped image's reason. Tests using it must not run in parallel.
func captureImageWarns(t *testing.T) *[]map[string]any {
	t.Helper()
	var got []map[string]any
	prev := imageWarnFn
	imageWarnFn = func(_, _ string, fields map[string]any) { got = append(got, fields) }
	t.Cleanup(func() { imageWarnFn = prev })
	return &got
}

func TestImageCollector_AddURL(t *testing.T) {
	tests := []struct {
		name       string
		url        string
		wantType   string // empty: no image kept
		wantReason string // empty: no WARN
	}{
		{name: "png data URL", url: pngDataURL(), wantType: "image/png"},
		{name: "declared type wins", url: "data:image/webp;base64," + base64.StdEncoding.EncodeToString(pngBytes), wantType: "image/webp"},
		{name: "no declared type is sniffed", url: "data:;base64," + base64.StdEncoding.EncodeToString(pngBytes), wantType: "image/png"},
		{name: "http URL not fetched", url: "https://cdn.example.com/img.png?sig=secret", wantReason: "image given as a remote URL, not fetched"},
		{name: "malformed base64", url: "data:image/png;base64,@@not base64@@", wantReason: "image base64 does not decode"},
		{name: "not base64 encoded", url: "data:image/png,%89PNG", wantReason: "image data URL is not base64"},
		{name: "no comma", url: "data:image/png;base64", wantReason: "image data URL is not base64"},
		{name: "empty", url: "data:image/png;base64,", wantReason: "image is empty"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			warns := captureImageWarns(t)
			c := NewImageCollector("gemini-image")
			c.AddURL(tc.url)
			imgs := c.Images()
			if tc.wantType == "" {
				if len(imgs) != 0 {
					t.Fatalf("Images() = %d, want none", len(imgs))
				}
			} else {
				if len(imgs) != 1 {
					t.Fatalf("Images() = %d, want 1", len(imgs))
				}
				if imgs[0].ContentType != tc.wantType {
					t.Errorf("ContentType = %q, want %q", imgs[0].ContentType, tc.wantType)
				}
				if !bytes.Equal(imgs[0].Data, pngBytes) {
					t.Errorf("Data = %q, want %q", imgs[0].Data, pngBytes)
				}
			}
			if tc.wantReason == "" {
				if len(*warns) != 0 {
					t.Fatalf("WARNs = %v, want none", *warns)
				}
				return
			}
			if len(*warns) != 1 {
				t.Fatalf("WARNs = %v, want 1", *warns)
			}
			w := (*warns)[0]
			if w["reason"] != tc.wantReason {
				t.Errorf("reason = %v, want %q", w["reason"], tc.wantReason)
			}
			if w["model"] != "gemini-image" {
				t.Errorf("model = %v, want gemini-image", w["model"])
			}
			if u, ok := w["url"].(string); ok && strings.Contains(u, "secret") {
				t.Errorf("logged URL %q keeps the query", u)
			}
		})
	}
}

func TestImageCollector_PerImageCap(t *testing.T) {
	warns := captureImageWarns(t)
	c := NewImageCollector("m")
	c.AddBase64(base64.StdEncoding.EncodeToString(make([]byte, MaxImageBytes+1)), "image/png")
	c.AddBase64(base64.StdEncoding.EncodeToString(make([]byte, MaxImageBytes)), "image/png")
	if got := len(c.Images()); got != 1 {
		t.Fatalf("Images() = %d, want 1 (the image at the cap)", got)
	}
	if len(*warns) != 1 || (*warns)[0]["reason"] != "image over the size cap" {
		t.Fatalf("WARNs = %v, want one over the size cap", *warns)
	}
}

func TestImageCollector_PerResponseCap(t *testing.T) {
	warns := captureImageWarns(t)
	c := NewImageCollector("m")
	full := base64.StdEncoding.EncodeToString(make([]byte, MaxImageBytes))
	small := base64.StdEncoding.EncodeToString(make([]byte, 1))
	for range MaxImagesBytes / MaxImageBytes {
		c.AddBase64(full, "image/png")
	}
	room := MaxImagesBytes % MaxImageBytes
	c.AddBase64(base64.StdEncoding.EncodeToString(make([]byte, room+1)), "image/png") // over by one
	c.AddBase64(small, "image/png")                                                   // still fits
	want := MaxImagesBytes/MaxImageBytes + 1
	if got := len(c.Images()); got != want {
		t.Fatalf("Images() = %d, want %d", got, want)
	}
	if len(*warns) != 1 || (*warns)[0]["reason"] != "images over the per-response size cap" {
		t.Fatalf("WARNs = %v, want one over the per-response cap", *warns)
	}
}

func TestParseResponse_Images(t *testing.T) {
	captureImageWarns(t)
	body := `{"model":"google/gemini-2.5-flash-image","choices":[{"message":{"role":"assistant","content":"Here it is.",` +
		`"images":[{"type":"image_url","image_url":{"url":"` + pngDataURL() + `"}},` +
		`{"type":"image_url","image_url":{"url":"https://example.com/a.png"}}]},"finish_reason":"stop"}]}`
	out, err := ParseResponse(strings.NewReader(body), nil)
	if err != nil {
		t.Fatalf("ParseResponse: %v", err)
	}
	if out.Content != "Here it is." {
		t.Errorf("Content = %q", out.Content)
	}
	if len(out.Images) != 1 || out.Images[0].ContentType != "image/png" || !bytes.Equal(out.Images[0].Data, pngBytes) {
		t.Fatalf("Images = %+v, want the one PNG", out.Images)
	}
}

func TestParseResponse_ImagesOnlyIsNormal(t *testing.T) {
	body := `{"model":"m","choices":[{"message":{"role":"assistant","content":"",` +
		`"images":[{"type":"image_url","image_url":{"url":"` + pngDataURL() + `"}}]},"finish_reason":"stop"}]}`
	out, err := ParseResponse(strings.NewReader(body), nil)
	if err != nil {
		t.Fatalf("ParseResponse: %v", err)
	}
	if !out.Normal || out.Content != "" || len(out.Images) != 1 {
		t.Fatalf("got Normal=%v Content=%q Images=%d, want a normal images-only reply", out.Normal, out.Content, len(out.Images))
	}
}

func TestAccumulateChatStream_Images(t *testing.T) {
	// OpenRouter streams each image whole in delta.images.
	stream := buildSSE(
		`{"model":"google/gemini-2.5-flash-image","choices":[{"delta":{"role":"assistant","content":"Here "}}]}`,
		`{"model":"google/gemini-2.5-flash-image","choices":[{"delta":{"images":[{"type":"image_url","index":0,"image_url":{"url":"`+pngDataURL()+`"}}]}}]}`,
		`{"model":"google/gemini-2.5-flash-image","choices":[{"delta":{"content":"it is."},"finish_reason":"stop"}]}`,
		`[DONE]`,
	)
	out, _, err := AccumulateChatStream(strings.NewReader(stream), func(string) {}, nil)
	if err != nil {
		t.Fatalf("AccumulateChatStream: %v", err)
	}
	if out.Content != "Here it is." {
		t.Errorf("Content = %q", out.Content)
	}
	if len(out.Images) != 1 || !bytes.Equal(out.Images[0].Data, pngBytes) {
		t.Fatalf("Images = %+v, want the one PNG", out.Images)
	}
}
