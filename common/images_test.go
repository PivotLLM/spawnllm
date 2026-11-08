package common

import (
	"bytes"
	"encoding/base64"
	"strings"
	"testing"

	"github.com/PivotLLM/spawnllm/protocoltypes"
)

// pngBytes starts with the PNG signature, so http.DetectContentType detects it
// as image/png.
var pngBytes = []byte("\x89PNG\r\n\x1a\nnot-really-a-png")

func pngDataURL() string {
	return "data:image/png;base64," + base64.StdEncoding.EncodeToString(pngBytes)
}

// captureImageWarns swaps the image WARN hook for one that records each
// WARN's fields. Tests using it must not run in parallel.
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
		name     string
		url      string
		want     []protocoltypes.GeneratedImage
		wantWarn bool
	}{
		{
			name: "png data URL", url: pngDataURL(),
			want: []protocoltypes.GeneratedImage{{ContentType: "image/png", Data: pngBytes, Size: int64(len(pngBytes))}},
		},
		{
			name: "declared type wins", url: "data:image/webp;base64," + base64.StdEncoding.EncodeToString(pngBytes),
			want: []protocoltypes.GeneratedImage{{ContentType: "image/webp", Data: pngBytes, Size: int64(len(pngBytes))}},
		},
		{
			name: "no declared type is detected", url: "data:;base64," + base64.StdEncoding.EncodeToString(pngBytes),
			want: []protocoltypes.GeneratedImage{{ContentType: "image/png", Data: pngBytes, Size: int64(len(pngBytes))}},
		},
		{
			name: "https URL passed through", url: "https://cdn.example.com/img.png?sig=secret",
			want: []protocoltypes.GeneratedImage{{URL: "https://cdn.example.com/img.png?sig=secret", Size: -1}},
		},
		{
			name: "http URL passed through", url: "http://example.com/a.png",
			want: []protocoltypes.GeneratedImage{{URL: "http://example.com/a.png", Size: -1}},
		},
		{
			name: "malformed base64 passed through", url: "data:image/png;base64,@@not base64@@",
			want:     []protocoltypes.GeneratedImage{{ContentType: "image/png", URL: "data:image/png;base64,@@not base64@@", Size: -1}},
			wantWarn: true,
		},
		{
			name: "not base64 encoded passed through", url: "data:image/png,%89PNG",
			want:     []protocoltypes.GeneratedImage{{ContentType: "image/png", URL: "data:image/png,%89PNG", Size: -1}},
			wantWarn: true,
		},
		{
			name: "no comma passed through", url: "data:image/png;base64",
			want:     []protocoltypes.GeneratedImage{{ContentType: "image/png", URL: "data:image/png;base64", Size: -1}},
			wantWarn: true,
		},
		{
			name: "empty payload passed through", url: "data:image/png;base64,",
			want:     []protocoltypes.GeneratedImage{{ContentType: "image/png", URL: "data:image/png;base64,", Size: -1}},
			wantWarn: true,
		},
		{name: "empty URL is no image", url: ""},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			warns := captureImageWarns(t)
			c := NewImageCollector("gemini-image")
			c.AddURL(tc.url)
			assertImages(t, c.Images(), tc.want)
			assertWarned(t, *warns, tc.wantWarn)
		})
	}
}

func TestImageCollector_AddBase64(t *testing.T) {
	tests := []struct {
		name     string
		b64      string
		want     []protocoltypes.GeneratedImage
		wantWarn bool
	}{
		{
			name: "decoded and detected", b64: base64.StdEncoding.EncodeToString(pngBytes),
			want: []protocoltypes.GeneratedImage{{ContentType: "image/png", Data: pngBytes, Size: int64(len(pngBytes))}},
		},
		{
			name: "malformed passed through as data URL", b64: "@@not base64@@",
			want:     []protocoltypes.GeneratedImage{{URL: "data:;base64,@@not base64@@", Size: -1}},
			wantWarn: true,
		},
		{name: "empty is no image", b64: ""},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			warns := captureImageWarns(t)
			c := NewImageCollector("gpt-image")
			c.AddBase64(tc.b64)
			assertImages(t, c.Images(), tc.want)
			assertWarned(t, *warns, tc.wantWarn)
		})
	}
}

// TestImageCollector_AnySize checks there is no size cap: large images, and
// a response whose images add up to more, are all returned whole.
func TestImageCollector_AnySize(t *testing.T) {
	const size = 24 << 20
	big := make([]byte, size)
	copy(big, pngBytes)
	b64 := base64.StdEncoding.EncodeToString(big)
	c := NewImageCollector("m")
	c.AddBase64(b64)
	c.AddBase64(b64)
	c.AddURL("data:image/png;base64," + b64)
	imgs := c.Images()
	if len(imgs) != 3 {
		t.Fatalf("Images() = %d, want 3", len(imgs))
	}
	for i, img := range imgs {
		if len(img.Data) != size || img.Size != size || img.URL != "" {
			t.Errorf("Images()[%d]: %d bytes, Size %d, URL %q; want %d bytes and no URL", i, len(img.Data), img.Size, img.URL, size)
		}
	}
}

func assertImages(t *testing.T, got, want []protocoltypes.GeneratedImage) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("Images() = %+v, want %+v", got, want)
	}
	for i := range want {
		g, w := got[i], want[i]
		if g.ContentType != w.ContentType || g.URL != w.URL || g.Size != w.Size || !bytes.Equal(g.Data, w.Data) {
			t.Errorf("Images()[%d] = {%q %q %d %q}, want {%q %q %d %q}", i, g.ContentType, g.URL, g.Size, g.Data, w.ContentType, w.URL, w.Size, w.Data)
		}
		if (len(g.Data) > 0) == (g.URL != "") {
			t.Errorf("Images()[%d]: exactly one of Data or URL must be set", i)
		}
	}
}

func assertWarned(t *testing.T, warns []map[string]any, want bool) {
	t.Helper()
	if !want {
		if len(warns) != 0 {
			t.Fatalf("WARNs = %v, want none", warns)
		}
		return
	}
	if len(warns) != 1 {
		t.Fatalf("WARNs = %v, want 1", warns)
	}
	if m := warns[0]["model"]; m == nil || m == "" {
		t.Errorf("WARN names no model: %v", warns[0])
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
	assertImages(t, out.Images, []protocoltypes.GeneratedImage{
		{ContentType: "image/png", Data: pngBytes, Size: int64(len(pngBytes))},
		{URL: "https://example.com/a.png", Size: -1},
	})
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
		`{"model":"google/gemini-2.5-flash-image","choices":[{"delta":{"images":[{"type":"image_url","index":1,"image_url":{"url":"https://example.com/b.png"}}]}}]}`,
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
	assertImages(t, out.Images, []protocoltypes.GeneratedImage{
		{ContentType: "image/png", Data: pngBytes, Size: int64(len(pngBytes))},
		{URL: "https://example.com/b.png", Size: -1},
	})
}
