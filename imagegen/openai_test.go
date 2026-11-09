package imagegen

import (
	"context"
	"encoding/base64"
	"errors"
	"io"
	"mime"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"github.com/PivotLLM/spawnllm/common"
)

var (
	pngBytes = []byte("\x89PNG\r\n\x1a\nnot-really-a-png")
	pngB64   = base64.StdEncoding.EncodeToString(pngBytes)
)

// redirectTransport sends every request to the test server, keeping its path,
// so a client built for openrouter.ai reaches it.
type redirectTransport struct{ target *url.URL }

func (rt redirectTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	out := req.Clone(req.Context())
	out.URL.Scheme = rt.target.Scheme
	out.URL.Host = rt.target.Host
	out.Host = rt.target.Host
	return http.DefaultTransport.RoundTrip(out)
}

func redirectClient(t *testing.T, srv *httptest.Server) *http.Client {
	t.Helper()
	u, err := url.Parse(srv.URL)
	if err != nil {
		t.Fatalf("parse server URL: %v", err)
	}
	return &http.Client{Transport: redirectTransport{target: u}}
}

func writeJSON(t *testing.T, w http.ResponseWriter, status int, body string) {
	t.Helper()
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if _, err := io.WriteString(w, body); err != nil {
		t.Errorf("write response: %v", err)
	}
}

func readBody(t *testing.T, r *http.Request) string {
	t.Helper()
	data, err := io.ReadAll(r.Body)
	if err != nil {
		t.Fatalf("read body: %v", err)
	}
	return string(data)
}

func TestOpenAIGenerate(t *testing.T) {
	tests := []struct {
		name     string
		req      GenerateRequest
		wantBody string
	}{
		{
			name:     "all fields",
			req:      GenerateRequest{Model: "gpt-image-2", Prompt: "A cat named Bob", N: 2, Size: "1024x1024", Quality: "high"},
			wantBody: `{"model":"gpt-image-2","n":2,"prompt":"A cat named Bob","quality":"high","size":"1024x1024"}`,
		},
		{
			name:     "only required",
			req:      GenerateRequest{Model: "dall-e-3", Prompt: "A dog"},
			wantBody: `{"model":"dall-e-3","prompt":"A dog"}`,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Method != http.MethodPost || r.URL.Path != "/v1/images/generations" {
					t.Errorf("request = %s %s", r.Method, r.URL.Path)
				}
				if r.Header.Get("Authorization") != "Bearer sk" || r.Header.Get("Content-Type") != "application/json" {
					t.Errorf("headers = %v", r.Header)
				}
				if got := readBody(t, r); got != tc.wantBody {
					t.Errorf("body = %s, want %s", got, tc.wantBody)
				}
				writeJSON(t, w, http.StatusOK, `{"created":1,"data":[`+
					`{"b64_json":"`+pngB64+`","revised_prompt":"x"},`+
					`{"url":"https://cdn.example.com/a.png"},`+
					`{"b64_json":"%%%not-base64"}],`+
					`"usage":{"input_tokens":10,"output_tokens":4000,"total_tokens":4010,"cost":9}}`)
			}))
			defer srv.Close()

			got, err := NewOpenAI(srv.URL+"/v1/", "sk").Generate(context.Background(), &tc.req)
			if err != nil {
				t.Fatalf("Generate: %v", err)
			}
			if len(got.Images) != 3 {
				t.Fatalf("images = %d, want 3", len(got.Images))
			}
			first := got.Images[0]
			if string(first.Data) != string(pngBytes) || first.ContentType != "image/png" ||
				first.Size != int64(len(pngBytes)) || first.URL != "" {
				t.Errorf("b64 image = %+v", first)
			}
			if second := got.Images[1]; second.URL != "https://cdn.example.com/a.png" || second.Size != -1 || second.Data != nil {
				t.Errorf("url image = %+v", second)
			}
			if third := got.Images[2]; third.URL != "data:;base64,%%%not-base64" || third.Size != -1 {
				t.Errorf("undecodable image = %+v", third)
			}
			if u := got.Usage; u == nil || u.InputTokens != 10 || u.OutputTokens != 4000 || u.TotalTokens != 4010 {
				t.Errorf("usage = %+v", u)
			}
			if got.CostUSD != nil {
				t.Errorf("CostUSD = %v, want nil off OpenRouter", *got.CostUSD)
			}
		})
	}
}

func TestOpenAIGenerate_NoUsage(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "" {
			t.Error("Authorization sent without a key")
		}
		writeJSON(t, w, http.StatusOK, `{"created":1,"data":[]}`)
	}))
	defer srv.Close()
	got, err := NewOpenAI(srv.URL, "").Generate(context.Background(), &GenerateRequest{Model: "m", Prompt: "p"})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if got.Images != nil || got.Usage != nil || got.CostUSD != nil {
		t.Errorf("result = %+v, want empty", got)
	}
}

func TestOpenAIGenerate_OpenRouter(t *testing.T) {
	svg := []byte("<svg xmlns='http://www.w3.org/2000/svg'/>")
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/v1/images" {
			t.Errorf("path = %s, want /api/v1/images", r.URL.Path)
		}
		if got, want := readBody(t, r), `{"model":"google/gemini-3-pro-image","prompt":"Alice"}`; got != want {
			t.Errorf("body = %s, want %s", got, want)
		}
		writeJSON(t, w, http.StatusOK, `{"created":1,"data":[`+
			`{"b64_json":"`+base64.StdEncoding.EncodeToString(svg)+`","media_type":"image/svg+xml"},`+
			`{"b64_json":"`+pngB64+`"}],`+
			`"usage":{"prompt_tokens":3,"completion_tokens":4175,"total_tokens":4178,"cost":0.04}}`)
	}))
	defer srv.Close()

	c := NewOpenAI("https://openrouter.ai/api/v1", "sk-or", WithHTTPClient(redirectClient(t, srv)))
	got, err := c.Generate(context.Background(), &GenerateRequest{Model: "google/gemini-3-pro-image", Prompt: "Alice"})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if len(got.Images) != 2 || got.Images[0].ContentType != "image/svg+xml" || string(got.Images[0].Data) != string(svg) ||
		got.Images[1].ContentType != "image/png" {
		t.Errorf("images = %+v", got.Images)
	}
	if u := got.Usage; u == nil || u.InputTokens != 3 || u.OutputTokens != 4175 || u.TotalTokens != 4178 {
		t.Errorf("usage = %+v", u)
	}
	if got.CostUSD == nil || *got.CostUSD != 0.04 {
		t.Errorf("CostUSD = %v, want 0.04", got.CostUSD)
	}

	_, err = c.Edit(context.Background(), &EditRequest{Model: "m", Prompt: "p", Images: []InputImage{{Data: strings.NewReader("x")}}})
	if !errors.Is(err, ErrEditUnsupported) {
		t.Errorf("Edit on OpenRouter: err = %v, want ErrEditUnsupported", err)
	}
}

type part struct{ field, filename, contentType, value string }

func readParts(t *testing.T, r *http.Request) []part {
	t.Helper()
	mediaType, params, err := mime.ParseMediaType(r.Header.Get("Content-Type"))
	if err != nil || mediaType != "multipart/form-data" {
		t.Fatalf("content type = %q", r.Header.Get("Content-Type"))
	}
	mr := multipart.NewReader(r.Body, params["boundary"])
	var out []part
	for {
		p, err := mr.NextPart()
		if errors.Is(err, io.EOF) {
			return out
		}
		if err != nil {
			t.Fatalf("next part: %v", err)
		}
		data, err := io.ReadAll(p)
		if err != nil {
			t.Fatalf("read part: %v", err)
		}
		out = append(out, part{p.FormName(), p.FileName(), p.Header.Get("Content-Type"), string(data)})
	}
}

func TestOpenAIEdit(t *testing.T) {
	tests := []struct {
		name  string
		req   EditRequest
		want  []part
		wantN bool
	}{
		{
			name: "one image, mask and options",
			req: EditRequest{
				Model: "gpt-image-2", Prompt: "Add a hat", N: 1, Size: "1024x1024", Quality: "low",
				Images: []InputImage{{Data: strings.NewReader("img1"), Filename: "bob.png", ContentType: "image/png"}},
				Mask:   &InputImage{Data: strings.NewReader("mask"), Filename: "mask.png", ContentType: "image/png"},
			},
			want: []part{
				{"model", "", "", "gpt-image-2"},
				{"prompt", "", "", "Add a hat"},
				{"n", "", "", "1"},
				{"size", "", "", "1024x1024"},
				{"quality", "", "", "low"},
				{"image", "bob.png", "image/png", "img1"},
				{"mask", "mask.png", "image/png", "mask"},
			},
		},
		{
			name: "several images",
			req: EditRequest{
				Model: "gpt-image-2", Prompt: "Combine",
				Images: []InputImage{{Data: strings.NewReader("a")}, {Data: strings.NewReader("b"), Filename: "b.webp", ContentType: "image/webp"}},
			},
			want: []part{
				{"model", "", "", "gpt-image-2"},
				{"prompt", "", "", "Combine"},
				{"image[]", "image", "application/octet-stream", "a"},
				{"image[]", "b.webp", "image/webp", "b"},
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Method != http.MethodPost || r.URL.Path != "/images/edits" {
					t.Errorf("request = %s %s", r.Method, r.URL.Path)
				}
				if r.Header.Get("Authorization") != "Bearer sk" {
					t.Errorf("Authorization = %q", r.Header.Get("Authorization"))
				}
				got := readParts(t, r)
				if len(got) != len(tc.want) {
					t.Fatalf("parts = %+v, want %+v", got, tc.want)
				}
				for i := range got {
					want := tc.want[i]
					if want.contentType == "" {
						want.contentType = got[i].contentType // plain fields: not checked
					}
					if got[i] != want {
						t.Errorf("part %d = %+v, want %+v", i, got[i], want)
					}
				}
				writeJSON(t, w, http.StatusOK, `{"created":1,"data":[{"b64_json":"`+pngB64+`"}]}`)
			}))
			defer srv.Close()

			got, err := NewOpenAI(srv.URL, "sk").Edit(context.Background(), &tc.req)
			if err != nil {
				t.Fatalf("Edit: %v", err)
			}
			if len(got.Images) != 1 || string(got.Images[0].Data) != string(pngBytes) {
				t.Errorf("images = %+v", got.Images)
			}
		})
	}
}

func TestOpenAI_Validation(t *testing.T) {
	c := NewOpenAI("http://127.0.0.1:1", "k")
	ctx := context.Background()
	if _, err := c.Generate(ctx, nil); err == nil {
		t.Error("nil generate request accepted")
	}
	if _, err := c.Generate(ctx, &GenerateRequest{Prompt: "p"}); err == nil {
		t.Error("missing model accepted")
	}
	if _, err := c.Generate(ctx, &GenerateRequest{Model: "m"}); err == nil {
		t.Error("missing prompt accepted")
	}
	if _, err := c.Edit(ctx, &EditRequest{Model: "m", Prompt: "p"}); err == nil {
		t.Error("edit without images accepted")
	}
	if _, err := c.Edit(ctx, &EditRequest{Model: "m", Prompt: "p", Images: []InputImage{{}}}); err == nil {
		t.Error("image without data accepted")
	}
	if _, err := c.Edit(ctx, nil); err == nil {
		t.Error("nil edit request accepted")
	}
}

func TestOpenAI_Errors(t *testing.T) {
	tests := []struct {
		name    string
		handler http.HandlerFunc
		check   func(t *testing.T, err error)
	}{
		{
			name: "400 keeps status and body",
			handler: func(w http.ResponseWriter, _ *http.Request) {
				writeJSON(t, w, http.StatusBadRequest, `{"error":{"code":"moderation_blocked","message":"rejected"}}`)
			},
			check: func(t *testing.T, err error) {
				t.Helper()
				statusErr, ok := errors.AsType[*common.HTTPStatusError](err)
				if !ok || statusErr.StatusCode != http.StatusBadRequest || !strings.Contains(statusErr.BodyPreview, "moderation_blocked") {
					t.Errorf("err = %v", err)
				}
			},
		},
		{
			name: "HTML",
			handler: func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", "text/html")
				writeRaw(t, w, []byte("<!DOCTYPE html><html></html>"))
			},
			check: func(t *testing.T, err error) {
				t.Helper()
				if err == nil || !strings.Contains(err.Error(), "returned HTML") {
					t.Errorf("err = %v", err)
				}
			},
		},
		{
			name: "malformed JSON",
			handler: func(w http.ResponseWriter, _ *http.Request) {
				writeJSON(t, w, http.StatusOK, `{"data":[`)
			},
			check: func(t *testing.T, err error) {
				t.Helper()
				if err == nil || !strings.Contains(err.Error(), "failed to decode response") {
					t.Errorf("err = %v", err)
				}
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			srv := httptest.NewServer(tc.handler)
			defer srv.Close()
			_, err := NewOpenAI(srv.URL, "sk").Generate(context.Background(), &GenerateRequest{Model: "m", Prompt: "p"})
			tc.check(t, err)
		})
	}
}

func writeRaw(t *testing.T, w http.ResponseWriter, data []byte) {
	t.Helper()
	if _, err := w.Write(data); err != nil {
		t.Errorf("write response: %v", err)
	}
}
