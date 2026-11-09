package speech

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/PivotLLM/spawnllm/common"
)

// geminiRequest is the part of a generateContent body the tests check.
type geminiRequest struct {
	Contents []struct {
		Role  string `json:"role"`
		Parts []struct {
			Text       string `json:"text"`
			InlineData *struct {
				MimeType string `json:"mimeType"`
				Data     string `json:"data"`
			} `json:"inlineData"`
		} `json:"parts"`
	} `json:"contents"`
	GenerationConfig map[string]any `json:"generationConfig"`
}

func decodeGemini(t *testing.T, r *http.Request) geminiRequest {
	t.Helper()
	var req geminiRequest
	if err := json.Unmarshal([]byte(readBody(t, r)), &req); err != nil {
		t.Fatalf("decode request: %v", err)
	}
	if len(req.Contents) != 1 {
		t.Fatalf("contents = %d, want 1", len(req.Contents))
	}
	return req
}

func checkGeminiHeaders(t *testing.T, r *http.Request, wantPath string) {
	t.Helper()
	if r.Method != http.MethodPost || r.URL.Path != wantPath {
		t.Errorf("request = %s %s, want POST %s", r.Method, r.URL.Path, wantPath)
	}
	if got := r.Header.Get("X-Goog-Api-Key"); got != "g-key" {
		t.Errorf("x-goog-api-key = %q", got)
	}
	if r.URL.RawQuery != "" {
		t.Errorf("query = %q, want the key in the header only", r.URL.RawQuery)
	}
}

func TestGeminiTranscribe(t *testing.T) {
	tests := []struct {
		name         string
		req          TranscribeRequest
		wantMime     string
		wantInPrompt []string
	}{
		{
			name:     "content type given",
			req:      TranscribeRequest{Model: "models/gemini-3.8-flash", Audio: audio(), ContentType: "audio/ogg", Filename: "a.mp3"},
			wantMime: "audio/ogg",
		},
		{
			name: "type from the extension, language and prompt",
			req: TranscribeRequest{
				Model: "gemini-3.8-flash", Audio: audio(), Filename: "Note.MP3", Language: "de", Prompt: "Alice, Bob",
			},
			wantMime:     "audio/mp3",
			wantInPrompt: []string{"language with code de", "Alice, Bob"},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				checkGeminiHeaders(t, r, "/v1beta/models/gemini-3.8-flash:generateContent")
				req := decodeGemini(t, r)
				parts := req.Contents[0].Parts
				if req.Contents[0].Role != "user" || len(parts) != 2 {
					t.Fatalf("contents = %+v", req.Contents)
				}
				if !strings.HasPrefix(parts[0].Text, geminiTranscribeInstruction) {
					t.Errorf("instruction = %q", parts[0].Text)
				}
				for _, s := range tc.wantInPrompt {
					if !strings.Contains(parts[0].Text, s) {
						t.Errorf("instruction %q lacks %q", parts[0].Text, s)
					}
				}
				in := parts[1].InlineData
				if in == nil || in.MimeType != tc.wantMime ||
					in.Data != base64.StdEncoding.EncodeToString([]byte("OggS-fake-audio")) {
					t.Errorf("inlineData = %+v", in)
				}
				if req.GenerationConfig != nil {
					t.Errorf("generationConfig = %v, want none", req.GenerationConfig)
				}
				writeJSON(t, w, http.StatusOK, `{"candidates":[{"content":{"role":"model","parts":[`+
					`{"text":"thinking...","thought":true},{"text":"Hello "},{"text":"Alice.\n"}]},"finishReason":"STOP"}],`+
					`"usageMetadata":{"promptTokenCount":100,"candidatesTokenCount":5,"totalTokenCount":105}}`)
			}))
			defer srv.Close()

			got, err := NewGemini(srv.URL+"/v1beta/", "g-key").Transcribe(context.Background(), &tc.req)
			if err != nil {
				t.Fatalf("Transcribe: %v", err)
			}
			if got.Text != "Hello Alice.\n" {
				t.Errorf("Text = %q", got.Text)
			}
			if got.Usage == nil || got.Usage.InputTokens != 100 || got.Usage.OutputTokens != 5 || got.Usage.TotalTokens != 105 {
				t.Errorf("usage = %+v", got.Usage)
			}
			if got.CostUSD != nil {
				t.Error("CostUSD set")
			}
		})
	}
}

func TestGeminiTranscribe_UnknownType(t *testing.T) {
	_, err := NewGemini("http://127.0.0.1:1", "k").Transcribe(context.Background(),
		&TranscribeRequest{Model: "m", Audio: audio(), Filename: "voice.xyz"})
	if err == nil || !strings.Contains(err.Error(), "cannot determine audio type") {
		t.Fatalf("err = %v", err)
	}
}

func TestNewGemini_DefaultBaseURL(t *testing.T) {
	if got := NewGemini("", "k").baseURL; got != GeminiBaseURL {
		t.Errorf("baseURL = %q, want %q", got, GeminiBaseURL)
	}
}

func TestGeminiTranscribe_Failures(t *testing.T) {
	tests := []struct {
		name     string
		status   int
		body     string
		wantErr  string
		wantHTTP int
	}{
		{name: "blocked", status: http.StatusOK, body: `{"promptFeedback":{"blockReason":"SAFETY"}}`, wantErr: "blocked the request: SAFETY"},
		{name: "no candidates", status: http.StatusOK, body: `{}`, wantErr: "no candidates"},
		{
			name: "API error", status: http.StatusBadRequest, wantHTTP: http.StatusBadRequest,
			body: `{"error":{"code":400,"message":"API key not valid","status":"INVALID_ARGUMENT"}}`,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				writeJSON(t, w, tc.status, tc.body)
			}))
			defer srv.Close()
			_, err := NewGemini(srv.URL, "g-key").Transcribe(context.Background(),
				&TranscribeRequest{Model: "m", Audio: audio(), ContentType: "audio/wav"})
			if tc.wantHTTP != 0 {
				statusErr, ok := errors.AsType[*common.HTTPStatusError](err)
				if !ok || statusErr.StatusCode != tc.wantHTTP || !strings.Contains(statusErr.BodyPreview, "API key not valid") {
					t.Fatalf("err = %v, want HTTPStatusError %d", err, tc.wantHTTP)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("err = %v, want %q", err, tc.wantErr)
			}
		})
	}
}

func TestGeminiSynthesize(t *testing.T) {
	pcm := []byte{0x01, 0x00, 0xff, 0x7f}
	tests := []struct {
		name       string
		req        SynthesizeRequest
		wantText   string
		wantConfig string
	}{
		{
			name:       "voice and instructions",
			req:        SynthesizeRequest{Model: "gemini-3.8-flash-tts", Text: "Have a wonderful day!", Voice: "Kore", Instructions: "Say cheerfully", Format: "mp3"},
			wantText:   "Say cheerfully: Have a wonderful day!",
			wantConfig: `{"responseModalities":["AUDIO"],"speechConfig":{"voiceConfig":{"prebuiltVoiceConfig":{"voiceName":"Kore"}}}}`,
		},
		{
			name:       "no voice",
			req:        SynthesizeRequest{Model: "gemini-3.8-flash-tts", Text: "Hi"},
			wantText:   "Hi",
			wantConfig: `{"responseModalities":["AUDIO"]}`,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				checkGeminiHeaders(t, r, "/models/gemini-3.8-flash-tts:generateContent")
				req := decodeGemini(t, r)
				if parts := req.Contents[0].Parts; len(parts) != 1 || parts[0].Text != tc.wantText {
					t.Errorf("parts = %+v, want the text %q", parts, tc.wantText)
				}
				config, err := json.Marshal(req.GenerationConfig)
				if err != nil {
					t.Fatalf("marshal config: %v", err)
				}
				if string(config) != tc.wantConfig {
					t.Errorf("generationConfig = %s, want %s", config, tc.wantConfig)
				}
				writeJSON(t, w, http.StatusOK, `{"candidates":[{"content":{"parts":[{"inlineData":{`+
					`"mimeType":"audio/L16;codec=pcm;rate=24000","data":"`+base64.StdEncoding.EncodeToString(pcm)+`"}}]}}],`+
					`"usageMetadata":{"promptTokenCount":7,"candidatesTokenCount":50,"totalTokenCount":57}}`)
			}))
			defer srv.Close()

			got, err := NewGemini(srv.URL, "g-key").Synthesize(context.Background(), &tc.req)
			if err != nil {
				t.Fatalf("Synthesize: %v", err)
			}
			if string(got.Audio) != string(pcm) || got.Size != int64(len(pcm)) {
				t.Errorf("audio = %v", got.Audio)
			}
			if got.ContentType != "audio/L16;codec=pcm;rate=24000" {
				t.Errorf("ContentType = %q, want the mimeType unchanged", got.ContentType)
			}
			u := got.Usage
			if u == nil || u.Characters != len(tc.req.Text) || u.InputTokens != 7 || u.OutputTokens != 50 || u.TotalTokens != 57 {
				t.Errorf("usage = %+v", u)
			}
		})
	}
}

func TestGeminiSynthesize_NoAudio(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(t, w, http.StatusOK, `{"candidates":[{"content":{"parts":[{"text":"no"}]},"finishReason":"OTHER"}]}`)
	}))
	defer srv.Close()
	got, err := NewGemini(srv.URL, "g-key").Synthesize(context.Background(), &SynthesizeRequest{Model: "m", Text: "Hi"})
	if err == nil || !strings.Contains(err.Error(), "no audio") || got != nil {
		t.Fatalf("got %v, err = %v, want a no-audio error", got, err)
	}
}
