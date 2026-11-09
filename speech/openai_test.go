package speech

import (
	"context"
	"encoding/base64"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/PivotLLM/spawnllm/common"
)

func TestOpenAITranscribe_Multipart(t *testing.T) {
	var gotForm map[string][]formPart
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/v1/audio/transcriptions" {
			t.Errorf("request = %s %s, want POST /v1/audio/transcriptions", r.Method, r.URL.Path)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer sk-test" {
			t.Errorf("Authorization = %q", got)
		}
		gotForm = readForm(t, r)
		writeJSON(t, w, http.StatusOK,
			`{"text":"Hello Bob.","language":"english","duration":2.5,`+
				`"usage":{"type":"tokens","input_tokens":12,"output_tokens":4,"total_tokens":16,"cost":0.5}}`)
	}))
	defer srv.Close()

	c := NewOpenAI(srv.URL+"/v1/", "sk-test")
	got, err := c.Transcribe(context.Background(), &TranscribeRequest{
		Model: "whisper-1", Audio: audio(), Filename: "/tmp/voice.ogg", ContentType: "audio/ogg",
		Language: "en", Prompt: "Alice and Bob",
	})
	if err != nil {
		t.Fatalf("Transcribe: %v", err)
	}

	file := field(t, gotForm, "file")
	if file.filename != "voice.ogg" || file.contentType != "audio/ogg" || file.value != "OggS-fake-audio" {
		t.Errorf("file part = %+v", file)
	}
	for name, want := range map[string]string{
		"model": "whisper-1", "response_format": "json", "language": "en", "prompt": "Alice and Bob",
	} {
		if got := field(t, gotForm, name).value; got != want {
			t.Errorf("field %s = %q, want %q", name, got, want)
		}
	}

	if got.Text != "Hello Bob." || got.Language != "english" || got.DurationSeconds != 2.5 {
		t.Errorf("result = %+v", got)
	}
	if got.Usage == nil || got.Usage.InputTokens != 12 || got.Usage.OutputTokens != 4 || got.Usage.TotalTokens != 16 {
		t.Errorf("usage = %+v", got.Usage)
	}
	// usage.cost is read only from OpenRouter, whose unit is known.
	if got.CostUSD != nil {
		t.Errorf("CostUSD = %v, want nil off OpenRouter", *got.CostUSD)
	}
}

func TestOpenAITranscribe_MultipartDefaultsAndOptionalFields(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Authorization"); got != "" {
			t.Errorf("Authorization = %q, want none without a key", got)
		}
		form := readForm(t, r)
		file := field(t, form, "file")
		if file.filename != "audio.ogg" || file.contentType != "audio/ogg; codecs=opus" {
			t.Errorf("file part = %+v, want audio.ogg typed audio/ogg; codecs=opus", file)
		}
		for _, name := range []string{"language", "prompt"} {
			if _, ok := form[name]; ok {
				t.Errorf("field %s sent although empty", name)
			}
		}
		writeJSON(t, w, http.StatusOK, `{"text":"hi"}`)
	}))
	defer srv.Close()

	got, err := NewOpenAI(srv.URL, "").Transcribe(context.Background(),
		&TranscribeRequest{Model: "whisper-large-v3", Audio: audio(), ContentType: "audio/ogg; codecs=opus"})
	if err != nil {
		t.Fatalf("Transcribe: %v", err)
	}
	if got.Text != "hi" || got.Language != "" || got.DurationSeconds != 0 || got.Usage != nil || got.CostUSD != nil {
		t.Errorf("result = %+v, want only the text", got)
	}
}

func TestOpenAITranscribe_DurationUsage(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(t, w, http.StatusOK, `{"text":"x","usage":{"type":"duration","seconds":7}}`)
	}))
	defer srv.Close()
	got, err := NewOpenAI(srv.URL, "k").Transcribe(context.Background(),
		&TranscribeRequest{Model: "whisper-1", Audio: audio(), Filename: "a.mp3"})
	if err != nil {
		t.Fatalf("Transcribe: %v", err)
	}
	if got.Usage == nil || got.Usage.Seconds != 7 {
		t.Errorf("usage = %+v, want 7 seconds", got.Usage)
	}
	// Only OpenRouter's usage.seconds stands in for the duration.
	if got.DurationSeconds != 0 {
		t.Errorf("DurationSeconds = %v, want 0", got.DurationSeconds)
	}
}

// TestOpenAITranscribe_OpenRouter checks the exact request OpenRouter's
// transcription endpoint receives (it is in production use) and that its
// usage is reported.
func TestOpenAITranscribe_OpenRouter(t *testing.T) {
	audioB64 := base64.StdEncoding.EncodeToString([]byte("OggS-fake-audio"))
	tests := []struct {
		name     string
		language string
		wantBody string
		response string
		wantCost *float64
		wantDur  float64
	}{
		{
			name:     "with usage and cost",
			wantBody: `{"input_audio":{"data":"` + audioB64 + `","format":"ogg"},"model":"openai/whisper-large-v3"}`,
			response: `{"text":"Hello Alice.","usage":{"seconds":9.2,"input_tokens":83,"output_tokens":30,"total_tokens":113,"cost":0.000508}}`,
			wantCost: new(0.000508),
			wantDur:  9.2,
		},
		{
			name:     "language and no usage",
			language: "fr",
			wantBody: `{"input_audio":{"data":"` + audioB64 + `","format":"ogg"},"language":"fr","model":"openai/whisper-large-v3"}`,
			response: `{"text":"Bonjour."}`,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Method != http.MethodPost || r.URL.Path != "/api/v1/audio/transcriptions" {
					t.Errorf("request = %s %s", r.Method, r.URL.Path)
				}
				if got := r.Header.Get("Content-Type"); got != "application/json" {
					t.Errorf("Content-Type = %q", got)
				}
				if got := r.Header.Get("Authorization"); got != "Bearer sk-or-test" {
					t.Errorf("Authorization = %q", got)
				}
				if got := readBody(t, r); got != tc.wantBody {
					t.Errorf("body =\n%s\nwant\n%s", got, tc.wantBody)
				}
				writeJSON(t, w, http.StatusOK, tc.response)
			}))
			defer srv.Close()

			// Chosen by host: the client is built for openrouter.ai and the
			// transport delivers to the test server.
			c := NewOpenAI("https://openrouter.ai/api/v1", "sk-or-test", WithHTTPClient(redirectClient(t, srv)))
			got, err := c.Transcribe(context.Background(), &TranscribeRequest{
				Model: "openai/whisper-large-v3", Audio: audio(), Filename: "Voice.OGG",
				Language: tc.language, Prompt: "not sent",
			})
			if err != nil {
				t.Fatalf("Transcribe: %v", err)
			}
			if got.DurationSeconds != tc.wantDur {
				t.Errorf("DurationSeconds = %v, want %v", got.DurationSeconds, tc.wantDur)
			}
			switch {
			case tc.wantCost == nil && got.CostUSD != nil:
				t.Errorf("CostUSD = %v, want nil", *got.CostUSD)
			case tc.wantCost != nil && (got.CostUSD == nil || *got.CostUSD != *tc.wantCost):
				t.Errorf("CostUSD = %v, want %v", got.CostUSD, *tc.wantCost)
			}
			if tc.wantCost != nil && (got.Usage == nil || got.Usage.TotalTokens != 113 || got.Usage.Seconds != 9.2) {
				t.Errorf("usage = %+v", got.Usage)
			}
		})
	}
}

func TestOpenAITranscribe_ExplicitAPI(t *testing.T) {
	var gotType string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotType = r.Header.Get("Content-Type")
		writeJSON(t, w, http.StatusOK, `{"text":"ok"}`)
	}))
	defer srv.Close()

	req := func() *TranscribeRequest {
		return &TranscribeRequest{Model: "m", Audio: audio(), Filename: "a.wav"}
	}
	if _, err := NewOpenAI(srv.URL, "k", WithTranscriptionAPI(TranscriptionInputAudio)).
		Transcribe(context.Background(), req()); err != nil {
		t.Fatalf("Transcribe: %v", err)
	}
	if gotType != "application/json" {
		t.Errorf("explicit input audio: Content-Type = %q", gotType)
	}
	c := NewOpenAI("https://openrouter.ai/api/v1", "k",
		WithTranscriptionAPI(TranscriptionMultipart), WithHTTPClient(redirectClient(t, srv)))
	if _, err := c.Transcribe(context.Background(), req()); err != nil {
		t.Fatalf("Transcribe: %v", err)
	}
	if !strings.HasPrefix(gotType, "multipart/form-data") {
		t.Errorf("explicit multipart on OpenRouter: Content-Type = %q", gotType)
	}
}

func TestNewOpenAI_TranscriptionAPIByHost(t *testing.T) {
	tests := map[string]TranscriptionAPI{
		"https://openrouter.ai/api/v1":        TranscriptionInputAudio,
		"https://eu.openrouter.ai/api/v1":     TranscriptionInputAudio,
		"https://api.openai.com/v1":           TranscriptionMultipart,
		"https://api.groq.com/openai/v1":      TranscriptionMultipart,
		"http://localhost:8080/v1":            TranscriptionMultipart,
		"https://notopenrouter.ai.example/v1": TranscriptionMultipart,
	}
	for base, want := range tests {
		if got := NewOpenAI(base, "").transcription; got != want {
			t.Errorf("%s: transcription API = %v, want %v", base, got, want)
		}
	}
}

func TestOpenAITranscribe_OpenRouterNeedsExtension(t *testing.T) {
	called := false
	srv := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { called = true }))
	defer srv.Close()
	_, err := NewOpenAI(srv.URL, "k", WithTranscriptionAPI(TranscriptionInputAudio)).
		Transcribe(context.Background(), &TranscribeRequest{Model: "m", Audio: audio(), Filename: "voice"})
	if err == nil || !strings.Contains(err.Error(), "cannot determine audio format") {
		t.Fatalf("err = %v, want a format error", err)
	}
	if called {
		t.Error("a request was sent")
	}
}

func TestDefaultSTTModel(t *testing.T) {
	tests := map[string]string{
		"https://api.groq.com/openai/v1": "whisper-large-v3",
		"https://api.openai.com/v1":      "whisper-1",
		"https://openrouter.ai/api/v1":   "openai/whisper-large-v3",
		"https://API.OpenAI.com:443/v1":  "whisper-1",
		"http://localhost:8080":          "",
		"":                               "",
		"::not a url":                    "",
	}
	for base, want := range tests {
		if got := DefaultSTTModel(base); got != want {
			t.Errorf("DefaultSTTModel(%q) = %q, want %q", base, got, want)
		}
	}
}

func TestOpenAITranscribe_Errors(t *testing.T) {
	tests := []struct {
		name    string
		handler http.HandlerFunc
		check   func(t *testing.T, err error)
	}{
		{
			name: "401 keeps status and body",
			handler: func(w http.ResponseWriter, _ *http.Request) {
				writeJSON(t, w, http.StatusUnauthorized, `{"error":{"message":"Incorrect API key provided"}}`)
			},
			check: func(t *testing.T, err error) {
				t.Helper()
				statusErr, ok := errors.AsType[*common.HTTPStatusError](err)
				if !ok || statusErr.StatusCode != http.StatusUnauthorized ||
					!strings.Contains(statusErr.BodyPreview, "Incorrect API key") {
					t.Errorf("err = %#v, want HTTPStatusError 401 with the body", err)
				}
			},
		},
		{
			name: "429 keeps Retry-After and caps the body",
			handler: func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Retry-After", "30")
				writeJSON(t, w, http.StatusTooManyRequests, strings.Repeat("x", 5000))
			},
			check: func(t *testing.T, err error) {
				t.Helper()
				statusErr, ok := errors.AsType[*common.HTTPStatusError](err)
				if !ok || statusErr.RetryAfter != 30*time.Second || len(statusErr.BodyPreview) > 1024+3 {
					t.Errorf("err = %#v", err)
				}
			},
		},
		{
			name: "HTML page",
			handler: func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", "text/html")
				writeRaw(t, w, []byte("<html><body>login</body></html>"))
			},
			check: func(t *testing.T, err error) {
				t.Helper()
				if err == nil || !strings.Contains(err.Error(), "returned HTML") {
					t.Errorf("err = %v, want the HTML error", err)
				}
			},
		},
		{
			name: "malformed JSON",
			handler: func(w http.ResponseWriter, _ *http.Request) {
				writeJSON(t, w, http.StatusOK, `{"text":`)
			},
			check: func(t *testing.T, err error) {
				t.Helper()
				if err == nil || !strings.Contains(err.Error(), "failed to decode response") {
					t.Errorf("err = %v, want a decode error", err)
				}
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			srv := httptest.NewServer(tc.handler)
			defer srv.Close()
			_, err := NewOpenAI(srv.URL, "k").Transcribe(context.Background(),
				&TranscribeRequest{Model: "whisper-1", Audio: audio(), Filename: "a.ogg"})
			tc.check(t, err)
		})
	}
}

func TestOpenAITranscribe_ContextDeadline(t *testing.T) {
	release := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { <-release }))
	defer srv.Close()
	defer close(release)

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	_, err := NewOpenAI(srv.URL, "k").Transcribe(ctx, &TranscribeRequest{Model: "m", Audio: audio(), Filename: "a.ogg"})
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("err = %v, want context.DeadlineExceeded", err)
	}
}

func TestValidation(t *testing.T) {
	c := NewOpenAI("http://127.0.0.1:1", "k")
	if _, err := c.Transcribe(context.Background(), nil); err == nil {
		t.Error("nil transcribe request accepted")
	}
	if _, err := c.Transcribe(context.Background(), &TranscribeRequest{Audio: audio()}); err == nil {
		t.Error("missing model accepted")
	}
	if _, err := c.Transcribe(context.Background(), &TranscribeRequest{Model: "m"}); err == nil {
		t.Error("missing audio accepted")
	}
	if _, err := c.Synthesize(context.Background(), &SynthesizeRequest{Model: "m", Voice: "v"}); err == nil {
		t.Error("missing text accepted")
	}
	if _, err := c.Synthesize(context.Background(), &SynthesizeRequest{Model: "m", Text: "t"}); err == nil {
		t.Error("missing voice accepted")
	}
	if _, err := c.Synthesize(context.Background(), &SynthesizeRequest{Text: "t", Voice: "v"}); err == nil {
		t.Error("missing model accepted")
	}
}

func TestOpenAISynthesize(t *testing.T) {
	tests := []struct {
		name            string
		req             SynthesizeRequest
		wantBody        string
		respType        string // "" sends no Content-Type
		wantContentType string
	}{
		{
			name:            "all fields",
			req:             SynthesizeRequest{Model: "gpt-4o-mini-tts", Text: "Héllo Bob", Voice: "coral", Format: "opus", Instructions: "Cheerful."},
			wantBody:        `{"input":"Héllo Bob","instructions":"Cheerful.","model":"gpt-4o-mini-tts","response_format":"opus","voice":"coral"}`,
			respType:        "audio/ogg; codecs=opus",
			wantContentType: "audio/ogg; codecs=opus",
		},
		{
			name:            "defaults, no Content-Type from the server",
			req:             SynthesizeRequest{Model: "tts-1", Text: "Hi", Voice: "alloy"},
			wantBody:        `{"input":"Hi","model":"tts-1","voice":"alloy"}`,
			wantContentType: "audio/mpeg",
		},
		{
			name:            "wav, no Content-Type from the server",
			req:             SynthesizeRequest{Model: "tts-1", Text: "Hi", Voice: "alloy", Format: "wav"},
			wantBody:        `{"input":"Hi","model":"tts-1","response_format":"wav","voice":"alloy"}`,
			wantContentType: "audio/wav",
		},
	}
	audioOut := []byte("\x00\x01binary-audio\xff")
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Method != http.MethodPost || r.URL.Path != "/v1/audio/speech" {
					t.Errorf("request = %s %s", r.Method, r.URL.Path)
				}
				if r.Header.Get("Authorization") != "Bearer k" || r.Header.Get("Content-Type") != "application/json" {
					t.Errorf("headers = %v", r.Header)
				}
				if got := readBody(t, r); got != tc.wantBody {
					t.Errorf("body = %s, want %s", got, tc.wantBody)
				}
				if tc.respType == "" {
					w.Header()["Content-Type"] = nil // no sniffing
				} else {
					w.Header().Set("Content-Type", tc.respType)
				}
				writeRaw(t, w, audioOut)
			}))
			defer srv.Close()

			got, err := NewOpenAI(srv.URL+"/v1", "k").Synthesize(context.Background(), &tc.req)
			if err != nil {
				t.Fatalf("Synthesize: %v", err)
			}
			if string(got.Audio) != string(audioOut) || got.Size != int64(len(audioOut)) {
				t.Errorf("audio = %q (size %d)", got.Audio, got.Size)
			}
			if got.ContentType != tc.wantContentType {
				t.Errorf("ContentType = %q, want %q", got.ContentType, tc.wantContentType)
			}
			if got.Usage == nil || got.Usage.Characters != len([]rune(tc.req.Text)) {
				t.Errorf("usage = %+v", got.Usage)
			}
		})
	}
}

func TestOpenAISynthesize_Error(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(t, w, http.StatusBadRequest, `{"error":{"message":"input too long"}}`)
	}))
	defer srv.Close()
	_, err := NewOpenAI(srv.URL, "k").Synthesize(context.Background(),
		&SynthesizeRequest{Model: "tts-1", Text: "x", Voice: "alloy"})
	statusErr, ok := errors.AsType[*common.HTTPStatusError](err)
	if !ok || statusErr.StatusCode != http.StatusBadRequest {
		t.Fatalf("err = %v, want HTTPStatusError 400", err)
	}
}

// TestOpenAITranscribe_UnknownFormat checks that audio whose format neither
// the file name nor the content type tells is refused before sending, on
// both request shapes.
func TestOpenAITranscribe_UnknownFormat(t *testing.T) {
	called := false
	srv := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { called = true }))
	defer srv.Close()
	for _, api := range []TranscriptionAPI{TranscriptionMultipart, TranscriptionInputAudio} {
		for _, req := range []TranscribeRequest{
			{Model: "m", Audio: audio()},
			{Model: "m", Audio: audio(), Filename: "voice", ContentType: "application/octet-stream"},
		} {
			_, err := NewOpenAI(srv.URL, "k", WithTranscriptionAPI(api)).Transcribe(context.Background(), &req)
			if err == nil || !strings.Contains(err.Error(), "cannot determine audio format") {
				t.Errorf("api %v, %+v: err = %v, want a format error", api, req, err)
			}
		}
	}
	if called {
		t.Error("a request was sent")
	}
}

func TestOpenAISynthesize_NotAudio(t *testing.T) {
	tests := []struct {
		name, contentType, body, want string
	}{
		{"JSON", "application/json", `{"error":{"message":"quota"}}`, "quota"},
		{"text", "text/plain; charset=utf-8", "upstream said no", "upstream said no"},
		{"HTML", "text/html", "<html><body>login</body></html>", "returned HTML"},
		{"HTML without a type", "", "<!DOCTYPE html><html></html>", "returned HTML"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				if tc.contentType == "" {
					w.Header()["Content-Type"] = nil
				} else {
					w.Header().Set("Content-Type", tc.contentType)
				}
				writeRaw(t, w, []byte(tc.body))
			}))
			defer srv.Close()
			got, err := NewOpenAI(srv.URL, "k").Synthesize(context.Background(),
				&SynthesizeRequest{Model: "tts-1", Text: "x", Voice: "alloy"})
			if err == nil || got != nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("got %v, err = %v, want an error containing %q", got, err, tc.want)
			}
		})
	}
}
