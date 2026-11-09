package speech

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/PivotLLM/spawnllm/common"
)

func TestElevenLabsTranscribe(t *testing.T) {
	tests := []struct {
		name     string
		req      TranscribeRequest
		response string
		want     Transcription
		wantLang bool
		wantFile string
	}{
		{
			name:     "all fields",
			req:      TranscribeRequest{Model: "scribe_v2", Audio: audio(), Filename: "memo.ogg", ContentType: "audio/ogg", Language: "en", Prompt: "ignored"},
			response: `{"language_code":"eng","language_probability":0.98,"text":"Hello Bob.","words":[],"audio_duration_secs":3.25}`,
			want:     Transcription{Text: "Hello Bob.", Language: "eng", DurationSeconds: 3.25},
			wantLang: true,
			wantFile: "memo.ogg",
		},
		{
			name:     "optional fields missing",
			req:      TranscribeRequest{Model: "scribe_v2", Audio: audio(), ContentType: "audio/mpeg"},
			response: `{"text":"Hi","words":[]}`,
			want:     Transcription{Text: "Hi"},
			wantFile: "audio.mp3",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Method != http.MethodPost || r.URL.Path != "/v1/speech-to-text" {
					t.Errorf("request = %s %s", r.Method, r.URL.Path)
				}
				if got := r.Header.Get("Xi-Api-Key"); got != "el-key" {
					t.Errorf("xi-api-key = %q", got)
				}
				if r.Header.Get("Authorization") != "" {
					t.Error("Authorization header sent")
				}
				form := readForm(t, r)
				if got := field(t, form, "model_id").value; got != "scribe_v2" {
					t.Errorf("model_id = %q", got)
				}
				if _, ok := form["prompt"]; ok {
					t.Error("prompt sent")
				}
				if _, ok := form["language_code"]; ok != tc.wantLang {
					t.Errorf("language_code present = %v, want %v", ok, tc.wantLang)
				}
				if tc.wantLang && field(t, form, "language_code").value != "en" {
					t.Error("language_code value")
				}
				file := field(t, form, "file")
				if file.value != "OggS-fake-audio" || file.filename != tc.wantFile {
					t.Errorf("file = %+v", file)
				}
				writeJSON(t, w, http.StatusOK, tc.response)
			}))
			defer srv.Close()

			got, err := NewElevenLabs(srv.URL, "el-key").Transcribe(context.Background(), &tc.req)
			if err != nil {
				t.Fatalf("Transcribe: %v", err)
			}
			if *got != tc.want {
				t.Errorf("result = %+v, want %+v", *got, tc.want)
			}
		})
	}
}

func TestElevenLabsSynthesize(t *testing.T) {
	tests := []struct {
		name      string
		req       SynthesizeRequest
		wantQuery string
	}{
		{
			name:      "format",
			req:       SynthesizeRequest{Model: "eleven_multilingual_v2", Text: "Hi Alice", Voice: "JBFqnCBsd6RMkjVDRZzb", Format: "mp3_44100_128", Instructions: "ignored"},
			wantQuery: "output_format=mp3_44100_128",
		},
		{
			name: "default format",
			req:  SynthesizeRequest{Model: "eleven_multilingual_v2", Text: "Hi Alice", Voice: "JBFqnCBsd6RMkjVDRZzb"},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Method != http.MethodPost || r.URL.Path != "/v1/text-to-speech/JBFqnCBsd6RMkjVDRZzb" {
					t.Errorf("request = %s %s", r.Method, r.URL.Path)
				}
				if r.URL.RawQuery != tc.wantQuery {
					t.Errorf("query = %q, want %q", r.URL.RawQuery, tc.wantQuery)
				}
				if r.Header.Get("Xi-Api-Key") != "el-key" || r.Header.Get("Content-Type") != "application/json" {
					t.Errorf("headers = %v", r.Header)
				}
				if got, want := readBody(t, r), `{"model_id":"eleven_multilingual_v2","text":"Hi Alice"}`; got != want {
					t.Errorf("body = %s, want %s", got, want)
				}
				w.Header().Set("Content-Type", "audio/mpeg")
				writeRaw(t, w, []byte("ID3-mp3"))
			}))
			defer srv.Close()

			got, err := NewElevenLabs(srv.URL+"/", "el-key").Synthesize(context.Background(), &tc.req)
			if err != nil {
				t.Fatalf("Synthesize: %v", err)
			}
			if string(got.Audio) != "ID3-mp3" || got.Size != 7 || got.ContentType != "audio/mpeg" {
				t.Errorf("speech = %+v", got)
			}
			if got.Usage == nil || got.Usage.Characters != 8 {
				t.Errorf("usage = %+v", got.Usage)
			}
		})
	}
}

func TestElevenLabs_Errors(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(t, w, http.StatusUnprocessableEntity, `{"detail":[{"loc":["body","model_id"],"msg":"field required","type":"missing"}]}`)
	}))
	defer srv.Close()
	c := NewElevenLabs(srv.URL, "el-key")
	_, err := c.Transcribe(context.Background(), &TranscribeRequest{Model: "scribe_v2", Audio: audio(), Filename: "a.ogg"})
	if statusErr, ok := errors.AsType[*common.HTTPStatusError](err); !ok || statusErr.StatusCode != http.StatusUnprocessableEntity {
		t.Errorf("transcribe err = %v", err)
	}
	_, err = c.Synthesize(context.Background(), &SynthesizeRequest{Model: "m", Text: "t", Voice: "v"})
	if statusErr, ok := errors.AsType[*common.HTTPStatusError](err); !ok || statusErr.StatusCode != http.StatusUnprocessableEntity {
		t.Errorf("synthesize err = %v", err)
	}
}

func TestNewElevenLabs_DefaultBaseURL(t *testing.T) {
	if got := NewElevenLabs("", "k").baseURL; got != ElevenLabsBaseURL {
		t.Errorf("baseURL = %q", got)
	}
}

func TestElevenLabsTranscribe_UnknownFormat(t *testing.T) {
	_, err := NewElevenLabs("http://127.0.0.1:1", "k").Transcribe(context.Background(),
		&TranscribeRequest{Model: "scribe_v2", Audio: audio()})
	if err == nil || !strings.Contains(err.Error(), "cannot determine audio format") {
		t.Fatalf("err = %v, want a format error", err)
	}
}

func TestElevenLabsSynthesize_ContentTypeFallback(t *testing.T) {
	tests := map[string]string{
		"":              "audio/mpeg",
		"mp3_44100_128": "audio/mpeg",
		"pcm_24000":     "audio/pcm",
		"ulaw_8000":     "audio/basic",
		"opus_48000_64": "audio/ogg",
		"wav_16000":     "audio/wav",
	}
	for format, want := range tests {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.Header()["Content-Type"] = nil
			writeRaw(t, w, []byte{0, 1, 2})
		}))
		got, err := NewElevenLabs(srv.URL, "k").Synthesize(context.Background(),
			&SynthesizeRequest{Model: "m", Text: "t", Voice: "v", Format: format})
		srv.Close()
		if err != nil || got.ContentType != want {
			t.Errorf("format %q: got %+v, err %v, want %s", format, got, err, want)
		}
	}
}

func TestElevenLabsSynthesize_NotAudio(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(t, w, http.StatusOK, `{"detail":{"status":"quota_exceeded"}}`)
	}))
	defer srv.Close()
	got, err := NewElevenLabs(srv.URL, "k").Synthesize(context.Background(),
		&SynthesizeRequest{Model: "m", Text: "t", Voice: "v"})
	if err == nil || got != nil || !strings.Contains(err.Error(), "quota_exceeded") {
		t.Fatalf("got %v, err = %v, want an error with the body", got, err)
	}
}
