/******************************************************************************
 * Copyright (c) 2026 Tenebris Technologies Inc.                              *
 * Please see LICENSE file for details.                                       *
 ******************************************************************************/

// Package speech calls speech-to-text (Transcriber) and text-to-speech
// (Synthesizer) APIs: OpenAI-style endpoints (OpenAI, Groq, local servers,
// OpenRouter), Gemini and ElevenLabs.
//
// Each call is one request. The caller supplies the base URL, API key and
// model, and owns every policy decision: model choice, fallback, cooldown,
// size limits. spawnllm returns what the API returned: no size caps and no
// rewriting of the audio or text. The request's timeout is the context's
// deadline; the default HTTP client sets none of its own.
//
// Errors keep the HTTP status: a non-2xx response is a
// *common.HTTPStatusError (status, Retry-After, a body excerpt of at most
// 1 KiB), so spawnllm.ClassifyError reads it as it does a chat provider's.
package speech

import (
	"context"
	"io"
	"net/http"
	"net/url"
	"strings"

	"github.com/PivotLLM/spawnllm/protocoltypes"
)

// Transcriber turns audio into text.
//
// Audio is an io.Reader so a backend can consume it as it arrives; the
// backends here read it to the end before sending, but a streaming backend
// (a local model fed audio as it is recorded) can implement the same
// interface without changing callers.
type Transcriber interface {
	Transcribe(ctx context.Context, req *TranscribeRequest) (*Transcription, error)
}

// Synthesizer turns text into audio.
type Synthesizer interface {
	Synthesize(ctx context.Context, req *SynthesizeRequest) (*Speech, error)
}

// TranscribeRequest is one transcription.
type TranscribeRequest struct {
	// Model is the model to call (required).
	Model string
	// Audio is read to the end (required).
	Audio io.Reader
	// Filename is the audio's file name. Its extension tells the OpenAI-style
	// endpoints the format; OpenRouter requires it (the format sent is the
	// extension).
	Filename string
	// ContentType is the audio's media type (audio/ogg, …). Gemini requires
	// it or a Filename with a known extension; the multipart backends send it
	// as the file part's type when set.
	ContentType string
	// Language is an optional ISO-639-1 hint (en, fr, …).
	Language string
	// Prompt is optional text that guides the transcription (names, terms,
	// the preceding text). Sent where the API has a field for it (OpenAI
	// style, Gemini); OpenRouter and ElevenLabs have none and ignore it.
	Prompt string
}

// Transcription is a transcription's result. Fields the API did not return
// are zero.
type Transcription struct {
	// Text is the transcript as the API returned it.
	Text string
	// Language is the detected language, when the API reports it.
	Language string
	// DurationSeconds is the audio's duration, when the API reports it.
	DurationSeconds float64
	// Usage is what the API reported it used; nil when it reported nothing.
	Usage *protocoltypes.TaskUsage
	// CostUSD is the cost the API reported (OpenRouter's usage.cost); nil
	// when it reported none, which is not a cost of $0.00.
	CostUSD *float64
}

// SynthesizeRequest is one speech synthesis.
type SynthesizeRequest struct {
	// Model is the model to call (required).
	Model string
	// Text is the text to speak (required).
	Text string
	// Voice names the voice. Required for OpenAI style and ElevenLabs (the
	// voice id); optional for Gemini (a prebuilt voice name).
	Voice string
	// Format is the output format in the API's own terms (OpenAI: mp3, opus,
	// wav, pcm, …; ElevenLabs: mp3_44100_128, pcm_24000, …). Empty uses the
	// API's default. Gemini has no choice and ignores it.
	Format string
	// Instructions is optional direction on how to speak (tone, pace).
	// OpenAI style sends it as instructions; Gemini puts it before the text
	// ("<instructions>: <text>"), the form Google documents for its
	// generateContent TTS models; ElevenLabs has no such field and ignores it.
	Instructions string
}

// Speech is a synthesis result.
type Speech struct {
	// Audio is the audio exactly as the API returned it.
	Audio []byte
	// ContentType is the audio's media type as the API declared it. Gemini
	// returns raw PCM (for example "audio/L16;codec=pcm;rate=24000": 16-bit
	// signed little-endian mono samples with no container); it is returned
	// unchanged and not converted, so the caller must wrap it (WAV, …) if it
	// needs a playable file.
	ContentType string
	// Size is len(Audio).
	Size int64
	// Usage always carries Characters (the text's length in code points);
	// Gemini also reports tokens.
	Usage *protocoltypes.TaskUsage
	// CostUSD is the cost the API reported; nil when it reported none.
	CostUSD *float64
}

// Option configures a backend.
type Option func(*options)

type options struct {
	httpClient    *http.Client
	transcription TranscriptionAPI
}

// WithHTTPClient sets the HTTP client. The default is a client with no
// timeout of its own: the context's deadline bounds each call.
func WithHTTPClient(c *http.Client) Option {
	return func(o *options) {
		if c != nil {
			o.httpClient = c
		}
	}
}

// TranscriptionAPI selects the request shape OpenAI.Transcribe sends.
type TranscriptionAPI int

const (
	// TranscriptionAuto picks TranscriptionInputAudio for an openrouter.ai
	// base URL and TranscriptionMultipart for any other.
	TranscriptionAuto TranscriptionAPI = iota
	// TranscriptionMultipart is OpenAI's multipart upload (OpenAI, Groq,
	// local Whisper servers).
	TranscriptionMultipart
	// TranscriptionInputAudio is OpenRouter's JSON body with base64 audio
	// (input_audio {data, format}).
	TranscriptionInputAudio
)

// WithTranscriptionAPI chooses the transcription request shape for an
// OpenAI-style backend instead of choosing it by host.
func WithTranscriptionAPI(api TranscriptionAPI) Option {
	return func(o *options) { o.transcription = api }
}

func applyOptions(opts []Option) options {
	o := options{httpClient: &http.Client{}}
	for _, opt := range opts {
		if opt != nil {
			opt(&o)
		}
	}
	return o
}

// openRouterHost is the host whose transcription endpoint takes OpenRouter's
// JSON/base64 body rather than the OpenAI multipart upload.
const openRouterHost = "openrouter.ai"

// sttDefaultModels maps the host of a known speech-to-text endpoint to its
// usual transcription model.
var sttDefaultModels = map[string]string{
	"api.groq.com":   "whisper-large-v3",
	"api.openai.com": "whisper-1",
	openRouterHost:   "openai/whisper-large-v3",
}

// DefaultSTTModel returns the usual transcription model for a base URL,
// chosen by its host, or "" when the host is not a known STT service. The
// backends never fill a model in themselves; this is for the caller.
func DefaultSTTModel(baseURL string) string {
	return sttDefaultModels[urlHost(baseURL)]
}

// urlHost returns the lower-cased host name (without port) of a URL, or ""
// when it cannot be parsed.
func urlHost(raw string) string {
	if raw == "" {
		return ""
	}
	u, err := url.Parse(raw)
	if err != nil {
		return ""
	}
	return strings.ToLower(u.Hostname())
}

// isOpenRouter reports whether a base URL is OpenRouter's.
func isOpenRouter(baseURL string) bool {
	host := urlHost(baseURL)
	return host == openRouterHost || strings.HasSuffix(host, "."+openRouterHost)
}

// characterUsage is the usage of a synthesis: the text's length in code
// points.
func characterUsage(text string) *protocoltypes.TaskUsage {
	return &protocoltypes.TaskUsage{Characters: len([]rune(text))}
}
