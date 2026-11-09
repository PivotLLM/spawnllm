/******************************************************************************
 * Copyright (c) 2026 Tenebris Technologies Inc.                              *
 * Please see LICENSE file for details.                                       *
 ******************************************************************************/

package speech

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"mime/multipart"
	"net/http"
	"net/url"
	"strings"

	"github.com/PivotLLM/spawnllm/internal/httpcall"
	"github.com/PivotLLM/spawnllm/logger"
)

// ElevenLabsBaseURL is the ElevenLabs API's base URL, used when
// NewElevenLabs is given none.
const ElevenLabsBaseURL = "https://api.elevenlabs.io"

// ElevenLabs calls the ElevenLabs API (key in xi-api-key):
//
//   - Transcribe: POST {base}/v1/speech-to-text, multipart model_id, file and
//     language_code (https://elevenlabs.io/docs/api-reference/speech-to-text/convert).
//   - Synthesize: POST {base}/v1/text-to-speech/{voice_id}?output_format=…
//     with JSON {text, model_id}
//     (https://elevenlabs.io/docs/api-reference/text-to-speech/convert).
type ElevenLabs struct {
	baseURL    string
	apiKey     string
	httpClient *http.Client
}

// NewElevenLabs returns an ElevenLabs client. An empty baseURL means
// ElevenLabsBaseURL.
func NewElevenLabs(baseURL, apiKey string, opts ...Option) *ElevenLabs {
	o := applyOptions(opts)
	if baseURL == "" {
		baseURL = ElevenLabsBaseURL
	}
	return &ElevenLabs{baseURL: strings.TrimRight(baseURL, "/"), apiKey: apiKey, httpClient: o.httpClient}
}

var (
	_ Transcriber = (*ElevenLabs)(nil)
	_ Synthesizer = (*ElevenLabs)(nil)
)

func (e *ElevenLabs) call(model string, expectJSON bool) httpcall.Call {
	return httpcall.Call{Component: component, Backend: "elevenlabs", Model: model, APIBase: e.baseURL, JSON: expectJSON}
}

// Transcribe sends the audio to Scribe (Model is the model_id, e.g.
// scribe_v2). Language is sent as language_code; Prompt is not sent (Scribe
// has no such field). Language is the detected language_code and
// DurationSeconds the audio_duration_secs, when returned. Scribe reports no
// usage or cost.
func (e *ElevenLabs) Transcribe(ctx context.Context, req *TranscribeRequest) (*Transcription, error) {
	if err := validateTranscribe(req); err != nil {
		return nil, err
	}
	if req.Prompt != "" {
		logger.DebugCF(component, "ElevenLabs transcription has no prompt field; prompt not sent",
			map[string]any{"model": req.Model})
	}
	var body bytes.Buffer
	w := multipart.NewWriter(&body)
	fields := [][2]string{{"model_id", req.Model}}
	if req.Language != "" {
		fields = append(fields, [2]string{"language_code", req.Language})
	}
	if err := httpcall.WriteFields(w, fields); err != nil {
		return nil, err
	}
	if err := httpcall.WriteFilePart(w, "file", audioFilename(req.Filename), req.ContentType, req.Audio); err != nil {
		return nil, err
	}
	if err := w.Close(); err != nil {
		return nil, fmt.Errorf("failed to close multipart writer: %w", err)
	}
	httpReq, err := httpcall.NewPost(ctx, e.baseURL+"/v1/speech-to-text", body.Bytes(), w.FormDataContentType())
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Xi-Api-Key", e.apiKey)

	resp, err := httpcall.Do(e.httpClient, httpReq, e.call(req.Model, true))
	if err != nil {
		return nil, err
	}
	var parsed struct {
		Text              string  `json:"text"`
		LanguageCode      string  `json:"language_code"`
		AudioDurationSecs float64 `json:"audio_duration_secs"`
	}
	if err := httpcall.DecodeJSON(resp.Body, &parsed); err != nil {
		return nil, err
	}
	return &Transcription{Text: parsed.Text, Language: parsed.LanguageCode, DurationSeconds: parsed.AudioDurationSecs}, nil
}

// Synthesize speaks the text in the voice whose id is Voice. Format is sent
// as output_format when set (the API's default is mp3_44100_128).
// Instructions is not sent (the API has no such field). ContentType is the
// response's Content-Type.
func (e *ElevenLabs) Synthesize(ctx context.Context, req *SynthesizeRequest) (*Speech, error) {
	if err := validateSynthesize(req, true); err != nil {
		return nil, err
	}
	if req.Instructions != "" {
		logger.DebugCF(component, "ElevenLabs speech has no instructions field; instructions not sent",
			map[string]any{"model": req.Model})
	}
	data, err := json.Marshal(map[string]string{"text": req.Text, "model_id": req.Model})
	if err != nil {
		return nil, fmt.Errorf("failed to encode request: %w", err)
	}
	endpoint := e.baseURL + "/v1/text-to-speech/" + url.PathEscape(req.Voice)
	if req.Format != "" {
		endpoint += "?" + url.Values{"output_format": {req.Format}}.Encode()
	}
	httpReq, err := httpcall.NewPost(ctx, endpoint, data, "application/json")
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Xi-Api-Key", e.apiKey)

	resp, err := httpcall.Do(e.httpClient, httpReq, e.call(req.Model, false))
	if err != nil {
		return nil, err
	}
	return &Speech{
		Audio:       resp.Body,
		ContentType: resp.Header.Get("Content-Type"),
		Size:        int64(len(resp.Body)),
		Usage:       characterUsage(req.Text),
	}, nil
}
