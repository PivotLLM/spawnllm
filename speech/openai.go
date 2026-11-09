/******************************************************************************
 * Copyright (c) 2026 Tenebris Technologies Inc.                              *
 * Please see LICENSE file for details.                                       *
 ******************************************************************************/

package speech

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"path/filepath"
	"strings"

	"github.com/PivotLLM/spawnllm/internal/httpcall"
	"github.com/PivotLLM/spawnllm/logger"
	"github.com/PivotLLM/spawnllm/protocoltypes"
)

const component = "speech"

// OpenAI calls OpenAI-style speech endpoints under one base URL (for example
// https://api.openai.com/v1, https://api.groq.com/openai/v1,
// https://openrouter.ai/api/v1 or a local server):
//
//   - Transcribe: POST {base}/audio/transcriptions, as a multipart upload
//     (https://developers.openai.com/api/reference/resources/audio,
//     "Create transcription") or, for OpenRouter, as JSON with base64 audio
//     (https://openrouter.ai/docs/api/api-reference/stt/create-transcription).
//     See TranscriptionAPI.
//   - Synthesize: POST {base}/audio/speech (OpenAI "Create speech";
//     OpenRouter's /audio/speech takes the same fields).
type OpenAI struct {
	baseURL       string
	apiKey        string
	httpClient    *http.Client
	transcription TranscriptionAPI
}

// NewOpenAI returns a client for an OpenAI-style base URL. apiKey is sent as
// a bearer token when not empty (a local server may need none).
func NewOpenAI(baseURL, apiKey string, opts ...Option) *OpenAI {
	o := applyOptions(opts)
	api := o.transcription
	if api == TranscriptionAuto {
		api = TranscriptionMultipart
		if isOpenRouter(baseURL) {
			api = TranscriptionInputAudio
		}
	}
	return &OpenAI{
		baseURL:       strings.TrimRight(baseURL, "/"),
		apiKey:        apiKey,
		httpClient:    o.httpClient,
		transcription: api,
	}
}

var (
	_ Transcriber = (*OpenAI)(nil)
	_ Synthesizer = (*OpenAI)(nil)
)

func (c *OpenAI) call(model string, expectJSON bool) httpcall.Call {
	return httpcall.Call{Component: component, Backend: "openai", Model: model, APIBase: c.baseURL, JSON: expectJSON}
}

func (c *OpenAI) authorize(req *http.Request) {
	if c.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.apiKey)
	}
}

// openAITranscription is the JSON transcription response of both shapes:
// OpenAI's {text, language?, duration?, usage?} (language and duration only
// with verbose_json, which this client does not request, but some
// OpenAI-compatible servers return them anyway) and OpenRouter's
// {text, usage {seconds, input_tokens, output_tokens, total_tokens, cost}}.
type openAITranscription struct {
	Text     string  `json:"text"`
	Language string  `json:"language"`
	Duration float64 `json:"duration"`
	Usage    *struct {
		InputTokens  int      `json:"input_tokens"`
		OutputTokens int      `json:"output_tokens"`
		TotalTokens  int      `json:"total_tokens"`
		Seconds      float64  `json:"seconds"`
		Cost         *float64 `json:"cost"`
	} `json:"usage"`
}

// Transcribe sends the audio to {base}/audio/transcriptions.
//
// Multipart (OpenAI, Groq, local servers): fields file (named Filename, or
// "audio" when empty, typed ContentType or application/octet-stream), model,
// response_format=json, and language and prompt when set.
//
// Input audio (OpenRouter): a JSON body {"input_audio": {"data": <base64>,
// "format": <Filename's extension>}, "model": …}, plus language when set.
// OpenRouter has no prompt field, so Prompt is not sent. Its usage.cost is
// returned as CostUSD and usage.seconds as DurationSeconds.
func (c *OpenAI) Transcribe(ctx context.Context, req *TranscribeRequest) (*Transcription, error) {
	if err := validateTranscribe(req); err != nil {
		return nil, err
	}
	var (
		httpReq *http.Request
		err     error
	)
	if c.transcription == TranscriptionInputAudio {
		httpReq, err = c.inputAudioRequest(ctx, req)
	} else {
		httpReq, err = c.multipartRequest(ctx, req)
	}
	if err != nil {
		return nil, err
	}
	c.authorize(httpReq)

	resp, err := httpcall.Do(c.httpClient, httpReq, c.call(req.Model, true))
	if err != nil {
		return nil, err
	}
	var parsed openAITranscription
	if err := httpcall.DecodeJSON(resp.Body, &parsed); err != nil {
		return nil, err
	}

	out := &Transcription{Text: parsed.Text, Language: parsed.Language, DurationSeconds: parsed.Duration}
	if u := parsed.Usage; u != nil {
		out.Usage = &protocoltypes.TaskUsage{
			InputTokens: u.InputTokens, OutputTokens: u.OutputTokens, TotalTokens: u.TotalTokens, Seconds: u.Seconds,
		}
		if c.transcription == TranscriptionInputAudio {
			// OpenRouter reports the audio's length only as usage.seconds,
			// and its usage.cost is in US dollars (credits).
			if out.DurationSeconds == 0 {
				out.DurationSeconds = u.Seconds
			}
			out.CostUSD = u.Cost
		}
	}
	return out, nil
}

func (c *OpenAI) multipartRequest(ctx context.Context, req *TranscribeRequest) (*http.Request, error) {
	var body bytes.Buffer
	w := multipart.NewWriter(&body)
	if err := httpcall.WriteFilePart(w, "file", audioFilename(req.Filename), req.ContentType, req.Audio); err != nil {
		return nil, err
	}
	fields := [][2]string{{"model", req.Model}, {"response_format", "json"}}
	if req.Language != "" {
		fields = append(fields, [2]string{"language", req.Language})
	}
	if req.Prompt != "" {
		fields = append(fields, [2]string{"prompt", req.Prompt})
	}
	if err := httpcall.WriteFields(w, fields); err != nil {
		return nil, err
	}
	if err := w.Close(); err != nil {
		return nil, fmt.Errorf("failed to close multipart writer: %w", err)
	}
	return httpcall.NewPost(ctx, c.baseURL+"/audio/transcriptions", body.Bytes(), w.FormDataContentType())
}

func (c *OpenAI) inputAudioRequest(ctx context.Context, req *TranscribeRequest) (*http.Request, error) {
	// OpenRouter needs the container format, taken from the file extension.
	format := strings.TrimPrefix(strings.ToLower(filepath.Ext(req.Filename)), ".")
	if format == "" {
		return nil, fmt.Errorf("cannot determine audio format from %q", filepath.Base(req.Filename))
	}
	raw, err := io.ReadAll(req.Audio)
	if err != nil {
		return nil, fmt.Errorf("failed to read audio: %w", err)
	}
	if req.Prompt != "" {
		logger.DebugCF(component, "OpenRouter transcription has no prompt field; prompt not sent",
			map[string]any{"model": req.Model})
	}
	payload := map[string]any{
		"input_audio": map[string]string{
			"data":   base64.StdEncoding.EncodeToString(raw),
			"format": format,
		},
		"model": req.Model,
	}
	if req.Language != "" {
		payload["language"] = req.Language
	}
	data, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to encode request: %w", err)
	}
	return httpcall.NewPost(ctx, c.baseURL+"/audio/transcriptions", data, "application/json")
}

// Synthesize sends {model, input, voice, response_format?, instructions?} to
// {base}/audio/speech and returns the body as the audio. ContentType is the
// response's Content-Type; when the server sends none it is the media type of
// the requested Format (mp3 when none was requested, OpenAI's default).
func (c *OpenAI) Synthesize(ctx context.Context, req *SynthesizeRequest) (*Speech, error) {
	if err := validateSynthesize(req, true); err != nil {
		return nil, err
	}
	payload := map[string]any{"model": req.Model, "input": req.Text, "voice": req.Voice}
	if req.Format != "" {
		payload["response_format"] = req.Format
	}
	if req.Instructions != "" {
		payload["instructions"] = req.Instructions
	}
	data, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to encode request: %w", err)
	}
	httpReq, err := httpcall.NewPost(ctx, c.baseURL+"/audio/speech", data, "application/json")
	if err != nil {
		return nil, err
	}
	c.authorize(httpReq)

	resp, err := httpcall.Do(c.httpClient, httpReq, c.call(req.Model, false))
	if err != nil {
		return nil, err
	}
	contentType := resp.Header.Get("Content-Type")
	if contentType == "" {
		contentType = openAIFormatType(req.Format)
	}
	return &Speech{
		Audio:       resp.Body,
		ContentType: contentType,
		Size:        int64(len(resp.Body)),
		Usage:       characterUsage(req.Text),
	}, nil
}

// openAIFormatType maps an OpenAI response_format to a media type, for a
// server that declares none.
func openAIFormatType(format string) string {
	switch strings.ToLower(format) {
	case "", "mp3":
		return "audio/mpeg"
	case "opus":
		return "audio/ogg"
	case "aac":
		return "audio/aac"
	case "flac":
		return "audio/flac"
	case "wav":
		return "audio/wav"
	case "pcm":
		return "audio/pcm"
	default:
		return "application/octet-stream"
	}
}

func validateTranscribe(req *TranscribeRequest) error {
	switch {
	case req == nil:
		return errors.New("transcribe request is nil")
	case req.Model == "":
		return errors.New("transcribe: model is required")
	case req.Audio == nil:
		return errors.New("transcribe: audio is required")
	}
	return nil
}

func validateSynthesize(req *SynthesizeRequest, voiceRequired bool) error {
	switch {
	case req == nil:
		return errors.New("synthesize request is nil")
	case req.Model == "":
		return errors.New("synthesize: model is required")
	case req.Text == "":
		return errors.New("synthesize: text is required")
	case voiceRequired && req.Voice == "":
		return errors.New("synthesize: voice is required")
	}
	return nil
}

// audioFilename is the file name sent for the audio: the request's, else
// "audio".
func audioFilename(name string) string {
	if name == "" {
		return "audio"
	}
	return name
}
