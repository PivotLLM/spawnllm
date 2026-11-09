/******************************************************************************
 * Copyright (c) 2026 Tenebris Technologies Inc.                              *
 * Please see LICENSE file for details.                                       *
 ******************************************************************************/

package speech

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"mime/multipart"
	"net/http"
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

// binaryCall is the call description for a response that must be audio.
func (c *OpenAI) binaryCall(model string) httpcall.Call {
	call := c.call(model, false)
	call.Binary = true
	return call
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
// Multipart (OpenAI, Groq, local servers): fields file, model,
// response_format=json, and language and prompt when set. The file part is
// typed ContentType (application/octet-stream when empty) and named
// Filename; the endpoints pick the decoder by the name's extension, so a
// name without one gets ContentType's ("audio.ogg" for an unnamed
// audio/ogg), and audio with neither is refused before sending.
//
// Input audio (OpenRouter): a JSON body {"input_audio": {"data": <base64>,
// "format": …}, "model": …}, plus language when set. The format is
// Filename's extension, else ContentType's, with oga and opus sent as ogg.
// OpenRouter also accepts the multipart upload; TranscriptionAuto picks
// input_audio for it by choice, the shape ClawEh has used in production.
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
	filename, err := uploadFilename(req.Filename, req.ContentType)
	if err != nil {
		return nil, err
	}
	var body bytes.Buffer
	w := multipart.NewWriter(&body)
	if err := httpcall.WriteFilePart(w, "file", filename, req.ContentType, req.Audio); err != nil {
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
	format, err := openRouterFormat(req.Filename, req.ContentType)
	if err != nil {
		return nil, err
	}
	if req.Prompt != "" {
		logger.DebugCF(component, "OpenRouter transcription has no prompt field; prompt not sent",
			map[string]any{"model": req.Model})
	}
	// Written by hand so the audio is base64-encoded straight into the body,
	// without a copy of the audio or of its encoding. Keys in sorted order,
	// as json.Marshal of a map writes them.
	var body bytes.Buffer
	body.WriteString(`{"input_audio":{"data":"`)
	if err := httpcall.WriteBase64(&body, req.Audio); err != nil {
		return nil, fmt.Errorf("audio: %w", err)
	}
	body.WriteString(`","format":`)
	httpcall.WriteJSONString(&body, format)
	body.WriteString(`}`)
	if req.Language != "" {
		body.WriteString(`,"language":`)
		httpcall.WriteJSONString(&body, req.Language)
	}
	body.WriteString(`,"model":`)
	httpcall.WriteJSONString(&body, req.Model)
	body.WriteString(`}`)
	return httpcall.NewPost(ctx, c.baseURL+"/audio/transcriptions", body.Bytes(), "application/json")
}

// Synthesize sends {model, input, voice, response_format?, instructions?} to
// {base}/audio/speech and returns the body as the audio. A 2xx answer that is
// text, JSON or HTML is an error, not audio. ContentType is the
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

	resp, err := httpcall.Do(c.httpClient, httpReq, c.binaryCall(req.Model))
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
