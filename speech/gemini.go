/******************************************************************************
 * Copyright (c) 2026 Tenebris Technologies Inc.                              *
 * Please see LICENSE file for details.                                       *
 ******************************************************************************/

package speech

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"path/filepath"
	"strings"

	"github.com/PivotLLM/spawnllm/internal/httpcall"
	"github.com/PivotLLM/spawnllm/logger"
	"github.com/PivotLLM/spawnllm/protocoltypes"
)

// GeminiBaseURL is the Gemini API's base URL, used when NewGemini is given
// none.
const GeminiBaseURL = "https://generativelanguage.googleapis.com/v1beta"

// geminiTranscribeInstruction is the instruction sent with the audio. The
// first sentence is the one Google's audio guide uses
// (https://ai.google.dev/gemini-api/docs/audio); the second keeps the reply
// to the transcript alone.
const geminiTranscribeInstruction = "Generate a transcript of the speech. Reply with the transcript only."

// Gemini calls the Gemini API's generateContent
// (POST {base}/models/{model}:generateContent, key in x-goog-api-key;
// https://ai.google.dev/api/generate-content):
//
//   - Transcribe sends the audio as inlineData with a transcription
//     instruction (https://ai.google.dev/gemini-api/docs/audio). The inline
//     request limit is 20 MB in all; spawnllm does not check it.
//   - Synthesize asks a TTS model for responseModalities ["AUDIO"] with
//     speechConfig.voiceConfig.prebuiltVoiceConfig.voiceName
//     (https://ai.google.dev/gemini-api/docs/speech-generation) and returns
//     the inlineData bytes and mimeType unchanged: raw PCM (16-bit, mono,
//     24 kHz, e.g. "audio/L16;codec=pcm;rate=24000"), not a playable file.
type Gemini struct {
	baseURL    string
	apiKey     string
	httpClient *http.Client
}

// NewGemini returns a Gemini client. An empty baseURL means GeminiBaseURL.
func NewGemini(baseURL, apiKey string, opts ...Option) *Gemini {
	o := applyOptions(opts)
	if baseURL == "" {
		baseURL = GeminiBaseURL
	}
	return &Gemini{baseURL: strings.TrimRight(baseURL, "/"), apiKey: apiKey, httpClient: o.httpClient}
}

var (
	_ Transcriber = (*Gemini)(nil)
	_ Synthesizer = (*Gemini)(nil)
)

type geminiInlineData struct {
	MimeType string `json:"mimeType"`
	Data     string `json:"data"`
}

type geminiPart struct {
	Text       string            `json:"text,omitempty"`
	InlineData *geminiInlineData `json:"inlineData,omitempty"`
	Thought    bool              `json:"thought,omitempty"`
}

type geminiResponse struct {
	Candidates []struct {
		Content struct {
			Parts []geminiPart `json:"parts"`
		} `json:"content"`
		FinishReason string `json:"finishReason"`
	} `json:"candidates"`
	PromptFeedback *struct {
		BlockReason string `json:"blockReason"`
	} `json:"promptFeedback"`
	UsageMetadata *struct {
		PromptTokenCount     int `json:"promptTokenCount"`
		CandidatesTokenCount int `json:"candidatesTokenCount"`
		TotalTokenCount      int `json:"totalTokenCount"`
	} `json:"usageMetadata"`
}

// Transcribe sends the audio with an instruction to transcribe it. Language
// and Prompt, when set, are added to the instruction. The transcript is the
// text of the first candidate's parts (thought parts left out), as returned.
// The audio's media type is ContentType, else the one Filename's extension
// names.
func (g *Gemini) Transcribe(ctx context.Context, req *TranscribeRequest) (*Transcription, error) {
	if err := validateTranscribe(req); err != nil {
		return nil, err
	}
	mimeType := req.ContentType
	if mimeType == "" {
		mimeType = geminiAudioType(req.Filename)
	}
	if mimeType == "" {
		return nil, fmt.Errorf("cannot determine audio type of %q: set the content type", filepath.Base(req.Filename))
	}
	raw, err := io.ReadAll(req.Audio)
	if err != nil {
		return nil, fmt.Errorf("failed to read audio: %w", err)
	}

	instruction := geminiTranscribeInstruction
	if req.Language != "" {
		instruction += " The speech is in the language with code " + req.Language + "."
	}
	if req.Prompt != "" {
		instruction += "\n\nContext for the transcription:\n" + req.Prompt
	}
	payload := map[string]any{
		"contents": []map[string]any{{
			"role": "user",
			"parts": []geminiPart{
				{Text: instruction},
				{InlineData: &geminiInlineData{MimeType: mimeType, Data: base64.StdEncoding.EncodeToString(raw)}},
			},
		}},
	}
	parsed, err := g.generate(ctx, req.Model, payload)
	if err != nil {
		return nil, err
	}
	var text strings.Builder
	for _, p := range parsed.Candidates[0].Content.Parts {
		if !p.Thought {
			text.WriteString(p.Text)
		}
	}
	return &Transcription{Text: text.String(), Usage: geminiUsage(parsed)}, nil
}

// Synthesize asks a Gemini TTS model to speak the text. Voice is a prebuilt
// voice name (Kore, Puck, …); empty lets the model choose. Instructions, when
// set, go before the text as "<instructions>: <text>". Format is ignored:
// Gemini returns PCM only.
func (g *Gemini) Synthesize(ctx context.Context, req *SynthesizeRequest) (*Speech, error) {
	if err := validateSynthesize(req, false); err != nil {
		return nil, err
	}
	if req.Format != "" {
		logger.DebugCF(component, "Gemini speech has no output format choice; format ignored",
			map[string]any{"model": req.Model, "format": req.Format})
	}
	text := req.Text
	if req.Instructions != "" {
		text = req.Instructions + ": " + req.Text
	}
	genConfig := map[string]any{"responseModalities": []string{"AUDIO"}}
	if req.Voice != "" {
		genConfig["speechConfig"] = map[string]any{
			"voiceConfig": map[string]any{
				"prebuiltVoiceConfig": map[string]any{"voiceName": req.Voice},
			},
		}
	}
	payload := map[string]any{
		"contents":         []map[string]any{{"role": "user", "parts": []geminiPart{{Text: text}}}},
		"generationConfig": genConfig,
	}
	parsed, err := g.generate(ctx, req.Model, payload)
	if err != nil {
		return nil, err
	}
	for _, p := range parsed.Candidates[0].Content.Parts {
		if p.InlineData == nil {
			continue
		}
		audio, err := base64.StdEncoding.DecodeString(p.InlineData.Data)
		if err != nil {
			return nil, fmt.Errorf("failed to decode response audio: %w", err)
		}
		usage := characterUsage(req.Text)
		if u := geminiUsage(parsed); u != nil {
			usage.InputTokens, usage.OutputTokens, usage.TotalTokens = u.InputTokens, u.OutputTokens, u.TotalTokens
		}
		return &Speech{Audio: audio, ContentType: p.InlineData.MimeType, Size: int64(len(audio)), Usage: usage}, nil
	}
	return nil, fmt.Errorf("gemini returned no audio (finish reason %q)", parsed.Candidates[0].FinishReason)
}

// generate posts a generateContent request and returns a response with at
// least one candidate.
func (g *Gemini) generate(ctx context.Context, model string, payload any) (*geminiResponse, error) {
	data, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to encode request: %w", err)
	}
	model = strings.TrimPrefix(model, "models/")
	endpoint := g.baseURL + "/models/" + url.PathEscape(model) + ":generateContent"
	httpReq, err := httpcall.NewPost(ctx, endpoint, data, "application/json")
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("X-Goog-Api-Key", g.apiKey)

	resp, err := httpcall.Do(g.httpClient, httpReq,
		httpcall.Call{Component: component, Backend: "gemini", Model: model, APIBase: g.baseURL, JSON: true})
	if err != nil {
		return nil, err
	}
	var parsed geminiResponse
	if err := httpcall.DecodeJSON(resp.Body, &parsed); err != nil {
		return nil, err
	}
	if len(parsed.Candidates) == 0 {
		if parsed.PromptFeedback != nil && parsed.PromptFeedback.BlockReason != "" {
			return nil, fmt.Errorf("gemini blocked the request: %s", parsed.PromptFeedback.BlockReason)
		}
		return nil, errors.New("gemini returned no candidates")
	}
	return &parsed, nil
}

func geminiUsage(r *geminiResponse) *protocoltypes.TaskUsage {
	if r.UsageMetadata == nil {
		return nil
	}
	return &protocoltypes.TaskUsage{
		InputTokens:  r.UsageMetadata.PromptTokenCount,
		OutputTokens: r.UsageMetadata.CandidatesTokenCount,
		TotalTokens:  r.UsageMetadata.TotalTokenCount,
	}
}

// geminiAudioTypes maps file extensions to the audio types Gemini lists
// (https://ai.google.dev/gemini-api/docs/audio).
var geminiAudioTypes = map[string]string{
	".wav":  "audio/wav",
	".mp3":  "audio/mp3",
	".aiff": "audio/aiff",
	".aac":  "audio/aac",
	".ogg":  "audio/ogg",
	".oga":  "audio/ogg",
	".opus": "audio/ogg",
	".flac": "audio/flac",
	".m4a":  "audio/m4a",
	".webm": "audio/webm",
}

func geminiAudioType(filename string) string {
	return geminiAudioTypes[strings.ToLower(filepath.Ext(filename))]
}
