/******************************************************************************
 * Copyright (c) 2026 Tenebris Technologies Inc.                              *
 * Please see LICENSE file for details.                                       *
 ******************************************************************************/

package imagegen

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"mime/multipart"
	"net/http"
	"net/url"
	"strconv"
	"strings"

	"github.com/PivotLLM/spawnllm/common"
	"github.com/PivotLLM/spawnllm/internal/httpcall"
	"github.com/PivotLLM/spawnllm/protocoltypes"
)

const component = "imagegen"

// ErrEditUnsupported is returned by OpenAI.Edit for an endpoint where
// /images/edits is not available (OpenRouter, whose /images takes input
// images as input_references instead).
var ErrEditUnsupported = errors.New("image edits are not supported by this endpoint")

// ErrNoImages is returned when a successful response carries no image.
var ErrNoImages = errors.New("response contains no images")

// OpenAI calls OpenAI-style image endpoints under one base URL
// (https://developers.openai.com/api/reference/resources/images):
//
//   - Generate: POST {base}/images/generations with JSON
//     {model, prompt, n?, size?, quality?}. On OpenRouter (an openrouter.ai
//     base URL) the same body goes to POST {base}/images, its image endpoint
//     (https://openrouter.ai/docs/api/api-reference/images/generate-an-image),
//     whose usage.cost is returned as CostUSD.
//   - Edit: POST {base}/images/edits, multipart model, prompt, the input
//     images, mask, n, size and quality. /images/edits is not available on
//     OpenRouter.
//
// Each data item's b64_json is decoded into Data (with the item's media_type
// when given, else the type detected from the bytes) and a url item is
// returned as URL with Size -1, never fetched. A 2xx response with no image
// (empty data, or items with neither b64_json nor url) is an error.
type OpenAI struct {
	baseURL    string
	apiKey     string
	httpClient *http.Client
	openRouter bool
}

// NewOpenAI returns a client for an OpenAI-style base URL. apiKey is sent as
// a bearer token when not empty.
func NewOpenAI(baseURL, apiKey string, opts ...Option) *OpenAI {
	o := applyOptions(opts)
	return &OpenAI{
		baseURL:    strings.TrimRight(baseURL, "/"),
		apiKey:     apiKey,
		httpClient: o.httpClient,
		openRouter: isOpenRouter(baseURL),
	}
}

var (
	_ Generator = (*OpenAI)(nil)
	_ Editor    = (*OpenAI)(nil)
)

// Generate creates images from the prompt.
func (c *OpenAI) Generate(ctx context.Context, req *GenerateRequest) (*Result, error) {
	switch {
	case req == nil:
		return nil, errors.New("generate request is nil")
	case req.Model == "":
		return nil, errors.New("generate: model is required")
	case req.Prompt == "":
		return nil, errors.New("generate: prompt is required")
	}
	payload := map[string]any{"model": req.Model, "prompt": req.Prompt}
	if req.N > 0 {
		payload["n"] = req.N
	}
	if req.Size != "" {
		payload["size"] = req.Size
	}
	if req.Quality != "" {
		payload["quality"] = req.Quality
	}
	data, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to encode request: %w", err)
	}
	path := "/images/generations"
	if c.openRouter {
		path = "/images"
	}
	httpReq, err := httpcall.NewPost(ctx, c.baseURL+path, data, "application/json")
	if err != nil {
		return nil, err
	}
	return c.send(httpReq, req.Model)
}

// Edit creates images from the prompt and the input images. A single image
// is sent as the form field "image", several as repeated "image[]" fields;
// the mask as "mask".
func (c *OpenAI) Edit(ctx context.Context, req *EditRequest) (*Result, error) {
	switch {
	case req == nil:
		return nil, errors.New("edit request is nil")
	case req.Model == "":
		return nil, errors.New("edit: model is required")
	case req.Prompt == "":
		return nil, errors.New("edit: prompt is required")
	case len(req.Images) == 0:
		return nil, errors.New("edit: at least one image is required")
	case c.openRouter:
		return nil, ErrEditUnsupported
	}

	var body bytes.Buffer
	w := multipart.NewWriter(&body)
	fields := [][2]string{{"model", req.Model}, {"prompt", req.Prompt}}
	if req.N > 0 {
		fields = append(fields, [2]string{"n", strconv.Itoa(req.N)})
	}
	if req.Size != "" {
		fields = append(fields, [2]string{"size", req.Size})
	}
	if req.Quality != "" {
		fields = append(fields, [2]string{"quality", req.Quality})
	}
	if err := httpcall.WriteFields(w, fields); err != nil {
		return nil, err
	}
	imageField := "image"
	if len(req.Images) > 1 {
		imageField = "image[]"
	}
	for i, img := range req.Images {
		if err := writeImage(w, imageField, img); err != nil {
			return nil, fmt.Errorf("image %d: %w", i+1, err)
		}
	}
	if req.Mask != nil {
		if err := writeImage(w, "mask", *req.Mask); err != nil {
			return nil, fmt.Errorf("mask: %w", err)
		}
	}
	if err := w.Close(); err != nil {
		return nil, fmt.Errorf("failed to close multipart writer: %w", err)
	}
	httpReq, err := httpcall.NewPost(ctx, c.baseURL+"/images/edits", body.Bytes(), w.FormDataContentType())
	if err != nil {
		return nil, err
	}
	return c.send(httpReq, req.Model)
}

func writeImage(w *multipart.Writer, field string, img InputImage) error {
	if img.Data == nil {
		return errors.New("no data")
	}
	name := img.Filename
	if name == "" {
		name = "image"
	}
	return httpcall.WriteFilePart(w, field, name, img.ContentType, img.Data)
}

// imagesResponse is the images response of OpenAI ({created, data: [{b64_json
// | url, revised_prompt}], usage {input_tokens, output_tokens, total_tokens}})
// and of OpenRouter ({created, data: [{b64_json, media_type}], usage
// {prompt_tokens, completion_tokens, total_tokens, cost}}).
type imagesResponse struct {
	Data []struct {
		B64JSON   string `json:"b64_json"`
		URL       string `json:"url"`
		MediaType string `json:"media_type"`
	} `json:"data"`
	Usage *struct {
		InputTokens      int      `json:"input_tokens"`
		OutputTokens     int      `json:"output_tokens"`
		PromptTokens     int      `json:"prompt_tokens"`
		CompletionTokens int      `json:"completion_tokens"`
		TotalTokens      int      `json:"total_tokens"`
		Cost             *float64 `json:"cost"`
	} `json:"usage"`
}

func (c *OpenAI) send(httpReq *http.Request, model string) (*Result, error) {
	if c.apiKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+c.apiKey)
	}
	resp, err := httpcall.Do(c.httpClient, httpReq,
		httpcall.Call{Component: component, Backend: "openai", Model: model, APIBase: c.baseURL, JSON: true})
	if err != nil {
		return nil, err
	}
	var parsed imagesResponse
	if err := httpcall.DecodeJSON(resp.Body, &parsed); err != nil {
		return nil, err
	}

	collector := common.NewImageCollector(model)
	for _, item := range parsed.Data {
		switch {
		case item.B64JSON != "" && item.MediaType != "":
			collector.AddURL("data:" + item.MediaType + ";base64," + item.B64JSON)
		case item.B64JSON != "":
			collector.AddBase64(item.B64JSON)
		default:
			collector.AddURL(item.URL)
		}
	}
	if len(collector.Images()) == 0 {
		return nil, fmt.Errorf("%w (%d data items)", ErrNoImages, len(parsed.Data))
	}
	out := &Result{Images: collector.Images()}
	if u := parsed.Usage; u != nil {
		out.Usage = &protocoltypes.TaskUsage{
			InputTokens:  max(u.InputTokens, u.PromptTokens),
			OutputTokens: max(u.OutputTokens, u.CompletionTokens),
			TotalTokens:  u.TotalTokens,
		}
		if c.openRouter {
			out.CostUSD = u.Cost
		}
	}
	return out, nil
}

// isOpenRouter reports whether a base URL is OpenRouter's.
func isOpenRouter(baseURL string) bool {
	u, err := url.Parse(baseURL)
	if err != nil {
		return false
	}
	host := strings.ToLower(u.Hostname())
	return host == "openrouter.ai" || strings.HasSuffix(host, ".openrouter.ai")
}
