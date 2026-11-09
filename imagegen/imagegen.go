/******************************************************************************
 * Copyright (c) 2026 Tenebris Technologies Inc.                              *
 * Please see LICENSE file for details.                                       *
 ******************************************************************************/

// Package imagegen calls image-generation APIs: OpenAI-style
// /images/generations and /images/edits (OpenAI, compatible servers) and
// OpenRouter's /images.
//
// Each call is one request. The caller supplies the base URL, API key and
// model, and owns every policy decision. Images come back as
// protocoltypes.GeneratedImage values, exactly as a chat model's generated
// images do (LLMResponse.Images): base64 decoded into Data, a URL returned
// as URL and never fetched (Size -1), nothing filtered or capped. The
// request's timeout is the context's deadline; the default HTTP client sets
// none of its own. A non-2xx response is a *common.HTTPStatusError, which
// spawnllm.ClassifyError reads as it does a chat provider's.
package imagegen

import (
	"context"
	"io"
	"net/http"

	"github.com/PivotLLM/spawnllm/protocoltypes"
)

// Generator creates images from a prompt.
type Generator interface {
	Generate(ctx context.Context, req *GenerateRequest) (*Result, error)
}

// Editor creates images from a prompt and input images.
type Editor interface {
	Edit(ctx context.Context, req *EditRequest) (*Result, error)
}

// GenerateRequest is one image generation. Empty optional fields are not
// sent, so the API's defaults apply.
type GenerateRequest struct {
	// Model is the model to call (required).
	Model string
	// Prompt describes the image (required).
	Prompt string
	// N is the number of images; 0 leaves it to the API (one).
	N int
	// Size is the size in the API's terms (1024x1024, auto, …).
	Size string
	// Quality is the quality in the API's terms (low, medium, high, auto, …).
	Quality string
}

// InputImage is an image sent for editing.
type InputImage struct {
	// Data is read to the end (required).
	Data io.Reader
	// Filename is the part's file name; its extension may tell the API the
	// format. Empty sends "image".
	Filename string
	// ContentType is the image's media type (image/png, …). Empty sends
	// application/octet-stream.
	ContentType string
}

// EditRequest is one image edit.
type EditRequest struct {
	// Model is the model to call (required).
	Model string
	// Prompt describes the result (required).
	Prompt string
	// Images are the input images (at least one).
	Images []InputImage
	// Mask optionally marks, by its transparent areas, where to edit.
	Mask *InputImage
	// N, Size and Quality are as for GenerateRequest.
	N       int
	Size    string
	Quality string
}

// Result is the images of one call, in the order the API returned them.
type Result struct {
	Images []protocoltypes.GeneratedImage
	// Usage is what the API reported it used; nil when it reported nothing.
	Usage *protocoltypes.TaskUsage
	// CostUSD is the cost the API reported (OpenRouter's usage.cost); nil
	// when it reported none, which is not a cost of $0.00.
	CostUSD *float64
}

// Option configures a client.
type Option func(*options)

type options struct {
	httpClient *http.Client
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

func applyOptions(opts []Option) options {
	o := options{httpClient: &http.Client{}}
	for _, opt := range opts {
		if opt != nil {
			opt(&o)
		}
	}
	return o
}
