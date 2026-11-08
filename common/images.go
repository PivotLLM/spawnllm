package common

import (
	"encoding/base64"
	"maps"
	"net/http"
	"strings"

	"github.com/PivotLLM/spawnllm/logger"
	"github.com/PivotLLM/spawnllm/protocoltypes"
)

// Size caps for the images a model generates (LLMResponse.Images). An image
// over MaxImageBytes, or one that would take a response past MaxImagesBytes,
// is left out with a WARN; the reply itself still succeeds.
const (
	MaxImageBytes  = 20 << 20 // 20 MiB decoded, per image
	MaxImagesBytes = 64 << 20 // 64 MiB decoded, all images of one response
)

// imageWarnFn is the WARN hook for skipped images; tests swap it.
var imageWarnFn = logger.WarnCF

// ImageCollector decodes the images of one response, applying the size caps.
// model names the model in the WARN logged for each image it leaves out.
type ImageCollector struct {
	model  string
	total  int
	images []protocoltypes.GeneratedImage
}

// NewImageCollector returns an empty collector for a response from model.
func NewImageCollector(model string) *ImageCollector {
	return &ImageCollector{model: model}
}

// Images returns the images collected so far, nil when there are none.
func (c *ImageCollector) Images() []protocoltypes.GeneratedImage {
	return c.images
}

// AddURL adds the image a URL carries. Only base64 data URLs are decoded; any
// other URL (http, https, …) is skipped with a WARN and never fetched, so the
// library makes no request the caller did not make.
func (c *ImageCollector) AddURL(url string) {
	rest, ok := strings.CutPrefix(url, "data:")
	if !ok {
		c.skip("image given as a remote URL, not fetched", map[string]any{"url": previewURL(url)})
		return
	}
	meta, payload, ok := strings.Cut(rest, ",")
	mediaType, params, _ := strings.Cut(meta, ";")
	if !ok || !hasBase64Param(params) {
		c.skip("image data URL is not base64", nil)
		return
	}
	c.AddBase64(payload, strings.TrimSpace(mediaType))
}

// AddBase64 adds a base64-encoded image. contentType is the declared media
// type; when empty it is sniffed from the decoded bytes.
func (c *ImageCollector) AddBase64(b64, contentType string) {
	b64 = strings.TrimSpace(b64)
	// DecodedLen is an upper bound (padding included); checking it first
	// refuses a clearly oversize image without allocating for it.
	if base64.StdEncoding.DecodedLen(len(b64))-2 > MaxImageBytes {
		c.skip("image over the size cap", map[string]any{"limit_bytes": MaxImageBytes})
		return
	}
	data, err := base64.StdEncoding.DecodeString(b64)
	if err != nil {
		c.skip("image base64 does not decode", map[string]any{"error": err.Error()})
		return
	}
	if len(data) == 0 {
		c.skip("image is empty", nil)
		return
	}
	if len(data) > MaxImageBytes {
		c.skip("image over the size cap", map[string]any{"bytes": len(data), "limit_bytes": MaxImageBytes})
		return
	}
	if c.total+len(data) > MaxImagesBytes {
		c.skip("images over the per-response size cap", map[string]any{"bytes": len(data), "limit_bytes": MaxImagesBytes})
		return
	}
	if contentType == "" {
		contentType = http.DetectContentType(data)
	}
	c.total += len(data)
	c.images = append(c.images, protocoltypes.GeneratedImage{ContentType: contentType, Data: data})
}

func (c *ImageCollector) skip(reason string, fields map[string]any) {
	f := map[string]any{"model": c.model, "reason": reason}
	maps.Copy(f, fields)
	imageWarnFn("common", "Generated image skipped", f)
}

// hasBase64Param reports whether a data URL's parameters include base64.
func hasBase64Param(params string) bool {
	for p := range strings.SplitSeq(params, ";") {
		if strings.EqualFold(strings.TrimSpace(p), "base64") {
			return true
		}
	}
	return false
}

// previewURL keeps a logged URL short: signed URLs can be long and carry
// credentials in the query, so only the part before it is kept.
func previewURL(url string) string {
	if i := strings.IndexAny(url, "?#"); i >= 0 {
		url = url[:i]
	}
	if len(url) > 200 {
		url = url[:200]
	}
	return url
}
