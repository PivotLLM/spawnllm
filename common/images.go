package common

import (
	"encoding/base64"
	"net/http"
	"strings"

	"github.com/PivotLLM/spawnllm/logger"
	"github.com/PivotLLM/spawnllm/protocoltypes"
)

// imageWarnFn is the WARN hook for images passed through undecoded; tests
// swap it.
var imageWarnFn = logger.WarnCF

// ImageCollector gathers the images of one response, in order. It passes
// through every image the model returned, whatever its size: base64 data is
// decoded into Data, anything else is returned as URL, never fetched and never
// dropped. model names the model in the WARN logged for data that does not
// decode.
type ImageCollector struct {
	model  string
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

// AddURL adds the image a URL carries. A base64 data URL is decoded into
// Data, with the data URL's media type, else the one detected from the bytes.
// Any other URL (http, https, …) is returned as URL, unfetched, so the library
// makes no request the caller did not make. A data URL that cannot be decoded
// (not base64, malformed, empty) is returned as URL too, unchanged, with the
// media type it declares, so the model's output is never lost. An empty url
// carries no image and is ignored.
func (c *ImageCollector) AddURL(url string) {
	if url == "" {
		return
	}
	rest, ok := strings.CutPrefix(url, "data:")
	if !ok {
		c.images = append(c.images, protocoltypes.GeneratedImage{URL: url, Size: -1})
		return
	}
	meta, payload, ok := strings.Cut(rest, ",")
	mediaType, params, _ := strings.Cut(meta, ";")
	mediaType = strings.TrimSpace(mediaType)
	if ok && hasBase64Param(params) {
		if data, ok := decodeImage(payload); ok {
			c.addData(data, mediaType)
			return
		}
	}
	c.passThrough(url, mediaType)
}

// AddBase64 adds a base64-encoded image, its media type detected from the
// decoded bytes. Data that does not decode is returned as URL in the form
// "data:;base64,<b64>": the original text, unchanged, behind a fixed prefix.
// An empty b64 carries no image and is ignored.
func (c *ImageCollector) AddBase64(b64 string) {
	if b64 == "" {
		return
	}
	if data, ok := decodeImage(b64); ok {
		c.addData(data, "")
		return
	}
	c.passThrough("data:;base64,"+b64, "")
}

func (c *ImageCollector) addData(data []byte, contentType string) {
	if contentType == "" {
		contentType = http.DetectContentType(data)
	}
	c.images = append(c.images, protocoltypes.GeneratedImage{ContentType: contentType, Data: data, Size: int64(len(data))})
}

// passThrough returns image data that could not be decoded as a data URL,
// with a WARN, so the caller still receives what the model returned.
// contentType is the media type the data URL declares, if any.
func (c *ImageCollector) passThrough(dataURL, contentType string) {
	imageWarnFn("common", "Generated image data does not decode, returned as URL",
		map[string]any{"model": c.model, "length": len(dataURL)})
	c.images = append(c.images, protocoltypes.GeneratedImage{ContentType: contentType, URL: dataURL, Size: -1})
}

// decodeImage decodes standard base64, reporting false when it is malformed
// or decodes to nothing.
func decodeImage(b64 string) ([]byte, bool) {
	data, err := base64.StdEncoding.DecodeString(strings.TrimSpace(b64))
	if err != nil || len(data) == 0 {
		return nil, false
	}
	return data, true
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
