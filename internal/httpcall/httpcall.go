/******************************************************************************
 * Copyright (c) 2026 Tenebris Technologies Inc.                              *
 * Please see LICENSE file for details.                                       *
 ******************************************************************************/

// Package httpcall sends the single HTTP request of a task call (speech,
// image generation) and reports a failure the way the chat providers do, so
// the host's error classifier treats both alike: a non-2xx status is a
// *common.HTTPStatusError carrying the status, a body excerpt of at most 1 KiB
// and any Retry-After; an HTML page where JSON was expected is the chat
// providers' HTML error.
package httpcall

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime"
	"net/http"
	"strings"
	"time"

	"github.com/PivotLLM/spawnllm/common"
	"github.com/PivotLLM/spawnllm/logger"
)

// Call describes one request for logging and error reporting.
type Call struct {
	// Component is the logger component ("speech", "imagegen").
	Component string
	// Backend names the API in the log ("openai", "gemini", …).
	Backend string
	// Model is the model requested, for the log.
	Model string
	// APIBase is the base URL, named in an HTML error.
	APIBase string
	// JSON marks a response that must be JSON: an HTML body with a 2xx status
	// is then an error (a proxy or login page), as in the chat providers.
	JSON bool
	// Binary marks a response that must be binary data (audio): a 2xx whose
	// Content-Type is text/* or application/json, or whose body looks like
	// HTML, is then an error rather than data.
	Binary bool
}

// drainLimit is how much of an unread body is discarded before closing, so
// the connection can be reused.
const drainLimit = 64 << 10

// Response is a successful (2xx) response, its body read in full.
type Response struct {
	StatusCode int
	Header     http.Header
	Body       []byte
}

// NewPost builds a POST request bound to ctx, so the caller's deadline is
// the request's timeout.
func NewPost(ctx context.Context, url string, body []byte, contentType string) (*http.Request, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", contentType)
	return req, nil
}

// Do sends req with client and returns the response read in full. Transport
// errors are wrapped with %w, so a context deadline stays
// context.DeadlineExceeded for errors.Is. The body is not capped: spawnllm
// returns what the API returned and leaves size policy to the caller.
func Do(client *http.Client, req *http.Request, c Call) (*Response, error) {
	start := time.Now()
	logger.DebugCF(c.Component, "Sending request", map[string]any{
		"backend":    c.Backend,
		"model":      c.Model,
		"url":        req.URL.Scheme + "://" + req.URL.Host + req.URL.Path,
		"bytes_sent": req.ContentLength,
	})

	resp, err := client.Do(req) //nolint:gosec // G704: the URL is the host-configured API endpoint
	if err != nil {
		logger.DebugCF(c.Component, "Request failed", map[string]any{
			"backend": c.Backend, "model": c.Model, "error": err.Error(),
			"duration_ms": time.Since(start).Milliseconds(),
		})
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer func() {
		_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, drainLimit)) //nolint:errcheck // best effort
		if closeErr := resp.Body.Close(); closeErr != nil {
			logger.DebugCF(c.Component, "closing response body", map[string]any{"error": closeErr.Error()})
		}
	}()

	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		respErr := common.HandleErrorResponse(resp, c.APIBase)
		logger.DebugCF(c.Component, "Request failed", map[string]any{
			"backend": c.Backend, "model": c.Model, "status_code": resp.StatusCode,
			"duration_ms": time.Since(start).Milliseconds(),
		})
		return nil, respErr
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}
	contentType := resp.Header.Get("Content-Type")
	if (c.JSON || c.Binary) && common.LooksLikeHTML(body, contentType) {
		return nil, common.WrapHTMLResponseError(resp.StatusCode, body, contentType, c.APIBase)
	}
	if c.Binary && isTextType(contentType) {
		// An API that answers with a JSON or text message where audio was
		// expected has not returned audio; report it as a status error so
		// the body reaches the caller and the classifier.
		return nil, &common.HTTPStatusError{
			StatusCode:  resp.StatusCode,
			APIBase:     c.APIBase,
			BodyPreview: common.ResponsePreview(body, 1024),
		}
	}

	logger.DebugCF(c.Component, "Received response", map[string]any{
		"backend":        c.Backend,
		"model":          c.Model,
		"status_code":    resp.StatusCode,
		"bytes_received": len(body),
		"duration_ms":    time.Since(start).Milliseconds(),
	})
	return &Response{StatusCode: resp.StatusCode, Header: resp.Header, Body: body}, nil
}

// DecodeJSON decodes a response body into v. The error says "failed to
// decode response", which the error classifier treats as a transient upstream
// failure, as it does for the chat providers.
func DecodeJSON(body []byte, v any) error {
	if err := json.Unmarshal(body, v); err != nil {
		return fmt.Errorf("failed to decode response: %w", err)
	}
	return nil
}

// isTextType reports whether a Content-Type is text or JSON, not binary data.
func isTextType(contentType string) bool {
	mediaType, _, err := mime.ParseMediaType(contentType)
	if err != nil {
		return false
	}
	return strings.HasPrefix(mediaType, "text/") || mediaType == "application/json" ||
		strings.HasSuffix(mediaType, "+json")
}
