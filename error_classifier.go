package spawnllm

import (
	"context"
	"errors"
	"regexp"
	"strings"

	"github.com/PivotLLM/spawnllm/common"
)

// Common patterns in Go HTTP error messages
var httpStatusPatterns = []*regexp.Regexp{
	regexp.MustCompile(`status[:\s]+(\d{3})`),
	regexp.MustCompile(`http[/\s]+\d*\.?\d*\s+(\d{3})`),
	regexp.MustCompile(`\b([3-5]\d{2})\b`),
}

// errorPattern defines a single pattern (string or regex) for error classification.
type errorPattern struct {
	substring string
	regex     *regexp.Regexp
}

func substr(s string) errorPattern { return errorPattern{substring: s} }
func rxp(r string) errorPattern    { return errorPattern{regex: regexp.MustCompile("(?i)" + r)} }

// Error patterns organized by FailoverReason, matching OpenClaw production (~40 patterns).
var (
	rateLimitPatterns = []errorPattern{
		rxp(`rate[_ ]limit`),
		substr("too many requests"),
		substr("429"),
		substr("exceeded your current quota"),
		rxp(`exceeded.*quota`),
		rxp(`resource has been exhausted`),
		rxp(`resource.*exhausted`),
		substr("resource_exhausted"),
		substr("quota exceeded"),
		substr("usage limit"),
	}

	overloadedPatterns = []errorPattern{
		rxp(`overloaded_error`),
		rxp(`"type"\s*:\s*"overloaded_error"`),
		substr("overloaded"),
	}

	timeoutPatterns = []errorPattern{
		substr("timeout"),
		substr("timed out"),
		substr("deadline exceeded"),
		substr("context deadline exceeded"),
	}

	billingPatterns = []errorPattern{
		rxp(`\b402\b`),
		substr("payment required"),
		substr("insufficient credits"),
		substr("credit balance"),
		substr("plans & billing"),
		substr("insufficient balance"),
		substr("out of credits"),
		substr("account balance is too low"),
	}

	// billingBodyPatterns are structured markers that a provider returns to
	// signal credits-exhausted regardless of HTTP status. OpenAI returns
	// insufficient_quota with a 429 status; treating that as rate_limit causes
	// cooldown cycling against a permanently-broken model. These markers must
	// match BEFORE rateLimitPatterns and BEFORE status-only classification.
	billingBodyPatterns = []errorPattern{
		rxp(`"code"\s*:\s*"insufficient_quota"`),
		rxp(`"type"\s*:\s*"insufficient_quota"`),
		rxp(`"code"\s*:\s*"credits_exhausted"`),
		rxp(`"billing_url"\s*:`),
	}

	authPatterns = []errorPattern{
		rxp(`invalid[_ ]?api[_ ]?key`),
		substr("incorrect api key"),
		substr("invalid token"),
		substr("authentication"),
		substr("re-authenticate"),
		substr("oauth token refresh failed"),
		substr("unauthorized"),
		substr("forbidden"),
		substr("access denied"),
		substr("expired"),
		substr("token has expired"),
		rxp(`\b401\b`),
		rxp(`\b403\b`),
		substr("no credentials found"),
		substr("no api key found"),
	}

	formatPatterns = []errorPattern{
		substr("string should match pattern"),
		substr("tool_use.id"),
		substr("tool_use_id"),
		substr("messages.1.content.1.tool_use.id"),
		substr("invalid request format"),
	}

	// Parse-error patterns: a malformed upstream response should not stop the
	// fallback chain. Classifying them as a transient (timeout-like) failure
	// makes the next configured candidate take over instead of bubbling a raw
	// JSON decode error to the agent.
	parseErrorPatterns = []errorPattern{
		substr("failed to parse json response"),
		substr("failed to decode response"),
		substr("failed to inspect response"),
	}

	contextLimitPatterns = []errorPattern{
		substr("request too large"),
		substr("payload too large"),
		substr("prompt is too long"),
		substr("context_length_exceeded"),
		rxp(`\btoo many tokens\b`),
		rxp(`\btoken.*limit.*exceeded\b`),
		rxp(`\bmaximum context length\b`),
	}

	imageDimensionPatterns = []errorPattern{
		rxp(`image dimensions exceed max`),
	}

	imageSizePatterns = []errorPattern{
		rxp(`image exceeds.*mb`),
	}

	// Transient HTTP status codes that map to timeout (server-side failures).
	// Includes the Cloudflare 52x family plus OpenRouter's commonly-seen
	// 504/520/525-528 transient set so a brief upstream blip falls through
	// to the configured fallbacks instead of stopping the chain.
	transientStatusCodes = map[int]bool{
		500: true, 502: true, 503: true, 504: true,
		520: true, 521: true, 522: true, 523: true, 524: true,
		525: true, 526: true, 527: true, 528: true, 529: true,
	}
)

// ClassifyError classifies an error into a FailoverError with reason.
// Returns nil if the error is not classifiable (unknown errors should not trigger fallback).
func ClassifyError(err error, provider, model string) *FailoverError {
	if err == nil {
		return nil
	}

	// Context cancellation: user abort, never fallback.
	if err == context.Canceled {
		return nil
	}

	// Context deadline exceeded: treat as timeout, always fallback.
	if errors.Is(err, context.DeadlineExceeded) {
		return &FailoverError{
			Reason:   FailoverTimeout,
			Provider: provider,
			Model:    model,
			Wrapped:  err,
		}
	}

	// Structured HTTP status error from common.HandleErrorResponse. Carries
	// the Retry-After hint so the fallback chain can size its cooldown to
	// the server's suggestion instead of the default exponential backoff.
	var statusErr *common.HTTPStatusError
	if errors.As(err, &statusErr) {
		// Structured billing markers in the body override status-based
		// classification. OpenAI returns insufficient_quota with HTTP 429
		// — without this check, the request would be treated as a transient
		// rate limit and cooldown-cycled against a model that is permanently
		// out of credits.
		if matchesAny(strings.ToLower(statusErr.BodyPreview), billingBodyPatterns) {
			return &FailoverError{
				Reason:     FailoverBilling,
				Provider:   provider,
				Model:      model,
				Status:     statusErr.StatusCode,
				RetryAfter: statusErr.RetryAfter,
				Wrapped:    err,
			}
		}
		if reason := classifyByStatus(statusErr.StatusCode); reason != "" {
			return &FailoverError{
				Reason:     reason,
				Provider:   provider,
				Model:      model,
				Status:     statusErr.StatusCode,
				RetryAfter: statusErr.RetryAfter,
				Wrapped:    err,
			}
		}
	}

	msg := strings.ToLower(err.Error())

	// Image dimension/size errors: non-retriable, non-fallback.
	if IsImageDimensionError(msg) || IsImageSizeError(msg) {
		return &FailoverError{
			Reason:   FailoverFormat,
			Provider: provider,
			Model:    model,
			Wrapped:  err,
		}
	}

	// Try HTTP status code extraction first.
	if status := extractHTTPStatus(msg); status > 0 {
		if reason := classifyByStatus(status); reason != "" {
			return &FailoverError{
				Reason:   reason,
				Provider: provider,
				Model:    model,
				Status:   status,
				Wrapped:  err,
			}
		}
	}

	// Message pattern matching (priority order from OpenClaw).
	if reason := classifyByMessage(msg); reason != "" {
		return &FailoverError{
			Reason:   reason,
			Provider: provider,
			Model:    model,
			Wrapped:  err,
		}
	}

	return nil
}

// classifyByStatus maps HTTP status codes to FailoverReason.
func classifyByStatus(status int) FailoverReason {
	switch {
	case status == 401 || status == 403:
		return FailoverAuth
	case status == 402:
		return FailoverBilling
	case status == 408:
		return FailoverTimeout
	case status == 429:
		return FailoverRateLimit
	case status == 413:
		return FailoverContextLimit
	case status == 400:
		return FailoverFormat
	case transientStatusCodes[status]:
		return FailoverTimeout
	}
	return ""
}

// classifyByMessage matches error messages against patterns.
// Priority order matters (from OpenClaw classifyFailoverReason).
func classifyByMessage(msg string) FailoverReason {
	// Structured billing markers run first: "insufficient_quota" appears with
	// a 429 status from OpenAI and otherwise gets eaten by rateLimitPatterns
	// (the substring "exceeded your current quota" travels with it).
	if matchesAny(msg, billingBodyPatterns) {
		return FailoverBilling
	}
	if matchesAny(msg, rateLimitPatterns) {
		return FailoverRateLimit
	}
	if matchesAny(msg, overloadedPatterns) {
		return FailoverRateLimit // Overloaded treated as rate_limit
	}
	if matchesAny(msg, billingPatterns) {
		return FailoverBilling
	}
	if matchesAny(msg, contextLimitPatterns) {
		return FailoverContextLimit
	}
	if matchesAny(msg, timeoutPatterns) {
		return FailoverTimeout
	}
	if matchesAny(msg, authPatterns) {
		return FailoverAuth
	}
	if matchesAny(msg, parseErrorPatterns) {
		// A malformed upstream response is transient from the caller's
		// perspective — treat as timeout so the configured fallback chain
		// gets the next candidate instead of surfacing a parse error.
		return FailoverTimeout
	}
	if matchesAny(msg, formatPatterns) {
		return FailoverFormat
	}
	return ""
}

// extractHTTPStatus extracts an HTTP status code from an error message.
// Looks for patterns like "status: 429", "status 429", "http/1.1 429", "http 429", or standalone "429".
func extractHTTPStatus(msg string) int {
	for _, p := range httpStatusPatterns {
		if m := p.FindStringSubmatch(msg); len(m) > 1 {
			return parseDigits(m[1])
		}
	}
	return 0
}

// IsImageDimensionError returns true if the message indicates an image dimension error.
func IsImageDimensionError(msg string) bool {
	return matchesAny(msg, imageDimensionPatterns)
}

// IsImageSizeError returns true if the message indicates an image file size error.
func IsImageSizeError(msg string) bool {
	return matchesAny(msg, imageSizePatterns)
}

// matchesAny checks if msg matches any of the patterns.
func matchesAny(msg string, patterns []errorPattern) bool {
	for _, p := range patterns {
		if p.regex != nil {
			if p.regex.MatchString(msg) {
				return true
			}
		} else if p.substring != "" {
			if strings.Contains(msg, p.substring) {
				return true
			}
		}
	}
	return false
}

// parseDigits converts a string of digits to an int.
func parseDigits(s string) int {
	n := 0
	for _, c := range s {
		if c >= '0' && c <= '9' {
			n = n*10 + int(c-'0')
		}
	}
	return n
}
