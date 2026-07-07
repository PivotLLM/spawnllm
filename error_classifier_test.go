package spawnllm

import (
	"context"
	"errors"
	"fmt"
	"testing"

	"github.com/PivotLLM/spawnllm/common"
)

func TestClassifyError_Nil(t *testing.T) {
	result := ClassifyError(nil, "openai", "gpt-4")
	if result != nil {
		t.Errorf("expected nil for nil error, got %+v", result)
	}
}

func TestClassifyError_ContextCanceled(t *testing.T) {
	result := ClassifyError(context.Canceled, "openai", "gpt-4")
	if result != nil {
		t.Errorf("expected nil for context.Canceled (user abort), got %+v", result)
	}
}

func TestClassifyError_ContextDeadlineExceeded(t *testing.T) {
	result := ClassifyError(context.DeadlineExceeded, "openai", "gpt-4")
	if result == nil {
		t.Fatal("expected non-nil for deadline exceeded")
	}
	if result.Reason != FailoverTimeout {
		t.Errorf("reason = %q, want timeout", result.Reason)
	}
}

func TestClassifyError_StatusCodes(t *testing.T) {
	tests := []struct {
		status int
		reason FailoverReason
	}{
		{401, FailoverAuth},
		{403, FailoverAuth},
		{402, FailoverBilling},
		{408, FailoverTimeout},
		{429, FailoverRateLimit},
		{400, FailoverFormat},
		{500, FailoverTimeout},
		{502, FailoverTimeout},
		{503, FailoverTimeout},
		{521, FailoverTimeout},
		{522, FailoverTimeout},
		{523, FailoverTimeout},
		{524, FailoverTimeout},
		{529, FailoverTimeout},
	}

	for _, tt := range tests {
		err := fmt.Errorf("API error: status: %d something went wrong", tt.status)
		result := ClassifyError(err, "test", "model")
		if result == nil {
			t.Errorf("status %d: expected non-nil", tt.status)
			continue
		}
		if result.Reason != tt.reason {
			t.Errorf("status %d: reason = %q, want %q", tt.status, result.Reason, tt.reason)
		}
	}
}

func TestClassifyError_LimitExceededIsRateLimitNotAuth(t *testing.T) {
	// The live OpenRouter per-key daily cap: HTTP 403 whose body says "Key limit
	// exceeded". Must classify as rate_limit, not auth, on the structured path
	// (what ClawEh's HTTP providers hit).
	structured := &common.HTTPStatusError{
		StatusCode:  403,
		BodyPreview: `{"error":{"message":"Key limit exceeded (daily limit). Manage it using https://openrouter.ai/...","code":403}}`,
	}
	if got := ClassifyError(structured, "openrouter", "gpt-5.5"); got == nil || got.Reason != FailoverRateLimit {
		t.Fatalf("structured 403 key-limit: reason = %v, want rate_limit", reasonOf(got))
	}

	// Same, arriving as a plain message with a co-occurring 403.
	msg := errors.New(`status: 403 body: {"error":{"message":"Key limit exceeded (daily limit)","code":403}}`)
	if got := ClassifyError(msg, "openrouter", "gpt-5.5"); got == nil || got.Reason != FailoverRateLimit {
		t.Fatalf("message 403 key-limit: reason = %v, want rate_limit", reasonOf(got))
	}

	// Guard: a token/context "limit exceeded" must stay context_limit, not be
	// swallowed by the generic rate-limit rule.
	tok := errors.New("token limit exceeded for this model")
	if got := ClassifyError(tok, "openai", "gpt-4"); got == nil || got.Reason != FailoverContextLimit {
		t.Fatalf("token limit exceeded: reason = %v, want context_limit", reasonOf(got))
	}

	// A genuine 403 auth failure (no "limit exceeded") stays auth.
	auth := &common.HTTPStatusError{
		StatusCode:  403,
		BodyPreview: `{"error":{"message":"No auth credentials found","code":403}}`,
	}
	if got := ClassifyError(auth, "openrouter", "gpt-5.5"); got == nil || got.Reason != FailoverAuth {
		t.Fatalf("403 auth: reason = %v, want auth", reasonOf(got))
	}
}

func reasonOf(fe *FailoverError) FailoverReason {
	if fe == nil {
		return "<nil>"
	}
	return fe.Reason
}

func TestClassifyError_RateLimitPatterns(t *testing.T) {
	patterns := []string{
		"rate limit exceeded",
		"rate_limit reached",
		"too many requests",
		"exceeded your current quota",
		"resource has been exhausted",
		"resource_exhausted",
		"quota exceeded",
		"usage limit reached",
	}

	for _, msg := range patterns {
		err := errors.New(msg)
		result := ClassifyError(err, "openai", "gpt-4")
		if result == nil {
			t.Errorf("pattern %q: expected non-nil", msg)
			continue
		}
		if result.Reason != FailoverRateLimit {
			t.Errorf("pattern %q: reason = %q, want rate_limit", msg, result.Reason)
		}
	}
}

func TestClassifyError_OverloadedPatterns(t *testing.T) {
	patterns := []string{
		"overloaded_error",
		`{"type": "overloaded_error"}`,
		"server is overloaded",
	}

	for _, msg := range patterns {
		err := errors.New(msg)
		result := ClassifyError(err, "anthropic", "claude")
		if result == nil {
			t.Errorf("pattern %q: expected non-nil", msg)
			continue
		}
		// Overloaded is treated as rate_limit
		if result.Reason != FailoverRateLimit {
			t.Errorf("pattern %q: reason = %q, want rate_limit", msg, result.Reason)
		}
	}
}

func TestClassifyError_BillingPatterns(t *testing.T) {
	patterns := []string{
		"payment required",
		"insufficient credits",
		"credit balance too low",
		"plans & billing page",
		"insufficient balance",
	}

	for _, msg := range patterns {
		err := errors.New(msg)
		result := ClassifyError(err, "openai", "gpt-4")
		if result == nil {
			t.Errorf("pattern %q: expected non-nil", msg)
			continue
		}
		if result.Reason != FailoverBilling {
			t.Errorf("pattern %q: reason = %q, want billing", msg, result.Reason)
		}
	}
}

func TestClassifyError_TimeoutPatterns(t *testing.T) {
	patterns := []string{
		"request timeout",
		"connection timed out",
		"deadline exceeded",
		"context deadline exceeded",
	}

	for _, msg := range patterns {
		err := errors.New(msg)
		result := ClassifyError(err, "openai", "gpt-4")
		if result == nil {
			t.Errorf("pattern %q: expected non-nil", msg)
			continue
		}
		if result.Reason != FailoverTimeout {
			t.Errorf("pattern %q: reason = %q, want timeout", msg, result.Reason)
		}
	}
}

func TestClassifyError_AuthPatterns(t *testing.T) {
	patterns := []string{
		"invalid api key",
		"invalid_api_key",
		"incorrect api key",
		"invalid token",
		"authentication failed",
		"re-authenticate",
		"oauth token refresh failed",
		"unauthorized access",
		"forbidden",
		"access denied",
		"expired",
		"token has expired",
		"no credentials found",
		"no api key found",
	}

	for _, msg := range patterns {
		err := errors.New(msg)
		result := ClassifyError(err, "openai", "gpt-4")
		if result == nil {
			t.Errorf("pattern %q: expected non-nil", msg)
			continue
		}
		if result.Reason != FailoverAuth {
			t.Errorf("pattern %q: reason = %q, want auth", msg, result.Reason)
		}
	}
}

func TestClassifyError_FormatPatterns(t *testing.T) {
	patterns := []string{
		"string should match pattern",
		"tool_use.id is required",
		"invalid tool_use_id",
		"messages.1.content.1.tool_use.id must be valid",
		"invalid request format",
	}

	for _, msg := range patterns {
		err := errors.New(msg)
		result := ClassifyError(err, "anthropic", "claude")
		if result == nil {
			t.Errorf("pattern %q: expected non-nil", msg)
			continue
		}
		if result.Reason != FailoverFormat {
			t.Errorf("pattern %q: reason = %q, want format", msg, result.Reason)
		}
	}
}

func TestClassifyError_ImageDimensionError(t *testing.T) {
	err := errors.New("image dimensions exceed max allowed 2048x2048")
	result := ClassifyError(err, "openai", "gpt-4o")
	if result == nil {
		t.Fatal("expected non-nil for image dimension error")
	}
	if result.Reason != FailoverFormat {
		t.Errorf("reason = %q, want format", result.Reason)
	}
	if result.IsRetriable() {
		t.Error("image dimension error should not be retriable")
	}
}

func TestClassifyError_ImageSizeError(t *testing.T) {
	err := errors.New("image exceeds 20 mb limit")
	result := ClassifyError(err, "openai", "gpt-4o")
	if result == nil {
		t.Fatal("expected non-nil for image size error")
	}
	if result.Reason != FailoverFormat {
		t.Errorf("reason = %q, want format", result.Reason)
	}
}

func TestClassifyError_UnknownError(t *testing.T) {
	err := errors.New("some completely random error")
	result := ClassifyError(err, "openai", "gpt-4")
	if result != nil {
		t.Errorf("expected nil for unknown error, got %+v", result)
	}
}

func TestClassifyError_ProviderModelPropagation(t *testing.T) {
	err := errors.New("rate limit exceeded")
	result := ClassifyError(err, "my-provider", "my-model")
	if result == nil {
		t.Fatal("expected non-nil")
	}
	if result.Provider != "my-provider" {
		t.Errorf("provider = %q, want my-provider", result.Provider)
	}
	if result.Model != "my-model" {
		t.Errorf("model = %q, want my-model", result.Model)
	}
}

func TestFailoverError_IsRetriable(t *testing.T) {
	tests := []struct {
		reason    FailoverReason
		retriable bool
	}{
		{FailoverAuth, true},
		{FailoverRateLimit, true},
		{FailoverBilling, true},
		{FailoverTimeout, true},
		{FailoverOverloaded, true},
		{FailoverFormat, false},
		{FailoverUnknown, true},
	}

	for _, tt := range tests {
		fe := &FailoverError{Reason: tt.reason}
		if fe.IsRetriable() != tt.retriable {
			t.Errorf("IsRetriable(%q) = %v, want %v", tt.reason, fe.IsRetriable(), tt.retriable)
		}
	}
}

func TestFailoverError_ErrorString(t *testing.T) {
	fe := &FailoverError{
		Reason:   FailoverRateLimit,
		Provider: "openai",
		Model:    "gpt-4",
		Status:   429,
		Wrapped:  errors.New("too many requests"),
	}
	s := fe.Error()
	if s == "" {
		t.Error("expected non-empty error string")
	}
}

func TestFailoverError_Unwrap(t *testing.T) {
	inner := errors.New("inner error")
	fe := &FailoverError{Reason: FailoverTimeout, Wrapped: inner}
	if fe.Unwrap() != inner {
		t.Error("Unwrap should return wrapped error")
	}
}

func TestExtractHTTPStatus(t *testing.T) {
	tests := []struct {
		msg  string
		want int
	}{
		{"status: 429 rate limited", 429},
		{"status 401 unauthorized", 401},
		{"http/1.1 502 bad gateway", 502},
		{"error 429", 429},
		{"no status code here", 0},
		{"random number 12345", 0},
	}

	for _, tt := range tests {
		got := extractHTTPStatus(tt.msg)
		if got != tt.want {
			t.Errorf("extractHTTPStatus(%q) = %d, want %d", tt.msg, got, tt.want)
		}
	}
}

func TestIsImageDimensionError(t *testing.T) {
	if !IsImageDimensionError("image dimensions exceed max 4096x4096") {
		t.Error("should match image dimensions exceed max")
	}
	if IsImageDimensionError("normal error message") {
		t.Error("should not match normal error")
	}
}

func TestIsImageSizeError(t *testing.T) {
	if !IsImageSizeError("image exceeds 20 mb") {
		t.Error("should match image exceeds mb")
	}
	if IsImageSizeError("normal error message") {
		t.Error("should not match normal error")
	}
}

// TestClassifyError_BillingBodyMarkers verifies that structured billing markers
// (insufficient_quota, credits_exhausted, billing_url) classify as billing even
// when the HTTP status would otherwise resolve to rate_limit. Without this the
// classifier cooldown-cycles a permanently-broken model that OpenAI signals
// with HTTP 429 + body.error.code=insufficient_quota.
func TestClassifyError_BillingBodyMarkers(t *testing.T) {
	tests := []struct {
		name   string
		status int
		body   string
		want   FailoverReason
	}{
		{
			name:   "429 + insufficient_quota code → billing",
			status: 429,
			body:   `{"error":{"code":"insufficient_quota","message":"You exceeded your current quota"}}`,
			want:   FailoverBilling,
		},
		{
			name:   "429 + insufficient_quota type → billing",
			status: 429,
			body:   `{"error":{"type":"insufficient_quota"}}`,
			want:   FailoverBilling,
		},
		{
			name:   "402 + credits_exhausted → billing",
			status: 402,
			body:   `{"error":{"code":"credits_exhausted","message":"out of credits"}}`,
			want:   FailoverBilling,
		},
		{
			name:   "200 with billing_url somehow → billing",
			status: 200,
			body:   `{"error":{"message":"too low","billing_url":"https://example.com/billing"}}`,
			want:   FailoverBilling,
		},
		{
			name:   "429 plain → rate_limit",
			status: 429,
			body:   `{"error":{"message":"too many requests"}}`,
			want:   FailoverRateLimit,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := &common.HTTPStatusError{
				StatusCode:  tt.status,
				BodyPreview: tt.body,
			}
			result := ClassifyError(err, "openai", "gpt-4")
			if result == nil {
				t.Fatalf("expected non-nil failover error")
			}
			if result.Reason != tt.want {
				t.Errorf("reason = %q, want %q (body=%s)", result.Reason, tt.want, tt.body)
			}
		})
	}
}

func TestClassifyError_BillingBodyMarkers_MessagePath(t *testing.T) {
	// When the error is a plain string (not an HTTPStatusError), the body
	// markers must still be picked up via classifyByMessage.
	cases := []struct {
		name string
		msg  string
	}{
		{"insufficient_quota in body", `oops: {"code":"insufficient_quota"}`},
		{"credits_exhausted in body", `boom {"code":"credits_exhausted"}`},
		{"billing_url field", `{"billing_url":"https://x"}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			result := ClassifyError(errors.New(tc.msg), "openai", "gpt-4")
			if result == nil {
				t.Fatal("expected non-nil")
			}
			if result.Reason != FailoverBilling {
				t.Errorf("reason=%q, want billing", result.Reason)
			}
		})
	}
}
