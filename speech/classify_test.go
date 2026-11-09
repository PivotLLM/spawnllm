package speech_test

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/PivotLLM/spawnllm"
	"github.com/PivotLLM/spawnllm/speech"
)

// TestErrorsClassify checks that speech errors reach the host's fallback
// classifier the way chat-provider errors do.
func TestErrorsClassify(t *testing.T) {
	tests := []struct {
		status int
		body   string
		want   spawnllm.FailoverReason
	}{
		{http.StatusUnauthorized, `{"error":{"message":"bad key"}}`, spawnllm.FailoverAuth},
		{http.StatusTooManyRequests, `{"error":{"message":"slow down"}}`, spawnllm.FailoverRateLimit},
		{http.StatusTooManyRequests, `{"error":{"code":"insufficient_quota"}}`, spawnllm.FailoverBilling},
		{http.StatusServiceUnavailable, `upstream down`, spawnllm.FailoverTimeout},
	}
	for _, tc := range tests {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(tc.status)
			if _, err := io.WriteString(w, tc.body); err != nil {
				t.Errorf("write response: %v", err)
			}
		}))
		_, err := speech.NewOpenAI(srv.URL, "k").Transcribe(context.Background(),
			&speech.TranscribeRequest{Model: "whisper-1", Audio: strings.NewReader("a")})
		srv.Close()
		fe := spawnllm.ClassifyError(err, "openai", "whisper-1")
		if fe == nil || fe.Reason != tc.want || fe.Status != tc.status {
			t.Errorf("status %d: classified %+v, want %s", tc.status, fe, tc.want)
		}
	}
}
