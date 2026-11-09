package httpcall

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/PivotLLM/spawnllm/common"
	"github.com/PivotLLM/spawnllm/logger"
)

type captureLog struct {
	mu    sync.Mutex
	lines []string
}

func (c *captureLog) Log(level, component, message string, fields map[string]any) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.lines = append(c.lines, level+" "+component+" "+message)
	for k, v := range fields {
		if s, ok := v.(string); ok && strings.Contains(s, "secret") {
			c.lines = append(c.lines, "LEAK "+k)
		}
	}
}

func (*captureLog) LogMessageContent() bool { return false }

func TestDo(t *testing.T) {
	logs := &captureLog{}
	logger.SetBackend(logs)
	defer logger.SetBackend(nil)

	tests := []struct {
		name       string
		status     int
		body       string
		expectJSON bool
		wantErr    bool
	}{
		{name: "200", status: http.StatusOK, body: "audio", wantErr: false},
		{name: "201 is success", status: http.StatusCreated, body: `{}`, expectJSON: true},
		{name: "302 is an error", status: http.StatusFound, body: "moved", wantErr: true},
		{name: "500 is an error", status: http.StatusInternalServerError, body: "boom", wantErr: true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				if tc.status == http.StatusFound {
					w.Header().Set("Location", "/elsewhere")
				}
				w.WriteHeader(tc.status)
				writeRaw(t, w, []byte(tc.body))
			}))
			defer srv.Close()

			req, err := NewPost(context.Background(), srv.URL+"/x?key=secret", []byte("in"), "text/plain")
			if err != nil {
				t.Fatalf("NewPost: %v", err)
			}
			client := &http.Client{CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
			resp, err := Do(client, req, Call{Component: "test", Backend: "b", Model: "m", APIBase: srv.URL, JSON: tc.expectJSON})
			if tc.wantErr {
				statusErr, ok := errors.AsType[*common.HTTPStatusError](err)
				if !ok || statusErr.StatusCode != tc.status {
					t.Fatalf("err = %v, want HTTPStatusError %d", err, tc.status)
				}
				return
			}
			if err != nil || string(resp.Body) != tc.body || resp.StatusCode != tc.status {
				t.Fatalf("resp = %+v, err = %v", resp, err)
			}
		})
	}

	logs.mu.Lock()
	defer logs.mu.Unlock()
	joined := strings.Join(logs.lines, "\n")
	if !strings.Contains(joined, "debug test Sending request") || !strings.Contains(joined, "debug test Received response") {
		t.Errorf("logs = %s", joined)
	}
	if strings.Contains(joined, "LEAK") {
		t.Errorf("the query string was logged: %s", joined)
	}
}

func TestDecodeJSON(t *testing.T) {
	var v struct{ A int }
	if err := DecodeJSON([]byte(`{"A":1}`), &v); err != nil || v.A != 1 {
		t.Fatalf("DecodeJSON = %v, %+v", err, v)
	}
	if err := DecodeJSON([]byte(`{`), &v); err == nil || !strings.Contains(err.Error(), "failed to decode response") {
		t.Fatalf("err = %v", err)
	}
}

func writeRaw(t *testing.T, w http.ResponseWriter, data []byte) {
	t.Helper()
	if _, err := w.Write(data); err != nil {
		t.Errorf("write response: %v", err)
	}
}
