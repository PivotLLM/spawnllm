package speech

import (
	"errors"
	"io"
	"mime"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
)

// redirectTransport sends every request to the test server, keeping its path
// and query, so a client built for a real host (openrouter.ai) reaches it.
type redirectTransport struct{ target *url.URL }

func (rt redirectTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	out := req.Clone(req.Context())
	out.URL.Scheme = rt.target.Scheme
	out.URL.Host = rt.target.Host
	out.Host = rt.target.Host
	return http.DefaultTransport.RoundTrip(out)
}

func redirectClient(t *testing.T, srv *httptest.Server) *http.Client {
	t.Helper()
	u, err := url.Parse(srv.URL)
	if err != nil {
		t.Fatalf("parse server URL: %v", err)
	}
	return &http.Client{Transport: redirectTransport{target: u}}
}

// formPart is one part of a parsed multipart request.
type formPart struct {
	filename    string
	contentType string
	value       string
}

// readForm parses a multipart request into its parts by field name, in order.
func readForm(t *testing.T, r *http.Request) map[string][]formPart {
	t.Helper()
	mediaType, params, err := mime.ParseMediaType(r.Header.Get("Content-Type"))
	if err != nil || mediaType != "multipart/form-data" {
		t.Fatalf("content type = %q, want multipart/form-data", r.Header.Get("Content-Type"))
	}
	mr := multipart.NewReader(r.Body, params["boundary"])
	out := map[string][]formPart{}
	for {
		p, err := mr.NextPart()
		if errors.Is(err, io.EOF) {
			return out
		}
		if err != nil {
			t.Fatalf("next part: %v", err)
		}
		data, err := io.ReadAll(p)
		if err != nil {
			t.Fatalf("read part: %v", err)
		}
		out[p.FormName()] = append(out[p.FormName()], formPart{
			filename: p.FileName(), contentType: p.Header.Get("Content-Type"), value: string(data),
		})
	}
}

// field returns the single value of a form field, failing when it is not
// there exactly once.
func field(t *testing.T, form map[string][]formPart, name string) formPart {
	t.Helper()
	parts := form[name]
	if len(parts) != 1 {
		t.Fatalf("field %q: %d parts, want 1", name, len(parts))
	}
	return parts[0]
}

func readBody(t *testing.T, r *http.Request) string {
	t.Helper()
	data, err := io.ReadAll(r.Body)
	if err != nil {
		t.Fatalf("read body: %v", err)
	}
	return string(data)
}

func writeJSON(t *testing.T, w http.ResponseWriter, status int, body string) {
	t.Helper()
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if _, err := io.WriteString(w, body); err != nil {
		t.Errorf("write response: %v", err)
	}
}

func audio() *strings.Reader { return strings.NewReader("OggS-fake-audio") }

func writeRaw(t *testing.T, w http.ResponseWriter, data []byte) {
	t.Helper()
	if _, err := w.Write(data); err != nil {
		t.Errorf("write response: %v", err)
	}
}
