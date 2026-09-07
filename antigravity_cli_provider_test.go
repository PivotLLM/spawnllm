package spawnllm

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// argCaptureScript builds a mock CLI that records the arguments it was invoked
// with, then prints the given envelope. Used to assert on the command line
// rather than only on the parsed result.
func argCaptureScript(t *testing.T, envelopeJSON string) (script, argsFile string) {
	t.Helper()
	if runtime.GOOS == "windows" {
		t.Skip("mock CLI scripts not supported on Windows")
	}
	dir := t.TempDir()
	argsFile = filepath.Join(dir, "args.txt")
	script = filepath.Join(dir, "cli")
	body := fmt.Sprintf(`#!/bin/sh
echo "$@" > '%s'
cat - > /dev/null
cat <<'EOFMOCK'
%s
EOFMOCK
`, argsFile, envelopeJSON)
	if err := os.WriteFile(script, []byte(body), 0o755); err != nil {
		t.Fatal(err)
	}
	return script, argsFile
}

// --- Compile-time interface check ---

var _ LLMProvider = (*AntigravityCliProvider)(nil)

func TestNewAntigravityCliProvider_DefaultCommand(t *testing.T) {
	p := NewAntigravityCliProvider("", "/ws", nil, nil)
	if p.command != "agy" {
		t.Errorf("command = %q, want agy", p.command)
	}
	if p.workspace != "/ws" {
		t.Errorf("workspace = %q, want /ws", p.workspace)
	}
	if got := NewAntigravityCliProvider("/opt/agy", "", nil, nil).command; got != "/opt/agy" {
		t.Errorf("explicit command = %q, want /opt/agy", got)
	}
}

func TestAntigravityCliProvider_GetDefaultModel(t *testing.T) {
	if got := NewAntigravityCliProvider("", "", nil, nil).GetDefaultModel(); got != "antigravity-cli" {
		t.Errorf("GetDefaultModel = %q, want antigravity-cli", got)
	}
}

// The prompt goes on stdin, and -p / --print must NEVER be passed.
//
// agy accepts both flags, but with either one it expects the prompt as a
// command-line ARGUMENT and stops reading stdin — so the conversation would be
// silently dropped and the model would answer an empty question. This is the
// one invocation detail that fails quietly rather than loudly, so it is pinned.
func TestAntigravityCliProvider_NeverPassesPrintFlag(t *testing.T) {
	script, argsFile := argCaptureScript(t,
		`{"conversation_id":"c","status":"SUCCESS","response":"ok"}`)
	p := NewAntigravityCliProvider(script, t.TempDir(), []string{"--dangerously-skip-permissions"}, nil)

	if _, err := p.Chat(context.Background(),
		[]Message{{Role: "user", Content: "hi"}}, nil, "", nil); err != nil {
		t.Fatalf("Chat: %v", err)
	}

	raw, err := os.ReadFile(argsFile)
	if err != nil {
		t.Fatalf("read args: %v", err)
	}
	args := strings.Fields(string(raw))
	for _, a := range args {
		if a == "-p" || a == "--print" || a == "--prompt" {
			t.Fatalf("passed %q — agy then reads the prompt from argv and ignores stdin.\nargs: %v", a, args)
		}
	}
	joined := strings.Join(args, " ")
	for _, want := range []string{"--input-format text", "--output-format json", "--dangerously-skip-permissions"} {
		if !strings.Contains(joined, want) {
			t.Errorf("args missing %q: %v", want, args)
		}
	}
}

// The flat envelope, and the token accounting that differs from the other CLIs:
// cached reads are prompt tokens served from cache, and thinking tokens are
// billed output.
func TestAntigravityCliProvider_Chat_ParsesResponse(t *testing.T) {
	const out = `{"conversation_id":"b2d93f9e","status":"SUCCESS","response":"ok\n","duration_seconds":0.895,"num_turns":1,"usage":{"input_tokens":5356,"output_tokens":27,"thinking_tokens":26,"cache_read_tokens":8129,"total_tokens":5383}}`
	script := createMockCLI(t, out, "", 0)
	p := NewAntigravityCliProvider(script, t.TempDir(), nil, nil)

	resp, err := p.Chat(context.Background(),
		[]Message{{Role: "user", Content: "say ok"}}, nil, "", nil)
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if resp.Content != "ok" {
		t.Errorf("Content = %q, want %q (trailing newline trimmed)", resp.Content, "ok")
	}
	if resp.Usage == nil {
		t.Fatal("Usage is nil")
	}
	if want := 5356 + 8129; resp.Usage.PromptTokens != want {
		t.Errorf("PromptTokens = %d, want %d (input + cache_read)", resp.Usage.PromptTokens, want)
	}
	if want := 27 + 26; resp.Usage.CompletionTokens != want {
		t.Errorf("CompletionTokens = %d, want %d (output + thinking)", resp.Usage.CompletionTokens, want)
	}
	if resp.Usage.TotalTokens != 5383 {
		t.Errorf("TotalTokens = %d, want the reported 5383", resp.Usage.TotalTokens)
	}
	if resp.Status == nil || !resp.Status.Success {
		t.Errorf("Status = %+v, want success", resp.Status)
	}
	if resp.Status.NumTurns != 1 {
		t.Errorf("NumTurns = %d, want 1", resp.Status.NumTurns)
	}
	if resp.Status.DurationMs != 895 {
		t.Errorf("DurationMs = %d, want 895 (from duration_seconds)", resp.Status.DurationMs)
	}
}

// Success is carried by status, not an is_error flag. Anything that is not
// SUCCESS is a failure — including a status this build has never seen, which
// must surface rather than be reported as a completed turn.
func TestAntigravityCliProvider_NonSuccessStatusIsAnError(t *testing.T) {
	for _, status := range []string{"ERROR", "CANCELLED", "SOMETHING_NEW"} {
		out := `{"conversation_id":"c","status":"` + status + `","response":"it went wrong"}`
		script := createMockCLI(t, out, "", 0)
		p := NewAntigravityCliProvider(script, t.TempDir(), nil, nil)

		resp, err := p.Chat(context.Background(),
			[]Message{{Role: "user", Content: "hi"}}, nil, "", nil)
		if err == nil {
			t.Errorf("status %q: no error returned", status)
			continue
		}
		if !strings.Contains(err.Error(), status) {
			t.Errorf("status %q: error does not name the status: %v", status, err)
		}
		if resp != nil && resp.Status != nil && resp.Status.Success {
			t.Errorf("status %q: DispatchStatus reports success", status)
		}
	}
}

// A model name is forwarded, but the protocol's own placeholders are not — they
// are ClawEh's identifier for the provider, not something agy would recognise.
func TestAntigravityCliProvider_ModelFlag(t *testing.T) {
	for _, tc := range []struct {
		model string
		want  bool
	}{
		{"gemini-3-pro", true},
		{"antigravity-cli", false},
		{"agy", false},
		{"", false},
	} {
		script, argsFile := argCaptureScript(t,
			`{"conversation_id":"c","status":"SUCCESS","response":"ok"}`)
		p := NewAntigravityCliProvider(script, t.TempDir(), nil, nil)
		if _, err := p.Chat(context.Background(),
			[]Message{{Role: "user", Content: "hi"}}, nil, tc.model, nil); err != nil {
			t.Fatalf("model %q: Chat: %v", tc.model, err)
		}
		raw, _ := os.ReadFile(argsFile)
		got := strings.Contains(string(raw), "--model")
		if got != tc.want {
			t.Errorf("model %q: --model passed = %v, want %v (args: %s)", tc.model, got, tc.want, raw)
		}
	}
}
