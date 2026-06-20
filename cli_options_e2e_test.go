package spawnllm

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/PivotLLM/spawnllm/openai_compat"
)

// stdinCaptureScript builds a mock CLI script that writes its stdin to a file
// and emits the supplied JSON envelope. Returns (scriptPath, stdinFile).
func stdinCaptureScript(t *testing.T, envelopeJSON string) (string, string) {
	t.Helper()
	if runtime.GOOS == "windows" {
		t.Skip("mock CLI scripts not supported on Windows")
	}
	dir := t.TempDir()
	stdinFile := filepath.Join(dir, "stdin.txt")
	script := filepath.Join(dir, "cli")
	body := fmt.Sprintf(`#!/bin/sh
cat - > '%s'
cat <<'EOFMOCK'
%s
EOFMOCK
`, stdinFile, envelopeJSON)
	if err := os.WriteFile(script, []byte(body), 0o755); err != nil {
		t.Fatal(err)
	}
	return script, stdinFile
}

func readStdin(t *testing.T, path string) string {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read stdin file: %v", err)
	}
	return string(b)
}

// --- Claude CLI ---

func TestClaudeCLI_AppendsFortification_WhenJSONObjectOptionSet(t *testing.T) {
	script, stdinFile := stdinCaptureScript(t,
		`{"type":"result","result":"{}","session_id":"t"}`)

	p := NewClaudeCliProvider("", t.TempDir(), nil, nil)
	p.command = script

	_, err := p.Chat(context.Background(),
		[]Message{{Role: "user", Content: "summarise"}},
		nil, "",
		map[string]any{openai_compat.ResponseFormatJSONObjectOption: true},
	)
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	got := readStdin(t, stdinFile)
	if !strings.Contains(got, JSONObjectFortification) {
		t.Fatalf("claude-cli stdin missing fortification.\n--- got ---\n%s", got)
	}
}

func TestClaudeCLI_NoFortification_WhenOptionUnset(t *testing.T) {
	script, stdinFile := stdinCaptureScript(t,
		`{"type":"result","result":"{}","session_id":"t"}`)

	p := NewClaudeCliProvider("", t.TempDir(), nil, nil)
	p.command = script

	_, err := p.Chat(context.Background(),
		[]Message{{Role: "user", Content: "summarise"}},
		nil, "", nil,
	)
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	got := readStdin(t, stdinFile)
	if strings.Contains(got, "single JSON object") {
		t.Fatalf("claude-cli stdin must not contain fortification when option unset.\n--- got ---\n%s", got)
	}
}

// --- Codex CLI ---

func TestCodexCLI_AppendsFortification_WhenJSONObjectOptionSet(t *testing.T) {
	script, stdinFile := stdinCaptureScript(t,
		`{"type":"item.completed","item":{"type":"agent_message","text":"{}"}}
{"type":"turn.completed","usage":{"input_tokens":1,"output_tokens":1}}`)

	p := NewCodexCliProvider("", t.TempDir(), nil, nil)
	p.command = script

	_, err := p.Chat(context.Background(),
		[]Message{{Role: "user", Content: "summarise"}},
		nil, "",
		map[string]any{openai_compat.ResponseFormatJSONObjectOption: true},
	)
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	got := readStdin(t, stdinFile)
	if !strings.Contains(got, JSONObjectFortification) {
		t.Fatalf("codex-cli stdin missing fortification.\n--- got ---\n%s", got)
	}
}

// --- Gemini CLI ---

func TestGeminiCLI_AppendsFortification_WhenJSONObjectOptionSet(t *testing.T) {
	script, stdinFile := stdinCaptureScript(t,
		`{"session_id":"t","response":"{}","stats":{"models":{}}}`)

	p := NewGeminiCliProvider("", t.TempDir(), nil, nil)
	p.command = script

	_, err := p.Chat(context.Background(),
		[]Message{{Role: "user", Content: "summarise"}},
		nil, "",
		map[string]any{openai_compat.ResponseFormatJSONObjectOption: true},
	)
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	got := readStdin(t, stdinFile)
	if !strings.Contains(got, JSONObjectFortification) {
		t.Fatalf("gemini-cli stdin missing fortification.\n--- got ---\n%s", got)
	}
}
