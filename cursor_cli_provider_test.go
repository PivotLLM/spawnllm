package spawnllm

import (
	"context"
	"os"
	"strings"
	"testing"
)

// --- Compile-time interface check ---

var _ LLMProvider = (*CursorCliProvider)(nil)

func TestNewCursorCliProvider_DefaultCommand(t *testing.T) {
	p := NewCursorCliProvider("", "/ws", nil, nil)
	if p.command != "cursor-agent" {
		t.Errorf("command = %q, want cursor-agent", p.command)
	}
	if p.workspace != "/ws" {
		t.Errorf("workspace = %q, want /ws", p.workspace)
	}
	if got := NewCursorCliProvider("agent", "", nil, nil).command; got != "agent" {
		t.Errorf("explicit command = %q, want agent", got)
	}
}

func TestCursorCliProvider_GetDefaultModel(t *testing.T) {
	if got := NewCursorCliProvider("", "", nil, nil).GetDefaultModel(); got != "cursor-agent" {
		t.Errorf("GetDefaultModel = %q, want cursor-agent", got)
	}
}

// TestCursorCliProvider_Chat_ParsesResult feeds a real Cursor-format JSON
// envelope and checks the result text and camelCase usage mapping.
func TestCursorCliProvider_Chat_ParsesResult(t *testing.T) {
	const out = `{"type":"result","subtype":"success","is_error":false,"duration_ms":1710,"duration_api_ms":1710,"result":"ok","session_id":"s","request_id":"r","usage":{"inputTokens":4385,"outputTokens":18,"cacheReadTokens":8576,"cacheWriteTokens":2}}`
	script := createMockCLI(t, out, "", 0)
	p := NewCursorCliProvider(script, t.TempDir(), nil, nil)

	resp, err := p.Chat(context.Background(),
		[]Message{{Role: "user", Content: "respond with ok"}}, nil, "", nil)
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if resp.Content != "ok" {
		t.Errorf("Content = %q, want ok", resp.Content)
	}
	if resp.Usage == nil {
		t.Fatal("expected usage")
	}
	// PromptTokens = input + cacheRead + cacheWrite = 4385 + 8576 + 2.
	if resp.Usage.PromptTokens != 4385+8576+2 {
		t.Errorf("PromptTokens = %d, want %d", resp.Usage.PromptTokens, 4385+8576+2)
	}
	if resp.Usage.CompletionTokens != 18 {
		t.Errorf("CompletionTokens = %d, want 18", resp.Usage.CompletionTokens)
	}
	if resp.Status == nil || !resp.Status.Success {
		t.Error("expected a successful DispatchStatus")
	}
}

// TestCursorCliProvider_Chat_Error surfaces an is_error envelope as an error.
func TestCursorCliProvider_Chat_Error(t *testing.T) {
	const out = `{"type":"result","subtype":"error","is_error":true,"result":"boom"}`
	script := createMockCLI(t, out, "", 0)
	p := NewCursorCliProvider(script, t.TempDir(), nil, nil)

	_, err := p.Chat(context.Background(),
		[]Message{{Role: "user", Content: "x"}}, nil, "", nil)
	if err == nil {
		t.Fatal("expected an error for an is_error envelope")
	}
}

// TestCursorCliProvider_Chat_Args checks the base args, that a placeholder model
// adds no --model flag, and that a real model id does.
func TestCursorCliProvider_Chat_Args(t *testing.T) {
	argsFile := t.TempDir() + "/args.txt"
	script := createArgCaptureCLI(t, argsFile)

	readArgs := func(model string) string {
		p := NewCursorCliProvider(script, t.TempDir(), []string{"--yolo"}, nil)
		if _, err := p.Chat(context.Background(),
			[]Message{{Role: "user", Content: "hi"}}, nil, model, nil); err != nil {
			t.Fatalf("Chat: %v", err)
		}
		b, err := os.ReadFile(argsFile)
		if err != nil {
			t.Fatalf("read args: %v", err)
		}
		return string(b)
	}

	base := readArgs("")
	for _, want := range []string{"-p", "--output-format", "json", "--yolo"} {
		if !strings.Contains(base, want) {
			t.Errorf("base args %q missing %q", strings.TrimSpace(base), want)
		}
	}
	if strings.Contains(base, "--model") {
		t.Errorf("placeholder model should not add --model: %q", strings.TrimSpace(base))
	}

	withModel := readArgs("gpt-5")
	if !strings.Contains(withModel, "--model gpt-5") {
		t.Errorf("real model should add --model gpt-5: %q", strings.TrimSpace(withModel))
	}
}
