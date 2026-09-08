package spawnllm

import (
	"context"
	"os"
	"strings"
	"testing"

	"github.com/PivotLLM/spawnllm/openai_compat"
)

func TestApplyCLIOptions_AppendsFortificationWhenJSONObjectRequested(t *testing.T) {
	base := "User: hi"
	out := applyCLIOptions("claude-cli", base,
		map[string]any{openai_compat.ResponseFormatJSONObjectOption: true})
	if !strings.HasPrefix(out, base) {
		t.Fatalf("output must preserve original prompt; got %q", out)
	}
	if !strings.HasSuffix(out, JSONObjectFortification) {
		t.Fatalf("output must end with JSONObjectFortification; got tail %q", out[len(out)-min(len(out), 80):])
	}
}

func TestApplyCLIOptions_NoChangeWhenOptionUnset(t *testing.T) {
	base := "User: hi"
	out := applyCLIOptions("claude-cli", base, nil)
	if out != base {
		t.Fatalf("prompt mutated when no options supplied; got %q want %q", out, base)
	}
	out = applyCLIOptions("claude-cli", base, map[string]any{})
	if out != base {
		t.Fatalf("prompt mutated when options empty; got %q want %q", out, base)
	}
}

func TestApplyCLIOptions_NoChangeWhenOptionFalse(t *testing.T) {
	base := "User: hi"
	out := applyCLIOptions("claude-cli", base,
		map[string]any{openai_compat.ResponseFormatJSONObjectOption: false})
	if out != base {
		t.Fatalf("prompt mutated when option false; got %q want %q", out, base)
	}
}

func TestApplyCLIOptions_AcceptsStringTrue(t *testing.T) {
	base := "User: hi"
	out := applyCLIOptions("claude-cli", base,
		map[string]any{openai_compat.ResponseFormatJSONObjectOption: "true"})
	if !strings.HasSuffix(out, JSONObjectFortification) {
		t.Fatalf("string \"true\" must trigger fortification; got tail %q", out[len(out)-min(len(out), 80):])
	}
}

func TestApplyCLIOptions_DoesNotDropUnknownSilently(t *testing.T) {
	// This test exercises the iteration path that produces DBG logs for
	// every unrecognised option. We can't intercept the logger trivially,
	// but we can confirm the function still returns the (fortified) prompt
	// and does not panic when extra keys are present — that's the contract
	// guaranteed by the regression we're guarding against.
	base := "User: hi"
	out := applyCLIOptions("claude-cli", base, map[string]any{
		openai_compat.ResponseFormatJSONObjectOption: true,
		"temperature": 0.2,
		"max_tokens":  1024,
		"unknown_x":   "y",
	})
	if !strings.HasSuffix(out, JSONObjectFortification) {
		t.Fatalf("fortification missing when option set alongside others; got tail %q",
			out[len(out)-min(len(out), 80):])
	}
}

// The exported base arguments must be what the providers actually run with, or
// the operator is shown a command line that is not the one being executed —
// which is the whole reason they are exported.
func TestCLIBaseArgs_MatchTheInvocation(t *testing.T) {
	tests := []struct {
		name     string
		base     []string
		envelope string
		build    func(command string) LLMProvider
	}{
		{
			name:     "claude",
			base:     ClaudeCliBaseArgs(),
			envelope: `{"type":"result","subtype":"success","is_error":false,"result":"ok"}`,
			build:    func(c string) LLMProvider { return NewClaudeCliProvider(c, ".", nil, nil) },
		},
		{
			name:     "antigravity",
			base:     AntigravityCliBaseArgs(),
			envelope: `{"conversation_id":"x","status":"SUCCESS","response":"ok"}`,
			build:    func(c string) LLMProvider { return NewAntigravityCliProvider(c, ".", nil, nil) },
		},
		{
			name:     "cursor",
			base:     CursorCliBaseArgs(),
			envelope: `{"type":"result","subtype":"success","is_error":false,"result":"ok"}`,
			build:    func(c string) LLMProvider { return NewCursorCliProvider(c, ".", nil, nil) },
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			script, argsFile := argCaptureScript(t, tc.envelope)
			p := tc.build(script)
			if _, err := p.Chat(context.Background(),
				[]Message{{Role: "user", Content: "hi"}}, nil, "", nil); err != nil {
				t.Fatalf("Chat: %v", err)
			}
			got, err := os.ReadFile(argsFile)
			if err != nil {
				t.Fatal(err)
			}
			line := strings.TrimSpace(string(got))
			prefix := strings.Join(tc.base, " ")
			if !strings.HasPrefix(line, prefix) {
				t.Errorf("invocation %q does not start with the exported base args %q", line, prefix)
			}
		})
	}
}
