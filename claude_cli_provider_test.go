package spawnllm

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

// --- Compile-time interface check ---

var _ LLMProvider = (*ClaudeCliProvider)(nil)

// --- Helper: create mock CLI scripts ---

// createMockCLI creates a temporary script that simulates the claude CLI.
// Uses files for stdout/stderr to avoid shell quoting issues with JSON.
func createMockCLI(t *testing.T, stdout, stderr string, exitCode int) string {
	t.Helper()
	if runtime.GOOS == "windows" {
		t.Skip("mock CLI scripts not supported on Windows")
	}

	dir := t.TempDir()

	if stdout != "" {
		if err := os.WriteFile(filepath.Join(dir, "stdout.txt"), []byte(stdout), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	if stderr != "" {
		if err := os.WriteFile(filepath.Join(dir, "stderr.txt"), []byte(stderr), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	var sb strings.Builder
	sb.WriteString("#!/bin/sh\n")
	if stderr != "" {
		sb.WriteString(fmt.Sprintf("cat '%s/stderr.txt' >&2\n", dir))
	}
	if stdout != "" {
		sb.WriteString(fmt.Sprintf("cat '%s/stdout.txt'\n", dir))
	}
	sb.WriteString(fmt.Sprintf("exit %d\n", exitCode))

	script := filepath.Join(dir, "claude")
	if err := os.WriteFile(script, []byte(sb.String()), 0o755); err != nil {
		t.Fatal(err)
	}
	return script
}

// createSlowMockCLI creates a script that sleeps before responding (for context cancellation tests).
func createSlowMockCLI(t *testing.T, sleepSeconds int) string {
	t.Helper()
	if runtime.GOOS == "windows" {
		t.Skip("mock CLI scripts not supported on Windows")
	}

	dir := t.TempDir()
	script := filepath.Join(dir, "claude")
	content := fmt.Sprintf("#!/bin/sh\nsleep %d\necho '{\"type\":\"result\",\"result\":\"late\"}'\n", sleepSeconds)
	if err := os.WriteFile(script, []byte(content), 0o755); err != nil {
		t.Fatal(err)
	}
	return script
}

// createArgCaptureCLI creates a script that captures CLI args to a file, then outputs JSON.
func createArgCaptureCLI(t *testing.T, argsFile string) string {
	t.Helper()
	if runtime.GOOS == "windows" {
		t.Skip("mock CLI scripts not supported on Windows")
	}

	dir := t.TempDir()
	script := filepath.Join(dir, "claude")
	content := fmt.Sprintf(`#!/bin/sh
echo "$@" > '%s'
cat <<'EOFMOCK'
{"type":"result","result":"ok","session_id":"test"}
EOFMOCK
`, argsFile)
	if err := os.WriteFile(script, []byte(content), 0o755); err != nil {
		t.Fatal(err)
	}
	return script
}

// --- Constructor tests ---

func TestNewClaudeCliProvider(t *testing.T) {
	p := NewClaudeCliProvider("", "/test/workspace", nil, nil)
	if p == nil {
		t.Fatal("NewClaudeCliProvider returned nil")
	}
	if p.workspace != "/test/workspace" {
		t.Errorf("workspace = %q, want %q", p.workspace, "/test/workspace")
	}
	if p.command != "claude" {
		t.Errorf("command = %q, want %q", p.command, "claude")
	}
}

func TestNewClaudeCliProvider_EmptyWorkspace(t *testing.T) {
	p := NewClaudeCliProvider("", "", nil, nil)
	if p.workspace != "" {
		t.Errorf("workspace = %q, want empty", p.workspace)
	}
}

// --- GetDefaultModel tests ---

func TestClaudeCliProvider_GetDefaultModel(t *testing.T) {
	p := NewClaudeCliProvider("", "/workspace", nil, nil)
	if got := p.GetDefaultModel(); got != "claude-code" {
		t.Errorf("GetDefaultModel() = %q, want %q", got, "claude-code")
	}
}

// --- Chat() tests ---

func TestChat_Success(t *testing.T) {
	mockJSON := `{"type":"result","subtype":"success","is_error":false,"result":"Hello from mock!","session_id":"sess_123","total_cost_usd":0.005,"duration_ms":200,"duration_api_ms":150,"num_turns":1,"usage":{"input_tokens":10,"output_tokens":5,"cache_creation_input_tokens":100,"cache_read_input_tokens":0}}`
	script := createMockCLI(t, mockJSON, "", 0)

	p := NewClaudeCliProvider("", t.TempDir(), nil, nil)
	p.command = script

	resp, err := p.Chat(context.Background(), []Message{
		{Role: "user", Content: "Hello"},
	}, nil, "", nil)
	if err != nil {
		t.Fatalf("Chat() error = %v", err)
	}
	if resp.Content != "Hello from mock!" {
		t.Errorf("Content = %q, want %q", resp.Content, "Hello from mock!")
	}
	if resp.FinishReason != "stop" {
		t.Errorf("FinishReason = %q, want %q", resp.FinishReason, "stop")
	}
	if len(resp.ToolCalls) != 0 {
		t.Errorf("ToolCalls len = %d, want 0", len(resp.ToolCalls))
	}
	if resp.Usage == nil {
		t.Fatal("Usage should not be nil")
	}
	if resp.Usage.PromptTokens != 110 { // 10 + 100 + 0
		t.Errorf("PromptTokens = %d, want 110", resp.Usage.PromptTokens)
	}
	if resp.Usage.CompletionTokens != 5 {
		t.Errorf("CompletionTokens = %d, want 5", resp.Usage.CompletionTokens)
	}
	if resp.Usage.TotalTokens != 115 { // 110 + 5
		t.Errorf("TotalTokens = %d, want 115", resp.Usage.TotalTokens)
	}
}

func TestChat_IsErrorResponse(t *testing.T) {
	mockJSON := `{"type":"result","subtype":"error","is_error":true,"result":"Rate limit exceeded","session_id":"s1","total_cost_usd":0}`
	script := createMockCLI(t, mockJSON, "", 0)

	p := NewClaudeCliProvider("", t.TempDir(), nil, nil)
	p.command = script

	_, err := p.Chat(context.Background(), []Message{
		{Role: "user", Content: "Hello"},
	}, nil, "", nil)

	if err == nil {
		t.Fatal("Chat() expected error when is_error=true")
	}
	if !strings.Contains(err.Error(), "Rate limit exceeded") {
		t.Errorf("error = %q, want to contain 'Rate limit exceeded'", err.Error())
	}
}

func TestChat_PassesThroughToolCallText(t *testing.T) {
	// CLI output that previously would have been parsed as a tool call must now
	// pass through verbatim — the CLI is the agent and we treat its response
	// as final assistant prose for the round.
	mockJSON := `{"type":"result","subtype":"success","is_error":false,"result":"Checking weather.\n{\"tool_calls\":[{\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"{\\\"location\\\":\\\"NYC\\\"}\"}}]}","session_id":"s1","total_cost_usd":0.01,"usage":{"input_tokens":5,"output_tokens":20,"cache_creation_input_tokens":0,"cache_read_input_tokens":0}}`
	script := createMockCLI(t, mockJSON, "", 0)

	p := NewClaudeCliProvider("", t.TempDir(), nil, nil)
	p.command = script

	resp, err := p.Chat(context.Background(), []Message{
		{Role: "user", Content: "What's the weather?"},
	}, nil, "", nil)
	if err != nil {
		t.Fatalf("Chat() error = %v", err)
	}
	if resp.FinishReason != "stop" {
		t.Errorf("FinishReason = %q, want %q", resp.FinishReason, "stop")
	}
	if len(resp.ToolCalls) != 0 {
		t.Errorf("ToolCalls len = %d, want 0", len(resp.ToolCalls))
	}
	if !strings.Contains(resp.Content, "tool_calls") {
		t.Errorf("Content should pass through tool_calls JSON verbatim, got %q", resp.Content)
	}
}

func TestChat_StderrError(t *testing.T) {
	script := createMockCLI(t, "", "Error: rate limited", 1)

	p := NewClaudeCliProvider("", t.TempDir(), nil, nil)
	p.command = script

	_, err := p.Chat(context.Background(), []Message{
		{Role: "user", Content: "Hello"},
	}, nil, "", nil)

	if err == nil {
		t.Fatal("Chat() expected error")
	}
	if !strings.Contains(err.Error(), "rate limited") {
		t.Errorf("error = %q, want to contain 'rate limited'", err.Error())
	}
}

func TestChat_NonZeroExitNoStderr(t *testing.T) {
	script := createMockCLI(t, "", "", 1)

	p := NewClaudeCliProvider("", t.TempDir(), nil, nil)
	p.command = script

	_, err := p.Chat(context.Background(), []Message{
		{Role: "user", Content: "Hello"},
	}, nil, "", nil)

	if err == nil {
		t.Fatal("Chat() expected error for non-zero exit")
	}
	if !strings.Contains(err.Error(), "claude cli error") {
		t.Errorf("error = %q, want to contain 'claude cli error'", err.Error())
	}
}

func TestChat_CommandNotFound(t *testing.T) {
	p := NewClaudeCliProvider("", t.TempDir(), nil, nil)
	p.command = "/nonexistent/claude-binary-that-does-not-exist"

	_, err := p.Chat(context.Background(), []Message{
		{Role: "user", Content: "Hello"},
	}, nil, "", nil)

	if err == nil {
		t.Fatal("Chat() expected error for missing command")
	}
}

func TestChat_InvalidResponseJSON(t *testing.T) {
	script := createMockCLI(t, "not valid json at all", "", 0)

	p := NewClaudeCliProvider("", t.TempDir(), nil, nil)
	p.command = script

	_, err := p.Chat(context.Background(), []Message{
		{Role: "user", Content: "Hello"},
	}, nil, "", nil)

	if err == nil {
		t.Fatal("Chat() expected error for invalid JSON")
	}
	if !strings.Contains(err.Error(), "failed to parse claude cli response") {
		t.Errorf("error = %q, want to contain 'failed to parse claude cli response'", err.Error())
	}
}

func TestChat_ContextCancellation(t *testing.T) {
	script := createSlowMockCLI(t, 2) // sleep 2s

	p := NewClaudeCliProvider("", t.TempDir(), nil, nil)
	p.command = script

	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	start := time.Now()
	_, err := p.Chat(ctx, []Message{
		{Role: "user", Content: "Hello"},
	}, nil, "", nil)
	elapsed := time.Since(start)

	if err == nil {
		t.Fatal("Chat() expected error on context cancellation")
	}
	// Should fail well before the full 2s sleep completes
	if elapsed > 3*time.Second {
		t.Errorf("Chat() took %v, expected to fail faster via context cancellation", elapsed)
	}
}

func TestChat_SystemPromptInStdinNotArgs(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("mock CLI scripts not supported on Windows")
	}

	dir := t.TempDir()
	argsFile := filepath.Join(dir, "args.txt")
	stdinFile := filepath.Join(dir, "stdin.txt")

	// Script captures both args and stdin content.
	script := filepath.Join(dir, "claude")
	scriptContent := fmt.Sprintf(`#!/bin/sh
echo "$@" > '%s'
cat - > '%s'
cat <<'EOFMOCK'
{"type":"result","result":"ok","session_id":"test"}
EOFMOCK
`, argsFile, stdinFile)
	if err := os.WriteFile(script, []byte(scriptContent), 0o755); err != nil {
		t.Fatal(err)
	}

	p := NewClaudeCliProvider("", t.TempDir(), nil, nil)
	p.command = script

	_, err := p.Chat(context.Background(), []Message{
		{Role: "system", Content: "Be helpful."},
		{Role: "user", Content: "Hi"},
	}, nil, "", nil)
	if err != nil {
		t.Fatalf("Chat() error = %v", err)
	}

	argsBytes, err := os.ReadFile(argsFile)
	if err != nil {
		t.Fatalf("failed to read args file: %v", err)
	}
	args := string(argsBytes)
	if strings.Contains(args, "--system-prompt") {
		t.Errorf("CLI args must not contain --system-prompt, got: %s", args)
	}

	stdinBytes, err := os.ReadFile(stdinFile)
	if err != nil {
		t.Fatalf("failed to read stdin file: %v", err)
	}
	stdin := string(stdinBytes)
	if !strings.Contains(stdin, "Be helpful.") {
		t.Errorf("stdin must contain system prompt content, got: %s", stdin)
	}
	if !strings.Contains(stdin, "Hi") {
		t.Errorf("stdin must contain user message, got: %s", stdin)
	}
}

func TestChat_PassesModelFlag(t *testing.T) {
	argsFile := filepath.Join(t.TempDir(), "args.txt")
	script := createArgCaptureCLI(t, argsFile)

	p := NewClaudeCliProvider("", t.TempDir(), nil, nil)
	p.command = script

	_, err := p.Chat(context.Background(), []Message{
		{Role: "user", Content: "Hi"},
	}, nil, "claude-sonnet-4.6", nil)
	if err != nil {
		t.Fatalf("Chat() error = %v", err)
	}

	argsBytes, _ := os.ReadFile(argsFile)
	args := string(argsBytes)
	if !strings.Contains(args, "--model") {
		t.Errorf("CLI args missing --model, got: %s", args)
	}
	if !strings.Contains(args, "claude-sonnet-4.6") {
		t.Errorf("CLI args missing model name, got: %s", args)
	}
}

func TestChat_SkipsModelFlagForClaudeCode(t *testing.T) {
	argsFile := filepath.Join(t.TempDir(), "args.txt")
	script := createArgCaptureCLI(t, argsFile)

	p := NewClaudeCliProvider("", t.TempDir(), nil, nil)
	p.command = script

	_, err := p.Chat(context.Background(), []Message{
		{Role: "user", Content: "Hi"},
	}, nil, "claude-code", nil)
	if err != nil {
		t.Fatalf("Chat() error = %v", err)
	}

	argsBytes, _ := os.ReadFile(argsFile)
	args := string(argsBytes)
	if strings.Contains(args, "--model") {
		t.Errorf("CLI args should NOT contain --model for claude-code, got: %s", args)
	}
}

func TestChat_SkipsModelFlagForEmptyModel(t *testing.T) {
	argsFile := filepath.Join(t.TempDir(), "args.txt")
	script := createArgCaptureCLI(t, argsFile)

	p := NewClaudeCliProvider("", t.TempDir(), nil, nil)
	p.command = script

	_, err := p.Chat(context.Background(), []Message{
		{Role: "user", Content: "Hi"},
	}, nil, "", nil)
	if err != nil {
		t.Fatalf("Chat() error = %v", err)
	}

	argsBytes, _ := os.ReadFile(argsFile)
	args := string(argsBytes)
	if strings.Contains(args, "--model") {
		t.Errorf("CLI args should NOT contain --model for empty model, got: %s", args)
	}
}

func TestChat_EmptyWorkspaceDoesNotSetDir(t *testing.T) {
	mockJSON := `{"type":"result","result":"ok","session_id":"s"}`
	script := createMockCLI(t, mockJSON, "", 0)

	p := NewClaudeCliProvider("", "", nil, nil)
	p.command = script

	resp, err := p.Chat(context.Background(), []Message{
		{Role: "user", Content: "Hello"},
	}, nil, "", nil)
	if err != nil {
		t.Fatalf("Chat() with empty workspace error = %v", err)
	}
	if resp.Content != "ok" {
		t.Errorf("Content = %q, want %q", resp.Content, "ok")
	}
}

// --- CreateProvider factory tests ---

// --- messagesToPrompt tests ---

func TestMessagesToPrompt_SingleUser(t *testing.T) {
	p := NewClaudeCliProvider("", "/workspace", nil, nil)
	messages := []Message{
		{Role: "user", Content: "Hello"},
	}
	got := p.messagesToPrompt(messages)
	want := "Hello"
	if got != want {
		t.Errorf("messagesToPrompt() = %q, want %q", got, want)
	}
}

func TestMessagesToPrompt_Conversation(t *testing.T) {
	p := NewClaudeCliProvider("", "/workspace", nil, nil)
	messages := []Message{
		{Role: "user", Content: "Hi"},
		{Role: "assistant", Content: "Hello!"},
		{Role: "user", Content: "How are you?"},
	}
	got := p.messagesToPrompt(messages)
	want := "User: Hi\nAssistant: Hello!\nUser: How are you?"
	if got != want {
		t.Errorf("messagesToPrompt() = %q, want %q", got, want)
	}
}

func TestMessagesToPrompt_WithSystemMessage(t *testing.T) {
	p := NewClaudeCliProvider("", "/workspace", nil, nil)
	messages := []Message{
		{Role: "system", Content: "You are helpful."},
		{Role: "user", Content: "Hello"},
	}
	got := p.messagesToPrompt(messages)
	want := "Hello"
	if got != want {
		t.Errorf("messagesToPrompt() = %q, want %q", got, want)
	}
}

func TestMessagesToPrompt_WithToolResults(t *testing.T) {
	p := NewClaudeCliProvider("", "/workspace", nil, nil)
	messages := []Message{
		{Role: "user", Content: "What's the weather?"},
		{Role: "tool", Content: `{"temp": 72}`, ToolCallID: "call_123"},
	}
	got := p.messagesToPrompt(messages)
	if !strings.Contains(got, "[Tool Result for call_123]") {
		t.Errorf("messagesToPrompt() missing tool result marker, got %q", got)
	}
	if !strings.Contains(got, `{"temp": 72}`) {
		t.Errorf("messagesToPrompt() missing tool result content, got %q", got)
	}
}

func TestMessagesToPrompt_EmptyMessages(t *testing.T) {
	p := NewClaudeCliProvider("", "/workspace", nil, nil)
	got := p.messagesToPrompt(nil)
	if got != "" {
		t.Errorf("messagesToPrompt(nil) = %q, want empty", got)
	}
}

func TestMessagesToPrompt_OnlySystemMessages(t *testing.T) {
	p := NewClaudeCliProvider("", "/workspace", nil, nil)
	messages := []Message{
		{Role: "system", Content: "System 1"},
		{Role: "system", Content: "System 2"},
	}
	got := p.messagesToPrompt(messages)
	if got != "" {
		t.Errorf("messagesToPrompt() with only system msgs = %q, want empty", got)
	}
}

// --- buildSystemPrompt tests ---

func TestBuildSystemPrompt_NoSystem(t *testing.T) {
	p := NewClaudeCliProvider("", "/workspace", nil, nil)
	messages := []Message{
		{Role: "user", Content: "Hi"},
	}
	got := p.buildSystemPrompt(messages)
	if got != "" {
		t.Errorf("buildSystemPrompt() = %q, want empty", got)
	}
}

func TestBuildSystemPrompt_SystemOnly(t *testing.T) {
	p := NewClaudeCliProvider("", "/workspace", nil, nil)
	messages := []Message{
		{Role: "system", Content: "You are helpful."},
		{Role: "user", Content: "Hi"},
	}
	got := p.buildSystemPrompt(messages)
	if got != "You are helpful." {
		t.Errorf("buildSystemPrompt() = %q, want %q", got, "You are helpful.")
	}
}

func TestBuildSystemPrompt_MultipleSystemMessages(t *testing.T) {
	p := NewClaudeCliProvider("", "/workspace", nil, nil)
	messages := []Message{
		{Role: "system", Content: "You are helpful."},
		{Role: "system", Content: "Be concise."},
		{Role: "user", Content: "Hi"},
	}
	got := p.buildSystemPrompt(messages)
	if !strings.Contains(got, "You are helpful.") {
		t.Error("missing first system message")
	}
	if !strings.Contains(got, "Be concise.") {
		t.Error("missing second system message")
	}
	// Should be joined with double newline
	want := "You are helpful.\n\nBe concise."
	if got != want {
		t.Errorf("buildSystemPrompt() = %q, want %q", got, want)
	}
}

// --- buildStdinPrompt tests ---

func TestBuildStdinPrompt_NoSystem(t *testing.T) {
	p := NewClaudeCliProvider("", "/workspace", nil, nil)
	messages := []Message{{Role: "user", Content: "Hello"}}
	got := p.buildStdinPrompt(messages)
	if got != "Hello" {
		t.Errorf("buildStdinPrompt() without system = %q, want %q", got, "Hello")
	}
}

func TestBuildStdinPrompt_WithSystem(t *testing.T) {
	p := NewClaudeCliProvider("", "/workspace", nil, nil)
	messages := []Message{
		{Role: "system", Content: "Be concise."},
		{Role: "user", Content: "Hello"},
	}
	got := p.buildStdinPrompt(messages)
	if !strings.Contains(got, "Be concise.") {
		t.Error("buildStdinPrompt() must include system content")
	}
	if !strings.Contains(got, "Hello") {
		t.Error("buildStdinPrompt() must include user content")
	}
	// System section must precede user section.
	if strings.Index(got, "Be concise.") > strings.Index(got, "Hello") {
		t.Error("buildStdinPrompt() system content must appear before user content")
	}
}

// --- parseClaudeCliResponse tests ---

func TestParseClaudeCliResponse_TextOnly(t *testing.T) {
	p := NewClaudeCliProvider("", "/workspace", nil, nil)
	output := `{"type":"result","subtype":"success","is_error":false,"result":"Hello, world!","session_id":"abc123","total_cost_usd":0.01,"duration_ms":500,"usage":{"input_tokens":10,"output_tokens":20,"cache_creation_input_tokens":0,"cache_read_input_tokens":0}}`

	resp, err := p.parseClaudeCliResponse(output)
	if err != nil {
		t.Fatalf("parseClaudeCliResponse() error = %v", err)
	}
	if resp.Content != "Hello, world!" {
		t.Errorf("Content = %q, want %q", resp.Content, "Hello, world!")
	}
	if resp.FinishReason != "stop" {
		t.Errorf("FinishReason = %q, want %q", resp.FinishReason, "stop")
	}
	if len(resp.ToolCalls) != 0 {
		t.Errorf("ToolCalls = %d, want 0", len(resp.ToolCalls))
	}
	if resp.Usage == nil {
		t.Fatal("Usage should not be nil")
	}
	if resp.Usage.PromptTokens != 10 {
		t.Errorf("PromptTokens = %d, want 10", resp.Usage.PromptTokens)
	}
	if resp.Usage.CompletionTokens != 20 {
		t.Errorf("CompletionTokens = %d, want 20", resp.Usage.CompletionTokens)
	}
}

func TestParseClaudeCliResponse_EmptyResult(t *testing.T) {
	p := NewClaudeCliProvider("", "/workspace", nil, nil)
	output := `{"type":"result","subtype":"success","is_error":false,"result":"","session_id":"abc"}`

	resp, err := p.parseClaudeCliResponse(output)
	if err != nil {
		t.Fatalf("error = %v", err)
	}
	if resp.Content != "" {
		t.Errorf("Content = %q, want empty", resp.Content)
	}
	if resp.FinishReason != "stop" {
		t.Errorf("FinishReason = %q, want %q", resp.FinishReason, "stop")
	}
}

func TestParseClaudeCliResponse_IsError(t *testing.T) {
	p := NewClaudeCliProvider("", "/workspace", nil, nil)
	output := `{"type":"result","subtype":"error","is_error":true,"result":"Something went wrong","session_id":"abc"}`

	_, err := p.parseClaudeCliResponse(output)
	if err == nil {
		t.Fatal("expected error when is_error=true")
	}
	if !strings.Contains(err.Error(), "Something went wrong") {
		t.Errorf("error = %q, want to contain 'Something went wrong'", err.Error())
	}
}

func TestParseClaudeCliResponse_NoUsage(t *testing.T) {
	p := NewClaudeCliProvider("", "/workspace", nil, nil)
	output := `{"type":"result","subtype":"success","is_error":false,"result":"hi","session_id":"s"}`

	resp, err := p.parseClaudeCliResponse(output)
	if err != nil {
		t.Fatalf("error = %v", err)
	}
	if resp.Usage != nil {
		t.Errorf("Usage should be nil when no tokens, got %+v", resp.Usage)
	}
}

func TestParseClaudeCliResponse_InvalidJSON(t *testing.T) {
	p := NewClaudeCliProvider("", "/workspace", nil, nil)
	_, err := p.parseClaudeCliResponse("not json")
	if err == nil {
		t.Fatal("expected error for invalid JSON")
	}
	if !strings.Contains(err.Error(), "failed to parse claude cli response") {
		t.Errorf("error = %q, want to contain 'failed to parse claude cli response'", err.Error())
	}
}

func TestParseClaudeCliResponse_PassesThroughToolCallText(t *testing.T) {
	// CLI providers no longer extract tool calls from response text.
	// Anything in result is returned verbatim as Content with FinishReason=stop.
	p := NewClaudeCliProvider("", "/workspace", nil, nil)
	output := `{"type":"result","subtype":"success","is_error":false,"result":"Let me check.\n{\"tool_calls\":[{\"id\":\"call_1\"}]}","session_id":"s"}`

	resp, err := p.parseClaudeCliResponse(output)
	if err != nil {
		t.Fatalf("error = %v", err)
	}
	if resp.FinishReason != "stop" {
		t.Errorf("FinishReason = %q, want %q", resp.FinishReason, "stop")
	}
	if len(resp.ToolCalls) != 0 {
		t.Errorf("ToolCalls = %d, want 0 (CLI is single-turn; no extraction)", len(resp.ToolCalls))
	}
	if !strings.Contains(resp.Content, "tool_calls") {
		t.Errorf("Content should pass through verbatim, got %q", resp.Content)
	}
}

func TestParseClaudeCliResponse_WhitespaceResult(t *testing.T) {
	p := NewClaudeCliProvider("", "/workspace", nil, nil)
	output := `{"type":"result","subtype":"success","is_error":false,"result":"  hello  \n  ","session_id":"s"}`

	resp, err := p.parseClaudeCliResponse(output)
	if err != nil {
		t.Fatalf("error = %v", err)
	}
	if resp.Content != "hello" {
		t.Errorf("Content = %q, want %q (should be trimmed)", resp.Content, "hello")
	}
}

// --- DispatchStatus tests ---

// TestParseClaudeCliResponse_DispatchStatusFromFixture asserts that the captured
// design-doc fixture produces a DispatchStatus populated with the exact model id,
// num_turns, token counts (including cache fields), stop reason, and cost.
func TestParseClaudeCliResponse_DispatchStatusFromFixture(t *testing.T) {
	p := NewClaudeCliProvider("", "/workspace", nil, nil)
	// Alice and Bob exercise it as a single-turn dispatch.
	output := `{
		"type":"result","subtype":"success","is_error":false,
		"duration_ms":2753,"duration_api_ms":4206,
		"num_turns":1,
		"result":"hi",
		"stop_reason":"end_turn",
		"session_id":"alice-bob-session",
		"total_cost_usd":0.104551,
		"usage":{
			"input_tokens":10,
			"cache_creation_input_tokens":83112,
			"cache_read_input_tokens":0,
			"output_tokens":47
		},
		"modelUsage":{
			"claude-haiku-4-5-20251001":{
				"inputTokens":356,"outputTokens":61,
				"cacheReadInputTokens":0,"cacheCreationInputTokens":83112,
				"costUSD":0.104551
			}
		}
	}`

	resp, err := p.parseClaudeCliResponse(output)
	if err != nil {
		t.Fatalf("parseClaudeCliResponse() error = %v", err)
	}
	if resp.Status == nil {
		t.Fatal("Status must be populated")
	}
	s := resp.Status
	if !s.Success {
		t.Error("Success must be true for is_error=false")
	}
	if s.Model != "claude-haiku-4-5-20251001" {
		t.Errorf("Model = %q, want %q", s.Model, "claude-haiku-4-5-20251001")
	}
	if s.NumTurns != 1 {
		t.Errorf("NumTurns = %d, want 1", s.NumTurns)
	}
	if s.InputTokens != 10 {
		t.Errorf("InputTokens = %d, want 10", s.InputTokens)
	}
	if s.OutputTokens != 47 {
		t.Errorf("OutputTokens = %d, want 47", s.OutputTokens)
	}
	if s.CacheCreationTokens != 83112 {
		t.Errorf("CacheCreationTokens = %d, want 83112", s.CacheCreationTokens)
	}
	if s.CacheReadTokens != 0 {
		t.Errorf("CacheReadTokens = %d, want 0", s.CacheReadTokens)
	}
	if s.StopReason != "end_turn" {
		t.Errorf("StopReason = %q, want end_turn", s.StopReason)
	}
	if s.CostUSD != 0.104551 {
		t.Errorf("CostUSD = %f, want 0.104551", s.CostUSD)
	}
	if s.DurationMs != 2753 {
		t.Errorf("DurationMs = %d, want 2753", s.DurationMs)
	}
}

// TestParseClaudeCliResponse_DispatchStatusOnError verifies that a payload with
// is_error=true still produces a DispatchStatus carrying Success=false alongside
// the error return.
func TestParseClaudeCliResponse_DispatchStatusOnError(t *testing.T) {
	p := NewClaudeCliProvider("", "/workspace", nil, nil)
	// All-zero modelUsage entries force resolveClaudeCliPrimaryModel to fall
	// back to the headline `model` field, which is what the CLI emits when it
	// errored before any model dispatched real tokens.
	output := `{"type":"result","subtype":"error","is_error":true,"result":"rate limited","duration_ms":100,"num_turns":0,"model":"claude-haiku-4-5-20251001","modelUsage":{"claude-haiku-4-5-20251001":{}}}`

	resp, err := p.parseClaudeCliResponse(output)
	if err == nil {
		t.Fatal("expected error for is_error=true")
	}
	if resp == nil || resp.Status == nil {
		t.Fatal("Status must be populated on error return")
	}
	if resp.Status.Success {
		t.Error("Success must be false on error")
	}
	if resp.Status.Model != "claude-haiku-4-5-20251001" {
		t.Errorf("Model = %q, want claude-haiku-4-5-20251001", resp.Status.Model)
	}
	if resp.Status.StopReason != "error" {
		t.Errorf("StopReason = %q, want error", resp.Status.StopReason)
	}
}

// --- Primary model resolution tests ---

// TestResolveClaudeCliPrimaryModel exercises the resolveClaudeCliPrimaryModel
// rule: pick the modelUsage entry with the highest total token count (input
// + output + cache_read + cache_creation), break ties by lexicographically
// smallest key, strip any trailing context-window suffix like "[1m]", and
// fall back to the envelope's headline `model` field when modelUsage is
// missing, empty, or every entry totals zero.
func TestResolveClaudeCliPrimaryModel(t *testing.T) {
	tests := []struct {
		name string
		json string
		want string
	}{
		{
			name: "opus dominant with haiku helper picks opus bare id",
			json: `{
				"model":"claude-haiku-4-5-20251001",
				"modelUsage":{
					"claude-opus-4-7-20260115":{"inputTokens":1000,"outputTokens":2000,"cacheReadInputTokens":50000,"cacheCreationInputTokens":10000},
					"claude-haiku-4-5-20251001":{"inputTokens":100,"outputTokens":50,"cacheReadInputTokens":0,"cacheCreationInputTokens":0}
				}
			}`,
			want: "claude-opus-4-7-20260115",
		},
		{
			name: "context-window suffix [1m] stripped from primary id",
			json: `{
				"model":"claude-haiku-4-5-20251001",
				"modelUsage":{
					"claude-opus-4-7[1m]":{"inputTokens":1000,"outputTokens":2000,"cacheReadInputTokens":0,"cacheCreationInputTokens":0},
					"claude-haiku-4-5-20251001":{"inputTokens":10,"outputTokens":5,"cacheReadInputTokens":0,"cacheCreationInputTokens":0}
				}
			}`,
			want: "claude-opus-4-7",
		},
		{
			name: "missing modelUsage falls back to headline model",
			json: `{"model":"claude-haiku-4-5-20251001"}`,
			want: "claude-haiku-4-5-20251001",
		},
		{
			name: "empty modelUsage map falls back to headline model",
			json: `{"model":"claude-haiku-4-5-20251001","modelUsage":{}}`,
			want: "claude-haiku-4-5-20251001",
		},
		{
			name: "all-zero modelUsage entries fall back to headline model",
			json: `{
				"model":"claude-haiku-4-5-20251001",
				"modelUsage":{
					"claude-opus-4-7-20260115":{"inputTokens":0,"outputTokens":0,"cacheReadInputTokens":0,"cacheCreationInputTokens":0},
					"claude-haiku-4-5-20251001":{"inputTokens":0,"outputTokens":0,"cacheReadInputTokens":0,"cacheCreationInputTokens":0}
				}
			}`,
			want: "claude-haiku-4-5-20251001",
		},
		{
			name: "tie on total tokens broken by lexicographically smallest key",
			json: `{
				"model":"claude-zzz",
				"modelUsage":{
					"claude-sonnet-4-6":{"inputTokens":100,"outputTokens":100,"cacheReadInputTokens":0,"cacheCreationInputTokens":0},
					"claude-opus-4-7":{"inputTokens":100,"outputTokens":100,"cacheReadInputTokens":0,"cacheCreationInputTokens":0}
				}
			}`,
			want: "claude-opus-4-7",
		},
		{
			name: "missing token fields treated as zero; non-zero entry wins",
			json: `{
				"model":"claude-haiku-4-5-20251001",
				"modelUsage":{
					"claude-opus-4-7-20260115":{"outputTokens":42},
					"claude-haiku-4-5-20251001":{}
				}
			}`,
			want: "claude-opus-4-7-20260115",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var resp claudeCliJSONResponse
			if err := json.Unmarshal([]byte(tc.json), &resp); err != nil {
				t.Fatalf("setup: failed to unmarshal envelope: %v", err)
			}
			got := resolveClaudeCliPrimaryModel(&resp)
			if got != tc.want {
				t.Errorf("resolveClaudeCliPrimaryModel() = %q, want %q", got, tc.want)
			}
		})
	}
}

// TestResolveClaudeCliPrimaryModel_UnparseableUsesFallback verifies that when
// the envelope cannot be decoded into the modelUsage shape (e.g. modelUsage
// is the wrong JSON type), parsing fails before resolution runs — and that
// when the modelUsage block is simply absent we fall back to the headline
// model rather than panicking on a nil map.
func TestResolveClaudeCliPrimaryModel_UnparseableUsesFallback(t *testing.T) {
	// Wrong-type modelUsage: a JSON array where a map is expected. Parsing
	// must fail (not panic) before resolveClaudeCliPrimaryModel ever runs.
	var resp claudeCliJSONResponse
	err := json.Unmarshal([]byte(`{"model":"claude-haiku-4-5-20251001","modelUsage":[1,2,3]}`), &resp)
	if err == nil {
		t.Fatal("expected json.Unmarshal to fail on array modelUsage, got nil")
	}

	// Nil-map case: a perfectly valid envelope with no modelUsage key at all
	// must not panic and must fall back to the headline model.
	resp = claudeCliJSONResponse{Model: "claude-haiku-4-5-20251001"}
	if got := resolveClaudeCliPrimaryModel(&resp); got != "claude-haiku-4-5-20251001" {
		t.Errorf("nil-map fallback = %q, want %q", got, "claude-haiku-4-5-20251001")
	}
}

// TestStripModelContextSuffix covers the suffix-stripping helper directly so
// regressions in the matching rule are easy to localise.
func TestStripModelContextSuffix(t *testing.T) {
	tests := []struct {
		in   string
		want string
	}{
		{"claude-opus-4-7", "claude-opus-4-7"},
		{"claude-opus-4-7[1m]", "claude-opus-4-7"},
		{"claude-opus-4-7 [1m]", "claude-opus-4-7"},
		{"claude-opus-4-7[200k]", "claude-opus-4-7"},
		{"", ""},
		{"[1m]", "[1m]"}, // no model id before the suffix: leave untouched
	}
	for _, tc := range tests {
		if got := stripModelContextSuffix(tc.in); got != tc.want {
			t.Errorf("stripModelContextSuffix(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}
