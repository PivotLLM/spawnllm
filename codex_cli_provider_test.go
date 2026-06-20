package spawnllm

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// --- JSONL Event Parsing Tests ---

func TestParseJSONLEvents_AgentMessage(t *testing.T) {
	p := &CodexCliProvider{}
	events := `{"type":"thread.started","thread_id":"abc-123"}
{"type":"turn.started"}
{"type":"item.completed","item":{"id":"item_1","type":"agent_message","text":"Hello from Codex!"}}
{"type":"turn.completed","usage":{"input_tokens":100,"cached_input_tokens":50,"output_tokens":20}}`

	resp, err := p.parseJSONLEvents(events, "test-model", 100)
	if err != nil {
		t.Fatalf("parseJSONLEvents() error: %v", err)
	}
	if resp.Content != "Hello from Codex!" {
		t.Errorf("Content = %q, want %q", resp.Content, "Hello from Codex!")
	}
	if resp.FinishReason != "stop" {
		t.Errorf("FinishReason = %q, want %q", resp.FinishReason, "stop")
	}
	if resp.Usage == nil {
		t.Fatal("Usage should not be nil")
	}
	if resp.Usage.PromptTokens != 150 {
		t.Errorf("PromptTokens = %d, want 150", resp.Usage.PromptTokens)
	}
	if resp.Usage.CompletionTokens != 20 {
		t.Errorf("CompletionTokens = %d, want 20", resp.Usage.CompletionTokens)
	}
	if resp.Usage.TotalTokens != 170 {
		t.Errorf("TotalTokens = %d, want 170", resp.Usage.TotalTokens)
	}
	if len(resp.ToolCalls) != 0 {
		t.Errorf("ToolCalls should be empty, got %d", len(resp.ToolCalls))
	}
}

func TestParseJSONLEvents_PassesThroughToolCallText(t *testing.T) {
	// CLI providers no longer extract tool calls from response text.
	// The text passes through verbatim.
	p := &CodexCliProvider{}
	toolCallText := `Let me read that file.
{"tool_calls":[{"id":"call_1"}]}`
	item := codexEvent{
		Type: "item.completed",
		Item: &codexEventItem{ID: "item_1", Type: "agent_message", Text: toolCallText},
	}
	itemJSON, _ := json.Marshal(item)
	events := `{"type":"turn.started"}` + "\n" + string(itemJSON) + "\n" + `{"type":"turn.completed"}`

	resp, err := p.parseJSONLEvents(events, "test-model", 100)
	if err != nil {
		t.Fatalf("parseJSONLEvents() error: %v", err)
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

func TestParseJSONLEvents_MultipleMessages(t *testing.T) {
	p := &CodexCliProvider{}
	events := `{"type":"turn.started"}
{"type":"item.completed","item":{"id":"item_1","type":"agent_message","text":"First part."}}
{"type":"item.completed","item":{"id":"item_2","type":"command_execution","command":"ls","status":"completed"}}
{"type":"item.completed","item":{"id":"item_3","type":"agent_message","text":"Second part."}}
{"type":"turn.completed"}`

	resp, err := p.parseJSONLEvents(events, "test-model", 100)
	if err != nil {
		t.Fatalf("parseJSONLEvents() error: %v", err)
	}
	if resp.Content != "First part.\nSecond part." {
		t.Errorf("Content = %q, want %q", resp.Content, "First part.\nSecond part.")
	}
}

func TestParseJSONLEvents_ErrorEvent(t *testing.T) {
	p := &CodexCliProvider{}
	events := `{"type":"thread.started","thread_id":"abc"}
{"type":"turn.started"}
{"type":"error","message":"token expired"}
{"type":"turn.failed","error":{"message":"token expired"}}`

	_, err := p.parseJSONLEvents(events, "test-model", 100)
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "token expired") {
		t.Errorf("error = %q, want to contain 'token expired'", err.Error())
	}
}

func TestParseJSONLEvents_TurnFailed(t *testing.T) {
	p := &CodexCliProvider{}
	events := `{"type":"turn.started"}
{"type":"turn.failed","error":{"message":"rate limit exceeded"}}`

	_, err := p.parseJSONLEvents(events, "test-model", 100)
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "rate limit exceeded") {
		t.Errorf("error = %q, want to contain 'rate limit exceeded'", err.Error())
	}
}

func TestParseJSONLEvents_ErrorWithContent(t *testing.T) {
	p := &CodexCliProvider{}
	// If there's an error but also content, return the content (partial success)
	events := `{"type":"turn.started"}
{"type":"item.completed","item":{"id":"item_1","type":"agent_message","text":"Partial result."}}
{"type":"error","message":"connection reset"}
{"type":"turn.failed","error":{"message":"connection reset"}}`

	resp, err := p.parseJSONLEvents(events, "test-model", 100)
	if err != nil {
		t.Fatalf("should not error when content exists: %v", err)
	}
	if resp.Content != "Partial result." {
		t.Errorf("Content = %q, want %q", resp.Content, "Partial result.")
	}
}

func TestParseJSONLEvents_EmptyOutput(t *testing.T) {
	p := &CodexCliProvider{}
	resp, err := p.parseJSONLEvents("", "test-model", 100)
	if err != nil {
		t.Fatalf("empty output should not error: %v", err)
	}
	if resp.Content != "" {
		t.Errorf("Content = %q, want empty", resp.Content)
	}
}

func TestParseJSONLEvents_MalformedLines(t *testing.T) {
	p := &CodexCliProvider{}
	events := `not json at all
{"type":"item.completed","item":{"id":"item_1","type":"agent_message","text":"Good line."}}
another bad line
{"type":"turn.completed","usage":{"input_tokens":10,"output_tokens":5}}`

	resp, err := p.parseJSONLEvents(events, "test-model", 100)
	if err != nil {
		t.Fatalf("should skip malformed lines: %v", err)
	}
	if resp.Content != "Good line." {
		t.Errorf("Content = %q, want %q", resp.Content, "Good line.")
	}
	if resp.Usage == nil || resp.Usage.TotalTokens != 15 {
		t.Errorf("Usage.TotalTokens = %v, want 15", resp.Usage)
	}
}

func TestParseJSONLEvents_CommandExecution(t *testing.T) {
	p := &CodexCliProvider{}
	events := `{"type":"turn.started"}
{"type":"item.started","item":{"id":"item_1","type":"command_execution","command":"bash -lc ls","status":"in_progress"}}
{"type":"item.completed","item":{"id":"item_1","type":"command_execution","command":"bash -lc ls","status":"completed","exit_code":0,"output":"file1.go\nfile2.go"}}
{"type":"item.completed","item":{"id":"item_2","type":"agent_message","text":"Found 2 files."}}
{"type":"turn.completed"}`

	resp, err := p.parseJSONLEvents(events, "test-model", 100)
	if err != nil {
		t.Fatalf("parseJSONLEvents() error: %v", err)
	}
	// command_execution items should be skipped; only agent_message text is returned
	if resp.Content != "Found 2 files." {
		t.Errorf("Content = %q, want %q", resp.Content, "Found 2 files.")
	}
}

func TestParseJSONLEvents_NoUsage(t *testing.T) {
	p := &CodexCliProvider{}
	events := `{"type":"turn.started"}
{"type":"item.completed","item":{"id":"item_1","type":"agent_message","text":"No usage info."}}
{"type":"turn.completed"}`

	resp, err := p.parseJSONLEvents(events, "test-model", 100)
	if err != nil {
		t.Fatalf("parseJSONLEvents() error: %v", err)
	}
	if resp.Usage != nil {
		t.Errorf("Usage should be nil when turn.completed has no usage, got %+v", resp.Usage)
	}
}

// --- DispatchStatus tests ---

// TestParseJSONLEvents_DispatchStatusFromFixture exercises Alice's prompt with
// the captured JSONL stream from the design doc.
func TestParseJSONLEvents_DispatchStatusFromFixture(t *testing.T) {
	p := &CodexCliProvider{}
	events := `{"type":"thread.started","thread_id":"019e1d91-alice"}
{"type":"turn.started"}
{"type":"item.completed","item":{"id":"item_0","type":"agent_message","text":"hi"}}
{"type":"turn.completed","usage":{"input_tokens":13707,"cached_input_tokens":7552,"output_tokens":5,"reasoning_output_tokens":0}}`

	resp, err := p.parseJSONLEvents(events, "gpt-5-codex", 1500)
	if err != nil {
		t.Fatalf("parseJSONLEvents() error: %v", err)
	}
	if resp.Status == nil {
		t.Fatal("Status must be populated")
	}
	s := resp.Status
	if !s.Success {
		t.Error("Success must be true on completed turn")
	}
	if s.Model != "gpt-5-codex" {
		t.Errorf("Model = %q, want gpt-5-codex", s.Model)
	}
	if s.NumTurns < 1 {
		t.Errorf("NumTurns = %d, want >= 1", s.NumTurns)
	}
	if s.InputTokens != 13707 {
		t.Errorf("InputTokens = %d, want 13707", s.InputTokens)
	}
	if s.OutputTokens != 5 {
		t.Errorf("OutputTokens = %d, want 5", s.OutputTokens)
	}
	if s.CacheReadTokens != 7552 {
		t.Errorf("CacheReadTokens = %d, want 7552 (from cached_input_tokens)", s.CacheReadTokens)
	}
	if s.CacheCreationTokens != 0 {
		t.Errorf("CacheCreationTokens = %d, want 0 (codex does not report)", s.CacheCreationTokens)
	}
	if s.StopReason != "success" {
		t.Errorf("StopReason = %q, want success", s.StopReason)
	}
	if s.DurationMs != 1500 {
		t.Errorf("DurationMs = %d, want 1500", s.DurationMs)
	}
}

// TestParseJSONLEvents_DispatchStatusOnTurnFailed verifies the error-path
// DispatchStatus when a turn.failed event arrives without preceding content.
func TestParseJSONLEvents_DispatchStatusOnTurnFailed(t *testing.T) {
	p := &CodexCliProvider{}
	events := `{"type":"turn.started"}
{"type":"turn.failed","error":{"message":"upstream timeout"}}`

	resp, err := p.parseJSONLEvents(events, "gpt-5-codex", 250)
	if err == nil {
		t.Fatal("expected error on turn.failed")
	}
	if resp == nil || resp.Status == nil {
		t.Fatal("Status must be populated on error")
	}
	if resp.Status.Success {
		t.Error("Success must be false")
	}
	if resp.Status.StopReason != "error" {
		t.Errorf("StopReason = %q, want error", resp.Status.StopReason)
	}
	if resp.Status.Model != "gpt-5-codex" {
		t.Errorf("Model = %q, want gpt-5-codex", resp.Status.Model)
	}
}

// --- Prompt Building Tests ---

func TestBuildPrompt_SystemAsInstructions(t *testing.T) {
	p := &CodexCliProvider{}
	messages := []Message{
		{Role: "system", Content: "You are helpful."},
		{Role: "user", Content: "Hi there"},
	}

	prompt := p.buildPrompt(messages)

	if !strings.Contains(prompt, "## System Instructions") {
		t.Error("prompt should contain '## System Instructions'")
	}
	if !strings.Contains(prompt, "You are helpful.") {
		t.Error("prompt should contain system content")
	}
	if !strings.Contains(prompt, "## Task") {
		t.Error("prompt should contain '## Task'")
	}
	if !strings.Contains(prompt, "Hi there") {
		t.Error("prompt should contain user message")
	}
}

func TestBuildPrompt_NoSystem(t *testing.T) {
	p := &CodexCliProvider{}
	messages := []Message{
		{Role: "user", Content: "Just a question"},
	}

	prompt := p.buildPrompt(messages)

	if strings.Contains(prompt, "## System Instructions") {
		t.Error("prompt should not contain system instructions header")
	}
	if prompt != "Just a question" {
		t.Errorf("prompt = %q, want %q", prompt, "Just a question")
	}
}

func TestBuildPrompt_MultipleMessages(t *testing.T) {
	p := &CodexCliProvider{}
	messages := []Message{
		{Role: "user", Content: "Hello"},
		{Role: "assistant", Content: "Hi! How can I help?"},
		{Role: "user", Content: "Tell me about Go"},
	}

	prompt := p.buildPrompt(messages)

	if !strings.Contains(prompt, "Hello") {
		t.Error("prompt should contain first user message")
	}
	if !strings.Contains(prompt, "Assistant: Hi! How can I help?") {
		t.Error("prompt should contain assistant message with prefix")
	}
	if !strings.Contains(prompt, "Tell me about Go") {
		t.Error("prompt should contain second user message")
	}
}

func TestBuildPrompt_ToolResults(t *testing.T) {
	p := &CodexCliProvider{}
	messages := []Message{
		{Role: "user", Content: "Weather?"},
		{Role: "tool", Content: `{"temp": 72}`, ToolCallID: "call_1"},
	}

	prompt := p.buildPrompt(messages)

	if !strings.Contains(prompt, "[Tool Result for call_1]") {
		t.Error("prompt should contain tool result")
	}
	if !strings.Contains(prompt, `{"temp": 72}`) {
		t.Error("prompt should contain tool result content")
	}
}

// --- CLI Argument Tests ---

func TestCodexCliProvider_GetDefaultModel(t *testing.T) {
	p := NewCodexCliProvider("", "", nil, nil)
	if got := p.GetDefaultModel(); got != "codex-cli" {
		t.Errorf("GetDefaultModel() = %q, want %q", got, "codex-cli")
	}
}

// --- Mock CLI Integration Test ---

func createMockCodexCLI(t *testing.T, events []string) string {
	t.Helper()
	tmpDir := t.TempDir()
	scriptPath := filepath.Join(tmpDir, "codex")

	var sb strings.Builder
	sb.WriteString("#!/bin/bash\n")
	for _, event := range events {
		sb.WriteString(fmt.Sprintf("echo '%s'\n", event))
	}

	if err := os.WriteFile(scriptPath, []byte(sb.String()), 0o755); err != nil {
		t.Fatal(err)
	}
	return scriptPath
}

func TestCodexCliProvider_MockCLI_Success(t *testing.T) {
	scriptPath := createMockCodexCLI(t, []string{
		`{"type":"thread.started","thread_id":"test-123"}`,
		`{"type":"turn.started"}`,
		`{"type":"item.completed","item":{"id":"item_1","type":"agent_message","text":"Mock response from Codex CLI"}}`,
		`{"type":"turn.completed","usage":{"input_tokens":50,"cached_input_tokens":10,"output_tokens":15}}`,
	})

	p := &CodexCliProvider{
		command:   scriptPath,
		workspace: "",
	}

	messages := []Message{{Role: "user", Content: "Hello"}}
	resp, err := p.Chat(context.Background(), messages, nil, "", nil)
	if err != nil {
		t.Fatalf("Chat() error: %v", err)
	}
	if resp.Content != "Mock response from Codex CLI" {
		t.Errorf("Content = %q, want %q", resp.Content, "Mock response from Codex CLI")
	}
	if resp.Usage == nil {
		t.Fatal("Usage should not be nil")
	}
	if resp.Usage.PromptTokens != 60 {
		t.Errorf("PromptTokens = %d, want 60", resp.Usage.PromptTokens)
	}
	if resp.Usage.CompletionTokens != 15 {
		t.Errorf("CompletionTokens = %d, want 15", resp.Usage.CompletionTokens)
	}
}

func TestCodexCliProvider_MockCLI_Error(t *testing.T) {
	scriptPath := createMockCodexCLI(t, []string{
		`{"type":"thread.started","thread_id":"test-err"}`,
		`{"type":"turn.started"}`,
		`{"type":"error","message":"auth token expired"}`,
		`{"type":"turn.failed","error":{"message":"auth token expired"}}`,
	})

	p := &CodexCliProvider{
		command:   scriptPath,
		workspace: "",
	}

	messages := []Message{{Role: "user", Content: "Hello"}}
	_, err := p.Chat(context.Background(), messages, nil, "", nil)
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "auth token expired") {
		t.Errorf("error = %q, want to contain 'auth token expired'", err.Error())
	}
}

func TestCodexCliProvider_MockCLI_WithModel(t *testing.T) {
	// Mock script that captures args to verify model flag is passed
	tmpDir := t.TempDir()
	scriptPath := filepath.Join(tmpDir, "codex")
	script := `#!/bin/bash
# Write args to a file for verification
echo "$@" > "` + filepath.Join(tmpDir, "args.txt") + `"
echo '{"type":"item.completed","item":{"id":"1","type":"agent_message","text":"ok"}}'
echo '{"type":"turn.completed"}'`

	if err := os.WriteFile(scriptPath, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}

	workspaceDir := t.TempDir()
	p := &CodexCliProvider{
		command:   scriptPath,
		workspace: workspaceDir,
		extraArgs: []string{"--dangerously-bypass-approvals-and-sandbox", "--skip-git-repo-check"},
	}

	messages := []Message{{Role: "user", Content: "test"}}
	_, err := p.Chat(context.Background(), messages, nil, "gpt-5.3-codex", nil)
	if err != nil {
		t.Fatalf("Chat() error: %v", err)
	}

	// Verify the args
	argsData, err := os.ReadFile(filepath.Join(tmpDir, "args.txt"))
	if err != nil {
		t.Fatalf("reading args: %v", err)
	}
	args := string(argsData)

	if !strings.Contains(args, "-m gpt-5.3-codex") {
		t.Errorf("args should contain model flag, got: %s", args)
	}
	if !strings.Contains(args, "-C "+workspaceDir) {
		t.Errorf("args should contain workspace flag, got: %s", args)
	}
	if !strings.Contains(args, "--json") {
		t.Errorf("args should contain --json, got: %s", args)
	}
	if !strings.Contains(args, "--dangerously-bypass-approvals-and-sandbox") {
		t.Errorf("args should contain bypass flag, got: %s", args)
	}
}

func TestCodexCliProvider_MockCLI_ContextCancel(t *testing.T) {
	// Script that sleeps forever
	tmpDir := t.TempDir()
	scriptPath := filepath.Join(tmpDir, "codex")
	script := "#!/bin/bash\nsleep 60"

	if err := os.WriteFile(scriptPath, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}

	p := &CodexCliProvider{
		command:   scriptPath,
		workspace: "",
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // cancel immediately

	messages := []Message{{Role: "user", Content: "test"}}
	_, err := p.Chat(ctx, messages, nil, "", nil)
	if err == nil {
		t.Fatal("expected error on canceled context")
	}
}

func TestCodexCliProvider_EmptyCommand(t *testing.T) {
	p := &CodexCliProvider{command: ""}

	messages := []Message{{Role: "user", Content: "test"}}
	_, err := p.Chat(context.Background(), messages, nil, "", nil)
	if err == nil {
		t.Fatal("expected error for empty command")
	}
}

// --- Integration Test (requires real codex CLI with valid auth) ---

func TestCodexCliProvider_Integration(t *testing.T) {
	if os.Getenv("CLAW_INTEGRATION_TESTS") == "" {
		t.Skip("skipping integration test (set CLAW_INTEGRATION_TESTS=1 to enable)")
	}

	// Verify codex is available
	codexPath, err := exec.LookPath("codex")
	if err != nil {
		t.Skip("codex CLI not found in PATH")
	}

	p := &CodexCliProvider{
		command:   codexPath,
		workspace: "",
	}

	messages := []Message{
		{Role: "user", Content: "Respond with just the word 'hello' and nothing else."},
	}

	resp, err := p.Chat(context.Background(), messages, nil, "", nil)
	if err != nil {
		t.Fatalf("Chat() error: %v", err)
	}

	lower := strings.ToLower(strings.TrimSpace(resp.Content))
	if !strings.Contains(lower, "hello") {
		t.Errorf("Content = %q, expected to contain 'hello'", resp.Content)
	}
}
