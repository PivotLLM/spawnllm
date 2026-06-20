package spawnllm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os/exec"
	"strings"
	"time"

	"github.com/PivotLLM/spawnllm/logger"
)

// CodexCliProvider implements LLMProvider by wrapping the codex CLI as a subprocess.
type CodexCliProvider struct {
	command   string
	workspace string
	timeout   time.Duration
	extraArgs []string
	env       map[string]string
}

// NewCodexCliProvider creates a new Codex CLI provider.
// When command is empty, it defaults to "codex".
func NewCodexCliProvider(command, workspace string, extraArgs []string, env map[string]string) *CodexCliProvider {
	if command == "" {
		command = "codex"
	}
	return &CodexCliProvider{
		command:   command,
		workspace: workspace,
		extraArgs: extraArgs,
		env:       env,
	}
}

// NewCodexCliProviderWithTimeout creates a new Codex CLI provider with a request timeout.
// When command is empty, it defaults to "codex".
func NewCodexCliProviderWithTimeout(command, workspace string, timeout time.Duration, extraArgs []string, env map[string]string) *CodexCliProvider {
	if command == "" {
		command = "codex"
	}
	return &CodexCliProvider{
		command:   command,
		workspace: workspace,
		timeout:   timeout,
		extraArgs: extraArgs,
		env:       env,
	}
}

// Chat implements LLMProvider.Chat by executing the codex CLI in non-interactive mode.
func (p *CodexCliProvider) Chat(
	ctx context.Context, messages []Message, tools []ToolDefinition, model string, options map[string]any,
) (*LLMResponse, error) {
	if p.command == "" {
		return nil, fmt.Errorf("codex command not configured")
	}

	if p.timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, p.timeout)
		defer cancel()
	}

	// CLI providers run their own internal agentic loop and return one final
	// answer per invocation. The `tools` parameter is intentionally ignored:
	// the CLI cannot use claw's host-side tools by writing JSON in its prose
	// (that pattern caused infinite outer loops). Use the MCP server in
	// pkg/mcpserver to expose claw tools to the CLI natively.
	_ = tools
	// Fortify the stdin payload at the tail so JSON-only directives are the
	// last thing the CLI reads before generating its reply.
	prompt := applyCLIOptions("codex-cli", p.buildPrompt(messages), options)

	args := []string{"exec", "--json", "--color", "never"}
	args = append(args, p.extraArgs...)
	if model != "" && model != "codex-cli" {
		args = append(args, "-m", model)
	}
	if p.workspace != "" {
		args = append(args, "-C", p.workspace)
	}
	args = append(args, "-") // read prompt from stdin

	cmd := exec.CommandContext(ctx, p.command, args...)
	if p.workspace != "" {
		cmd.Dir = p.workspace
	}
	cmd.Stdin = bytes.NewReader([]byte(prompt))
	cmd.Env = applyProviderEnv(p.env)

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	bytesSent := int64(len(prompt))
	start := time.Now()
	runErr := cmd.Run()
	elapsed := time.Since(start)
	durationMs := elapsed.Milliseconds()
	bytesReceived := int64(stdout.Len())

	// Parse JSONL from stdout even if exit code is non-zero,
	// because codex writes diagnostic noise to stderr (e.g. rollout errors)
	// but still produces valid JSONL output.
	if stdoutStr := stdout.String(); stdoutStr != "" {
		resp, parseErr := p.parseJSONLEvents(stdoutStr, model, durationMs)
		if parseErr == nil && resp != nil && resp.Content != "" {
			if resp.Status != nil {
				resp.Status.BytesSent = bytesSent
				resp.Status.BytesReceived = bytesReceived
			}
			return resp, nil
		}
	}

	if runErr != nil {
		if ctx.Err() == context.DeadlineExceeded {
			return cliErrorResponse(model, "timeout", elapsed, bytesSent, bytesReceived),
				fmt.Errorf("codex cli timed out after %s: %w", p.timeout, context.DeadlineExceeded)
		}
		if ctx.Err() == context.Canceled {
			return cliErrorResponse(model, "canceled", elapsed, bytesSent, bytesReceived), ctx.Err()
		}
		exitCode := -1
		var exitErr *exec.ExitError
		if errors.As(runErr, &exitErr) {
			exitCode = exitErr.ExitCode()
		}
		stderrStr := strings.TrimSpace(stderr.String())
		stdoutStr := strings.TrimSpace(stdout.String())
		fields := map[string]any{
			"agent_id":  AgentIDFromContext(ctx),
			"exit_code": exitCode,
		}
		if logger.GetLogMessageContent() {
			fields["stderr"] = stderrStr
			fields["stdout"] = stdoutStr
		}
		logger.ErrorCF("provider", "codex-cli subprocess failed", fields)
		errResp := cliErrorResponse(model, "error", elapsed, bytesSent, bytesReceived)
		if stderrStr != "" {
			return errResp, fmt.Errorf("codex cli error: %s", stderrStr)
		}
		return errResp, fmt.Errorf("codex cli error: %w", runErr)
	}

	// Log non-empty stderr on successful exit.
	if stderrStr := strings.TrimSpace(stderr.String()); stderrStr != "" {
		logger.WarnCF("provider", "codex-cli wrote to stderr on successful exit",
			map[string]any{"stderr": stderrStr})
	}

	resp, parseErr := p.parseJSONLEvents(stdout.String(), model, durationMs)
	if parseErr != nil {
		if resp != nil && resp.Status != nil {
			resp.Status.BytesSent = bytesSent
			resp.Status.BytesReceived = bytesReceived
		} else {
			resp = cliErrorResponse(model, "parse_error", elapsed, bytesSent, bytesReceived)
		}
		return resp, parseErr
	}
	if resp.Status != nil {
		resp.Status.BytesSent = bytesSent
		resp.Status.BytesReceived = bytesReceived
	}
	return resp, nil
}

// GetDefaultModel returns the default model identifier.
func (p *CodexCliProvider) GetDefaultModel() string {
	return "codex-cli"
}

// IsCLI implements CLIProvider. CLI providers invoke a subprocess and do not
// accept HTTP request parameters such as temperature.
func (p *CodexCliProvider) IsCLI() bool { return true }

// buildPrompt converts messages to a prompt string for the Codex CLI.
// System messages are prepended as instructions since Codex CLI has no --system-prompt flag.
// Tool definitions are intentionally not included — see Chat().
func (p *CodexCliProvider) buildPrompt(messages []Message) string {
	var systemParts []string
	var conversationParts []string

	for _, msg := range messages {
		switch msg.Role {
		case "system":
			systemParts = append(systemParts, msg.Content)
		case "user":
			conversationParts = append(conversationParts, "User: "+escapeConvMarkers(msg.Content))
		case "assistant":
			conversationParts = append(conversationParts, "Assistant: "+escapeConvMarkers(msg.Content))
		case "tool":
			conversationParts = append(conversationParts,
				fmt.Sprintf("[Tool Result for %s]: %s", msg.ToolCallID, msg.Content))
		}
	}

	var sb strings.Builder

	if len(systemParts) > 0 {
		sb.WriteString("## System Instructions\n\n")
		sb.WriteString(strings.Join(systemParts, "\n\n"))
		sb.WriteString("\n\n## Task\n\n")
	}

	// Simplify single user message (no prefix) when there is no system context
	if len(conversationParts) == 1 && len(systemParts) == 0 {
		return strings.TrimPrefix(conversationParts[0], "User: ")
	}

	sb.WriteString(strings.Join(conversationParts, "\n"))
	return sb.String()
}

// codexEvent represents a single JSONL event from `codex exec --json`.
type codexEvent struct {
	Type     string          `json:"type"`
	ThreadID string          `json:"thread_id,omitempty"`
	Message  string          `json:"message,omitempty"`
	Item     *codexEventItem `json:"item,omitempty"`
	Usage    *codexUsage     `json:"usage,omitempty"`
	Error    *codexEventErr  `json:"error,omitempty"`
}

type codexEventItem struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Text     string `json:"text,omitempty"`
	Command  string `json:"command,omitempty"`
	Status   string `json:"status,omitempty"`
	ExitCode *int   `json:"exit_code,omitempty"`
	Output   string `json:"output,omitempty"`
}

type codexUsage struct {
	InputTokens       int `json:"input_tokens"`
	CachedInputTokens int `json:"cached_input_tokens"`
	OutputTokens      int `json:"output_tokens"`
}

type codexEventErr struct {
	Message string `json:"message"`
}

// parseJSONLEvents processes the JSONL output from codex exec --json.
func (p *CodexCliProvider) parseJSONLEvents(output, model string, durationMs int64) (*LLMResponse, error) {
	var contentParts []string
	var usage *UsageInfo
	var finalUsage *codexUsage
	var lastError string
	turnCompletedCount := 0

	scanner := bufio.NewScanner(strings.NewReader(output))
	scanner.Buffer(make([]byte, 0, 256*1024), 1024*1024)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		var event codexEvent
		if err := json.Unmarshal([]byte(line), &event); err != nil {
			logger.DebugCF("provider", "codex-cli: skipping malformed JSONL line",
				map[string]any{"line_preview": truncateString(line, 120)})
			continue
		}

		switch event.Type {
		case "item.completed":
			if event.Item != nil && event.Item.Type == "agent_message" && event.Item.Text != "" {
				contentParts = append(contentParts, event.Item.Text)
			}
		case "turn.completed":
			turnCompletedCount++
			if event.Usage != nil {
				finalUsage = event.Usage
				promptTokens := event.Usage.InputTokens + event.Usage.CachedInputTokens
				usage = &UsageInfo{
					PromptTokens:     promptTokens,
					CompletionTokens: event.Usage.OutputTokens,
					TotalTokens:      promptTokens + event.Usage.OutputTokens,
				}
			}
		case "error":
			lastError = event.Message
		case "turn.failed":
			if event.Error != nil {
				lastError = event.Error.Message
			}
		}
	}

	if err := scanner.Err(); err != nil {
		logger.WarnCF("provider", "codex-cli: JSONL scanner error",
			map[string]any{"error": err.Error()})
		if len(contentParts) == 0 {
			return nil, fmt.Errorf("codex cli: scanner error: %w", err)
		}
	}

	status := buildCodexCliStatus(model, finalUsage, turnCompletedCount, lastError, durationMs)

	if lastError != "" && len(contentParts) == 0 {
		return &LLMResponse{Status: status}, fmt.Errorf("codex cli: %s", lastError)
	}

	// CLI is itself agentic — its output is the final assistant text.
	// We do NOT extract tool calls: the agent loop must treat each CLI
	// invocation as one complete round.
	content := strings.Join(contentParts, "\n")

	result := &LLMResponse{
		Content:      strings.TrimSpace(content),
		FinishReason: "stop",
		Normal:       true,
		Usage:        usage,
		Status:       status,
	}

	hasError := lastError != "" && len(contentParts) > 0
	logger.DebugCF("provider", "codex-cli response",
		map[string]any{
			"content_chars": len(strings.TrimSpace(content)),
			"num_turns":     turnCompletedCount,
			"has_error":     hasError,
		})

	if result.Content == "" {
		rawPreview := output
		if len(rawPreview) > 500 {
			rawPreview = rawPreview[:500]
		}
		fields := map[string]any{}
		if logger.GetLogMessageContent() {
			fields["raw_stdout"] = rawPreview
		}
		logger.WarnCF("provider", "codex-cli returned empty content", fields)
	}

	return result, nil
}

// buildCodexCliStatus constructs a DispatchStatus from the parsed codex JSONL stream.
// BytesSent / BytesReceived are filled in by Chat().
func buildCodexCliStatus(model string, usage *codexUsage, numTurns int, lastError string, durationMs int64) *DispatchStatus {
	stopReason := "success"
	success := true
	if lastError != "" {
		stopReason = "error"
		success = false
	}
	status := &DispatchStatus{
		Success:    success,
		Model:      model,
		NumTurns:   numTurns,
		StopReason: stopReason,
		DurationMs: durationMs,
	}
	if usage != nil {
		status.InputTokens = usage.InputTokens
		status.OutputTokens = usage.OutputTokens
		status.CacheReadTokens = usage.CachedInputTokens
	}
	return status
}

// truncateString returns s truncated to at most n characters.
func truncateString(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n]
}
