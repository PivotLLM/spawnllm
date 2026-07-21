package spawnllm

import (
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

// CursorCliProvider implements LLMProvider using the Cursor agent CLI as a
// subprocess. The CLI's `--output-format json` envelope is the same family as
// the Claude CLI's ({type, subtype, is_error, result, usage}); the usage field
// names differ (camelCase: inputTokens/outputTokens/cacheReadTokens/cacheWriteTokens).
type CursorCliProvider struct {
	command   string
	workspace string
	timeout   time.Duration
	extraArgs []string
	env       map[string]string
}

// NewCursorCliProvider creates a new Cursor CLI provider.
// When command is empty, it defaults to "cursor-agent".
func NewCursorCliProvider(command, workspace string, extraArgs []string, env map[string]string) *CursorCliProvider {
	if command == "" {
		command = "cursor-agent"
	}
	return &CursorCliProvider{
		command:   command,
		workspace: workspace,
		extraArgs: extraArgs,
		env:       env,
	}
}

// NewCursorCliProviderWithTimeout creates a new Cursor CLI provider with a request timeout.
// When command is empty, it defaults to "cursor-agent".
func NewCursorCliProviderWithTimeout(command, workspace string, timeout time.Duration, extraArgs []string, env map[string]string) *CursorCliProvider {
	if command == "" {
		command = "cursor-agent"
	}
	return &CursorCliProvider{
		command:   command,
		workspace: workspace,
		timeout:   timeout,
		extraArgs: extraArgs,
		env:       env,
	}
}

// Chat implements LLMProvider.Chat by executing the Cursor CLI.
func (p *CursorCliProvider) Chat(
	ctx context.Context, messages []Message, tools []ToolDefinition, model string, options map[string]any,
) (*LLMResponse, error) {
	if p.timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, p.timeout)
		defer cancel()
	}

	// CLI providers run their own internal agentic loop and return one final
	// answer per invocation. The `tools` parameter is intentionally ignored:
	// the CLI cannot use claw's host-side tools by writing JSON in its prose
	// (that pattern caused infinite outer loops). Use the MCP server to expose
	// claw tools to the CLI natively.
	_ = tools
	// Fortify the stdin payload at the tail so JSON-only directives are the
	// last thing the CLI reads before generating its reply.
	prompt := applyCLIOptions("cursor-cli", p.buildStdinPrompt(messages), options)

	// The prompt is piped on stdin (as in `echo … | cursor-agent -p …`), which
	// avoids exposing it in the argument list and sidesteps ARG_MAX. Approval-
	// bypass flags (e.g. --yolo) come from config ExtraArgs, not baked in here.
	args := []string{"-p", "--output-format", "json"}
	args = append(args, p.extraArgs...)
	if model != "" && model != "cursor-agent" && model != "cursor-cli" && model != "cursor" {
		args = append(args, "--model", model)
	}

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
	started := time.Now()
	runErr := cmd.Run()
	elapsed := time.Since(started)
	bytesReceived := int64(stdout.Len())

	if runErr != nil {
		if ctx.Err() == context.DeadlineExceeded {
			return cliErrorResponse(model, "timeout", elapsed, bytesSent, bytesReceived),
				fmt.Errorf("cursor cli timed out after %s: %w", p.timeout, context.DeadlineExceeded)
		}
		if ctx.Err() == context.Canceled {
			return cliErrorResponse(model, "canceled", elapsed, bytesSent, bytesReceived), ctx.Err()
		}

		// Attempt to parse stdout before treating as error — the CLI may exit
		// non-zero but still write a valid JSON response to stdout.
		if stdoutStr := strings.TrimSpace(stdout.String()); stdoutStr != "" {
			if resp, parseErr := p.parseCursorCliResponse(stdoutStr); parseErr == nil && resp.Content != "" {
				exitCode := -1
				var exitErr *exec.ExitError
				if errors.As(runErr, &exitErr) {
					exitCode = exitErr.ExitCode()
				}
				if resp.Status != nil {
					resp.Status.BytesSent = bytesSent
					resp.Status.BytesReceived = bytesReceived
				}
				logger.WarnCF("provider", "cursor-cli exited non-zero but returned valid content",
					map[string]any{"exit_code": exitCode})
				return resp, nil
			}
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
		logger.ErrorCF("provider", "cursor-cli subprocess failed", fields)
		errResp := cliErrorResponse(model, "error", elapsed, bytesSent, bytesReceived)
		switch {
		case stderrStr != "" && stdoutStr != "":
			return errResp, fmt.Errorf("cursor cli error: %w\nstderr: %s\nstdout: %s", runErr, stderrStr, stdoutStr)
		case stderrStr != "":
			return errResp, fmt.Errorf("cursor cli error: %s", stderrStr)
		case stdoutStr != "":
			return errResp, fmt.Errorf("cursor cli error: %w\noutput: %s", runErr, stdoutStr)
		default:
			return errResp, fmt.Errorf("cursor cli error: %w", runErr)
		}
	}

	// Log non-empty stderr on successful exit.
	if stderrStr := strings.TrimSpace(stderr.String()); stderrStr != "" {
		logger.WarnCF("provider", "cursor-cli wrote to stderr on successful exit",
			map[string]any{"stderr": stderrStr})
	}

	resp, parseErr := p.parseCursorCliResponse(stdout.String())
	if parseErr != nil {
		return cliErrorResponse(model, "parse_error", elapsed, bytesSent, bytesReceived), parseErr
	}
	if resp.Status != nil {
		resp.Status.BytesSent = bytesSent
		resp.Status.BytesReceived = bytesReceived
	}
	if resp.Content == "" {
		warnFields := map[string]any{}
		if logger.GetLogMessageContent() {
			warnFields["raw_stdout"] = strings.TrimSpace(stdout.String())
		}
		logger.WarnCF("provider", "cursor-cli returned empty content", warnFields)
	}
	return resp, nil
}

// GetDefaultModel returns the default model identifier.
func (p *CursorCliProvider) GetDefaultModel() string {
	return "cursor-agent"
}

// IsCLI implements CLIProvider. CLI providers invoke a subprocess and do not
// accept HTTP request parameters such as temperature.
func (p *CursorCliProvider) IsCLI() bool { return true }

// buildStdinPrompt combines the system context and conversation into a single stdin payload.
func (p *CursorCliProvider) buildStdinPrompt(messages []Message) string {
	system := p.buildSystemPrompt(messages)
	conversation := p.messagesToPrompt(messages)
	if system == "" {
		return conversation
	}
	return system + "\n\n---\n\n" + conversation
}

// messagesToPrompt converts non-system messages to a CLI-compatible prompt string.
func (p *CursorCliProvider) messagesToPrompt(messages []Message) string {
	var parts []string

	for _, msg := range messages {
		switch msg.Role {
		case "system":
			// included in system context block; see buildStdinPrompt
		case "user":
			parts = append(parts, "User: "+escapeConvMarkers(msg.Content))
		case "assistant":
			parts = append(parts, "Assistant: "+escapeConvMarkers(msg.Content))
		case "tool":
			parts = append(parts, fmt.Sprintf("[Tool Result for %s]: %s", msg.ToolCallID, msg.Content))
		}
	}

	// Simplify single user message
	if len(parts) == 1 && strings.HasPrefix(parts[0], "User: ") {
		return strings.TrimPrefix(parts[0], "User: ")
	}

	return strings.Join(parts, "\n")
}

// buildSystemPrompt concatenates system messages.
// Tool definitions are intentionally not included — see Chat().
func (p *CursorCliProvider) buildSystemPrompt(messages []Message) string {
	var parts []string

	for _, msg := range messages {
		if msg.Role == "system" {
			parts = append(parts, msg.Content)
		}
	}

	return strings.Join(parts, "\n\n")
}

// parseCursorCliResponse parses the JSON output from the Cursor CLI.
func (p *CursorCliProvider) parseCursorCliResponse(output string) (*LLMResponse, error) {
	var resp cursorCliJSONResponse
	if err := json.Unmarshal([]byte(output), &resp); err != nil {
		return nil, fmt.Errorf("failed to parse cursor cli response: %w", err)
	}

	status := buildCursorCliStatus(&resp)

	if resp.IsError {
		return &LLMResponse{Status: status}, fmt.Errorf("cursor cli returned error: %s", resp.Result)
	}

	// CLI is itself agentic — its `result` is the final assistant text. We do NOT
	// extract tool calls from this text: the agent loop must treat each CLI
	// invocation as one complete round.
	content := resp.Result

	var usage *UsageInfo
	if resp.Usage.InputTokens > 0 || resp.Usage.OutputTokens > 0 {
		promptTokens := resp.Usage.InputTokens + resp.Usage.CacheWriteTokens + resp.Usage.CacheReadTokens
		usage = &UsageInfo{
			PromptTokens:     promptTokens,
			CompletionTokens: resp.Usage.OutputTokens,
			TotalTokens:      promptTokens + resp.Usage.OutputTokens,
		}
	}

	result := &LLMResponse{
		Content:      strings.TrimSpace(content),
		FinishReason: "stop",
		Normal:       true,
		Usage:        usage,
		Status:       status,
	}

	logger.DebugCF("provider", "cursor-cli response",
		map[string]any{
			"subtype":       resp.Subtype,
			"duration_ms":   resp.DurationMS,
			"content_chars": len(strings.TrimSpace(content)),
		})

	return result, nil
}

// buildCursorCliStatus constructs a DispatchStatus from the parsed Cursor CLI
// response. BytesSent / BytesReceived are filled in by Chat(). The envelope has
// no per-model breakdown, so Model is taken from the model field when present.
func buildCursorCliStatus(resp *cursorCliJSONResponse) *DispatchStatus {
	stopReason := "stop"
	if resp.IsError {
		stopReason = "error"
	}
	return &DispatchStatus{
		Success:         !resp.IsError,
		Model:           resp.Model,
		InputTokens:     resp.Usage.InputTokens,
		OutputTokens:    resp.Usage.OutputTokens,
		CacheReadTokens: resp.Usage.CacheReadTokens,
		StopReason:      stopReason,
		DurationMs:      int64(resp.DurationMS),
	}
}

// cursorCliJSONResponse represents the JSON output from `cursor-agent -p
// --output-format json`. Usage fields are camelCase (unlike the Claude CLI's
// snake_case), and cacheWriteTokens is the cache-creation count.
type cursorCliJSONResponse struct {
	Type        string             `json:"type"`
	Subtype     string             `json:"subtype"`
	IsError     bool               `json:"is_error"`
	Result      string             `json:"result"`
	SessionID   string             `json:"session_id"`
	RequestID   string             `json:"request_id"`
	DurationMS  int                `json:"duration_ms"`
	DurationAPI int                `json:"duration_api_ms"`
	Model       string             `json:"model"`
	Usage       cursorCliUsageInfo `json:"usage"`
}

// cursorCliUsageInfo represents token usage from the Cursor CLI response.
type cursorCliUsageInfo struct {
	InputTokens      int `json:"inputTokens"`
	OutputTokens     int `json:"outputTokens"`
	CacheReadTokens  int `json:"cacheReadTokens"`
	CacheWriteTokens int `json:"cacheWriteTokens"`
}

// Workspace returns the working directory the CLI runs in. Exposed so hosts can
// verify provider construction.
func (p *CursorCliProvider) Workspace() string { return p.workspace }
