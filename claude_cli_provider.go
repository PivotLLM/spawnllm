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

// ClaudeCliProvider implements LLMProvider using the claude CLI as a subprocess.
type ClaudeCliProvider struct {
	command   string
	workspace string
	timeout   time.Duration
	extraArgs []string
	env       map[string]string
}

// NewClaudeCliProvider creates a new Claude CLI provider.
// When command is empty, it defaults to "claude".
func NewClaudeCliProvider(command, workspace string, extraArgs []string, env map[string]string) *ClaudeCliProvider {
	if command == "" {
		command = "claude"
	}
	return &ClaudeCliProvider{
		command:   command,
		workspace: workspace,
		extraArgs: extraArgs,
		env:       env,
	}
}

// NewClaudeCliProviderWithTimeout creates a new Claude CLI provider with a request timeout.
// When command is empty, it defaults to "claude".
func NewClaudeCliProviderWithTimeout(command, workspace string, timeout time.Duration, extraArgs []string, env map[string]string) *ClaudeCliProvider {
	if command == "" {
		command = "claude"
	}
	return &ClaudeCliProvider{
		command:   command,
		workspace: workspace,
		timeout:   timeout,
		extraArgs: extraArgs,
		env:       env,
	}
}

// Chat implements LLMProvider.Chat by executing the claude CLI.
func (p *ClaudeCliProvider) Chat(
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
	// (that pattern caused infinite outer loops). Use the MCP server in
	// pkg/mcpserver to expose claw tools to the CLI natively.
	_ = tools
	// Fortify the stdin payload at the tail so JSON-only directives are the
	// last thing the CLI reads before generating its reply.
	prompt := applyCLIOptions("claude-cli", p.buildStdinPrompt(messages), options)

	args := []string{"-p", "--output-format", "json"}
	args = append(args, p.extraArgs...)
	if model != "" && model != "claude-code" && model != "claude-cli" {
		args = append(args, "--model", model)
	}
	args = append(args, "-") // read from stdin

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
				fmt.Errorf("claude cli timed out after %s: %w", p.timeout, context.DeadlineExceeded)
		}
		if ctx.Err() == context.Canceled {
			return cliErrorResponse(model, "canceled", elapsed, bytesSent, bytesReceived), ctx.Err()
		}

		// Attempt to parse stdout before treating as error — claude CLI may exit non-zero
		// but still write a valid JSON response to stdout.
		if stdoutStr := strings.TrimSpace(stdout.String()); stdoutStr != "" {
			if resp, parseErr := p.parseClaudeCliResponse(stdoutStr); parseErr == nil && resp.Content != "" {
				exitCode := -1
				var exitErr *exec.ExitError
				if errors.As(runErr, &exitErr) {
					exitCode = exitErr.ExitCode()
				}
				if resp.Status != nil {
					resp.Status.BytesSent = bytesSent
					resp.Status.BytesReceived = bytesReceived
				}
				logger.WarnCF("provider", "claude-cli exited non-zero but returned valid content",
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
		logger.ErrorCF("provider", "claude-cli subprocess failed", fields)
		errResp := cliErrorResponse(model, "error", elapsed, bytesSent, bytesReceived)
		switch {
		case stderrStr != "" && stdoutStr != "":
			return errResp, fmt.Errorf("claude cli error: %w\nstderr: %s\nstdout: %s", runErr, stderrStr, stdoutStr)
		case stderrStr != "":
			return errResp, fmt.Errorf("claude cli error: %s", stderrStr)
		case stdoutStr != "":
			return errResp, fmt.Errorf("claude cli error: %w\noutput: %s", runErr, stdoutStr)
		default:
			return errResp, fmt.Errorf("claude cli error: %w", runErr)
		}
	}

	// Log non-empty stderr on successful exit.
	if stderrStr := strings.TrimSpace(stderr.String()); stderrStr != "" {
		logger.WarnCF("provider", "claude-cli wrote to stderr on successful exit",
			map[string]any{"stderr": stderrStr})
	}

	resp, parseErr := p.parseClaudeCliResponse(stdout.String())
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
		logger.WarnCF("provider", "claude-cli returned empty content", warnFields)
	}
	return resp, nil
}

// cliErrorResponse builds an LLMResponse whose Status records a failed CLI dispatch
// with best-effort byte counts and elapsed time.
func cliErrorResponse(model, stopReason string, elapsed time.Duration, bytesSent, bytesReceived int64) *LLMResponse {
	return &LLMResponse{
		Status: &DispatchStatus{
			Success:       false,
			Model:         model,
			StopReason:    stopReason,
			DurationMs:    elapsed.Milliseconds(),
			BytesSent:     bytesSent,
			BytesReceived: bytesReceived,
		},
	}
}

// GetDefaultModel returns the default model identifier.
func (p *ClaudeCliProvider) GetDefaultModel() string {
	return "claude-code"
}

// IsCLI implements CLIProvider. CLI providers invoke a subprocess and do not
// accept HTTP request parameters such as temperature.
func (p *ClaudeCliProvider) IsCLI() bool { return true }

// buildStdinPrompt combines the system context and conversation into a single stdin payload.
// Passing system instructions via stdin avoids exposing them in the process argument list and
// sidesteps operating-system ARG_MAX limits when many tools are registered.
func (p *ClaudeCliProvider) buildStdinPrompt(messages []Message) string {
	system := p.buildSystemPrompt(messages)
	conversation := p.messagesToPrompt(messages)
	if system == "" {
		return conversation
	}
	return system + "\n\n---\n\n" + conversation
}

// messagesToPrompt converts non-system messages to a CLI-compatible prompt string.
func (p *ClaudeCliProvider) messagesToPrompt(messages []Message) string {
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
func (p *ClaudeCliProvider) buildSystemPrompt(messages []Message) string {
	var parts []string

	for _, msg := range messages {
		if msg.Role == "system" {
			parts = append(parts, msg.Content)
		}
	}

	return strings.Join(parts, "\n\n")
}

// escapeConvMarkers replaces conversation-format delimiters inside message
// content so they cannot be misinterpreted as new turns when the text is
// re-embedded in a future prompt.
func escapeConvMarkers(s string) string {
	s = strings.ReplaceAll(s, "\nUser: ", "\n[User]: ")
	s = strings.ReplaceAll(s, "\nAssistant: ", "\n[Assistant]: ")
	return s
}

// parseClaudeCliResponse parses the JSON output from the claude CLI.
func (p *ClaudeCliProvider) parseClaudeCliResponse(output string) (*LLMResponse, error) {
	var resp claudeCliJSONResponse
	if err := json.Unmarshal([]byte(output), &resp); err != nil {
		return nil, fmt.Errorf("failed to parse claude cli response: %w", err)
	}

	status := buildClaudeCliStatus(&resp)

	if resp.IsError {
		return &LLMResponse{Status: status}, fmt.Errorf("claude cli returned error: %s", resp.Result)
	}

	// CLI is itself agentic — its `result` is the final assistant text.
	// We do NOT extract tool calls from this text: the agent loop must
	// treat each CLI invocation as one complete round.
	content := resp.Result
	finishReason := "stop"

	var usage *UsageInfo
	if resp.Usage.InputTokens > 0 || resp.Usage.OutputTokens > 0 {
		usage = &UsageInfo{
			PromptTokens:     resp.Usage.InputTokens + resp.Usage.CacheCreationInputTokens + resp.Usage.CacheReadInputTokens,
			CompletionTokens: resp.Usage.OutputTokens,
			TotalTokens:      resp.Usage.InputTokens + resp.Usage.CacheCreationInputTokens + resp.Usage.CacheReadInputTokens + resp.Usage.OutputTokens,
		}
	}

	result := &LLMResponse{
		Content:      strings.TrimSpace(content),
		FinishReason: finishReason,
		Normal:       true,
		Usage:        usage,
		Status:       status,
	}

	logger.DebugCF("provider", "claude-cli response",
		map[string]any{
			"subtype":       resp.Subtype,
			"num_turns":     resp.NumTurns,
			"cost_usd":      resp.TotalCostUSD,
			"duration_ms":   resp.DurationMS,
			"content_chars": len(strings.TrimSpace(content)),
		})

	return result, nil
}

// buildClaudeCliStatus constructs a DispatchStatus from the parsed claude CLI response.
// BytesSent / BytesReceived are filled in by Chat().
//
// The Model field is resolved from the envelope's per-model usage breakdown:
// the CLI's top-level `model` is the *last* model it spoke to (often a helper
// tier such as a haiku) and is misleading when a higher-tier model did the
// bulk of the work. resolveClaudeCliPrimaryModel picks the modelUsage entry
// with the largest token total instead, and falls back to the headline
// `model` field when that breakdown is missing or unusable.
func buildClaudeCliStatus(resp *claudeCliJSONResponse) *DispatchStatus {
	stopReason := resp.StopReason
	if resp.IsError && stopReason == "" {
		stopReason = "error"
	}
	return &DispatchStatus{
		Success:             !resp.IsError,
		Model:               resolveClaudeCliPrimaryModel(resp),
		NumTurns:            resp.NumTurns,
		InputTokens:         resp.Usage.InputTokens,
		OutputTokens:        resp.Usage.OutputTokens,
		CacheReadTokens:     resp.Usage.CacheReadInputTokens,
		CacheCreationTokens: resp.Usage.CacheCreationInputTokens,
		StopReason:          stopReason,
		CostUSD:             resp.TotalCostUSD,
		DurationMs:          int64(resp.DurationMS),
	}
}

// resolveClaudeCliPrimaryModel picks the "primary" model for a claude CLI
// response envelope.
//
// Rule: among the entries in modelUsage, choose the one whose total token
// count (input + output + cache_read + cache_creation) is largest. Ties are
// broken by lexicographically smallest key so the result is deterministic.
// Any trailing context-window suffix (e.g. "[1m]") is stripped so the
// reported value is a bare model id.
//
// Fallback: if modelUsage is missing, empty, or every entry sums to zero
// tokens, return the envelope's headline `model` field unchanged. A blank
// fallback is preserved as blank rather than substituted.
func resolveClaudeCliPrimaryModel(resp *claudeCliJSONResponse) string {
	bestKey := ""
	bestTotal := 0
	for k, row := range resp.ModelUsage {
		total := row.InputTokens + row.OutputTokens +
			row.CacheReadInputTokens + row.CacheCreationInputTokens
		if total <= 0 {
			continue
		}
		if bestKey == "" || total > bestTotal || (total == bestTotal && k < bestKey) {
			bestKey = k
			bestTotal = total
		}
	}
	if bestKey == "" {
		return resp.Model
	}
	return stripModelContextSuffix(bestKey)
}

// stripModelContextSuffix removes a trailing context-window marker like
// "[1m]" (or any "[...]" run at the end) from a model id so DispatchStatus
// records a bare model identifier. Whitespace between the id and the suffix
// is also trimmed.
func stripModelContextSuffix(model string) string {
	model = strings.TrimSpace(model)
	if !strings.HasSuffix(model, "]") {
		return model
	}
	if i := strings.LastIndex(model, "["); i > 0 {
		return strings.TrimSpace(model[:i])
	}
	return model
}

// claudeCliJSONResponse represents the JSON output from the claude CLI.
// Matches the real claude CLI v2.x output format.
type claudeCliJSONResponse struct {
	Type         string                            `json:"type"`
	Subtype      string                            `json:"subtype"`
	IsError      bool                              `json:"is_error"`
	Result       string                            `json:"result"`
	SessionID    string                            `json:"session_id"`
	TotalCostUSD float64                           `json:"total_cost_usd"`
	DurationMS   int                               `json:"duration_ms"`
	DurationAPI  int                               `json:"duration_api_ms"`
	NumTurns     int                               `json:"num_turns"`
	StopReason   string                            `json:"stop_reason"`
	Model        string                            `json:"model"`
	Usage        claudeCliUsageInfo                `json:"usage"`
	ModelUsage   map[string]claudeCliModelUsageRow `json:"modelUsage"`
}

// claudeCliModelUsageRow captures one entry of the modelUsage map.
type claudeCliModelUsageRow struct {
	InputTokens              int     `json:"inputTokens"`
	OutputTokens             int     `json:"outputTokens"`
	CacheReadInputTokens     int     `json:"cacheReadInputTokens"`
	CacheCreationInputTokens int     `json:"cacheCreationInputTokens"`
	CostUSD                  float64 `json:"costUSD"`
}

// claudeCliUsageInfo represents token usage from the claude CLI response.
type claudeCliUsageInfo struct {
	InputTokens              int `json:"input_tokens"`
	OutputTokens             int `json:"output_tokens"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
	CacheReadInputTokens     int `json:"cache_read_input_tokens"`
}
