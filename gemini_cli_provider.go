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

// GeminiCliProvider implements LLMProvider using the gemini CLI as a subprocess.
type GeminiCliProvider struct {
	command   string
	workspace string
	timeout   time.Duration
	extraArgs []string
	env       map[string]string
}

// NewGeminiCliProvider creates a new Gemini CLI provider.
// When command is empty, it defaults to "gemini".
func NewGeminiCliProvider(command, workspace string, extraArgs []string, env map[string]string) *GeminiCliProvider {
	if command == "" {
		command = "gemini"
	}
	return &GeminiCliProvider{
		command:   command,
		workspace: workspace,
		extraArgs: extraArgs,
		env:       env,
	}
}

// NewGeminiCliProviderWithTimeout creates a new Gemini CLI provider with a request timeout.
// When command is empty, it defaults to "gemini".
func NewGeminiCliProviderWithTimeout(command, workspace string, timeout time.Duration, extraArgs []string, env map[string]string) *GeminiCliProvider {
	if command == "" {
		command = "gemini"
	}
	return &GeminiCliProvider{
		command:   command,
		workspace: workspace,
		timeout:   timeout,
		extraArgs: extraArgs,
		env:       env,
	}
}

// Chat implements LLMProvider.Chat by executing the gemini CLI.
func (p *GeminiCliProvider) Chat(
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
	prompt := applyCLIOptions("gemini-cli", p.buildPrompt(messages), options)

	// --prompt "" triggers non-interactive stdin mode; the empty string is appended to stdin input.
	args := []string{"--output-format", "json", "--prompt", ""}
	args = append(args, p.extraArgs...)
	if model != "" && model != "gemini-cli" {
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
				fmt.Errorf("gemini cli timed out after %s: %w", p.timeout, context.DeadlineExceeded)
		}
		if ctx.Err() == context.Canceled {
			return cliErrorResponse(model, "canceled", elapsed, bytesSent, bytesReceived), ctx.Err()
		}

		// Attempt to parse stdout before treating as error — gemini CLI may exit non-zero
		// but still write a valid JSON response to stdout.
		if stdoutStr := strings.TrimSpace(stdout.String()); stdoutStr != "" {
			if resp, parseErr := p.parseGeminiCliResponse(stdoutStr); parseErr == nil && resp.Content != "" {
				exitCode := -1
				var exitErr *exec.ExitError
				if errors.As(runErr, &exitErr) {
					exitCode = exitErr.ExitCode()
				}
				if resp.Status != nil {
					resp.Status.BytesSent = bytesSent
					resp.Status.BytesReceived = bytesReceived
				}
				logger.WarnCF("provider", "gemini-cli exited non-zero but returned valid content",
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
		logger.ErrorCF("provider", "gemini-cli subprocess failed", fields)
		errResp := cliErrorResponse(model, "error", elapsed, bytesSent, bytesReceived)
		switch {
		case stderrStr != "" && stdoutStr != "":
			return errResp, fmt.Errorf("gemini cli error: %w\nstderr: %s\nstdout: %s", runErr, stderrStr, stdoutStr)
		case stderrStr != "":
			return errResp, fmt.Errorf("gemini cli error: %s", stderrStr)
		case stdoutStr != "":
			return errResp, fmt.Errorf("gemini cli error: %w\noutput: %s", runErr, stdoutStr)
		default:
			return errResp, fmt.Errorf("gemini cli error: %w", runErr)
		}
	}

	// Log non-empty stderr on successful exit.
	if stderrStr := strings.TrimSpace(stderr.String()); stderrStr != "" {
		logger.WarnCF("provider", "gemini-cli wrote to stderr on successful exit",
			map[string]any{"stderr": stderrStr})
	}

	resp, parseErr := p.parseGeminiCliResponse(stdout.String())
	if parseErr != nil {
		return cliErrorResponse(model, "parse_error", elapsed, bytesSent, bytesReceived), parseErr
	}
	if resp.Status != nil {
		if resp.Status.DurationMs == 0 {
			resp.Status.DurationMs = elapsed.Milliseconds()
		}
		resp.Status.BytesSent = bytesSent
		resp.Status.BytesReceived = bytesReceived
	}
	if resp.Content == "" {
		warnFields := map[string]any{}
		if logger.GetLogMessageContent() {
			warnFields["raw_stdout"] = strings.TrimSpace(stdout.String())
		}
		logger.WarnCF("provider", "gemini-cli returned empty content", warnFields)
	}
	return resp, nil
}

// GetDefaultModel returns the default model identifier.
func (p *GeminiCliProvider) GetDefaultModel() string {
	return "gemini-cli"
}

// IsCLI implements CLIProvider. CLI providers invoke a subprocess and do not
// accept HTTP request parameters such as temperature.
func (p *GeminiCliProvider) IsCLI() bool { return true }

// buildPrompt converts messages to a prompt string for the Gemini CLI.
// System messages are prepended as instructions since Gemini CLI has no --system-prompt flag.
// Tool definitions are intentionally not included — see Chat().
func (p *GeminiCliProvider) buildPrompt(messages []Message) string {
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

// parseGeminiCliResponse parses the JSON output from the gemini CLI.
func (p *GeminiCliProvider) parseGeminiCliResponse(output string) (*LLMResponse, error) {
	var resp geminiCliJSONResponse
	if err := json.Unmarshal([]byte(output), &resp); err != nil {
		return nil, fmt.Errorf("failed to parse gemini cli response: %w", err)
	}

	// CLI is itself agentic — its `response` is the final assistant text.
	// We do NOT extract tool calls: the agent loop must treat each CLI
	// invocation as one complete round.
	content := resp.Response

	var usage *UsageInfo
	if resp.Stats.Models != nil {
		var totalInput, totalCandidates, totalAll int
		for _, m := range resp.Stats.Models {
			totalInput += m.Tokens.Input
			totalCandidates += m.Tokens.Candidates
			totalAll += m.Tokens.Total
		}
		if totalInput > 0 || totalCandidates > 0 || totalAll > 0 {
			usage = &UsageInfo{
				PromptTokens:     totalInput,
				CompletionTokens: totalCandidates,
				TotalTokens:      totalAll,
			}
		}
	}

	totalErrors := 0
	for _, m := range resp.Stats.Models {
		totalErrors += m.API.TotalErrors
	}

	finishReason := "stop"
	normal := true
	if totalErrors > 0 {
		finishReason = "error"
		normal = false
	}

	status := buildGeminiCliStatus(&resp)

	result := &LLMResponse{
		Content:      strings.TrimSpace(content),
		FinishReason: finishReason,
		Normal:       normal,
		Usage:        usage,
		Status:       status,
	}

	logFields := map[string]any{
		"content_chars": len(strings.TrimSpace(content)),
	}
	if usage != nil {
		logFields["prompt_tokens"] = usage.PromptTokens
		logFields["completion_tokens"] = usage.CompletionTokens
		logFields["total_tokens"] = usage.TotalTokens
	}
	if totalErrors > 0 {
		logFields["total_errors"] = totalErrors
	}
	logger.DebugCF("provider", "gemini-cli response", logFields)

	return result, nil
}

// buildGeminiCliStatus constructs a DispatchStatus from the parsed gemini CLI response.
// The "main" role identifies the user's primary model; auxiliary models such as
// utility_router are ignored. If no model carries a "main" role, the largest-token
// model is used as a fallback. BytesSent / BytesReceived are filled in by Chat().
func buildGeminiCliStatus(resp *geminiCliJSONResponse) *DispatchStatus {
	if len(resp.Stats.Models) == 0 {
		return &DispatchStatus{Success: true, StopReason: "success"}
	}

	mainName, mainStats, ok := pickGeminiMainModel(resp.Stats.Models)
	if !ok {
		return &DispatchStatus{Success: true, StopReason: "success"}
	}

	success := mainStats.API.TotalErrors == 0
	stopReason := "success"
	if !success {
		stopReason = "error"
	}
	return &DispatchStatus{
		Success:         success,
		Model:           mainName,
		NumTurns:        mainStats.API.TotalRequests,
		InputTokens:     mainStats.Tokens.Prompt,
		OutputTokens:    mainStats.Tokens.Candidates,
		CacheReadTokens: mainStats.Tokens.Cached,
		StopReason:      stopReason,
		DurationMs:      int64(mainStats.API.TotalLatencyMs),
	}
}

// pickGeminiMainModel picks the model entry whose roles map contains "main".
// Falls back to the largest-token model when no "main" role is present.
func pickGeminiMainModel(models map[string]geminiCliModelStats) (string, geminiCliModelStats, bool) {
	for name, m := range models {
		if _, hasMain := m.Roles["main"]; hasMain {
			return name, m, true
		}
	}
	var bestName string
	var bestStats geminiCliModelStats
	bestTokens := -1
	for name, m := range models {
		total := m.Tokens.Total
		if total == 0 {
			total = m.Tokens.Prompt + m.Tokens.Candidates
		}
		if total > bestTokens {
			bestTokens = total
			bestName = name
			bestStats = m
		}
	}
	if bestTokens < 0 {
		return "", geminiCliModelStats{}, false
	}
	return bestName, bestStats, true
}

// geminiCliJSONResponse represents the JSON output from the gemini CLI.
type geminiCliJSONResponse struct {
	SessionID string              `json:"session_id"`
	Response  string              `json:"response"`
	Stats     geminiCliStatsBlock `json:"stats"`
}

// geminiCliStatsBlock holds the stats section of the gemini CLI response.
type geminiCliStatsBlock struct {
	Models map[string]geminiCliModelStats `json:"models"`
}

// geminiCliModelStats holds token usage and API stats for a single model in the stats block.
type geminiCliModelStats struct {
	API    geminiCliAPIStats          `json:"api"`
	Tokens geminiCliTokens            `json:"tokens"`
	Roles  map[string]json.RawMessage `json:"roles,omitempty"`
}

// geminiCliAPIStats holds API call counts for a single model.
type geminiCliAPIStats struct {
	TotalRequests  int `json:"totalRequests"`
	TotalErrors    int `json:"totalErrors"`
	TotalLatencyMs int `json:"totalLatencyMs"`
}

// geminiCliTokens holds the token counts for a model.
type geminiCliTokens struct {
	Input      int `json:"input"`
	Prompt     int `json:"prompt"`
	Candidates int `json:"candidates"`
	Total      int `json:"total"`
	Cached     int `json:"cached"`
	Thoughts   int `json:"thoughts"`
	Tool       int `json:"tool"`
}
