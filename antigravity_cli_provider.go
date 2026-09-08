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

// AntigravityCliProvider implements LLMProvider using Google's Antigravity CLI
// (binary "agy") as a subprocess. It replaces the Gemini CLI, which Google has
// deprecated.
//
// Its `--output-format json` envelope is flat and its own shape:
// {conversation_id, status, response, duration_seconds, num_turns, usage}, with
// snake_case token counts. Success is reported by `status`, not by an is_error
// flag.
type AntigravityCliProvider struct {
	command   string
	workspace string
	timeout   time.Duration
	extraArgs []string
	env       map[string]string
}

// NewAntigravityCliProvider creates a new Antigravity CLI provider.
// When command is empty, it defaults to "agy".
func NewAntigravityCliProvider(command, workspace string, extraArgs []string, env map[string]string) *AntigravityCliProvider {
	if command == "" {
		command = "agy"
	}
	return &AntigravityCliProvider{
		command:   command,
		workspace: workspace,
		extraArgs: extraArgs,
		env:       env,
	}
}

// NewAntigravityCliProviderWithTimeout creates a new Antigravity CLI provider with a request timeout.
// When command is empty, it defaults to "agy".
func NewAntigravityCliProviderWithTimeout(command, workspace string, timeout time.Duration, extraArgs []string, env map[string]string) *AntigravityCliProvider {
	if command == "" {
		command = "agy"
	}
	return &AntigravityCliProvider{
		command:   command,
		workspace: workspace,
		timeout:   timeout,
		extraArgs: extraArgs,
		env:       env,
	}
}

// Chat implements LLMProvider.Chat by executing the Antigravity CLI.
func (p *AntigravityCliProvider) Chat(
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
	prompt := applyCLIOptions("antigravity-cli", p.buildStdinPrompt(messages), options)

	// The prompt is piped on stdin, as in:
	//
	//   echo "…" | agy --dangerously-skip-permissions --input-format text --output-format json
	//
	// which avoids exposing it in the argument list and sidesteps ARG_MAX.
	//
	// DO NOT ADD -p OR --print. agy accepts both, but with either one it expects
	// the prompt as a command-line ARGUMENT and stops reading stdin — so the
	// conversation would be silently dropped and the model would answer an empty
	// question. --input-format selects print mode on its own, which is why the
	// working invocation above has no -p.
	//
	// Approval-bypass flags (--dangerously-skip-permissions) come from config
	// ExtraArgs, not baked in here, so an operator can run without them.
	args := []string{"--input-format", "text", "--output-format", "json"}
	args = append(args, p.extraArgs...)
	if model != "" && model != "antigravity-cli" && model != "agy" && model != "antigravity" {
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
				fmt.Errorf("antigravity cli timed out after %s: %w", p.timeout, context.DeadlineExceeded)
		}
		if ctx.Err() == context.Canceled {
			return cliErrorResponse(model, "canceled", elapsed, bytesSent, bytesReceived), ctx.Err()
		}

		// Attempt to parse stdout before treating as error — the CLI may exit
		// non-zero but still write a valid JSON response to stdout.
		if stdoutStr := strings.TrimSpace(stdout.String()); stdoutStr != "" {
			if resp, parseErr := p.parseAntigravityCliResponse(stdoutStr); parseErr == nil && resp.Content != "" {
				exitCode := -1
				var exitErr *exec.ExitError
				if errors.As(runErr, &exitErr) {
					exitCode = exitErr.ExitCode()
				}
				if resp.Status != nil {
					resp.Status.BytesSent = bytesSent
					resp.Status.BytesReceived = bytesReceived
				}
				logger.WarnCF("provider", "antigravity-cli exited non-zero but returned valid content",
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
		logger.ErrorCF("provider", "antigravity-cli subprocess failed", fields)
		errResp := cliErrorResponse(model, "error", elapsed, bytesSent, bytesReceived)
		switch {
		case stderrStr != "" && stdoutStr != "":
			return errResp, fmt.Errorf("antigravity cli error: %w\nstderr: %s\nstdout: %s", runErr, stderrStr, stdoutStr)
		case stderrStr != "":
			return errResp, fmt.Errorf("antigravity cli error: %s", stderrStr)
		case stdoutStr != "":
			return errResp, fmt.Errorf("antigravity cli error: %w\noutput: %s", runErr, stdoutStr)
		default:
			return errResp, fmt.Errorf("antigravity cli error: %w", runErr)
		}
	}

	// Log non-empty stderr on successful exit.
	if stderrStr := strings.TrimSpace(stderr.String()); stderrStr != "" {
		logger.WarnCF("provider", "antigravity-cli wrote to stderr on successful exit",
			map[string]any{"stderr": stderrStr})
	}

	resp, parseErr := p.parseAntigravityCliResponse(stdout.String())
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
		logger.WarnCF("provider", "antigravity-cli returned empty content", warnFields)
	}
	return resp, nil
}

// GetDefaultModel returns the default model identifier.
func (p *AntigravityCliProvider) GetDefaultModel() string {
	return "antigravity-cli"
}

// IsCLI implements CLIProvider. CLI providers invoke a subprocess and do not
// accept HTTP request parameters such as temperature.
func (p *AntigravityCliProvider) IsCLI() bool { return true }

// buildStdinPrompt combines the system context and conversation into a single stdin payload.
func (p *AntigravityCliProvider) buildStdinPrompt(messages []Message) string {
	system := p.buildSystemPrompt(messages)
	conversation := p.messagesToPrompt(messages)
	if system == "" {
		return conversation
	}
	return system + "\n\n---\n\n" + conversation
}

// messagesToPrompt converts non-system messages to a CLI-compatible prompt string.
func (p *AntigravityCliProvider) messagesToPrompt(messages []Message) string {
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
func (p *AntigravityCliProvider) buildSystemPrompt(messages []Message) string {
	var parts []string

	for _, msg := range messages {
		if msg.Role == "system" {
			parts = append(parts, msg.Content)
		}
	}

	return strings.Join(parts, "\n\n")
}

// parseAntigravityCliResponse parses the JSON output from the Antigravity CLI.
func (p *AntigravityCliProvider) parseAntigravityCliResponse(output string) (*LLMResponse, error) {
	var resp antigravityCliJSONResponse
	if err := json.Unmarshal([]byte(output), &resp); err != nil {
		return nil, fmt.Errorf("failed to parse antigravity cli response: %w", err)
	}

	status := buildAntigravityCliStatus(&resp)

	// Success is carried by Status; there is no is_error flag. An unrecognised
	// status is treated as failure rather than assumed good — a new terminal
	// state should surface, not be silently reported as a completed turn.
	if !strings.EqualFold(resp.Status, antigravityStatusSuccess) {
		detail := strings.TrimSpace(resp.Response)
		if detail == "" {
			detail = "no response text"
		}
		return &LLMResponse{Status: status},
			fmt.Errorf("antigravity cli status %q: %s", resp.Status, detail)
	}

	// A turn that produced no text because its tool calls were refused is
	// reported as SUCCESS, which would otherwise reach the user as an empty
	// reply with nothing to diagnose. Name the denied action instead.
	if strings.TrimSpace(resp.Response) == "" && len(resp.DeniedActions) > 0 {
		return &LLMResponse{Status: status},
			fmt.Errorf("antigravity cli denied %s and produced no answer: "+
				"add --dangerously-skip-permissions to the model's extra_args, "+
				"or an allow-rule in the CLI's own settings",
				strings.Join(deniedActionNames(resp.DeniedActions), ", "))
	}

	// CLI is itself agentic — its `response` is the final assistant text. We do
	// NOT extract tool calls from this text: the agent loop must treat each CLI
	// invocation as one complete round.
	content := resp.Response

	var usage *UsageInfo
	if resp.Usage.InputTokens > 0 || resp.Usage.OutputTokens > 0 {
		// Cached reads are prompt tokens that were served from cache; they are
		// reported separately, so add them back to get what the model actually
		// read. Thinking tokens are billed output.
		promptTokens := resp.Usage.InputTokens + resp.Usage.CacheReadTokens
		completionTokens := resp.Usage.OutputTokens + resp.Usage.ThinkingTokens
		total := resp.Usage.TotalTokens
		if total == 0 {
			total = promptTokens + completionTokens
		}
		usage = &UsageInfo{
			PromptTokens:     promptTokens,
			CompletionTokens: completionTokens,
			TotalTokens:      total,
		}
	}

	result := &LLMResponse{
		Content:      strings.TrimSpace(content),
		FinishReason: "stop",
		Normal:       true,
		Usage:        usage,
		Status:       status,
	}

	logger.DebugCF("provider", "antigravity-cli response",
		map[string]any{
			"status":        resp.Status,
			"num_turns":     resp.NumTurns,
			"content_chars": len(strings.TrimSpace(content)),
		})

	return result, nil
}

// buildAntigravityCliStatus constructs a DispatchStatus from the parsed Antigravity CLI
// response. BytesSent / BytesReceived are filled in by Chat(). The envelope has
// no per-model breakdown, so Model is taken from the model field when present.
func buildAntigravityCliStatus(resp *antigravityCliJSONResponse) *DispatchStatus {
	success := strings.EqualFold(resp.Status, antigravityStatusSuccess)
	stopReason := "stop"
	if !success {
		stopReason = "error"
	}
	return &DispatchStatus{
		Success:         success,
		Model:           resp.Model,
		NumTurns:        resp.NumTurns,
		InputTokens:     resp.Usage.InputTokens,
		OutputTokens:    resp.Usage.OutputTokens,
		CacheReadTokens: resp.Usage.CacheReadTokens,
		StopReason:      stopReason,
		DurationMs:      int64(resp.DurationSeconds * 1000),
	}
}

// antigravityCliJSONResponse represents the JSON output from
// `agy --input-format text --output-format json`, e.g.
//
//	{"conversation_id":"…","status":"SUCCESS","response":"ok\n",
//	 "duration_seconds":0.895,"num_turns":1,
//	 "usage":{"input_tokens":5356,"output_tokens":27,"thinking_tokens":26,
//	          "cache_read_tokens":8129,"total_tokens":5383}}
//
// Unlike the Claude and Cursor CLIs there is no is_error flag: success is
// carried by Status, and anything other than "SUCCESS" is a failure.
type antigravityCliJSONResponse struct {
	ConversationID  string                  `json:"conversation_id"`
	Status          string                  `json:"status"`
	Response        string                  `json:"response"`
	DurationSeconds float64                 `json:"duration_seconds"`
	NumTurns        int                     `json:"num_turns"`
	Model           string                  `json:"model"`
	Usage           antigravityCliUsageInfo `json:"usage"`
	// DeniedActions lists tool calls the CLI refused. Headless mode cannot
	// prompt for approval, so without --dangerously-skip-permissions (or an
	// allow-rule in the CLI's own settings) it auto-denies and returns
	// status SUCCESS with an empty response — a completed turn that says
	// nothing. Reported so the caller sees a reason instead of silence.
	DeniedActions []antigravityCliDeniedAction `json:"denied_actions"`
}

// antigravityCliDeniedAction is one refused tool call.
type antigravityCliDeniedAction struct {
	Action      string `json:"action"`
	DisplayName string `json:"display_name"`
}

// antigravityCliUsageInfo is the token accounting from an Antigravity CLI
// response. ThinkingTokens are billed output the model spent reasoning; they
// are counted toward completion so cost accounting is not understated.
type antigravityCliUsageInfo struct {
	InputTokens     int `json:"input_tokens"`
	OutputTokens    int `json:"output_tokens"`
	ThinkingTokens  int `json:"thinking_tokens"`
	CacheReadTokens int `json:"cache_read_tokens"`
	TotalTokens     int `json:"total_tokens"`
}

// deniedActionNames returns the human names of refused actions, falling back to
// the machine name when the CLI supplies no display name.
func deniedActionNames(actions []antigravityCliDeniedAction) []string {
	out := make([]string, 0, len(actions))
	for _, a := range actions {
		if a.DisplayName != "" {
			out = append(out, a.DisplayName)
			continue
		}
		out = append(out, a.Action)
	}
	return out
}

// antigravityStatusSuccess is the only status the CLI reports for a completed
// turn.
const antigravityStatusSuccess = "SUCCESS"

// Workspace returns the working directory the CLI subprocess runs in.
func (p *AntigravityCliProvider) Workspace() string { return p.workspace }
