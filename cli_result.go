package spawnllm

import (
	"context"
	"errors"
	"fmt"
	"os/exec"
	"strings"
	"time"

	"github.com/PivotLLM/spawnllm/logger"
)

// jsonCLICall describes one Chat call to a CLI that reads its prompt on stdin
// and answers with a single JSON document on stdout. The claude, cursor and
// antigravity providers share this path; they differ only in their arguments
// and in how stdout is parsed.
type jsonCLICall struct {
	// label names the CLI in returned errors ("<label> cli …") and in log
	// messages ("<label>-cli …").
	label     string
	command   string
	workspace string // working directory; empty keeps the caller's
	timeout   time.Duration
	args      []string
	prompt    string
	baseEnv   []string
	env       map[string]string
	model     string
	parse     func(stdout string) (*LLMResponse, error)
}

// runJSONCLI runs c.command with c.args, piping c.prompt on stdin, bounded by
// c.timeout when it is set, and turns the outcome into the provider response.
// The provider's environment is layered over c.baseEnv (see applyProviderEnv).
// A CLI that exits non-zero but still wrote a usable response is treated as a
// success.
func runJSONCLI(ctx context.Context, c *jsonCLICall) (*LLMResponse, error) {
	if c.timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, c.timeout)
		defer cancel()
	}

	cmd := newCLICommand(ctx, c.command, c.args...)
	if c.workspace != "" {
		cmd.Dir = c.workspace
	}
	cmd.Stdin = strings.NewReader(c.prompt)
	cmd.Env = applyProviderEnv(c.baseEnv, c.env)

	bytesSent := int64(len(c.prompt))
	run, runErr := runCLI(ctx, cmd)
	return jsonCLIResult(ctx, c, bytesSent, run, runErr)
}

// jsonCLIResult maps a finished run of c to the provider response and error.
func jsonCLIResult(ctx context.Context, c *jsonCLICall, bytesSent int64, run *cliRun, runErr error) (*LLMResponse, error) {
	label, model := c.label, c.model
	stdout, stderr, elapsed := run.stdout, run.stderr, run.elapsed
	bytesReceived := stdout.Received()
	if errors.Is(runErr, errCLIOutputCapExceeded) {
		return cliErrorResponse(model, "output_cap", elapsed, bytesSent, bytesReceived),
			fmt.Errorf("%s cli: %w", label, runErr)
	}

	if runErr != nil {
		if ctx.Err() == context.DeadlineExceeded {
			return cliErrorResponse(model, "timeout", elapsed, bytesSent, bytesReceived),
				fmt.Errorf("%s cli timed out after %s: %w", label, c.timeout, context.DeadlineExceeded)
		}
		if ctx.Err() == context.Canceled {
			return cliErrorResponse(model, "canceled", elapsed, bytesSent, bytesReceived), ctx.Err()
		}

		// Attempt to parse stdout before treating as error — the CLI may exit
		// non-zero but still write a valid JSON response to stdout.
		if stdoutStr := strings.TrimSpace(stdout.String()); stdoutStr != "" {
			if resp, parseErr := c.parse(stdoutStr); parseErr == nil && resp.Content != "" {
				exitCode := -1
				if exitErr, ok := errors.AsType[*exec.ExitError](runErr); ok {
					exitCode = exitErr.ExitCode()
				}
				if resp.Status != nil {
					resp.Status.BytesSent = bytesSent
					resp.Status.BytesReceived = bytesReceived
				}
				logger.WarnCF("provider", label+"-cli exited non-zero but returned valid content",
					map[string]any{"exit_code": exitCode})
				return resp, nil
			}
		}

		exitCode := -1
		if exitErr, ok := errors.AsType[*exec.ExitError](runErr); ok {
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
		logger.ErrorCF("provider", label+"-cli subprocess failed", fields)
		errResp := cliErrorResponse(model, "error", elapsed, bytesSent, bytesReceived)
		switch {
		case stderrStr != "" && stdoutStr != "":
			return errResp, fmt.Errorf("%s cli error: %w\nstderr: %s\nstdout: %s", label, runErr, stderrStr, stdoutStr)
		case stderrStr != "":
			return errResp, fmt.Errorf("%s cli error: %s", label, stderrStr)
		case stdoutStr != "":
			return errResp, fmt.Errorf("%s cli error: %w\noutput: %s", label, runErr, stdoutStr)
		default:
			return errResp, fmt.Errorf("%s cli error: %w", label, runErr)
		}
	}

	// Log non-empty stderr on successful exit.
	if stderrStr := strings.TrimSpace(stderr.String()); stderrStr != "" {
		logger.WarnCF("provider", label+"-cli wrote to stderr on successful exit",
			map[string]any{"stderr": stderrStr})
	}

	resp, parseErr := c.parse(stdout.String())
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
		logger.WarnCF("provider", label+"-cli returned empty content", warnFields)
	}
	return resp, nil
}
