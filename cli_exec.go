package spawnllm

import (
	"context"
	"errors"
	"fmt"
	"os/exec"
	"time"

	"github.com/PivotLLM/spawnllm/logger"
)

// Subprocess execution shared by the CLI providers.
//
// A CLI such as claude or codex spawns its own children (MCP servers, shells).
// Killing only the CLI binary on timeout orphans those children, and a
// grandchild that inherited the stdout pipe keeps cmd.Wait blocked past the
// deadline. newCLICommand therefore places the CLI in its own process group
// (unix), runCLI signals the whole group on cancellation, bounds how long Wait
// may linger on open pipes, and caps how much output is retained in memory.

const (
	// cliStdoutCapBytes bounds the stdout retained from a CLI invocation.
	cliStdoutCapBytes = 64 << 20
	// cliStderrCapBytes bounds the stderr retained from a CLI invocation.
	cliStderrCapBytes = 1 << 20
	// cliWaitDelay bounds how long Wait blocks after the context is done or the
	// CLI has exited while a descendant still holds the output pipes. When it
	// elapses the pipes are closed and the process group receives SIGKILL.
	cliWaitDelay = 5 * time.Second
)

// errCLIOutputCapExceeded reports that a CLI wrote more output than the
// provider retains. Callers match it with errors.Is.
var errCLIOutputCapExceeded = errors.New("cli output exceeded retention cap")

// cappedBuffer retains at most limit bytes of what is written to it. Writes past
// the limit are counted and discarded rather than refused, so the child's pipe
// keeps draining and the child never blocks on a full pipe.
type cappedBuffer struct {
	limit    int
	buf      []byte
	received int64
	exceeded bool
}

func newCappedBuffer(limit int) *cappedBuffer {
	return &cappedBuffer{limit: limit}
}

// Write implements io.Writer. It never returns an error.
func (b *cappedBuffer) Write(p []byte) (int, error) {
	b.received += int64(len(p))
	if b.exceeded {
		return len(p), nil
	}
	room := b.limit - len(b.buf)
	if len(p) > room {
		b.buf = append(b.buf, p[:room]...)
		b.exceeded = true
		return len(p), nil
	}
	b.buf = append(b.buf, p...)
	return len(p), nil
}

// String returns the retained output.
func (b *cappedBuffer) String() string { return string(b.buf) }

// Len returns the number of retained bytes.
func (b *cappedBuffer) Len() int { return len(b.buf) }

// Received returns the total bytes written, including any that were discarded.
func (b *cappedBuffer) Received() int64 { return b.received }

// Exceeded reports whether writes went past the retention limit.
func (b *cappedBuffer) Exceeded() bool { return b.exceeded }

// cliRun holds the captured output and timing of one CLI invocation.
type cliRun struct {
	stdout  *cappedBuffer
	stderr  *cappedBuffer
	elapsed time.Duration
}

// newCLICommand builds the exec.Cmd for a CLI invocation. The command is tied
// to ctx, placed in its own process group where the platform supports it, and
// given a WaitDelay so a descendant holding the output pipes cannot block Wait
// indefinitely. Callers set Dir, Stdin and Env, then pass the command to runCLI.
func newCLICommand(ctx context.Context, name string, args ...string) *exec.Cmd {
	cmd := exec.CommandContext(ctx, name, args...) //nolint:gosec // running the host-configured CLI is this function's purpose; no shell is involved
	cmd.WaitDelay = cliWaitDelay
	setCLIProcessGroup(cmd)
	return cmd
}

// runCLI runs a command built by newCLICommand, capturing stdout and stderr
// up to cliStdoutCapBytes and cliStderrCapBytes. On context cancellation the
// process group is signalled to terminate; once Wait returns after a
// cancellation, or after the CLI exited but left its pipes open, any survivors
// in the group are killed. The returned error is the Wait error, or an error
// wrapping errCLIOutputCapExceeded when stdout went past its cap.
func runCLI(ctx context.Context, cmd *exec.Cmd) (*cliRun, error) {
	run := &cliRun{
		stdout: newCappedBuffer(cliStdoutCapBytes),
		stderr: newCappedBuffer(cliStderrCapBytes),
	}
	cmd.Stdout = run.stdout
	cmd.Stderr = run.stderr

	started := time.Now()
	runErr := cmd.Run()
	run.elapsed = time.Since(started)

	pipesHeldOpen := errors.Is(runErr, exec.ErrWaitDelay)
	if ctx.Err() != nil || pipesHeldOpen {
		// Wait has returned, so Go has already killed the CLI itself (or it
		// exited). Anything left in the group is a descendant that ignored or
		// never received the termination signal: finish it now.
		killCLIProcessGroup(cmd)
	}
	if pipesHeldOpen {
		// The CLI exited successfully; its output is complete even though a
		// descendant kept the pipes open until WaitDelay elapsed.
		logger.WarnCF("provider", "cli exited but a descendant held its output pipes open; killed process group",
			map[string]any{"command": cmd.Path, "wait_delay": cliWaitDelay.String()})
		runErr = nil
	}

	if run.stdout.Exceeded() {
		return run, fmt.Errorf("%w: stdout exceeded %d MiB", errCLIOutputCapExceeded, cliStdoutCapBytes>>20)
	}
	if run.stderr.Exceeded() {
		logger.WarnCF("provider", "cli stderr exceeded retention cap; output truncated",
			map[string]any{"command": cmd.Path, "cap_bytes": cliStderrCapBytes})
	}
	return run, runErr
}
