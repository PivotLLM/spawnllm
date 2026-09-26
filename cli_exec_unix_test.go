//go:build !windows

package spawnllm

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"
)

// killSlack is the tolerance added to the timeout+WaitDelay bound. The bound
// is a hard property of the helper, so the slack only covers scheduler noise.
const killSlack = 2 * time.Second

// writeFakeCLI writes an executable sh script and returns its path.
func writeFakeCLI(t *testing.T, body string) string {
	t.Helper()
	script := filepath.Join(t.TempDir(), "cli")
	if err := os.WriteFile(script, []byte("#!/bin/sh\n"+body), 0o755); err != nil {
		t.Fatal(err)
	}
	return script
}

// readPID reads a pid written by a fake CLI, waiting briefly for the file to
// appear because the script writes it asynchronously with the provider call.
func readPID(t *testing.T, path string) int {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for {
		b, err := os.ReadFile(path)
		if err == nil && strings.TrimSpace(string(b)) != "" {
			pid, convErr := strconv.Atoi(strings.TrimSpace(string(b)))
			if convErr != nil {
				t.Fatalf("pid file %q: %v", string(b), convErr)
			}
			return pid
		}
		if time.Now().After(deadline) {
			t.Fatalf("pid file %s never written", path)
		}
		time.Sleep(20 * time.Millisecond)
	}
}

// processGone reports whether pid no longer exists, polling for up to wait so
// the kernel has time to reap the orphan after SIGKILL.
func processGone(pid int, wait time.Duration) bool {
	deadline := time.Now().Add(wait)
	for {
		err := syscall.Kill(pid, 0)
		if errors.Is(err, syscall.ESRCH) {
			return true
		}
		if time.Now().After(deadline) {
			return false
		}
		time.Sleep(20 * time.Millisecond)
	}
}

// TestCLI_TimeoutKillsProcessGroup runs a fake CLI that spawns a grandchild
// holding stdout and then blocks. On timeout the provider must return within
// timeout+WaitDelay and the grandchild must be gone.
func TestCLI_TimeoutKillsProcessGroup(t *testing.T) {
	const timeout = 500 * time.Millisecond

	tests := []struct {
		name       string
		ignoreTerm bool
		// minElapsed distinguishes the two exit paths: a grandchild that
		// honours SIGTERM lets Wait return right after the deadline; one that
		// ignores it holds the stdout pipe until WaitDelay closes it, after
		// which only the group SIGKILL can remove it.
		minElapsed time.Duration
		maxElapsed time.Duration
	}{
		{
			name:       "grandchild honours SIGTERM",
			minElapsed: timeout,
			maxElapsed: timeout + cliWaitDelay/2,
		},
		{
			name:       "grandchild ignores SIGTERM and holds stdout",
			ignoreTerm: true,
			minElapsed: timeout + cliWaitDelay,
			maxElapsed: timeout + cliWaitDelay + killSlack,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			pidFile := filepath.Join(t.TempDir(), "grandchild.pid")
			spawn := "sleep 300 &"
			if tc.ignoreTerm {
				// An ignored disposition survives exec, so sleep ignores TERM too.
				spawn = `sh -c 'trap "" TERM; exec sleep 300' &`
			}
			script := writeFakeCLI(t, fmt.Sprintf("%s\necho $! > '%s'\nwait\n", spawn, pidFile))

			p := NewClaudeCliProviderWithTimeout(script, t.TempDir(), timeout, nil, nil)

			started := time.Now()
			resp, err := p.Chat(context.Background(), []Message{{Role: "user", Content: "hi"}}, nil, "", nil)
			elapsed := time.Since(started)

			if !errors.Is(err, context.DeadlineExceeded) {
				t.Fatalf("err = %v, want DeadlineExceeded", err)
			}
			if resp == nil || resp.Status == nil || resp.Status.StopReason != "timeout" {
				t.Fatalf("resp.Status = %+v, want StopReason timeout", resp)
			}
			if elapsed < tc.minElapsed || elapsed > tc.maxElapsed {
				t.Fatalf("Chat returned after %s, want within [%s, %s]", elapsed, tc.minElapsed, tc.maxElapsed)
			}

			grandchild := readPID(t, pidFile)
			if !processGone(grandchild, 3*time.Second) {
				_ = syscall.Kill(grandchild, syscall.SIGKILL)
				t.Fatalf("grandchild %d survived the timeout", grandchild)
			}
		})
	}
}

// TestCLI_ExitWithPipesHeldOpen covers a CLI that exits successfully but leaves
// a descendant holding stdout: the answer must still be returned once WaitDelay
// releases Wait, and the descendant must be killed.
func TestCLI_ExitWithPipesHeldOpen(t *testing.T) {
	pidFile := filepath.Join(t.TempDir(), "grandchild.pid")
	script := writeFakeCLI(t, fmt.Sprintf(`sleep 300 &
echo $! > '%s'
echo '{"type":"result","result":"done","session_id":"t"}'
exit 0
`, pidFile))

	p := NewClaudeCliProviderWithTimeout(script, t.TempDir(), time.Minute, nil, nil)

	started := time.Now()
	resp, err := p.Chat(context.Background(), []Message{{Role: "user", Content: "hi"}}, nil, "", nil)
	elapsed := time.Since(started)

	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if resp.Content != "done" {
		t.Fatalf("Content = %q, want done", resp.Content)
	}
	if elapsed > cliWaitDelay+killSlack {
		t.Fatalf("Chat returned after %s, want within WaitDelay %s", elapsed, cliWaitDelay+killSlack)
	}

	grandchild := readPID(t, pidFile)
	if !processGone(grandchild, 3*time.Second) {
		_ = syscall.Kill(grandchild, syscall.SIGKILL)
		t.Fatalf("grandchild %d survived after the CLI exited", grandchild)
	}
}

// TestCLI_StdoutCapExceeded runs a fake CLI that writes past the stdout cap.
// The provider must drain the pipe, return an error naming the cap, and record
// the true byte count.
func TestCLI_StdoutCapExceeded(t *testing.T) {
	overflow := cliStdoutCapBytes + 4096
	script := writeFakeCLI(t, fmt.Sprintf("head -c %d /dev/zero\n", overflow))

	p := NewClaudeCliProviderWithTimeout(script, t.TempDir(), time.Minute, nil, nil)

	resp, err := p.Chat(context.Background(), []Message{{Role: "user", Content: "hi"}}, nil, "", nil)
	if !errors.Is(err, errCLIOutputCapExceeded) {
		t.Fatalf("err = %v, want errCLIOutputCapExceeded", err)
	}
	if !strings.Contains(err.Error(), "64 MiB") {
		t.Fatalf("error should name the cap: %q", err.Error())
	}
	if resp == nil || resp.Status == nil {
		t.Fatal("expected a status-bearing error response")
	}
	if resp.Status.StopReason != "output_cap" {
		t.Fatalf("StopReason = %q, want output_cap", resp.Status.StopReason)
	}
	if resp.Status.BytesReceived != int64(overflow) {
		t.Fatalf("BytesReceived = %d, want %d (pipe must be fully drained)", resp.Status.BytesReceived, overflow)
	}
}
