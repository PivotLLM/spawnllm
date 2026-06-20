package spawnllm

import (
	"strings"
	"testing"
)

// Tests for resolveCodexAuthPath edge cases.

func TestResolveCodexAuthPath_WithCODEX_HOME(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("CODEX_HOME", dir)

	path, err := resolveCodexAuthPath()
	if err != nil {
		t.Fatalf("resolveCodexAuthPath() error = %v", err)
	}
	if !strings.HasPrefix(path, dir) {
		t.Errorf("path = %q, want prefix %q", path, dir)
	}
	if !strings.HasSuffix(path, "auth.json") {
		t.Errorf("path = %q, want suffix 'auth.json'", path)
	}
}

func TestResolveCodexAuthPath_WithoutCODEX_HOME(t *testing.T) {
	t.Setenv("CODEX_HOME", "")

	path, err := resolveCodexAuthPath()
	if err != nil {
		t.Fatalf("resolveCodexAuthPath() error = %v (may fail if no home dir)", err)
	}
	if !strings.Contains(path, ".codex") {
		t.Errorf("path = %q, want '.codex' in path", path)
	}
	if !strings.HasSuffix(path, "auth.json") {
		t.Errorf("path = %q, want suffix 'auth.json'", path)
	}
}
