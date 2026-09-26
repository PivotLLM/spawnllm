// ClawEh
// License: MIT

package spawnllm

import (
	"os"
	"slices"
	"strings"
	"testing"
)

func TestApplyProviderEnv_Nil(t *testing.T) {
	if got := applyProviderEnv(nil, nil); got != nil {
		t.Errorf("expected nil for nil map, got %v", got)
	}
	if got := applyProviderEnv(nil, map[string]string{}); got != nil {
		t.Errorf("expected nil for empty map, got %v", got)
	}
}

func TestApplyProviderEnv_AppendsAfterOSEnviron(t *testing.T) {
	// Set a known var in our process env so we can verify the override order.
	key := "CLAW_TEST_APPLY_ENV_KEY"
	if err := os.Setenv(key, "from-parent"); err != nil {
		t.Fatalf("setenv: %v", err)
	}
	defer os.Unsetenv(key)

	got := applyProviderEnv(nil, map[string]string{
		key:                 "from-model",
		"CLAW_TEST_NEW_KEY": "fresh",
	})

	var parentIdx, modelIdx, freshIdx int = -1, -1, -1
	for i, kv := range got {
		switch {
		case kv == key+"=from-parent":
			parentIdx = i
		case kv == key+"=from-model":
			modelIdx = i
		case kv == "CLAW_TEST_NEW_KEY=fresh":
			freshIdx = i
		}
	}

	if parentIdx == -1 {
		t.Error("expected parent env entry to be present")
	}
	if modelIdx == -1 {
		t.Fatal("expected per-model override entry to be present")
	}
	if freshIdx == -1 {
		t.Error("expected new per-model key to be present")
	}
	// Per-model entries must come AFTER os.Environ entries so exec.Cmd
	// (which uses the last occurrence for duplicates) picks them.
	if modelIdx < parentIdx {
		t.Errorf("per-model override at %d appears before parent entry at %d — override would not win", modelIdx, parentIdx)
	}
}

func TestApplyProviderEnv_SortedForDeterminism(t *testing.T) {
	got := applyProviderEnv(nil, map[string]string{
		"ZZZ": "1",
		"AAA": "2",
		"MMM": "3",
	})
	// Collect only the appended entries (everything past os.Environ).
	base := len(os.Environ())
	if len(got) < base+3 {
		t.Fatalf("expected at least %d entries, got %d", base+3, len(got))
	}
	appended := got[base:]
	wantOrder := []string{"AAA=", "MMM=", "ZZZ="}
	for i, prefix := range wantOrder {
		if !strings.HasPrefix(appended[i], prefix) {
			t.Errorf("appended[%d] = %q, want prefix %q", i, appended[i], prefix)
		}
	}
}

// A supplied base replaces os.Environ() entirely: the host's own variables do
// not reach the subprocess, and the per-model overlay still comes last.
func TestApplyProviderEnv_BaseReplacesOSEnviron(t *testing.T) {
	t.Setenv("CLAW_TEST_HOST_SECRET", "secret")

	base := []string{"PATH=/usr/bin", "HOME=/home/alice"}
	got := applyProviderEnv(base, map[string]string{"HOME": "/srv/model", "CODEX_HOME": "/srv/codex"})

	want := []string{"PATH=/usr/bin", "HOME=/home/alice", "CODEX_HOME=/srv/codex", "HOME=/srv/model"}
	if !slices.Equal(got, want) {
		t.Errorf("got %v, want %v", got, want)
	}
	if !slices.Equal(base, []string{"PATH=/usr/bin", "HOME=/home/alice"}) {
		t.Errorf("base was modified: %v", base)
	}

	// An explicit empty base means "no environment at all", not "inherit".
	if got := applyProviderEnv([]string{}, nil); got == nil || len(got) != 0 {
		t.Errorf("empty base: got %v, want empty non-nil slice", got)
	}
}

// Every CLI provider accepts a base environment through the shared interface.
func TestCLIProviders_ImplementBaseEnvSetter(t *testing.T) {
	providers := []BaseEnvSetter{
		NewClaudeCliProvider("", "", nil, nil),
		NewCodexCliProvider("", "", nil, nil),
		NewCursorCliProvider("", "", nil, nil),
		NewAntigravityCliProvider("", "", nil, nil),
	}
	base := []string{"PATH=/usr/bin"}
	for _, p := range providers {
		p.SetBaseEnv(base)
	}
	if got := providers[0].(*ClaudeCliProvider).baseEnv; !slices.Equal(got, base) {
		t.Errorf("claude baseEnv = %v, want %v", got, base)
	}
	if got := providers[1].(*CodexCliProvider).baseEnv; !slices.Equal(got, base) {
		t.Errorf("codex baseEnv = %v, want %v", got, base)
	}
	if got := providers[2].(*CursorCliProvider).baseEnv; !slices.Equal(got, base) {
		t.Errorf("cursor baseEnv = %v, want %v", got, base)
	}
	if got := providers[3].(*AntigravityCliProvider).baseEnv; !slices.Equal(got, base) {
		t.Errorf("antigravity baseEnv = %v, want %v", got, base)
	}
}
