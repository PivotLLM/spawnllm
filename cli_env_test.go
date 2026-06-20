// ClawEh
// License: MIT

package spawnllm

import (
	"os"
	"strings"
	"testing"
)

func TestApplyProviderEnv_Nil(t *testing.T) {
	if got := applyProviderEnv(nil); got != nil {
		t.Errorf("expected nil for nil map, got %v", got)
	}
	if got := applyProviderEnv(map[string]string{}); got != nil {
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

	got := applyProviderEnv(map[string]string{
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
	got := applyProviderEnv(map[string]string{
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
