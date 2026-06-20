// ClawEh
// License: MIT

package spawnllm

import (
	"os"
	"sort"
)

// applyProviderEnv returns the environment to use for a CLI subprocess.
// It starts from os.Environ() and appends "K=V" entries from extra, so
// per-model values win over values already set in claw's process env.
// Returns nil when extra is empty so the caller can leave cmd.Env unset
// (Go's default: inherit the parent environment).
func applyProviderEnv(extra map[string]string) []string {
	if len(extra) == 0 {
		return nil
	}
	env := os.Environ()
	keys := make([]string, 0, len(extra))
	for k := range extra {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		env = append(env, k+"="+extra[k])
	}
	return env
}
