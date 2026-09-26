// ClawEh
// License: MIT

package spawnllm

import (
	"os"
	"sort"
)

// BaseEnvSetter is implemented by the CLI providers. SetBaseEnv replaces the
// environment the CLI subprocess starts from. By default that is the host
// process's own environment (os.Environ()), which hands the CLI every secret
// the host holds; a host that wants the CLI to see only an allowlisted subset
// passes that subset here. The per-model env is still overlaid on top, and
// passing nil restores the default.
type BaseEnvSetter interface {
	SetBaseEnv(env []string)
}

// applyProviderEnv returns the environment to use for a CLI subprocess.
// It starts from base — os.Environ() when base is nil — and appends "K=V"
// entries from extra in sorted key order, so per-model values win over values
// already present (exec.Cmd uses the last occurrence of a duplicate key).
// Returns nil when base is nil and extra is empty so the caller can leave
// cmd.Env unset (Go's default: inherit the parent environment).
func applyProviderEnv(base []string, extra map[string]string) []string {
	if base == nil && len(extra) == 0 {
		return nil
	}
	if base == nil {
		base = os.Environ()
	}
	env := make([]string, 0, len(base)+len(extra))
	env = append(env, base...)
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
