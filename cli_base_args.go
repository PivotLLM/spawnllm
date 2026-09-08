package spawnllm

// Base arguments for the CLI providers.
//
// Each CLI is driven headlessly: print the answer and encode it as JSON, read
// the prompt from stdin. Those flags are not configurable and never were, which
// made them invisible — a caller inspecting its configuration saw only the
// permission flags and had no way to learn what else was being run.
//
// Exported so a caller can show the operator the full command line rather than
// the part that happens to live in config. Each provider builds its arguments
// from these, so the two cannot drift.

// ClaudeCliBaseArgs are the arguments the Claude CLI is always invoked with.
// A trailing "-" is appended at call time to read the prompt from stdin, after
// any extra arguments and the optional --model.
func ClaudeCliBaseArgs() []string {
	return []string{"-p", "--output-format", "json"}
}

// CodexCliBaseArgs are the arguments the Codex CLI is always invoked with.
func CodexCliBaseArgs() []string {
	return []string{"exec", "--json", "--color", "never"}
}

// AntigravityCliBaseArgs are the arguments the Antigravity CLI is always
// invoked with. Note the absence of -p/--print: with either, agy reads the
// prompt from argv and ignores stdin.
func AntigravityCliBaseArgs() []string {
	return []string{"--input-format", "text", "--output-format", "json"}
}

// CursorCliBaseArgs are the arguments the Cursor CLI is always invoked with.
func CursorCliBaseArgs() []string {
	return []string{"-p", "--output-format", "json"}
}

// StdinArg is the trailing argument the Claude and Codex CLIs take to read the
// prompt from stdin.
const StdinArg = "-"
