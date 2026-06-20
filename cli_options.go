package spawnllm

import (
	"github.com/PivotLLM/spawnllm/logger"
	"github.com/PivotLLM/spawnllm/openai_compat"
)

// JSONObjectFortification is the prompt directive appended to the CLI-provider
// stdin payload when the caller sets ResponseFormatJSONObjectOption=true. CLI
// subprocesses do not expose a structured response_format flag, so JSON-only
// output is enforced via hard prompt language — the only available mechanism.
//
// Exported so tests in this package and downstream call sites can assert on the
// exact directive without copy-pasting the string.
const JSONObjectFortification = "\n\nIMPORTANT OUTPUT CONSTRAINT: Your entire response MUST be a single JSON object. Start with { and end with }. No prose before or after, no markdown fences, no preamble, no explanation, no refusal."

// wantsJSONObjectOption mirrors openai_compat.wantsJSONObject without
// re-exporting it: callers may set the option as bool true or string "true".
func wantsJSONObjectOption(options map[string]any) bool {
	if options == nil {
		return false
	}
	switch v := options[openai_compat.ResponseFormatJSONObjectOption].(type) {
	case bool:
		return v
	case string:
		return v == "true"
	}
	return false
}

// applyCLIOptions translates the provider options map into adjustments the CLI
// subprocess can honour. It returns the (possibly fortified) prompt. Options
// the provider recognises but cannot translate to a CLI flag are logged at DBG
// so the parity break that motivated this helper cannot recur silently.
//
// providerName is included in the log context only.
func applyCLIOptions(providerName, prompt string, options map[string]any) string {
	if wantsJSONObjectOption(options) {
		prompt += JSONObjectFortification
	}
	for k := range options {
		if k == openai_compat.ResponseFormatJSONObjectOption {
			continue
		}
		logger.DebugCF("provider", "cli provider has no equivalent for option",
			map[string]any{
				"provider": providerName,
				"option":   k,
			})
	}
	return prompt
}
