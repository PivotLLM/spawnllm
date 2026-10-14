package common

// AnthropicThinking maps a thinking_level to Anthropic adaptive thinking.
//
//	"adaptive"                                  → thinking {type: adaptive}, no effort (model default)
//	"low" | "medium" | "high" | "xhigh" | "max" → thinking {type: adaptive} + output_config {effort: <level>}
//	"", "off" or any other value                → no thinking and no output_config
//
// Budget thinking ({type: enabled, budget_tokens}) and {type: disabled} are
// never sent: current Claude models refuse them. on reports whether thinking
// is sent; effort is "" when no output_config effort is sent.
func AnthropicThinking(level string) (effort string, on bool) {
	switch level {
	case "adaptive":
		return "", true
	case "low", "medium", "high", "xhigh", "max":
		return level, true
	default:
		return "", false
	}
}
