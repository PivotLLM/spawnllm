package common

import "github.com/PivotLLM/spawnllm/logger"

// MinThinkingBudget is the smallest budget_tokens the Anthropic Messages API
// accepts for extended thinking.
const MinThinkingBudget = 1024

// ThinkingBudget maps a thinking_level to Anthropic's budget_tokens. It returns
// 0 for "off", "adaptive" (which has no budget) and unknown levels.
//
//	low    =  4,096  — simple reasoning, quick debugging
//	medium = 16,384  — Anthropic's recommended sweet spot for most tasks
//	high   = 32,000  — complex architecture, deep analysis
//	xhigh  = 64,000  — extreme reasoning, research problems
func ThinkingBudget(level string) int64 {
	switch level {
	case "low":
		return 4096
	case "medium":
		return 16384
	case "high":
		return 32000
	case "xhigh":
		return 64000
	default:
		return 0
	}
}

// ClampThinkingBudget fits budget below maxTokens, as the API requires
// budget_tokens < max_tokens. It reports false when no valid budget fits
// (max_tokens too small for the API minimum), in which case thinking must not
// be sent. component names the caller in log lines.
func ClampThinkingBudget(budget, maxTokens int64, component string) (int64, bool) {
	if budget <= 0 {
		return 0, false
	}
	if budget >= maxTokens {
		logger.WarnCF(component, "budget_tokens clamped to max_tokens-1",
			map[string]any{"budget_tokens": budget, "clamped_to": maxTokens - 1})
		budget = maxTokens - 1
	} else if budget > maxTokens*80/100 {
		logger.WarnCF(component, "thinking budget exceeds 80% of max_tokens, output may be truncated",
			map[string]any{"budget_tokens": budget, "max_tokens": maxTokens})
	}
	if budget < MinThinkingBudget {
		logger.WarnCF(component, "thinking not sent: max_tokens is too small for the minimum thinking budget",
			map[string]any{"max_tokens": maxTokens, "min_budget_tokens": MinThinkingBudget})
		return 0, false
	}
	return budget, true
}
