package spawnllm

import "github.com/PivotLLM/spawnllm/common"

// TextDeltaOption is the options-map key a caller sets to opt into token
// streaming for providers that support it (openai_compat, openai_responses).
// The value must be a TextDeltaFunc. When the key is absent or the value is
// nil, providers use their existing single-shot code path unchanged.
//
//	resp, err := provider.Chat(ctx, msgs, tools, model, map[string]any{
//	    spawnllm.TextDeltaOption: spawnllm.TextDeltaFunc(func(delta string) {
//	        fmt.Print(delta)
//	    }),
//	})
const TextDeltaOption = common.TextDeltaOption

// TextDeltaFunc receives each non-empty assistant text delta in arrival order
// as the stream is consumed. See common.TextDeltaFunc for semantics.
type TextDeltaFunc = common.TextDeltaFunc
