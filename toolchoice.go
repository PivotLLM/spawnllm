package spawnllm

import "github.com/PivotLLM/spawnllm/common"

// ToolChoiceOption is the options-map key a caller sets to choose whether the
// model may call tools on this request. The value is ToolChoiceAuto (the
// default when the key is absent or holds any other value) or ToolChoiceNone,
// which keeps the tools declared (so prompt caches and earlier tool calls in
// the history stay valid) but makes the model answer in text. It applies only
// when the request carries tools, and only to the HTTP providers
// (anthropic_messages, openai_compat, azure, openai_responses); CLI providers
// ignore it.
//
//	resp, err := provider.Chat(ctx, msgs, tools, model, map[string]any{
//	    spawnllm.ToolChoiceOption: spawnllm.ToolChoiceNone,
//	})
const ToolChoiceOption = common.ToolChoiceOption

const (
	// ToolChoiceAuto lets the model decide whether to call a tool.
	ToolChoiceAuto = common.ToolChoiceAuto
	// ToolChoiceNone forbids tool calls on the request.
	ToolChoiceNone = common.ToolChoiceNone
)
