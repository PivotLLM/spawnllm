// ClawEh - Personal AI Assistant
// Inspired by and based on nanobot: https://github.com/HKUDS/nanobot
// License: MIT
//
// Copyright (c) 2026 PicoClaw contributors

package spawnllm

import (
	"context"

	"github.com/PivotLLM/spawnllm/openai_compat"
)

type HTTPProvider struct {
	delegate *openai_compat.Provider
}

func NewHTTPProviderWithOptions(apiKey, apiBase, proxy string, opts ...openai_compat.Option) *HTTPProvider {
	return &HTTPProvider{
		delegate: openai_compat.NewProvider(apiKey, apiBase, proxy, opts...),
	}
}

func (p *HTTPProvider) Chat(
	ctx context.Context,
	messages []Message,
	tools []ToolDefinition,
	model string,
	options map[string]any,
) (*LLMResponse, error) {
	return p.delegate.Chat(ctx, messages, tools, model, options)
}

func (p *HTTPProvider) GetDefaultModel() string {
	return ""
}

// Delegate returns the underlying openai_compat.Provider so tests can inspect
// per-entry state (ResponseLogFile, ReasoningEffort, ...) attached at
// construction time. Internal callers should keep going through the
// LLMProvider interface.
func (p *HTTPProvider) Delegate() *openai_compat.Provider {
	return p.delegate
}
