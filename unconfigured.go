package spawnllm

import (
	"context"
	"fmt"
)

// UnconfiguredProvider is used when no model is configured at gateway startup.
// It returns a user-friendly error for any chat request, allowing the gateway
// to start and serve the config UI while no model is set up yet.
type UnconfiguredProvider struct{}

func NewUnconfiguredProvider() *UnconfiguredProvider {
	return &UnconfiguredProvider{}
}

func (u *UnconfiguredProvider) Chat(ctx context.Context, messages []Message, tools []ToolDefinition, model string, options map[string]any) (*LLMResponse, error) {
	return nil, fmt.Errorf("no model configured — enable a model and add your API key in the configuration")
}

func (u *UnconfiguredProvider) GetDefaultModel() string {
	return ""
}
