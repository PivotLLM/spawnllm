package spawnllm

import (
	"testing"

	"github.com/PivotLLM/spawnllm/azure"
)

// Tests for azure sub-package functions not covered.

func TestAzureProvider_GetDefaultModel(t *testing.T) {
	p := azure.NewProvider("test-key", "https://example.openai.azure.com", "")
	model := p.GetDefaultModel()
	// Azure deployments are user-configured; GetDefaultModel returns empty string.
	if model != "" {
		t.Errorf("GetDefaultModel() = %q, want empty (Azure uses deployment names)", model)
	}
}
