package spawnllm

import (
	"strings"
	"testing"
)

// --- Compile-time interface check ---

var _ LLMProvider = (*GeminiCliProvider)(nil)

// --- Constructor tests ---

func TestNewGeminiCliProvider(t *testing.T) {
	p := NewGeminiCliProvider("", "/test/workspace", nil, nil)
	if p == nil {
		t.Fatal("NewGeminiCliProvider returned nil")
	}
	if p.workspace != "/test/workspace" {
		t.Errorf("workspace = %q, want %q", p.workspace, "/test/workspace")
	}
	if p.command != "gemini" {
		t.Errorf("command = %q, want %q", p.command, "gemini")
	}
}

// --- GetDefaultModel tests ---

func TestGeminiCliProvider_GetDefaultModel(t *testing.T) {
	p := NewGeminiCliProvider("", "/workspace", nil, nil)
	if got := p.GetDefaultModel(); got != "gemini-cli" {
		t.Errorf("GetDefaultModel() = %q, want %q", got, "gemini-cli")
	}
}

// --- buildPrompt tests ---

func TestGeminiCliProvider_BuildPrompt_SingleUser(t *testing.T) {
	p := NewGeminiCliProvider("", "/workspace", nil, nil)
	messages := []Message{
		{Role: "user", Content: "Hello"},
	}
	got := p.buildPrompt(messages)
	// Single user message with no system or tools should be simplified (no prefix)
	want := "Hello"
	if got != want {
		t.Errorf("buildPrompt() = %q, want %q", got, want)
	}
}

func TestGeminiCliProvider_BuildPrompt_WithSystem(t *testing.T) {
	p := NewGeminiCliProvider("", "/workspace", nil, nil)
	messages := []Message{
		{Role: "system", Content: "You are a helpful assistant."},
		{Role: "user", Content: "What is Go?"},
	}
	got := p.buildPrompt(messages)
	if !strings.Contains(got, "## System Instructions") {
		t.Errorf("buildPrompt() missing ## System Instructions header, got %q", got)
	}
	if !strings.Contains(got, "You are a helpful assistant.") {
		t.Errorf("buildPrompt() missing system message content, got %q", got)
	}
	if !strings.Contains(got, "## Task") {
		t.Errorf("buildPrompt() missing ## Task header, got %q", got)
	}
	if !strings.Contains(got, "What is Go?") {
		t.Errorf("buildPrompt() missing user message content, got %q", got)
	}
}

// --- parseGeminiCliResponse tests ---

func TestGeminiCliProvider_ParseResponse_Basic(t *testing.T) {
	p := NewGeminiCliProvider("", "/workspace", nil, nil)
	output := `{
		"session_id": "abc123",
		"response": "Hello! How can I assist you?",
		"stats": {
			"models": {
				"gemini-2.5-flash-lite": {
					"tokens": {
						"input": 2634,
						"candidates": 29,
						"total": 2735
					}
				},
				"gemini-3-flash-preview": {
					"tokens": {
						"input": 16921,
						"candidates": 14,
						"total": 16935
					}
				}
			}
		}
	}`

	resp, err := p.parseGeminiCliResponse(output)
	if err != nil {
		t.Fatalf("parseGeminiCliResponse() error = %v", err)
	}
	if resp.Content != "Hello! How can I assist you?" {
		t.Errorf("Content = %q, want %q", resp.Content, "Hello! How can I assist you?")
	}
	if resp.FinishReason != "stop" {
		t.Errorf("FinishReason = %q, want %q", resp.FinishReason, "stop")
	}
	if len(resp.ToolCalls) != 0 {
		t.Errorf("ToolCalls len = %d, want 0", len(resp.ToolCalls))
	}
	if resp.Usage == nil {
		t.Fatal("Usage should not be nil")
	}
	// Summed input: 2634 + 16921 = 19555
	if resp.Usage.PromptTokens != 19555 {
		t.Errorf("PromptTokens = %d, want 19555", resp.Usage.PromptTokens)
	}
	// Summed candidates: 29 + 14 = 43
	if resp.Usage.CompletionTokens != 43 {
		t.Errorf("CompletionTokens = %d, want 43", resp.Usage.CompletionTokens)
	}
	// Summed total: 2735 + 16935 = 19670
	if resp.Usage.TotalTokens != 19670 {
		t.Errorf("TotalTokens = %d, want 19670", resp.Usage.TotalTokens)
	}
}

func TestGeminiCliProvider_ParseResponse_PassesThroughToolCallText(t *testing.T) {
	p := NewGeminiCliProvider("", "/workspace", nil, nil)
	// CLI output that previously would have been parsed as a tool call must now
	// pass through verbatim — the CLI is the agent and we treat its response
	// as final assistant prose for the round.
	output := `{"session_id":"s1","response":"Checking weather.\n{\"tool_calls\":[{\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"{\\\"location\\\":\\\"NYC\\\"}\"}}]}","stats":{}}`

	resp, err := p.parseGeminiCliResponse(output)
	if err != nil {
		t.Fatalf("parseGeminiCliResponse() error = %v", err)
	}
	if resp.FinishReason != "stop" {
		t.Errorf("FinishReason = %q, want %q", resp.FinishReason, "stop")
	}
	if len(resp.ToolCalls) != 0 {
		t.Errorf("ToolCalls len = %d, want 0", len(resp.ToolCalls))
	}
	if !strings.Contains(resp.Content, "tool_calls") {
		t.Errorf("Content should pass through tool_calls JSON verbatim, got %q", resp.Content)
	}
}

func TestGeminiCliProvider_ParseResponse_InvalidJSON(t *testing.T) {
	p := NewGeminiCliProvider("", "/workspace", nil, nil)
	_, err := p.parseGeminiCliResponse("not valid json")
	if err == nil {
		t.Fatal("expected error for invalid JSON")
	}
	if !strings.Contains(err.Error(), "failed to parse gemini cli response") {
		t.Errorf("error = %q, want to contain 'failed to parse gemini cli response'", err.Error())
	}
}

func TestGeminiCliProvider_ParseResponse_NoStats(t *testing.T) {
	p := NewGeminiCliProvider("", "/workspace", nil, nil)
	output := `{"session_id":"s","response":"hello"}`

	resp, err := p.parseGeminiCliResponse(output)
	if err != nil {
		t.Fatalf("error = %v", err)
	}
	if resp.Content != "hello" {
		t.Errorf("Content = %q, want %q", resp.Content, "hello")
	}
	if resp.Usage != nil {
		t.Errorf("Usage should be nil when no stats, got %+v", resp.Usage)
	}
}

// --- Factory tests ---

// --- DispatchStatus tests ---

// TestParseGeminiCliResponse_DispatchStatus_PrefersMainRole verifies that when
// multiple models share the stats block, the one whose roles map contains
// "main" wins — auxiliary models like utility_router are ignored.
func TestParseGeminiCliResponse_DispatchStatus_PrefersMainRole(t *testing.T) {
	p := NewGeminiCliProvider("", "/workspace", nil, nil)
	output := `{
		"session_id":"alice-bob-session","response":"hi",
		"stats":{
			"models":{
				"gemini-3-flash-preview":{
					"api":{"totalRequests":1,"totalErrors":0,"totalLatencyMs":3205},
					"tokens":{"input":2470,"prompt":64881,"candidates":1,"total":65054,"cached":62411,"thoughts":172,"tool":0},
					"roles":{"main":{"totalRequests":1,"totalErrors":0,"totalLatencyMs":3205}}
				},
				"gemini-2.5-flash-lite":{
					"api":{"totalRequests":2,"totalErrors":0,"totalLatencyMs":200},
					"tokens":{"input":50,"prompt":50,"candidates":5,"total":55,"cached":0,"thoughts":0,"tool":0},
					"roles":{"utility_router":{"totalRequests":2}}
				}
			}
		}
	}`
	resp, err := p.parseGeminiCliResponse(output)
	if err != nil {
		t.Fatalf("parseGeminiCliResponse() error = %v", err)
	}
	if resp.Status == nil {
		t.Fatal("Status must be populated")
	}
	s := resp.Status
	if s.Model != "gemini-3-flash-preview" {
		t.Errorf("Model = %q, want gemini-3-flash-preview (the main role)", s.Model)
	}
	if !s.Success {
		t.Error("Success must be true with no errors")
	}
	if s.InputTokens != 64881 {
		t.Errorf("InputTokens = %d, want 64881 (tokens.prompt of main)", s.InputTokens)
	}
	if s.OutputTokens != 1 {
		t.Errorf("OutputTokens = %d, want 1 (tokens.candidates of main)", s.OutputTokens)
	}
	if s.CacheReadTokens != 62411 {
		t.Errorf("CacheReadTokens = %d, want 62411 (tokens.cached of main)", s.CacheReadTokens)
	}
	if s.NumTurns != 1 {
		t.Errorf("NumTurns = %d, want 1 (totalRequests of main)", s.NumTurns)
	}
	if s.DurationMs != 3205 {
		t.Errorf("DurationMs = %d, want 3205 (totalLatencyMs of main)", s.DurationMs)
	}
	if s.StopReason != "success" {
		t.Errorf("StopReason = %q, want success", s.StopReason)
	}
}

// TestParseGeminiCliResponse_DispatchStatus_FallbackToLargest covers older CLI
// output that omits the roles map: the largest-token model wins.
func TestParseGeminiCliResponse_DispatchStatus_FallbackToLargest(t *testing.T) {
	p := NewGeminiCliProvider("", "/workspace", nil, nil)
	output := `{
		"session_id":"sess","response":"hi",
		"stats":{
			"models":{
				"gemini-3-pro":{
					"api":{"totalRequests":1,"totalErrors":0,"totalLatencyMs":4000},
					"tokens":{"input":1000,"prompt":1200,"candidates":3,"total":1203,"cached":0}
				},
				"gemini-2.5-flash-lite":{
					"api":{"totalRequests":2,"totalErrors":0,"totalLatencyMs":200},
					"tokens":{"input":50,"prompt":50,"candidates":5,"total":55,"cached":0}
				}
			}
		}
	}`
	resp, err := p.parseGeminiCliResponse(output)
	if err != nil {
		t.Fatalf("parseGeminiCliResponse() error = %v", err)
	}
	if resp.Status == nil {
		t.Fatal("Status must be populated")
	}
	if resp.Status.Model != "gemini-3-pro" {
		t.Errorf("Model = %q, want gemini-3-pro (largest-token fallback)", resp.Status.Model)
	}
}

// TestParseGeminiCliResponse_DispatchStatus_ErrorsFlipSuccess verifies that
// totalErrors > 0 produces Success=false and StopReason="error".
func TestParseGeminiCliResponse_DispatchStatus_ErrorsFlipSuccess(t *testing.T) {
	p := NewGeminiCliProvider("", "/workspace", nil, nil)
	output := `{
		"session_id":"sess","response":"",
		"stats":{
			"models":{
				"gemini-3-flash-preview":{
					"api":{"totalRequests":1,"totalErrors":1,"totalLatencyMs":1500},
					"tokens":{"input":0,"prompt":0,"candidates":0,"total":0,"cached":0},
					"roles":{"main":{"totalRequests":1}}
				}
			}
		}
	}`
	resp, err := p.parseGeminiCliResponse(output)
	if err != nil {
		t.Fatalf("parseGeminiCliResponse() error = %v", err)
	}
	if resp.Status == nil {
		t.Fatal("Status must be populated")
	}
	if resp.Status.Success {
		t.Error("Success must be false when totalErrors > 0")
	}
	if resp.Status.StopReason != "error" {
		t.Errorf("StopReason = %q, want error", resp.Status.StopReason)
	}
}
