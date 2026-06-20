package spawnllm

import (
	"context"
	"errors"
	"testing"

	"github.com/PivotLLM/toolspec"
)

// scriptedProvider returns a queued sequence of responses, one per Chat call.
type scriptedProvider struct {
	resps []*LLMResponse
	errs  []error
	calls int
	cli   bool
}

func (p *scriptedProvider) Chat(_ context.Context, _ []Message, _ []ToolDefinition, _ string, _ map[string]any) (*LLMResponse, error) {
	i := p.calls
	p.calls++
	if i < len(p.errs) && p.errs[i] != nil {
		return nil, p.errs[i]
	}
	return p.resps[i], nil
}
func (p *scriptedProvider) GetDefaultModel() string { return "test-model" }
func (p *scriptedProvider) IsCLI() bool             { return p.cli }

func TestNew_RequiresProvider(t *testing.T) {
	if _, err := New(WithMaxIterations(3)); err == nil {
		t.Fatal("expected error when no provider configured")
	}
}

func TestNew_UnknownKind(t *testing.T) {
	if _, err := New(WithProvider(ProviderSpec{Kind: "nope"})); err == nil {
		t.Fatal("expected error for unknown provider kind")
	}
}

func TestRun_APILoop_ExecutesInjectedToolThenAnswers(t *testing.T) {
	var gotArgs map[string]any
	tool := toolspec.ToolDefinition{
		Name:        "echo",
		Description: "echo a value",
		Handler: func(call *toolspec.ToolCall) (*toolspec.Result, error) {
			gotArgs = call.Args
			return &toolspec.Result{ForLLM: "echoed:hi"}, nil
		},
	}
	prov := &scriptedProvider{resps: []*LLMResponse{
		{ToolCalls: []ToolCall{{ID: "c1", Name: "echo", Arguments: map[string]any{"v": "hi"}}}},
		{Content: "all done"},
	}}

	w, err := New(WithProviderInstance(prov, "m"), WithTools([]toolspec.ToolDefinition{tool}))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	res, err := w.Run(context.Background(), []Message{{Role: "user", Content: "go"}})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if res.Content != "all done" {
		t.Errorf("content = %q, want 'all done'", res.Content)
	}
	if res.Iterations != 2 {
		t.Errorf("iterations = %d, want 2", res.Iterations)
	}
	if gotArgs["v"] != "hi" {
		t.Errorf("tool received args %v, want v=hi", gotArgs)
	}
	// Transcript must include the assistant tool-call turn and the tool result.
	var sawToolResult bool
	for _, m := range res.Messages {
		if m.Role == "tool" && m.ToolCallID == "c1" && m.Content == "echoed:hi" {
			sawToolResult = true
		}
	}
	if !sawToolResult {
		t.Errorf("transcript missing the tool result message: %+v", res.Messages)
	}
}

func TestRun_APILoop_MaxIterations(t *testing.T) {
	// Provider always asks for a tool → never completes → cap trips.
	loop := &LLMResponse{ToolCalls: []ToolCall{{ID: "x", Name: "echo"}}}
	prov := &scriptedProvider{resps: []*LLMResponse{loop, loop, loop, loop}}
	tool := toolspec.ToolDefinition{Name: "echo", Handler: func(*toolspec.ToolCall) (*toolspec.Result, error) {
		return &toolspec.Result{ForLLM: "again"}, nil
	}}
	w, _ := New(WithProviderInstance(prov, "m"), WithTools([]toolspec.ToolDefinition{tool}), WithMaxIterations(3))
	_, err := w.Run(context.Background(), nil)
	if err == nil {
		t.Fatal("expected max-iterations error")
	}
}

func TestRun_APILoop_ToolError_FeedsBackToModel(t *testing.T) {
	prov := &scriptedProvider{resps: []*LLMResponse{
		{ToolCalls: []ToolCall{{ID: "c1", Name: "boom"}}},
		{Content: "recovered"},
	}}
	tool := toolspec.ToolDefinition{Name: "boom", Handler: func(*toolspec.ToolCall) (*toolspec.Result, error) {
		return nil, errors.New("kaboom")
	}}
	w, _ := New(WithProviderInstance(prov, "m"), WithTools([]toolspec.ToolDefinition{tool}))
	res, err := w.Run(context.Background(), nil)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if res.Content != "recovered" {
		t.Errorf("content = %q, want 'recovered'", res.Content)
	}
	var sawErr bool
	for _, m := range res.Messages {
		if m.Role == "tool" && m.Content == "error: kaboom" {
			sawErr = true
		}
	}
	if !sawErr {
		t.Errorf("tool error should be fed back as a tool message: %+v", res.Messages)
	}
}

func TestRun_CLI_SingleCallNoLoop(t *testing.T) {
	prov := &scriptedProvider{cli: true, resps: []*LLMResponse{{Content: "cli output"}}}
	w, _ := New(WithProviderInstance(prov, "m"))
	res, err := w.Run(context.Background(), []Message{{Role: "user", Content: "go"}})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if res.Content != "cli output" {
		t.Errorf("content = %q", res.Content)
	}
	if prov.calls != 1 {
		t.Errorf("CLI provider should be called exactly once, got %d", prov.calls)
	}
}
