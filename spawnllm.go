package spawnllm

import (
	"context"
	"fmt"
	"time"

	anthropicmessages "github.com/PivotLLM/spawnllm/anthropic_messages"
	"github.com/PivotLLM/spawnllm/azure"
	"github.com/PivotLLM/spawnllm/openai_compat"
	"github.com/PivotLLM/spawnllm/openai_responses"
	"github.com/PivotLLM/toolspec"
)

// ProviderSpec is a host-resolved description of one LLM endpoint. The host
// (ClawEh, Maestro) decides which model to call and resolves credentials/knobs
// from its own config, then hands spawnllm a fully-resolved spec — spawnllm
// performs no model selection, config lookup, or credential resolution.
type ProviderSpec struct {
	// Kind selects the wire protocol / transport: "openai-chat",
	// "openai-responses", "azure", "anthropic", "anthropic-messages",
	// "claude-cli", "codex-cli", "gemini-cli".
	Kind string

	// API providers.
	BaseURL string
	APIKey  string
	Proxy   string

	// Model is the raw model id sent on the wire.
	Model string

	// CLI providers.
	CLIPath   string            // path to the CLI binary
	Workspace string            // working directory for the subprocess
	ExtraArgs []string          // extra CLI arguments
	Env       map[string]string // extra environment variables

	// Timeout bounds a single request (API) or the subprocess run (CLI). Zero
	// leaves the provider's own default.
	Timeout time.Duration
}

// Event is a progress signal emitted once per loop iteration (API providers).
type Event struct {
	Iteration int
	ToolCalls int
}

// Worker is a disposable, single-provider dispatch unit. Build one with New,
// call Run, discard it. Workers are not meant to be pooled; share a single
// *http.Client across them (WithHTTPClient on the provider, host-side) if you
// want connection reuse.
type Worker struct {
	provider      LLMProvider
	model         string
	tools         []toolspec.ToolDefinition
	toolByName    map[string]toolspec.ToolDefinition
	maxIterations int
	llmOptions    map[string]any
	progress      func(Event)

	// identity threaded onto each tool call (so injected tools can scope work)
	agentID, session, channel, chatID string
}

// Result is the outcome of a Run.
type Result struct {
	Content    string    // final model-facing content
	Messages   []Message // full transcript (API loop); nil for CLI
	Iterations int
}

// Option configures a Worker. Options are applied in order by New.
type Option func(*Worker) error

// WithProvider builds the provider from a resolved spec. This is the primary way
// to supply a provider.
func WithProvider(spec ProviderSpec) Option {
	return func(w *Worker) error {
		p, model, err := buildProvider(spec)
		if err != nil {
			return err
		}
		w.provider, w.model = p, model
		return nil
	}
}

// WithProviderInstance injects an already-constructed provider. Useful for tests
// and advanced hosts that build their own clients.
func WithProviderInstance(p LLMProvider, model string) Option {
	return func(w *Worker) error {
		w.provider, w.model = p, model
		return nil
	}
}

// WithTools injects the tools the model may call (API providers). They are the
// portable toolspec contract; the host supplies the published names and handlers.
func WithTools(defs []toolspec.ToolDefinition) Option {
	return func(w *Worker) error {
		w.tools = defs
		w.toolByName = make(map[string]toolspec.ToolDefinition, len(defs))
		for _, d := range defs {
			w.toolByName[d.Name] = d
		}
		return nil
	}
}

// WithMaxIterations caps the API tool loop (default 25).
func WithMaxIterations(n int) Option {
	return func(w *Worker) error {
		if n > 0 {
			w.maxIterations = n
		}
		return nil
	}
}

// WithLLMOptions sets per-request options passed to the provider (temperature,
// reasoning, etc.). CLI providers ignore HTTP-only options.
func WithLLMOptions(m map[string]any) Option {
	return func(w *Worker) error {
		w.llmOptions = m
		return nil
	}
}

// WithProgress installs a per-iteration progress callback (API providers).
func WithProgress(fn func(Event)) Option {
	return func(w *Worker) error {
		w.progress = fn
		return nil
	}
}

// WithIdentity threads agent/session/channel/chat identifiers onto each tool
// call, so injected tools can scope their work to the originating context.
func WithIdentity(agentID, session, channel, chatID string) Option {
	return func(w *Worker) error {
		w.agentID, w.session, w.channel, w.chatID = agentID, session, channel, chatID
		return nil
	}
}

// New builds a Worker. It returns an error if no provider was supplied.
func New(opts ...Option) (*Worker, error) {
	w := &Worker{maxIterations: 25}
	for _, o := range opts {
		if err := o(w); err != nil {
			return nil, err
		}
	}
	if w.provider == nil {
		return nil, fmt.Errorf("spawnllm: no provider configured (use WithProvider or WithProviderInstance)")
	}
	return w, nil
}

// Run dispatches the worker to completion. For a CLI provider it issues a single
// call and returns the result (the CLI drives its own tool use). For an API
// provider it runs the LLM↔tool-call loop over the injected tools until the model
// stops requesting tools or the iteration cap is hit.
func (w *Worker) Run(ctx context.Context, messages []Message) (*Result, error) {
	if cli, ok := w.provider.(CLIProvider); ok && cli.IsCLI() {
		resp, err := w.provider.Chat(ctx, messages, nil, w.model, w.llmOptions)
		if err != nil {
			return nil, err
		}
		return &Result{Content: resp.Content, Iterations: 1}, nil
	}

	toolDefs := w.providerToolDefs()
	msgs := append([]Message(nil), messages...)
	for i := 1; i <= w.maxIterations; i++ {
		resp, err := w.provider.Chat(ctx, msgs, toolDefs, w.model, w.llmOptions)
		if err != nil {
			return nil, err
		}
		if w.progress != nil {
			w.progress(Event{Iteration: i, ToolCalls: len(resp.ToolCalls)})
		}
		if len(resp.ToolCalls) == 0 {
			return &Result{Content: resp.Content, Messages: msgs, Iterations: i}, nil
		}
		msgs = append(msgs, Message{Role: "assistant", Content: resp.Content, ToolCalls: resp.ToolCalls})
		for _, tc := range resp.ToolCalls {
			msgs = append(msgs, Message{
				Role:       "tool",
				ToolCallID: tc.ID,
				Content:    w.execTool(ctx, tc),
			})
		}
	}
	return &Result{Messages: msgs, Iterations: w.maxIterations},
		fmt.Errorf("spawnllm: max iterations (%d) reached without completion", w.maxIterations)
}

// providerToolDefs renders the injected toolspec tools as wire tool definitions.
func (w *Worker) providerToolDefs() []ToolDefinition {
	if len(w.tools) == 0 {
		return nil
	}
	defs := make([]ToolDefinition, 0, len(w.tools))
	for _, t := range w.tools {
		defs = append(defs, ToolDefinition{
			Type: "function",
			Function: ToolFunctionDefinition{
				Name:        t.Name,
				Description: t.Description,
				Parameters:  t.Schema(),
			},
		})
	}
	return defs
}

// execTool runs one tool call against the injected tool and returns its
// model-facing content (or an error string the model can read).
func (w *Worker) execTool(ctx context.Context, tc ToolCall) string {
	def, ok := w.toolByName[tc.Name]
	if !ok {
		return fmt.Sprintf("error: tool %q is not available", tc.Name)
	}
	if def.Handler == nil {
		return fmt.Sprintf("error: tool %q has no handler", tc.Name)
	}
	res, err := def.Handler(&toolspec.ToolCall{
		Ctx:     ctx,
		Args:    tc.Arguments,
		AgentID: w.agentID,
		Session: w.session,
		Channel: w.channel,
		ChatID:  w.chatID,
	})
	if err != nil {
		return fmt.Sprintf("error: %v", err)
	}
	if res == nil {
		return ""
	}
	return res.ForLLM
}

// buildProvider constructs the LLM client for a resolved spec. Host config →
// rich per-provider option mapping stays in the host; spawnllm applies the
// universal knobs (timeout, model label).
func buildProvider(spec ProviderSpec) (LLMProvider, string, error) {
	secs := int(spec.Timeout / time.Second)
	switch spec.Kind {
	case "openai-chat", "anthropic":
		return NewHTTPProviderWithOptions(spec.APIKey, spec.BaseURL, spec.Proxy,
			openai_compat.WithRequestTimeout(spec.Timeout),
			openai_compat.WithModelLabel(spec.Model),
		), spec.Model, nil
	case "openai-responses":
		return openai_responses.NewProvider(spec.APIKey, spec.BaseURL, spec.Proxy,
			openai_responses.WithRequestTimeout(spec.Timeout),
			openai_responses.WithModelLabel(spec.Model),
		), spec.Model, nil
	case "azure":
		return azure.NewProviderWithTimeout(spec.APIKey, spec.BaseURL, spec.Proxy, secs), spec.Model, nil
	case "anthropic-messages":
		return anthropicmessages.NewProviderWithTimeout(spec.APIKey, spec.BaseURL, secs), spec.Model, nil
	case "claude-cli":
		return NewClaudeCliProviderWithTimeout(spec.CLIPath, spec.Workspace, spec.Timeout, spec.ExtraArgs, spec.Env), spec.Model, nil
	case "codex-cli":
		return NewCodexCliProviderWithTimeout(spec.CLIPath, spec.Workspace, spec.Timeout, spec.ExtraArgs, spec.Env), spec.Model, nil
	case "gemini-cli":
		return NewGeminiCliProviderWithTimeout(spec.CLIPath, spec.Workspace, spec.Timeout, spec.ExtraArgs, spec.Env), spec.Model, nil
	default:
		return nil, "", fmt.Errorf("spawnllm: unknown provider kind %q", spec.Kind)
	}
}
