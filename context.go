package spawnllm

import "context"

// providerCtxKey is the unexported key type for provider-scoped context values.
// Pointer-typed vars guarantee no collision with keys from other packages.
type providerCtxKey struct{ name string }

var ctxKeyAgentID = &providerCtxKey{"agent_id"}

// WithAgentID returns a child context that carries the given agent ID.
// The ID is extracted by providers to annotate error log entries.
func WithAgentID(ctx context.Context, agentID string) context.Context {
	return context.WithValue(ctx, ctxKeyAgentID, agentID)
}

// AgentIDFromContext returns the agent ID stored in ctx, or "" if not set.
func AgentIDFromContext(ctx context.Context) string {
	v, _ := ctx.Value(ctxKeyAgentID).(string)
	return v
}
