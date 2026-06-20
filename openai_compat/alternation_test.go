package openai_compat

import "testing"

func roles(ms []Message) []string {
	out := make([]string, len(ms))
	for i, m := range ms {
		out[i] = m.Role
	}
	return out
}

func TestNormalizeAlternation_FoldsSystemAndMergesRoles(t *testing.T) {
	in := []Message{
		{Role: "system", Content: "You are Amber."},
		{Role: "user", Content: "hi"},
		{Role: "assistant", Content: "", ToolCalls: []ToolCall{{ID: "t1"}}},
		{Role: "tool", Content: "result A", ToolCallID: "t1"},
		{Role: "tool", Content: "result B", ToolCallID: "t2"},
		{Role: "assistant", Content: "done"},
	}
	out := normalizeAlternation(in)

	got := roles(out)
	want := []string{"user", "assistant", "user", "assistant"}
	if len(got) != len(want) {
		t.Fatalf("roles = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("roles = %v, want %v", got, want)
		}
	}

	// System folded into the first user turn.
	if !contains(out[0].Content, "You are Amber.") || !contains(out[0].Content, "hi") {
		t.Errorf("first user turn should carry system + user content: %q", out[0].Content)
	}
	// No system or tool roles survive.
	for _, m := range out {
		if m.Role == "system" || m.Role == "tool" {
			t.Errorf("role %q must not survive normalization", m.Role)
		}
		if len(m.ToolCalls) != 0 || m.ToolCallID != "" {
			t.Errorf("tool_calls/tool_call_id must be dropped: %+v", m)
		}
	}
	// The two consecutive tool results merged into one user turn.
	if !contains(out[2].Content, "result A") || !contains(out[2].Content, "result B") {
		t.Errorf("merged tool results missing: %q", out[2].Content)
	}
}

func TestNormalizeAlternation_NoUserCreatesOne(t *testing.T) {
	out := normalizeAlternation([]Message{{Role: "system", Content: "sys only"}})
	if len(out) != 1 || out[0].Role != "user" || !contains(out[0].Content, "sys only") {
		t.Fatalf("system-only input should yield a single user turn, got %+v", out)
	}
}

func contains(s, sub string) bool {
	return len(s) >= len(sub) && (s == sub || indexOf(s, sub) >= 0)
}

func indexOf(s, sub string) int {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return i
		}
	}
	return -1
}
