package common

import "testing"

func TestThinkingBudget(t *testing.T) {
	tests := []struct {
		level string
		want  int64
	}{
		{"low", 4096},
		{"medium", 16384},
		{"high", 32000},
		{"xhigh", 64000},
		{"adaptive", 0},
		{"off", 0},
		{"", 0},
		{"unknown", 0},
	}
	for _, tt := range tests {
		t.Run(tt.level, func(t *testing.T) {
			if got := ThinkingBudget(tt.level); got != tt.want {
				t.Errorf("ThinkingBudget(%q) = %d, want %d", tt.level, got, tt.want)
			}
		})
	}
}

func TestClampThinkingBudget(t *testing.T) {
	tests := []struct {
		name      string
		budget    int64
		maxTokens int64
		want      int64
		wantOK    bool
	}{
		{"fits", 4096, 16000, 4096, true},
		{"clamped below max_tokens", 32000, 4096, 4095, true},
		{"equal to max_tokens clamped", 4096, 4096, 4095, true},
		{"too small after clamp", 4096, 1024, 0, false},
		{"minimum after clamp", 4096, 1025, 1024, true},
		{"no budget", 0, 16000, 0, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, ok := ClampThinkingBudget(tt.budget, tt.maxTokens, "test")
			if got != tt.want || ok != tt.wantOK {
				t.Errorf("ClampThinkingBudget(%d, %d) = (%d, %v), want (%d, %v)",
					tt.budget, tt.maxTokens, got, ok, tt.want, tt.wantOK)
			}
		})
	}
}
