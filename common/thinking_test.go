package common

import "testing"

func TestAnthropicThinking(t *testing.T) {
	tests := []struct {
		level      string
		wantEffort string
		wantOn     bool
	}{
		{"adaptive", "", true},
		{"low", "low", true},
		{"medium", "medium", true},
		{"high", "high", true},
		{"xhigh", "xhigh", true},
		{"max", "max", true},
		{"off", "", false},
		{"", "", false},
		{"unknown", "", false},
		{"High", "", false},
	}
	for _, tt := range tests {
		t.Run(tt.level, func(t *testing.T) {
			effort, on := AnthropicThinking(tt.level)
			if effort != tt.wantEffort || on != tt.wantOn {
				t.Errorf("AnthropicThinking(%q) = (%q, %v), want (%q, %v)",
					tt.level, effort, on, tt.wantEffort, tt.wantOn)
			}
		})
	}
}
