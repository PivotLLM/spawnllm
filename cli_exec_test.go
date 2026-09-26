package spawnllm

import "testing"

func TestCappedBuffer(t *testing.T) {
	tests := []struct {
		name         string
		limit        int
		writes       []string
		wantRetained string
		wantReceived int64
		wantExceeded bool
	}{
		{
			name:         "under limit",
			limit:        10,
			writes:       []string{"abc", "de"},
			wantRetained: "abcde",
			wantReceived: 5,
		},
		{
			name:         "exactly limit",
			limit:        5,
			writes:       []string{"abc", "de"},
			wantRetained: "abcde",
			wantReceived: 5,
		},
		{
			name:         "single write past limit keeps prefix",
			limit:        4,
			writes:       []string{"abcdefgh"},
			wantRetained: "abcd",
			wantReceived: 8,
			wantExceeded: true,
		},
		{
			name:         "later writes drained but discarded",
			limit:        4,
			writes:       []string{"abc", "def", "ghi"},
			wantRetained: "abcd",
			wantReceived: 9,
			wantExceeded: true,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			b := newCappedBuffer(tc.limit)
			for _, w := range tc.writes {
				n, err := b.Write([]byte(w))
				if err != nil {
					t.Fatalf("Write returned error: %v", err)
				}
				if n != len(w) {
					t.Fatalf("Write returned n=%d, want %d (must report full drain)", n, len(w))
				}
			}
			if got := b.String(); got != tc.wantRetained {
				t.Errorf("String() = %q, want %q", got, tc.wantRetained)
			}
			if got := b.Len(); got != len(tc.wantRetained) {
				t.Errorf("Len() = %d, want %d", got, len(tc.wantRetained))
			}
			if got := b.Received(); got != tc.wantReceived {
				t.Errorf("Received() = %d, want %d", got, tc.wantReceived)
			}
			if got := b.Exceeded(); got != tc.wantExceeded {
				t.Errorf("Exceeded() = %v, want %v", got, tc.wantExceeded)
			}
		})
	}
}
