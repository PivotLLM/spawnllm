package common

import "testing"

func TestSanitizeModelContent(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want string
	}{
		{"no token untouched", "Here is the weather for Toronto.", "Here is the weather for Toronto."},
		{"dangling opener stripped", "Let me check the docs.\n<｜DSML｜function_calls", "Let me check the docs."},
		{"balanced token stripped", "before <｜tool▁calls▁begin｜> after", "before  after"},
		{"empty stays empty", "", ""},
		{"only token becomes empty", "<｜DSML｜function_calls", ""},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := SanitizeModelContent(c.in); got != c.want {
				t.Fatalf("SanitizeModelContent(%q) = %q, want %q", c.in, got, c.want)
			}
		})
	}
}
