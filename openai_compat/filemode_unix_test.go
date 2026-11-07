//go:build unix

package openai_compat

import (
	"io/fs"
	"os"
	"path/filepath"
	"syscall"
	"testing"
)

// TestResponseLogFileModes checks the modes of the response log file and of
// the parent directories created for it. It sets the process umask to 0, so it
// must not run in parallel with other tests.
func TestResponseLogFileModes(t *testing.T) {
	oldMask := syscall.Umask(0)
	t.Cleanup(func() { syscall.Umask(oldMask) })

	tests := []struct {
		name     string
		opts     []Option
		wantDir  fs.FileMode
		wantFile fs.FileMode
	}{
		{name: "defaults", wantDir: 0o700, wantFile: 0o600},
		{name: "custom", opts: []Option{WithFileModes(0o750, 0o640)}, wantDir: 0o750, wantFile: 0o640},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			base := t.TempDir()
			dir := filepath.Join(base, "a", "b")
			path := filepath.Join(dir, "log.jsonl")
			opts := append([]Option{WithResponseLogFile(path)}, tt.opts...)
			p := NewProvider("key", "http://127.0.0.1", "", opts...)

			p.writeLogEntry(logEntry{Dir: "request", Model: "m"})
			if p.logFile != nil {
				t.Cleanup(func() {
					if err := p.logFile.Close(); err != nil {
						t.Errorf("close log file: %v", err)
					}
				})
			}

			for _, d := range []string{filepath.Join(base, "a"), dir} {
				info, err := os.Stat(d)
				if err != nil {
					t.Fatalf("stat %s: %v", d, err)
				}
				if !info.IsDir() || info.Mode().Perm() != tt.wantDir {
					t.Errorf("dir %s mode = %v, want dir %v", d, info.Mode(), tt.wantDir)
				}
			}
			info, err := os.Stat(path)
			if err != nil {
				t.Fatalf("stat log file: %v", err)
			}
			if info.Mode().Perm() != tt.wantFile {
				t.Errorf("file mode = %v, want %v", info.Mode().Perm(), tt.wantFile)
			}
			if info.Size() == 0 {
				t.Error("log file is empty, want one record")
			}
		})
	}
}
