//go:build unix

package common

import (
	"io/fs"
	"os"
	"path/filepath"
	"syscall"
	"testing"
)

// The tests in this file set the process umask, so none of them runs in
// parallel.

func setUmask(t *testing.T, mask int) {
	t.Helper()
	old := syscall.Umask(mask)
	t.Cleanup(func() { syscall.Umask(old) })
}

func permOf(t *testing.T, path string) fs.FileMode {
	t.Helper()
	info, err := os.Lstat(path)
	if err != nil {
		t.Fatalf("lstat %s: %v", path, err)
	}
	return info.Mode().Perm()
}

func TestMkdirAllExact(t *testing.T) {
	setUmask(t, 0o077)

	tests := []struct {
		name string
		mode fs.FileMode
	}{
		{name: "group writable", mode: 0o770},
		{name: "private", mode: 0o700},
		{name: "group readable", mode: 0o750},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			base := t.TempDir()
			existing := filepath.Join(base, "existing")
			if err := os.Mkdir(existing, 0o700); err != nil {
				t.Fatal(err)
			}
			if err := os.Chmod(existing, 0o711); err != nil {
				t.Fatal(err)
			}
			target := filepath.Join(existing, "a", "b")

			if err := MkdirAllExact(target, tt.mode); err != nil {
				t.Fatalf("MkdirAllExact: %v", err)
			}
			for _, d := range []string{filepath.Join(existing, "a"), target} {
				if got := permOf(t, d); got != tt.mode {
					t.Errorf("%s mode = %v, want %v", d, got, tt.mode)
				}
			}
			if got := permOf(t, existing); got != 0o711 {
				t.Errorf("existing folder mode = %v, want it left at 0711", got)
			}

			// A second call changes nothing that already exists.
			if err := os.Chmod(target, 0o700); err != nil {
				t.Fatal(err)
			}
			if err := MkdirAllExact(target, tt.mode); err != nil {
				t.Fatalf("MkdirAllExact again: %v", err)
			}
			if got := permOf(t, target); got != 0o700 {
				t.Errorf("existing target mode = %v, want it left at 0700", got)
			}
		})
	}
}

func TestMkdirAllExactNotDir(t *testing.T) {
	base := t.TempDir()
	file := filepath.Join(base, "file")
	if err := os.WriteFile(file, nil, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := MkdirAllExact(filepath.Join(file, "sub"), 0o700); err == nil {
		t.Fatal("MkdirAllExact under a file: want an error, got nil")
	}
}

func TestOpenAppendExact(t *testing.T) {
	setUmask(t, 0o077)

	tests := []struct {
		name     string
		existing fs.FileMode // 0: the file does not exist yet
		mode     fs.FileMode
	}{
		{name: "new group writable", mode: 0o660},
		{name: "new private", mode: 0o600},
		{name: "existing narrower", existing: 0o600, mode: 0o660},
		{name: "existing wider", existing: 0o666, mode: 0o640},
		{name: "existing same", existing: 0o660, mode: 0o660},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "log.jsonl")
			if tt.existing != 0 {
				if err := os.WriteFile(path, []byte("old\n"), 0o600); err != nil {
					t.Fatal(err)
				}
				if err := os.Chmod(path, tt.existing); err != nil {
					t.Fatal(err)
				}
			}

			f, err := OpenAppendExact(path, tt.mode)
			if err != nil {
				t.Fatalf("OpenAppendExact: %v", err)
			}
			if _, werr := f.WriteString("new\n"); werr != nil {
				t.Fatalf("write: %v", werr)
			}
			if cerr := f.Close(); cerr != nil {
				t.Fatalf("close: %v", cerr)
			}

			if got := permOf(t, path); got != tt.mode {
				t.Errorf("file mode = %v, want %v", got, tt.mode)
			}
			data, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			want := "new\n"
			if tt.existing != 0 {
				want = "old\nnew\n"
			}
			if string(data) != want {
				t.Errorf("content = %q, want %q", data, want)
			}
		})
	}
}
