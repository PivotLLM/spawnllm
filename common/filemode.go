package common

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
)

// MkdirAllExact creates dir and any missing parents with exactly mode,
// whatever the process umask. Each folder this call creates is made with
// os.Mkdir and then given mode through a descriptor opened on it; folders
// that already exist are left as they are.
func MkdirAllExact(dir string, mode fs.FileMode) error {
	var missing []string
	for d := filepath.Clean(dir); ; {
		info, err := os.Stat(d)
		if err == nil {
			if !info.IsDir() {
				return fmt.Errorf("%s: %w", d, errNotDir)
			}
			break
		}
		if !errors.Is(err, fs.ErrNotExist) {
			return err
		}
		missing = append(missing, d)
		parent := filepath.Dir(d)
		if parent == d {
			break
		}
		d = parent
	}
	for _, d := range slices.Backward(missing) {
		if err := os.Mkdir(d, mode); err != nil {
			if errors.Is(err, fs.ErrExist) {
				// Created by someone else meanwhile: not ours to change.
				continue
			}
			return err
		}
		if err := chmodCreatedDir(d, mode); err != nil {
			return err
		}
	}
	return nil
}

var errNotDir = errors.New("not a directory")

// chmodCreatedDir sets mode on the folder just created at d through a
// descriptor, refusing if what is at d now is not the folder that was opened
// (a symlink or another entry swapped in).
func chmodCreatedDir(d string, mode fs.FileMode) error {
	f, err := os.Open(d) //nolint:gosec // G304: the folder this call just created
	if err != nil {
		return err
	}
	defer f.Close() //nolint:errcheck // read-only descriptor
	opened, err := f.Stat()
	if err != nil {
		return err
	}
	current, err := os.Lstat(d)
	if err != nil {
		return err
	}
	if !current.IsDir() || !os.SameFile(opened, current) {
		return fmt.Errorf("%s: changed while being created", d)
	}
	return f.Chmod(mode)
}

// OpenAppendExact opens path for appending, creating it if needed, and makes
// its permission bits exactly mode, whatever the process umask. The mode is
// set on the open descriptor, and only when it differs, so an existing file
// that already has mode is never changed.
func OpenAppendExact(path string, mode fs.FileMode) (*os.File, error) {
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, mode) //nolint:gosec // G304: path set by the host
	if err != nil {
		return nil, err
	}
	info, err := f.Stat()
	if err == nil && info.Mode().Perm() != mode.Perm() {
		err = f.Chmod(mode)
	}
	if err != nil {
		return nil, errors.Join(err, f.Close())
	}
	return f, nil
}
