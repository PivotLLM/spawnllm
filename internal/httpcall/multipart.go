/******************************************************************************
 * Copyright (c) 2026 Tenebris Technologies Inc.                              *
 * Please see LICENSE file for details.                                       *
 ******************************************************************************/

package httpcall

import (
	"fmt"
	"io"
	"mime/multipart"
	"net/textproto"
	"path/filepath"
	"strings"
)

// WriteFilePart writes r as a multipart file part. filename defaults
// to "file"; contentType to application/octet-stream (what
// multipart.Writer.CreateFormFile uses).
func WriteFilePart(w *multipart.Writer, field, filename, contentType string, r io.Reader) error {
	if filename == "" {
		filename = "file"
	}
	if contentType == "" {
		contentType = "application/octet-stream"
	}
	h := make(textproto.MIMEHeader)
	h.Set("Content-Disposition", fmt.Sprintf(`form-data; name="%s"; filename="%s"`,
		quoteEscaper.Replace(field), quoteEscaper.Replace(filepath.Base(filename))))
	h.Set("Content-Type", contentType)
	part, err := w.CreatePart(h)
	if err != nil {
		return fmt.Errorf("failed to create form file: %w", err)
	}
	if _, err := io.Copy(part, r); err != nil {
		return fmt.Errorf("failed to read file part: %w", err)
	}
	return nil
}

// quoteEscaper escapes a Content-Disposition parameter as mime/multipart does.
var quoteEscaper = strings.NewReplacer("\\", "\\\\", `"`, "\\\"")

// WriteFields writes plain form fields in order.
func WriteFields(w *multipart.Writer, fields [][2]string) error {
	for _, f := range fields {
		if err := w.WriteField(f[0], f[1]); err != nil {
			return fmt.Errorf("failed to write %s field: %w", f[0], err)
		}
	}
	return nil
}
