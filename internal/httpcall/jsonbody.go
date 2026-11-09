/******************************************************************************
 * Copyright (c) 2026 Tenebris Technologies Inc.                              *
 * Please see LICENSE file for details.                                       *
 ******************************************************************************/

package httpcall

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
)

// WriteJSONString writes s to buf as a JSON string literal.
func WriteJSONString(buf *bytes.Buffer, s string) {
	data, err := json.Marshal(s)
	if err != nil {
		// json.Marshal of a string cannot fail; invalid UTF-8 is replaced.
		panic(fmt.Sprintf("marshal string: %v", err))
	}
	buf.Write(data)
}

// WriteBase64 reads r to the end and writes it to buf as standard base64,
// with no intermediate copy of the data or of its encoding. The base64
// alphabet needs no escaping inside a JSON string.
func WriteBase64(buf *bytes.Buffer, r io.Reader) error {
	enc := base64.NewEncoder(base64.StdEncoding, buf)
	if _, err := io.Copy(enc, r); err != nil {
		return fmt.Errorf("failed to read data: %w", err)
	}
	if err := enc.Close(); err != nil {
		return fmt.Errorf("failed to encode data: %w", err)
	}
	return nil
}
