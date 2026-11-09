/******************************************************************************
 * Copyright (c) 2026 Tenebris Technologies Inc.                              *
 * Please see LICENSE file for details.                                       *
 ******************************************************************************/

package speech

import (
	"fmt"
	"mime"
	"path/filepath"
	"strings"
)

// audioTypeExtensions maps an audio media type (without parameters, lower
// case) to the file extension the OpenAI-style endpoints recognise for it.
var audioTypeExtensions = map[string]string{
	"audio/ogg":    ".ogg",
	"audio/opus":   ".ogg",
	"audio/mpeg":   ".mp3",
	"audio/mp3":    ".mp3",
	"audio/wav":    ".wav",
	"audio/wave":   ".wav",
	"audio/x-wav":  ".wav",
	"audio/flac":   ".flac",
	"audio/x-flac": ".flac",
	"audio/mp4":    ".m4a",
	"audio/m4a":    ".m4a",
	"audio/x-m4a":  ".m4a",
	"audio/aac":    ".aac",
	"audio/webm":   ".webm",
	"video/webm":   ".webm",
	"audio/aiff":   ".aiff",
	"audio/x-aiff": ".aiff",
}

// extensionForType returns the file extension for an audio media type, or ""
// when the type is empty or not known.
func extensionForType(contentType string) string {
	mediaType, _, err := mime.ParseMediaType(contentType)
	if err != nil {
		return ""
	}
	return audioTypeExtensions[mediaType]
}

// errUnknownFormat is the error for audio whose format neither the file name
// nor the content type tells.
func errUnknownFormat(filename, contentType string) error {
	return fmt.Errorf("cannot determine audio format from filename %q or content type %q: "+
		"give a filename with an extension or a known audio content type", filename, contentType)
}

// uploadFilename is the file name sent with the audio. The OpenAI-style
// endpoints pick the decoder by its extension, so a name without one gets the
// extension of ContentType ("audio" plus it when no name was given); with
// neither, the request is refused before anything is sent.
func uploadFilename(filename, contentType string) (string, error) {
	base := ""
	if filename != "" {
		base = filepath.Base(filename)
	}
	if filepath.Ext(base) != "" {
		return base, nil
	}
	ext := extensionForType(contentType)
	if ext == "" {
		return "", errUnknownFormat(filename, contentType)
	}
	if base == "" || base == "." || base == string(filepath.Separator) {
		base = "audio"
	}
	return base + ext, nil
}

// openRouterFormat is the input_audio format OpenRouter is sent: the file
// name's extension, else the content type's, lower case without the dot.
// oga and opus are sent as ogg, the container they are (Telegram voice notes
// are .oga).
func openRouterFormat(filename, contentType string) (string, error) {
	ext := filepath.Ext(filename)
	if ext == "" {
		ext = extensionForType(contentType)
	}
	format := strings.TrimPrefix(strings.ToLower(ext), ".")
	switch format {
	case "":
		return "", errUnknownFormat(filename, contentType)
	case "oga", "opus":
		return "ogg", nil
	}
	return format, nil
}
