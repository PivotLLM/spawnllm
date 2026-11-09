package speech

import "testing"

func TestUploadFilename(t *testing.T) {
	tests := []struct {
		filename, contentType, want string
		wantErr                     bool
	}{
		{filename: "/tmp/voice.oga", want: "voice.oga"},
		{filename: "voice.MP3", contentType: "audio/ogg", want: "voice.MP3"},
		{contentType: "audio/ogg", want: "audio.ogg"},
		{contentType: "Audio/OGG; codecs=opus", want: "audio.ogg"},
		{filename: "voice", contentType: "audio/mpeg", want: "voice.mp3"},
		{contentType: "audio/x-wav", want: "audio.wav"},
		{contentType: "audio/mp4", want: "audio.m4a"},
		{wantErr: true},
		{filename: "voice", wantErr: true},
		{contentType: "application/octet-stream", wantErr: true},
		{contentType: "not a type;;", wantErr: true},
	}
	for _, tc := range tests {
		got, err := uploadFilename(tc.filename, tc.contentType)
		if (err != nil) != tc.wantErr || got != tc.want {
			t.Errorf("uploadFilename(%q, %q) = %q, %v; want %q, error %v",
				tc.filename, tc.contentType, got, err, tc.want, tc.wantErr)
		}
	}
}

func TestOpenRouterFormat(t *testing.T) {
	tests := []struct {
		filename, contentType, want string
		wantErr                     bool
	}{
		{filename: "voice.oga", want: "ogg"},
		{filename: "voice.OPUS", want: "ogg"},
		{filename: "Voice.OGG", want: "ogg"},
		{filename: "a.mp3", contentType: "audio/wav", want: "mp3"},
		{filename: "voice", contentType: "audio/wav", want: "wav"},
		{contentType: "audio/ogg; codecs=opus", want: "ogg"},
		{contentType: "audio/opus", want: "ogg"},
		{contentType: "audio/flac", want: "flac"},
		{wantErr: true},
		{filename: "voice", contentType: "application/octet-stream", wantErr: true},
	}
	for _, tc := range tests {
		got, err := openRouterFormat(tc.filename, tc.contentType)
		if (err != nil) != tc.wantErr || got != tc.want {
			t.Errorf("openRouterFormat(%q, %q) = %q, %v; want %q, error %v",
				tc.filename, tc.contentType, got, err, tc.want, tc.wantErr)
		}
	}
}
