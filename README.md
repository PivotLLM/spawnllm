# spawnllm

`github.com/PivotLLM/spawnllm` is the ClawEh ecosystem's **LLM-dispatch core**.
It calls an LLM and drives it to completion:

- **CLI providers** (claude-cli, codex-cli, cursor-cli, antigravity-cli; gemini-cli is an alias for antigravity-cli): run the subprocess, return the result.
- **API providers** (OpenAI-chat/responses, Azure, Anthropic): run the LLM↔tool-call loop, giving the model access to host-injected `toolspec.Tool`s until it is done.

## Boundaries

spawnllm imports only [`toolspec`](https://github.com/PivotLLM/toolspec) and the
standard library (plus provider SDKs). It **never imports a host**. Policy —
model selection, fallback, cooldown, config, results handling — stays in the host
(ClawEh, Maestro). The host resolves a `ProviderSpec`, injects tools, and calls
the worker.

Logging is host-injectable via the `logger` subpackage (`logger.SetBackend`), so
provider/loop logs flow into the host's own logger; spawnllm is silent until a
backend is installed.

## Generated images

Images a model returns come back in `LLMResponse.Images` (and, for a whole
run, `Result.Images`), passed through unfiltered and whatever their size:
spawnllm executes the call and leaves judging the data to the caller. Each
`GeneratedImage` has exactly one of `Data` (decoded base64, with its
`ContentType`) or `URL` (an image given as a URL, never fetched; also base64
that does not decode, returned as a data URL so nothing is lost). `Size` is
the decoded length in bytes, or -1 (unknown) for a `URL` image.

## Speech and image generation

Single-shot task calls live in their own packages. Each call is one request:
the caller supplies the base URL, API key and model and keeps every policy
decision (model choice, fallback, cooldown, limits). Results are passed
through: no size caps, no rewriting, a URL stays a URL. The timeout is the
context's deadline (the default HTTP client has none of its own; supply one
with `WithHTTPClient`). A non-2xx response is a `*common.HTTPStatusError`
(status, Retry-After, a body excerpt of at most 1 KiB), so `ClassifyError`
handles it as it does a chat provider's error.

- `speech` — `Transcriber` (speech to text) and `Synthesizer` (text to speech).
  - `NewOpenAI(baseURL, apiKey, ...)`: `POST {base}/audio/transcriptions` and
    `/audio/speech` (OpenAI, Groq, local servers, OpenRouter). Transcription
    is a multipart upload, or OpenRouter's JSON body with base64
    `input_audio` for an `openrouter.ai` base URL (or with
    `WithTranscriptionAPI`); OpenRouter's `usage.cost` is returned as
    `CostUSD`. `DefaultSTTModel(baseURL)` gives the usual model for Groq,
    OpenAI and OpenRouter.
  - `NewGemini(baseURL, apiKey, ...)`: `generateContent` with inline audio,
    and TTS models with `responseModalities: ["AUDIO"]`. The audio is returned
    as Gemini sends it: raw PCM (e.g. `audio/L16;codec=pcm;rate=24000`), not
    converted.
  - `NewElevenLabs(baseURL, apiKey, ...)`: Scribe (`/v1/speech-to-text`) and
    `/v1/text-to-speech/{voice_id}`.
- `imagegen` — `Generator` and `Editor`. `NewOpenAI(baseURL, apiKey, ...)`:
  `POST {base}/images/generations` (OpenRouter: `{base}/images`) and
  `/images/edits` (multipart). Images are `GeneratedImage` values, as for chat
  image output: base64 decoded into `Data`, a URL returned as `URL` with
  `Size` -1, never fetched.

## Building and testing

`make test` is the gate. It runs `./test.sh`, which verifies formatting
(`golangci-lint fmt --diff`), then runs `go vet`, `golangci-lint run` and
`go test -race -count=1 ./...`, prints a summary, and exits non-zero on any
failure. It never modifies the working tree. A plain `make` runs the gate and
then `make build` (compile only: this is a library with no binaries).

Other targets: `make fmt` rewrites formatting, `make lint` and `make vet` run
those checks on their own, and `make clean` drops the Go test cache.

The gate needs [golangci-lint](https://golangci-lint.run) v2. If it is missing,
the gate fails and prints the install command:

    go install github.com/golangci/golangci-lint/v2/cmd/golangci-lint@latest
