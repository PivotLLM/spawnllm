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
