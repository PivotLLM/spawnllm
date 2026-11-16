// Package spawnllm is an LLM-dispatch core for host applications: it calls an LLM
// and drives it to completion. For CLI-backed providers it runs the subprocess
// and returns the result; for API providers it runs the LLM↔tool-call loop,
// giving the model access to host-injected tools until it is done.
//
// spawnllm is deliberately dependency-free of any host: it imports only
// github.com/PivotLLM/toolspec (the tool contract) and the standard library
// (plus provider SDKs). Policy — which model to call, fallback, cooldown,
// config, results handling — stays in the host application. The host
// resolves a ProviderSpec and injects tools; spawnllm dispatches.
package spawnllm
