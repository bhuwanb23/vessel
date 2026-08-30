#pragma once

#include "model_manager.h"
#include <string>

// =============================================================================
// Step 14 — OpenAI-Compatible API Server
// =============================================================================
// Starts an HTTP server that implements the OpenAI API specification.
// Any OpenAI-compatible client (Open WebUI, SillyTavern, Continue, Cursor,
// etc.) can connect to this server and use local GGUF models with
// auto-optimized performance.
//
// Endpoints:
//   GET  /v1/models              — list local models
//   GET  /v1/models/:id          — model details
//   POST /v1/chat/completions    — chat completions (streaming)
//   POST /v1/completions         — raw completions (streaming)
//   GET  /v1/hardware            — hardware profile
//   GET  /v1/health              — health check
// =============================================================================

// Start the API server. Blocks until stopped (Ctrl+C).
// port: default 11434 (same as Ollama for drop-in replacement)
// host: "127.0.0.1" (default) or "0.0.0.0"
// model_dirs: directories to scan for .gguf files
// Returns true if the server started and ran successfully.
bool start_api_server(
    int port,
    const std::string& host,
    const std::vector<std::string>& model_dirs
);
