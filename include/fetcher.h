#pragma once

#include "types.h"

// =============================================================================
// Metadata Fetcher (Step 2)
// =============================================================================
// Fetches model metadata from HuggingFace URL via HTTP range request.
// Supports both GGUF header parsing and config.json fallback.
// =============================================================================

// Fetch metadata from URL (auto-detects GGUF vs config.json)
ModelSpec fetch_metadata(const std::string& url);

// Fetch metadata from GGUF file (local file path)
ModelSpec fetch_gguf_metadata(const std::string& file_path);

// Fetch metadata from HuggingFace GGUF URL (range request)
ModelSpec fetch_gguf_metadata_from_url(const std::string& url);

// Fetch metadata from config.json fallback
ModelSpec fetch_config_metadata(const std::string& repo_url);

// Check if URL points to a GGUF file
bool is_gguf_url(const std::string& url);

// =============================================================================
// Error Reporting (Phase F)
// =============================================================================
const std::string& get_fetch_error();
int get_fetch_http_status();
