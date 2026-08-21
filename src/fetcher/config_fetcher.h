#pragma once

#include <string>
#include "gguf_parser.h"

// Fetch and parse config.json from a Hugging Face repository URL
// Input: repository URL (e.g., https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
// Output: populated ModelMetadata struct
// Returns true on success
bool fetch_config_json(const std::string& repo_url, ModelMetadata& metadata);
