#include "config_fetcher.h"
#include "http_fetcher.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// Simple JSON value extractor (no external dependencies)
// Finds "key": value patterns in JSON text
static bool json_get_string(const std::string& json, const std::string& key, std::string& value) {
    std::string search = "\"" + key + "\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return false;

    // Find the colon after the key
    pos = json.find(':', pos + search.length());
    if (pos == std::string::npos) return false;

    // Find the opening quote
    pos = json.find('"', pos + 1);
    if (pos == std::string::npos) return false;

    // Find the closing quote
    size_t end = json.find('"', pos + 1);
    if (end == std::string::npos) return false;

    value = json.substr(pos + 1, end - pos - 1);
    return true;
}

static bool json_get_int(const std::string& json, const std::string& key, int& value) {
    std::string search = "\"" + key + "\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return false;

    // Find the colon
    pos = json.find(':', pos + search.length());
    if (pos == std::string::npos) return false;

    // Skip whitespace
    pos++;
    while (pos < json.length() && (json[pos] == ' ' || json[pos] == '\t')) pos++;

    // Parse number (handle negative)
    bool negative = false;
    if (json[pos] == '-') {
        negative = true;
        pos++;
    }

    value = 0;
    while (pos < json.length() && json[pos] >= '0' && json[pos] <= '9') {
        value = value * 10 + (json[pos] - '0');
        pos++;
    }

    if (negative) value = -value;
    return true;
}

// Construct config.json URL from repository URL
static std::string make_config_url(const std::string& repo_url) {
    std::string url = repo_url;

    // Remove trailing slash
    if (!url.empty() && url.back() == '/') {
        url.pop_back();
    }

    // Remove /resolve/main/... suffix if present
    size_t resolve_pos = url.find("/resolve/");
    if (resolve_pos != std::string::npos) {
        url = url.substr(0, resolve_pos);
    }

    // Append config.json path
    return url + "/resolve/main/config.json";
}

bool fetch_config_json(const std::string& repo_url, ModelMetadata& metadata) {
    std::string config_url = make_config_url(repo_url);

    printf("Fetching config.json from:\n  %s\n\n", config_url.c_str());

    // Fetch full JSON (no range request - it's small)
    std::vector<uint8_t> buffer;
    if (!fetch_full(config_url, buffer)) {
        fprintf(stderr, "Error: Failed to fetch config.json\n");
        return false;
    }

    // Convert to string
    std::string json(buffer.begin(), buffer.end());

    // Extract fields
    std::string model_type;
    if (json_get_string(json, "model_type", model_type)) {
        metadata.architecture = model_type;
    }

    std::string name;
    if (json_get_string(json, "name", name)) {
        metadata.name = name;
    } else {
        // Try model_type as name fallback
        metadata.name = model_type;
    }

    int value;

    if (json_get_int(json, "num_hidden_layers", value)) {
        metadata.block_count = static_cast<uint32_t>(value);
    }

    if (json_get_int(json, "hidden_size", value)) {
        metadata.embedding_length = static_cast<uint32_t>(value);
    }

    if (json_get_int(json, "num_attention_heads", value)) {
        metadata.head_count = static_cast<uint32_t>(value);
    }

    if (json_get_int(json, "num_key_value_heads", value)) {
        metadata.head_count_kv = static_cast<uint32_t>(value);
    }

    if (json_get_int(json, "max_position_embeddings", value)) {
        metadata.context_length = static_cast<uint32_t>(value);
    }

    if (json_get_int(json, "intermediate_size", value)) {
        metadata.feed_forward_length = static_cast<uint32_t>(value);
    }

    // config.json doesn't have quantization - it's for safetensors (F16/BF16)
    metadata.file_type = 0;  // F32 placeholder
    metadata.file_type_name = "safetensors (quantization unknown)";

    return !metadata.architecture.empty();
}
