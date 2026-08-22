#include "fetcher.h"
#include "../fetcher/http_fetcher.h"
#include "../fetcher/gguf_parser.h"
#include "../fetcher/config_fetcher.h"
#include <algorithm>

// =============================================================================
// Metadata Fetcher (Step 2)
// =============================================================================

ModelSpec fetch_metadata(const std::string& url) {
    if (is_gguf_url(url)) {
        return fetch_gguf_metadata_from_url(url);
    } else {
        return fetch_config_metadata(url);
    }
}

ModelSpec fetch_gguf_metadata(const std::string& file_path) {
    ModelSpec model;
    
    // Read GGUF header from local file
    std::vector<uint8_t> header = read_file_header(file_path, 65536);
    if (header.empty()) return model;
    
    // Parse GGUF header
    GGUFMetadata metadata = parse_gguf_header(header);
    if (!metadata.valid) return model;
    
    // Convert to ModelSpec
    model.architecture = metadata.architecture;
    model.name = metadata.name;
    model.param_count = metadata.param_count;
    model.layers = metadata.block_count;
    model.embedding_dim = metadata.embedding_length;
    model.attention_heads = metadata.attention_head_count;
    model.kv_heads = metadata.head_count_kv;
    model.head_dim = (model.attention_heads > 0) ? (model.embedding_dim / model.attention_heads) : 0;
    model.ffn_dim = metadata.feed_forward_length;
    model.context_length = metadata.context_length;
    model.quant_type = metadata.quant_type;
    model.bits_per_weight = get_bits_per_weight(metadata.file_type);
    model.source = MetadataSource::GGUF_HEADER;
    
    return model;
}

ModelSpec fetch_gguf_metadata_from_url(const std::string& url) {
    ModelSpec model;
    
    // Fetch first 64KB via range request
    std::vector<uint8_t> header = fetch_range(url, 0, 65535);
    if (header.empty()) return model;
    
    // Parse GGUF header
    GGUFMetadata metadata = parse_gguf_header(header);
    if (!metadata.valid) return model;
    
    // Convert to ModelSpec
    model.architecture = metadata.architecture;
    model.name = metadata.name;
    model.param_count = metadata.param_count;
    model.layers = metadata.block_count;
    model.embedding_dim = metadata.embedding_length;
    model.attention_heads = metadata.attention_head_count;
    model.kv_heads = metadata.head_count_kv;
    model.head_dim = (model.attention_heads > 0) ? (model.embedding_dim / model.attention_heads) : 0;
    model.ffn_dim = metadata.feed_forward_length;
    model.context_length = metadata.context_length;
    model.quant_type = metadata.quant_type;
    model.bits_per_weight = get_bits_per_weight(metadata.file_type);
    model.source = MetadataSource::GGUF_HEADER;
    
    return model;
}

ModelSpec fetch_config_metadata(const std::string& repo_url) {
    ModelSpec model;
    
    // Build config.json URL
    std::string config_url = repo_url;
    if (config_url.back() != '/') config_url += '/';
    config_url += "resolve/main/config.json";
    
    // Fetch config.json
    std::string json = fetch_full(config_url);
    if (json.empty()) return model;
    
    // Parse config.json (simplified parser)
    // Look for key fields
    auto get_int = [&](const std::string& key) -> uint32_t {
        size_t pos = json.find("\"" + key + "\"");
        if (pos == std::string::npos) return 0;
        pos = json.find(':', pos);
        if (pos == std::string::npos) return 0;
        pos++;
        while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\n')) pos++;
        uint32_t val = 0;
        while (pos < json.size() && json[pos] >= '0' && json[pos] <= '9') {
            val = val * 10 + (json[pos] - '0');
            pos++;
        }
        return val;
    };
    
    auto get_string = [&](const std::string& key) -> std::string {
        size_t pos = json.find("\"" + key + "\"");
        if (pos == std::string::npos) return "";
        pos = json.find(':', pos);
        if (pos == std::string::npos) return "";
        pos = json.find('"', pos + 1);
        if (pos == std::string::npos) return "";
        pos++;
        size_t end = json.find('"', pos);
        if (end == std::string::npos) return "";
        return json.substr(pos, end - pos);
    };
    
    model.architecture = get_string("model_type");
    model.name = get_string("name");
    model.layers = get_int("num_hidden_layers");
    model.embedding_dim = get_int("hidden_size");
    model.attention_heads = get_int("num_attention_heads");
    model.kv_heads = get_int("num_key_value_heads");
    model.head_dim = (model.attention_heads > 0) ? (model.embedding_dim / model.attention_heads) : 0;
    model.ffn_dim = get_int("intermediate_size");
    model.context_length = get_int("max_position_embeddings");
    model.quant_type = "safetensors (unknown)";
    model.bits_per_weight = 0.0;  // Unknown for config.json
    model.source = MetadataSource::CONFIG_JSON;
    
    // Estimate parameters if not provided
    model.estimate_parameters();
    
    return model;
}

bool is_gguf_url(const std::string& url) {
    // Check if URL ends with .gguf
    if (url.size() < 5) return false;
    std::string end_str = url.substr(url.size() - 5);
    return (end_str == ".gguf");
}
