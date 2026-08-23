#include "fetcher.h"
#include "../fetcher/http_fetcher.h"
#include "../fetcher/gguf_parser.h"
#include "../fetcher/config_fetcher.h"
#include <algorithm>
#include <cstdio>

// =============================================================================
// Metadata Fetcher (Step 2) — Error Handling (Phase F)
// =============================================================================

static std::string fetch_error_msg;
static int fetch_http_status = 0;

const std::string& get_fetch_error() {
    return fetch_error_msg;
}

int get_fetch_http_status() {
    return fetch_http_status;
}

void clear_fetch_error() {
    fetch_error_msg.clear();
    fetch_http_status = 0;
}

// Helper: Read file header (local files)
static std::vector<uint8_t> read_file_header_local(const std::string& file_path, size_t bytes) {
    std::vector<uint8_t> header(bytes);
    FILE* f = fopen(file_path.c_str(), "rb");
    if (!f) {
        fetch_error_msg = "Could not open file: " + file_path;
        return {};
    }
    size_t read = fread(header.data(), 1, bytes, f);
    fclose(f);
    header.resize(read);
    return header;
}

// Detect if URL looks like a HuggingFace repository (not a direct file URL)
static bool looks_like_repo_url(const std::string& url) {
    // HuggingFace repo URLs: https://huggingface.co/owner/repo
    // Direct file URLs: https://huggingface.co/owner/repo/resolve/main/file.gguf
    if (url.find("huggingface.co") != std::string::npos) {
        // Count path segments after huggingface.co/
        size_t base = url.find("huggingface.co/");
        if (base != std::string::npos) {
            std::string path = url.substr(base + 16);  // skip "huggingface.co/"
            // If no /resolve/ in path, it's likely a repo root
            if (path.find("/resolve/") == std::string::npos &&
                path.find(".gguf") == std::string::npos) {
                return true;
            }
        }
    }
    return false;
}

ModelSpec fetch_metadata(const std::string& url_or_path) {
    clear_fetch_error();
    
    // Check if it's a local file (not a URL)
    bool is_local = false;
    if (url_or_path.find("http://") == std::string::npos
        && url_or_path.find("https://") == std::string::npos) {
        // Not a URL — treat as local path
        is_local = true;
    }
    
    if (is_local) {
        return fetch_gguf_metadata(url_or_path);
    } else if (is_gguf_url(url_or_path)) {
        return fetch_gguf_metadata_from_url(url_or_path);
    } else {
        // Not a GGUF URL — try config.json or suggest fix
        if (looks_like_repo_url(url_or_path)) {
            fprintf(stderr, "Info: URL appears to be a repository, not a GGUF file.\n");
            fprintf(stderr, "  Trying config.json fallback...\n");
        } else {
            fprintf(stderr, "Warning: URL does not point to a .gguf file.\n");
            fprintf(stderr, "  If this is a HuggingFace repo, append the specific GGUF filename.\n");
            fprintf(stderr, "  Example: .../resolve/main/Model-Q4_K_M.gguf\n");
        }
        return fetch_config_metadata(url_or_path);
    }
}

ModelSpec fetch_gguf_metadata(const std::string& file_path) {
    ModelSpec model;
    
    // Read GGUF header from local file
    std::vector<uint8_t> header = read_file_header_local(file_path, 65536);
    if (header.empty()) return model;
    
    // Parse GGUF header
    ModelMetadata metadata;
    if (!parse_gguf_header(header.data(), header.size(), metadata)) return model;
    
    // Convert to ModelSpec
    model.architecture = metadata.architecture;
    model.name = metadata.name;
    model.param_count = metadata.parameter_count;
    model.layers = metadata.block_count;
    model.embedding_dim = metadata.embedding_length;
    model.attention_heads = metadata.head_count;
    model.kv_heads = metadata.head_count_kv;
    model.head_dim = (model.attention_heads > 0) ? (model.embedding_dim / model.attention_heads) : 0;
    model.ffn_dim = metadata.feed_forward_length;
    model.context_length = metadata.context_length;
    model.quant_type = metadata.file_type_name;
    model.bits_per_weight = get_bits_per_weight(metadata.file_type);
    model.source = MetadataSource::GGUF_HEADER;
    
    // MoE fields
    model.is_moe = metadata.is_moe;
    model.expert_count = metadata.expert_count;
    model.expert_used_count = metadata.expert_used_count;
    model.expert_shared_count = metadata.expert_shared_count;
    model.expert_ffn_dim = metadata.expert_ffn_dim;
    model.expert_weights_scale = metadata.expert_weights_scale;
    if (model.is_moe) model.model_type = ModelType::MOE;
    model.raw_kv_uint32 = metadata.raw_kv_uint32;
    
    // Estimate parameters if not provided in GGUF
    model.estimate_parameters();
    model.calculate_moe_parameters();
    
    return model;
}

ModelSpec fetch_gguf_metadata_from_url(const std::string& url) {
    ModelSpec model;
    
    // Fetch first 64KB via range request
    std::vector<uint8_t> header;
    if (!fetch_gguf_header(url, header)) return model;
    
    // Parse GGUF header
    ModelMetadata metadata;
    if (!parse_gguf_header(header.data(), header.size(), metadata)) return model;
    
    // Convert to ModelSpec
    model.architecture = metadata.architecture;
    model.name = metadata.name;
    model.param_count = metadata.parameter_count;
    model.layers = metadata.block_count;
    model.embedding_dim = metadata.embedding_length;
    model.attention_heads = metadata.head_count;
    model.kv_heads = metadata.head_count_kv;
    model.head_dim = (model.attention_heads > 0) ? (model.embedding_dim / model.attention_heads) : 0;
    model.ffn_dim = metadata.feed_forward_length;
    model.context_length = metadata.context_length;
    model.quant_type = metadata.file_type_name;
    model.bits_per_weight = get_bits_per_weight(metadata.file_type);
    model.source = MetadataSource::GGUF_HEADER;
    
    // MoE fields
    model.is_moe = metadata.is_moe;
    model.expert_count = metadata.expert_count;
    model.expert_used_count = metadata.expert_used_count;
    model.expert_shared_count = metadata.expert_shared_count;
    model.expert_ffn_dim = metadata.expert_ffn_dim;
    model.expert_weights_scale = metadata.expert_weights_scale;
    if (model.is_moe) model.model_type = ModelType::MOE;
    model.raw_kv_uint32 = metadata.raw_kv_uint32;
    
    // Estimate parameters if not provided in GGUF
    model.estimate_parameters();
    model.calculate_moe_parameters();
    
    return model;
}

ModelSpec fetch_config_metadata(const std::string& repo_url) {
    ModelSpec model;
    
    // Build config.json URL
    std::string config_url = repo_url;
    if (config_url.back() != '/') config_url += '/';
    config_url += "resolve/main/config.json";
    
    // Fetch config.json
    std::vector<uint8_t> buffer;
    if (!fetch_full(config_url, buffer)) return model;
    std::string json(buffer.begin(), buffer.end());
    if (json.empty()) return model;
    
    // Parse config.json (simplified parser)
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
