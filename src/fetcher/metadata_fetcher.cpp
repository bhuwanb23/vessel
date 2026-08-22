#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include "http_fetcher.h"
#include "gguf_parser.h"
#include "config_fetcher.h"

// Format a uint64 with commas (e.g., 3212749824 -> "3,212,749,824")
static std::string format_number(uint64_t n) {
    std::string s = std::to_string(n);
    int len = static_cast<int>(s.length());
    if (len <= 3) return s;

    std::string result;
    int first_group = len % 3;
    if (first_group > 0) {
        result += s.substr(0, first_group);
    }
    for (int i = first_group; i < len; i += 3) {
        if (!result.empty()) result += ",";
        result += s.substr(i, 3);
    }
    return result;
}

// Check if URL points to a GGUF file
static bool is_gguf_url(const std::string& url) {
    if (url.length() < 5) return false;
    std::string end_str = url.substr(url.length() - 5);
    for (auto& c : end_str) c = static_cast<char>(tolower(c));
    return end_str == ".gguf";
}

// Calculate parameter count estimate from dimensions
static uint64_t estimate_parameter_count(const ModelMetadata& m) {
    // Rough estimate: this is architecture-dependent, but a reasonable approximation
    // For transformer: params ≈ (2 * embedding * context) + (layers * (4 * ffn + 8 * embedding * heads / kv_heads))
    // This is a rough heuristic — not exact
    if (m.block_count == 0 || m.embedding_length == 0) return 0;
    uint64_t emb = m.embedding_length;
    uint64_t ffn = m.feed_forward_length > 0 ? m.feed_forward_length : emb * 4;
    uint64_t layers = m.block_count;
    uint64_t kv_ratio = m.head_count_kv > 0 ? m.head_count / m.head_count_kv : 1;
    // Per-layer: Q/K/V projections + FFN gate/up/down
    uint64_t per_layer = (8 * emb * emb / kv_ratio) + (3 * ffn * emb);
    return layers * per_layer + emb * emb;  // + embeddings
}

int main(int argc, char* argv[]) {
    // Default URL: the test GGUF model
    std::string url = "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf";

    if (argc > 1) {
        url = argv[1];
    }

    ModelMetadata metadata;
    bool success = false;
    bool is_gguf = is_gguf_url(url);
    std::string fetch_method;
    size_t bytes_downloaded = 0;

    if (is_gguf) {
        fetch_method = "64KB via HTTP Range (206 Partial Content)";
        std::vector<uint8_t> header;
        if (fetch_gguf_header(url, header)) {
            bytes_downloaded = header.size();
            success = parse_gguf_header(header.data(), header.size(), metadata);
        }
    } else {
        fetch_method = "Full GET (config.json fallback)";
        success = fetch_config_json(url, metadata);
    }

    if (!success) {
        fprintf(stderr, "Error: Failed to extract metadata\n");
        return 1;
    }

    // If parameter count not available, estimate from dimensions
    if (metadata.parameter_count == 0 && metadata.block_count > 0) {
        metadata.parameter_count = estimate_parameter_count(metadata);
    }

    // === Clean Report Output ===
    printf("=== LLM Deployment Planner - Model Metadata ===\n");
    printf("Source: %s\n", url.c_str());
    printf("Fetch:  %s\n\n", fetch_method.c_str());

    // Model Identity
    printf("--- Model Identity ---\n");
    printf("Name:            %s\n", metadata.name.c_str());
    printf("Architecture:    %s\n", metadata.architecture.c_str());
    printf("Quantization:    %s (file_type=%u)\n\n", metadata.file_type_name.c_str(), metadata.file_type);

    // Dimensions
    printf("--- Dimensions ---\n");
    if (metadata.parameter_count > 0) {
        double param_b = metadata.parameter_count / 1e9;
        printf("Parameters:      %s (~%.1fB)\n", format_number(metadata.parameter_count).c_str(), param_b);
    } else {
        printf("Parameters:      N/A (cannot calculate)\n");
    }
    printf("Layers:          %u\n", metadata.block_count);
    printf("Embedding Dim:   %u\n", metadata.embedding_length);
    printf("Attention Heads: %u\n", metadata.head_count);

    if (metadata.head_count_kv > 0 && metadata.head_count > metadata.head_count_kv) {
        uint32_t gqa_ratio = metadata.head_count / metadata.head_count_kv;
        printf("KV Heads:        %u (GQA ratio %u:1)\n", metadata.head_count_kv, gqa_ratio);
    } else if (metadata.head_count_kv > 0) {
        printf("KV Heads:        %u (MHA)\n", metadata.head_count_kv);
    }

    printf("FFN Dim:         %u\n", metadata.feed_forward_length);

    if (metadata.context_length > 0) {
        if (metadata.context_length >= 1024 && (metadata.context_length % 1024) == 0) {
            printf("Context Length:  %s (%uK)\n", format_number(metadata.context_length).c_str(), metadata.context_length / 1024);
        } else {
            printf("Context Length:  %s\n", format_number(metadata.context_length).c_str());
        }
    }
    printf("\n");

    // Derived values
    if (metadata.embedding_length > 0 && metadata.head_count > 0) {
        printf("--- Derived ---\n");
        uint32_t head_dim = metadata.embedding_length / metadata.head_count;
        printf("Head Dim:        %u  (embedding_dim / attention_heads)\n", head_dim);
        printf("KV Dim per Head: %u\n", head_dim);
        printf("\n");
    }

    printf("=================================================\n");
    return 0;
}
