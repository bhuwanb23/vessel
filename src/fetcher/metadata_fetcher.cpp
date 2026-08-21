#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include "http_fetcher.h"
#include "gguf_parser.h"
#include "config_fetcher.h"

// Check if URL points to a GGUF file
static bool is_gguf_url(const std::string& url) {
    // Check if URL ends with .gguf (case-insensitive)
    std::string lower_url = url;
    for (auto& c : lower_url) c = static_cast<char>(tolower(c));
    return lower_url.length() >= 4 &&
           lower_url.substr(lower_url.length() - 4) == ".gguf";
}

int main(int argc, char* argv[]) {
    // Default URL: the test GGUF model
    std::string url = "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf";

    if (argc > 1) {
        url = argv[1];
    }

    printf("=== GGUF Metadata Fetcher ===\n\n");
    printf("URL: %s\n", url.c_str());

    ModelMetadata metadata;
    bool success = false;

    if (is_gguf_url(url)) {
        // GGUF path: fetch header via range request
        printf("Detected: GGUF file (using range request)\n\n");
        printf("Fetching first 64KB via HTTP range request...\n\n");

        std::vector<uint8_t> header;
        if (fetch_gguf_header(url, header)) {
            printf("Downloaded: %zu bytes\n\n", header.size());
            success = parse_gguf_header(header.data(), header.size(), metadata);
        }
    } else {
        // config.json path: fetch full JSON
        printf("Detected: Repository URL (using config.json fallback)\n\n");
        success = fetch_config_json(url, metadata);
    }

    if (!success) {
        fprintf(stderr, "Error: Failed to extract metadata\n");
        return 1;
    }

    // Print metadata
    printf("=== Model Metadata ===\n");
    printf("Architecture:       %s\n", metadata.architecture.c_str());
    printf("Name:               %s\n", metadata.name.c_str());
    if (metadata.parameter_count > 0) {
        printf("Parameter Count:    %llu (%.1fB)\n", metadata.parameter_count,
               metadata.parameter_count / 1e9);
    } else {
        printf("Parameter Count:    N/A (must calculate from dimensions)\n");
    }
    printf("Quantization:       %s (type=%u)\n", metadata.file_type_name.c_str(), metadata.file_type);
    printf("\n");

    printf("--- %s-specific ---\n", metadata.architecture.c_str());
    printf("Context Length:     %u\n", metadata.context_length);
    printf("Block Count:        %u (layers)\n", metadata.block_count);
    printf("Embedding Length:   %u\n", metadata.embedding_length);
    printf("Attention Heads:    %u\n", metadata.head_count);
    printf("KV Heads:           %u\n", metadata.head_count_kv);
    printf("FFN Length:         %u\n", metadata.feed_forward_length);
    if (metadata.layer_norm_rms_epsilon > 0) {
        printf("RMS Epsilon:        %g\n", metadata.layer_norm_rms_epsilon);
    }
    printf("\n");

    printf("=================================================\n");
    return 0;
}
