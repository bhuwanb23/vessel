#include <cstdio>
#include <string>
#include <vector>
#include "http_fetcher.h"
#include "gguf_parser.h"

int main(int argc, char* argv[]) {
    // Default URL: the test model
    std::string url = "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf";

    if (argc > 1) {
        url = argv[1];
    }

    printf("=== GGUF Metadata Fetcher ===\n\n");
    printf("URL: %s\n", url.c_str());
    printf("Fetching first 64KB via HTTP range request...\n\n");

    // Fetch header
    std::vector<uint8_t> header;
    if (!fetch_gguf_header(url, header)) {
        fprintf(stderr, "Error: Failed to fetch GGUF header\n");
        return 1;
    }

    printf("Downloaded: %zu bytes\n\n", header.size());

    // Parse header
    ModelMetadata metadata;
    if (!parse_gguf_header(header.data(), header.size(), metadata)) {
        fprintf(stderr, "Error: Failed to parse GGUF header\n");
        return 1;
    }

    // Print metadata
    printf("=== Model Metadata ===\n");
    printf("Architecture:       %s\n", metadata.architecture.c_str());
    printf("Name:               %s\n", metadata.name.c_str());
    printf("Parameter Count:    %llu (%.1fB)\n", metadata.parameter_count,
           metadata.parameter_count / 1e9);
    printf("Quantization:       %s (type=%u)\n", metadata.file_type_name.c_str(), metadata.file_type);
    printf("\n");

    printf("--- %s-specific ---\n", metadata.architecture.c_str());
    printf("Context Length:     %u\n", metadata.context_length);
    printf("Block Count:        %u (layers)\n", metadata.block_count);
    printf("Embedding Length:   %u\n", metadata.embedding_length);
    printf("Attention Heads:    %u\n", metadata.head_count);
    printf("KV Heads:           %u\n", metadata.head_count_kv);
    printf("FFN Length:         %u\n", metadata.feed_forward_length);
    printf("RMS Epsilon:        %g\n", metadata.layer_norm_rms_epsilon);
    printf("\n");

    printf("=================================================\n");
    return 0;
}
