#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <variant>

// GGUF value types
enum class GgufValueType : uint32_t {
    UINT8 = 0,
    INT8 = 1,
    UINT16 = 2,
    INT16 = 3,
    UINT32 = 4,
    INT32 = 5,
    FLOAT32 = 6,
    BOOL = 7,
    STRING = 8,
    ARRAY = 9,
    UINT64 = 10,
    INT64 = 11,
    FLOAT64 = 12,
};

// Parsed model metadata
struct ModelMetadata {
    // General metadata
    std::string architecture;
    std::string name;
    uint64_t parameter_count = 0;
    uint32_t file_type = 0;
    std::string file_type_name;  // human-readable quant name

    // Architecture-specific (populated based on general.architecture)
    uint32_t context_length = 0;
    uint32_t block_count = 0;
    uint32_t embedding_length = 0;
    uint32_t head_count = 0;
    uint32_t head_count_kv = 0;
    uint32_t feed_forward_length = 0;
    float layer_norm_rms_epsilon = 0.0f;

    // Raw metadata for debugging
    std::unordered_map<std::string, std::string> raw_kv_strings;
};

// Parse GGUF header from raw bytes
// Returns true on success, false if magic/version mismatch
bool parse_gguf_header(const uint8_t* data, size_t data_size, ModelMetadata& metadata);

// Get human-readable name for quantization type
const char* get_quant_name(uint32_t file_type);
