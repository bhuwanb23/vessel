#include "gguf_parser.h"

#include <cstdio>
#include <cstring>
#include <map>

// GGUF magic number: "GGUF" in little-endian
static constexpr uint32_t GGUF_MAGIC = 0x46554747;
static constexpr uint32_t GGUF_VERSION_3 = 3;

// Quantization type lookup table
static const std::map<uint32_t, const char*> QUANT_NAMES = {
    {0, "F32"},
    {1, "F16"},
    {2, "Q4_0"},
    {3, "Q4_1"},
    {6, "Q5_0"},
    {7, "Q5_1"},
    {8, "Q8_0"},
    {10, "Q2_K"},
    {11, "Q3_K_S"},
    {12, "Q3_K_M"},
    {13, "Q3_K_L"},
    {14, "Q4_K_S"},
    {15, "Q4_K_M"},
    {16, "Q4_K_L"},
    {17, "Q5_K_S"},
    {18, "Q5_K_M"},
    {19, "Q6_K"},
    {20, "IQ2_XXS"},
    {21, "IQ2_XS"},
    {22, "IQ3_XXS"},
    {23, "IQ1_S"},
    {24, "IQ4_NL"},
    {28, "Q4_0_4_4"},
    {29, "Q4_0_4_8"},
    {30, "Q4_0_8_8"},
};

const char* get_quant_name(uint32_t file_type) {
    auto it = QUANT_NAMES.find(file_type);
    if (it != QUANT_NAMES.end()) {
        return it->second;
    }
    return "unknown";
}

// Helper: read a little-endian value from buffer
template<typename T>
static T read_le(const uint8_t* data, size_t offset) {
    T value = 0;
    memcpy(&value, data + offset, sizeof(T));
    return value;
}

// Helper: read a GGUF string (8-byte length prefix + UTF-8 bytes)
static std::string read_gguf_string(const uint8_t* data, size_t data_size, size_t& offset) {
    if (offset + 8 > data_size) return "";

    uint64_t length = read_le<uint64_t>(data, offset);
    offset += 8;

    if (offset + length > data_size) return "";

    std::string str(reinterpret_cast<const char*>(data + offset), length);
    offset += length;
    return str;
}

// Helper: skip a value based on its type
static bool skip_gguf_value(const uint8_t* data, size_t data_size, size_t& offset, uint32_t type_id) {
    switch (type_id) {
        case 0: // UINT8
        case 1: // INT8
        case 7: // BOOL
            offset += 1;
            return true;
        case 2: // UINT16
        case 3: // INT16
            offset += 2;
            return true;
        case 4: // UINT32
        case 5: // INT32
        case 6: // FLOAT32
            offset += 4;
            return true;
        case 10: // UINT64
        case 11: // INT64
        case 12: // FLOAT64
            offset += 8;
            return true;
        case 8: { // STRING
            if (offset + 8 > data_size) return false;
            uint64_t length = read_le<uint64_t>(data, offset);
            offset += 8 + length;
            return true;
        }
        case 9: { // ARRAY
            if (offset + 12 > data_size) return false;
            uint32_t elem_type = read_le<uint32_t>(data, offset);
            offset += 4;
            uint64_t count = read_le<uint64_t>(data, offset);
            offset += 8;
            // Skip each element
            for (uint64_t i = 0; i < count; i++) {
                if (!skip_gguf_value(data, data_size, offset, elem_type)) return false;
            }
            return true;
        }
        default:
            fprintf(stderr, "Warning: Unknown value type %u at offset %zu\n", type_id, offset);
            return false;
    }
}

bool parse_gguf_header(const uint8_t* data, size_t data_size, ModelMetadata& metadata) {
    // Minimum header size: 24 bytes (magic + version + n_tensors + n_kv)
    if (data_size < 24) {
        fprintf(stderr, "Error: Data too small for GGUF header (%zu bytes)\n", data_size);
        return false;
    }

    size_t offset = 0;

    // Read magic number
    uint32_t magic = read_le<uint32_t>(data, offset);
    offset += 4;
    if (magic != GGUF_MAGIC) {
        fprintf(stderr, "Error: Invalid GGUF magic number 0x%08X (expected 0x%08X)\n", magic, GGUF_MAGIC);
        return false;
    }

    // Read version
    uint32_t version = read_le<uint32_t>(data, offset);
    offset += 4;
    if (version != GGUF_VERSION_3) {
        fprintf(stderr, "Warning: GGUF version %u (expected %u)\n", version, GGUF_VERSION_3);
    }

    // Read n_tensors (skip)
    uint64_t n_tensors = read_le<uint64_t>(data, offset);
    offset += 8;

    // Read n_kv (number of metadata entries)
    uint64_t n_kv = read_le<uint64_t>(data, offset);
    offset += 8;

    // Parse KV pairs
    for (uint64_t i = 0; i < n_kv && offset < data_size; i++) {
        // Read key
        std::string key = read_gguf_string(data, data_size, offset);
        if (key.empty()) break;

        // Read value type
        if (offset + 4 > data_size) break;
        uint32_t type_id = read_le<uint32_t>(data, offset);
        offset += 4;

        // Read value based on type
        switch (type_id) {
            case 4: { // UINT32
                if (offset + 4 > data_size) goto done;
                uint32_t value = read_le<uint32_t>(data, offset);
                offset += 4;

                if (key == "general.file_type") {
                    metadata.file_type = value;
                    metadata.file_type_name = get_quant_name(value);
                } else if (key.find(".context_length") != std::string::npos) {
                    metadata.context_length = value;
                } else if (key.find(".block_count") != std::string::npos) {
                    metadata.block_count = value;
                } else if (key.find(".embedding_length") != std::string::npos) {
                    metadata.embedding_length = value;
                } else if (key.find(".attention.head_count_kv") != std::string::npos) {
                    metadata.head_count_kv = value;
                } else if (key.find(".attention.head_count") != std::string::npos) {
                    metadata.head_count = value;
                } else if (key.find(".feed_forward_length") != std::string::npos) {
                    metadata.feed_forward_length = value;
                }
                break;
            }
            case 10: { // UINT64
                if (offset + 8 > data_size) goto done;
                uint64_t value = read_le<uint64_t>(data, offset);
                offset += 8;

                if (key == "general.parameter_count") {
                    metadata.parameter_count = value;
                }
                break;
            }
            case 6: { // FLOAT32
                if (offset + 4 > data_size) goto done;
                float value = read_le<float>(data, offset);
                offset += 4;

                if (key.find(".layer_norm_rms_epsilon") != std::string::npos) {
                    metadata.layer_norm_rms_epsilon = value;
                }
                break;
            }
            case 8: { // STRING
                std::string value = read_gguf_string(data, data_size, offset);

                if (key == "general.architecture") {
                    metadata.architecture = value;
                } else if (key == "general.name") {
                    metadata.name = value;
                }

                // Store raw value for debugging
                metadata.raw_kv_strings[key] = value;
                break;
            }
            default: {
                // Skip unknown types
                if (!skip_gguf_value(data, data_size, offset, type_id)) {
                    fprintf(stderr, "Warning: Failed to skip value for key '%s'\n", key.c_str());
                    goto done;
                }
                break;
            }
        }
    }

done:
    return !metadata.architecture.empty() || !metadata.name.empty();
}
