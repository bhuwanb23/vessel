#include "hotcold/mask_file.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>

// =============================================================================
// Constants
// =============================================================================

static const char MASK_MAGIC[] = "HOTM";
static const uint32_t MASK_VERSION = 1;

// =============================================================================
// Path helpers
// =============================================================================

std::string get_mask_file_path(const std::string& model_path) {
    std::string path = model_path;
    size_t gguf_pos = path.rfind(".gguf");
    if (gguf_pos != std::string::npos) {
        path = path.substr(0, gguf_pos);
    }
    return path + ".hot_neurons.bin";
}

bool mask_file_exists(const std::string& model_path) {
    std::string path = get_mask_file_path(model_path);
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    fclose(f);
    return true;
}

// =============================================================================
// Save mask file
// =============================================================================

bool save_mask_file(const HotNeuronProfile& profile, const std::string& path) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "[MaskFile] Error: Could not write to %s\n", path.c_str());
        return false;
    }

    // Header
    fwrite(MASK_MAGIC, 4, 1, f);
    fwrite(&MASK_VERSION, sizeof(uint32_t), 1, f);
    fwrite(&profile.num_layers, sizeof(uint32_t), 1, f);
    fwrite(&profile.ffn_dim, sizeof(uint32_t), 1, f);

    // Metadata
    uint32_t name_len = static_cast<uint32_t>(profile.model_name.size());
    fwrite(&name_len, sizeof(uint32_t), 1, f);
    fwrite(profile.model_name.c_str(), name_len, 1, f);

    uint32_t activation = static_cast<uint32_t>(profile.activation);
    fwrite(&activation, sizeof(uint32_t), 1, f);
    fwrite(&profile.hot_ratio, sizeof(double), 1, f);
    fwrite(&profile.num_prompts_used, sizeof(uint32_t), 1, f);
    fwrite(&profile.avg_activation_rate, sizeof(double), 1, f);

    // Per-layer masks
    uint32_t mask_bytes = (profile.ffn_dim + 7) / 8;  // Round up to bytes
    for (uint32_t i = 0; i < profile.num_layers && i < profile.layers.size(); i++) {
        const LayerHotSet& layer = profile.layers[i];
        fwrite(&layer.n_hot, sizeof(uint32_t), 1, f);

        // Build bit vector
        std::vector<uint8_t> mask(mask_bytes, 0);
        for (uint32_t idx : layer.hot_indices) {
            if (idx < profile.ffn_dim) {
                mask[idx / 8] |= (1 << (idx % 8));
            }
        }
        fwrite(mask.data(), 1, mask_bytes, f);

        // Redundant hot count for validation
        fwrite(&layer.n_hot, sizeof(uint32_t), 1, f);
    }

    fclose(f);
    fprintf(stderr, "[MaskFile] Saved profile to %s (%u layers, %.0f%% hot)\n",
            path.c_str(), profile.num_layers, profile.hot_ratio * 100.0);
    return true;
}

// =============================================================================
// Load mask file
// =============================================================================

HotNeuronProfile load_mask_file(const std::string& path) {
    HotNeuronProfile profile;

    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        return profile;  // Empty profile on failure
    }

    // Header
    char magic[4];
    if (fread(magic, 4, 1, f) != 1 || memcmp(magic, MASK_MAGIC, 4) != 0) {
        fprintf(stderr, "[MaskFile] Invalid magic in %s\n", path.c_str());
        fclose(f);
        return profile;
    }

    uint32_t version;
    fread(&version, sizeof(uint32_t), 1, f);
    if (version != MASK_VERSION) {
        fprintf(stderr, "[MaskFile] Unsupported version %u in %s\n", version, path.c_str());
        fclose(f);
        return profile;
    }

    fread(&profile.num_layers, sizeof(uint32_t), 1, f);
    fread(&profile.ffn_dim, sizeof(uint32_t), 1, f);

    // Metadata
    uint32_t name_len;
    fread(&name_len, sizeof(uint32_t), 1, f);
    profile.model_name.resize(name_len);
    if (name_len > 0) fread(&profile.model_name[0], name_len, 1, f);

    uint32_t activation;
    fread(&activation, sizeof(uint32_t), 1, f);
    profile.activation = static_cast<ActivationType>(activation);
    fread(&profile.hot_ratio, sizeof(double), 1, f);
    fread(&profile.num_prompts_used, sizeof(uint32_t), 1, f);
    fread(&profile.avg_activation_rate, sizeof(double), 1, f);

    // Per-layer masks
    uint32_t mask_bytes = (profile.ffn_dim + 7) / 8;
    profile.layers.resize(profile.num_layers);

    for (uint32_t i = 0; i < profile.num_layers; i++) {
        LayerHotSet& layer = profile.layers[i];
        layer.layer_index = i;
        layer.ffn_dim = profile.ffn_dim;

        fread(&layer.n_hot, sizeof(uint32_t), 1, f);

        // Read bit vector
        std::vector<uint8_t> mask(mask_bytes);
        if (mask_bytes > 0) fread(mask.data(), 1, mask_bytes, f);

        // Decode hot indices from bit vector
        layer.hot_indices.clear();
        layer.cold_indices.clear();
        for (uint32_t bit = 0; bit < profile.ffn_dim; bit++) {
            if (mask[bit / 8] & (1 << (bit % 8))) {
                layer.hot_indices.push_back(bit);
            } else {
                layer.cold_indices.push_back(bit);
            }
        }
        layer.n_cold = static_cast<uint32_t>(layer.cold_indices.size());

        // Validate redundant count
        uint32_t stored_count;
        fread(&stored_count, sizeof(uint32_t), 1, f);
        if (stored_count != layer.n_hot) {
            fprintf(stderr, "[MaskFile] Warning: Layer %u hot count mismatch "
                    "(stored=%u, decoded=%u)\n", i, stored_count, layer.n_hot);
        }
    }

    fclose(f);
    return profile;
}

// =============================================================================
// Validate mask file
// =============================================================================

std::string validate_mask_file(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return "File not found: " + path;

    // Check magic
    char magic[4];
    if (fread(magic, 4, 1, f) != 1 || memcmp(magic, MASK_MAGIC, 4) != 0) {
        fclose(f);
        return "Invalid magic bytes";
    }

    // Check version
    uint32_t version;
    fread(&version, sizeof(uint32_t), 1, f);
    if (version != MASK_VERSION) {
        fclose(f);
        return "Unsupported version: " + std::to_string(version);
    }

    // Check dimensions
    uint32_t n_layers, ffn_dim;
    fread(&n_layers, sizeof(uint32_t), 1, f);
    fread(&ffn_dim, sizeof(uint32_t), 1, f);

    if (n_layers == 0 || n_layers > 200) {
        fclose(f);
        return "Invalid layer count: " + std::to_string(n_layers);
    }
    if (ffn_dim == 0 || ffn_dim > 100000) {
        fclose(f);
        return "Invalid FFN dimension: " + std::to_string(ffn_dim);
    }

    // Check file size matches expected
    uint32_t mask_bytes = (ffn_dim + 7) / 8;
    // Header(16) + name_len(4) + name + activation(4) + hot_ratio(8) + n_prompts(4) + avg_rate(8)
    // + n_layers * (4 + mask_bytes + 4)
    long expected_size = 16 + 4 + 0 + 4 + 8 + 4 + 8 + n_layers * (4 + mask_bytes + 4);

    // Get actual file size
    fseek(f, 0, SEEK_END);
    long actual_size = ftell(f);
    fclose(f);

    // Allow some slack for the model name
    if (actual_size < expected_size - 1000) {
        return "File truncated (expected ~" + std::to_string(expected_size) +
               " bytes, got " + std::to_string(actual_size) + ")";
    }

    return "";  // Valid
}

// =============================================================================
// Print mask file info
// =============================================================================

bool print_mask_file_info(const std::string& path) {
    std::string error = validate_mask_file(path);
    if (!error.empty()) {
        fprintf(stderr, "[MaskFile] Invalid: %s — %s\n", path.c_str(), error.c_str());
        return false;
    }

    HotNeuronProfile profile = load_mask_file(path);

    printf("=== Hot Neuron Mask File ===\n");
    printf("Path:     %s\n", path.c_str());
    printf("Model:    %s\n", profile.model_name.c_str());
    printf("Layers:   %u\n", profile.num_layers);
    printf("FFN dim:  %u\n", profile.ffn_dim);
    printf("Hot ratio: %.0f%%\n", profile.hot_ratio * 100.0);
    printf("Prompts:  %u\n", profile.num_prompts_used);
    printf("Avg activation rate: %.1f%%\n", profile.avg_activation_rate * 100.0);
    printf("\nPer-layer hot neuron counts:\n");

    uint32_t total_hot = 0;
    for (uint32_t i = 0; i < profile.num_layers && i < profile.layers.size(); i++) {
        const LayerHotSet& layer = profile.layers[i];
        printf("  Layer %2u: %u / %u hot (%.0f%%)\n",
               i, layer.n_hot, layer.ffn_dim,
               (layer.ffn_dim > 0) ? 100.0 * layer.n_hot / layer.ffn_dim : 0.0);
        total_hot += layer.n_hot;
    }

    printf("\nTotal hot neurons: %u / %u (%.1f%%)\n",
           total_hot, profile.num_layers * profile.ffn_dim,
           (profile.num_layers > 0 && profile.ffn_dim > 0) ?
               100.0 * total_hot / (profile.num_layers * profile.ffn_dim) : 0.0);

    return true;
}
