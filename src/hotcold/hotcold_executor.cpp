#include "hotcold/hotcold_executor.h"
#include "hotcold/mask_file.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>

// =============================================================================
// Hot/cold config creation
// =============================================================================

HotColdExecConfig create_hotcold_config(const std::string& model_path) {
    HotColdExecConfig config;
    
    // Check if mask file exists
    std::string mask_path = get_mask_file_path(model_path);
    if (!mask_file_exists(model_path)) {
        return config;  // enabled = false
    }
    
    return create_hotcold_config_from_mask(model_path, mask_path);
}

HotColdExecConfig create_hotcold_config_from_mask(
    const std::string& model_path,
    const std::string& mask_path)
{
    HotColdExecConfig config;
    
    // Load the mask file
    config.profile = load_mask_file(mask_path);
    if (config.profile.num_layers == 0 || config.profile.ffn_dim == 0) {
        fprintf(stderr, "[HotCold] Error: Invalid mask file: %s\n", mask_path.c_str());
        return config;
    }
    
    config.enabled = true;
    config.mask_file_path = mask_path;
    config.hot_ratio = config.profile.hot_ratio;
    
    // Get hot neurons per layer from the first layer (assumed uniform)
    if (!config.profile.layers.empty()) {
        config.hot_neurons_per_layer = config.profile.layers[0].n_hot;
    }
    config.total_neurons_per_layer = config.profile.ffn_dim;
    
    fprintf(stderr, "[HotCold] Loaded mask: %u layers, %u/%u hot neurons (%.0f%%)\n",
            config.profile.num_layers, config.hot_neurons_per_layer,
            config.total_neurons_per_layer, config.hot_ratio * 100.0);
    
    return config;
}

// =============================================================================
// Tensor split calculation
// =============================================================================

void calculate_tensor_split(
    const HotNeuronProfile& profile,
    uint32_t hidden_dim,
    uint32_t ffn_dim,
    uint32_t n_layers,
    double bytes_per_param,
    float& out_gpu_fraction,
    float& out_cpu_fraction)
{
    if (profile.layers.empty() || ffn_dim == 0 || hidden_dim == 0 || n_layers == 0) {
        // No mask data — default to full GPU
        out_gpu_fraction = 1.0f;
        out_cpu_fraction = 0.0f;
        return;
    }
    
    uint32_t n_hot = profile.layers[0].n_hot;
    uint32_t n_cold = ffn_dim - n_hot;
    
    // Calculate weight bytes for different components
    // Attention weights (always on GPU if possible)
    // Per layer: Q[embd,embd] + K[embd,kv_dim] + V[embd,kv_dim] + O[embd,embd]
    // Simplified: ~4 * hidden_dim^2 per layer (for standard MHA)
    uint64_t attention_per_layer = 4ULL * hidden_dim * hidden_dim;
    uint64_t total_attention = attention_per_layer * n_layers;
    
    // FFN weights per layer: gate[ffn,hidden] + up[ffn,hidden] + down[hidden,ffn]
    // = 3 * ffn_dim * hidden_dim per layer
    uint64_t ffn_per_layer = 3ULL * ffn_dim * hidden_dim;
    uint64_t total_ffn = ffn_per_layer * n_layers;
    
    // Hot FFN weights (should be on GPU)
    uint64_t hot_ffn_per_layer = 3ULL * n_hot * hidden_dim;
    uint64_t total_hot_ffn = hot_ffn_per_layer * n_layers;
    
    // Cold FFN weights (should be on CPU)
    uint64_t cold_ffn_per_layer = 3ULL * n_cold * hidden_dim;
    uint64_t total_cold_ffn = cold_ffn_per_layer * n_layers;
    
    // Embeddings + output head (always on GPU)
    uint64_t embedding_params = 2ULL * hidden_dim * hidden_dim;  // input + output
    uint64_t embedding_bytes = static_cast<uint64_t>(embedding_params * bytes_per_param);
    
    // Norms (negligible, ~hidden_dim per layer)
    uint64_t norm_bytes = n_layers * hidden_dim * sizeof(float);
    
    // Total bytes
    uint64_t total_bytes = static_cast<uint64_t>(
        (total_attention + total_ffn) * bytes_per_param) + embedding_bytes + norm_bytes;
    
    // GPU should hold: attention + hot FFN + embeddings + norms
    uint64_t gpu_bytes = static_cast<uint64_t>(
        (total_attention + total_hot_ffn) * bytes_per_param) + embedding_bytes + norm_bytes;
    
    // CPU should hold: cold FFN
    uint64_t cpu_bytes = static_cast<uint64_t>(total_cold_ffn * bytes_per_param);
    
    // Calculate fractions
    if (total_bytes > 0) {
        out_gpu_fraction = static_cast<float>(gpu_bytes) / static_cast<float>(total_bytes);
        out_cpu_fraction = static_cast<float>(cpu_bytes) / static_cast<float>(total_bytes);
    } else {
        out_gpu_fraction = 1.0f;
        out_cpu_fraction = 0.0f;
    }
    
    // Clamp to valid range
    out_gpu_fraction = std::max(0.0f, std::min(1.0f, out_gpu_fraction));
    out_cpu_fraction = std::max(0.0f, std::min(1.0f, out_cpu_fraction));
    
    // Ensure they sum to ~1.0
    float sum = out_gpu_fraction + out_cpu_fraction;
    if (sum > 0.0f) {
        out_gpu_fraction /= sum;
        out_cpu_fraction /= sum;
    }
    
    fprintf(stderr, "[HotCold] Tensor split: %.1f%% GPU, %.1f%% CPU\n",
            out_gpu_fraction * 100.0f, out_cpu_fraction * 100.0f);
    fprintf(stderr, "[HotCold] GPU weights: %.2f MB, CPU weights: %.2f MB\n",
            gpu_bytes / 1e6, cpu_bytes / 1e6);
}

// =============================================================================
// Print info
// =============================================================================

void print_hotcold_exec_info(const HotColdExecConfig& config) {
    if (!config.enabled) {
        printf("  Hot/Cold: Not active (no mask file found)\n");
        return;
    }
    
    printf("  Hot/Cold: ACTIVE\n");
    printf("  Mask file: %s\n", config.mask_file_path.c_str());
    printf("  Hot neurons: %u / %u (%.0f%%)\n",
           config.hot_neurons_per_layer, config.total_neurons_per_layer,
           config.hot_ratio * 100.0);
    printf("  Tensor split: %.1f%% GPU, %.1f%% CPU\n",
           config.tensor_split_gpu * 100.0f, config.tensor_split_cpu * 100.0);
}
