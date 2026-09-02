#include "memory_predictor.h"

// =============================================================================
// Memory Prediction Functions
// =============================================================================

// Convert kv_quant_bits to bytes per KV element.
// Used by both predict_kv_cache_memory and predict_kv_bytes_per_token.
static double kv_bytes_per_element(uint32_t kv_quant_bits) {
    switch (kv_quant_bits) {
        case 4:  return 0.5;   // Q4 KV cache
        case 8:  return 1.0;   // Q8 KV cache
        case 16: return 2.0;   // FP16 (default)
        default: return 2.0;
    }
}

uint64_t predict_weight_memory(const ModelSpec& model) {
    if (model.param_count == 0 || model.bits_per_weight == 0) return 0;

    // Formula: param_count * bits_per_weight / 8
    // This gives the raw weight storage in bytes
    double bits = static_cast<double>(model.param_count) * model.bits_per_weight;
    return static_cast<uint64_t>(bits / 8.0);
}

uint64_t predict_kv_cache_memory(const ModelSpec& model, uint32_t context_length, 
                                  uint32_t kv_quant_bits, uint32_t batch_size) {
    if (model.layers == 0) return 0;
    
    // Default to batch_size=1 if not provided
    if (batch_size == 0) batch_size = 1;
    
    double bytes_per_kv_element = kv_bytes_per_element(kv_quant_bits);
    
    // Check if this is an MLA model (DeepSeek/Kimi-class)
    bool is_mla = (model.architecture == "deepseek2" || model.architecture == "deepseek_v2");
    
    uint64_t kv_bytes = 0;
    
    if (is_mla && model.kv_lora_rank > 0) {
        // MLA attention: compressed KV cache
        // Formula: 2 * layers * kv_lora_rank * context * batch * bytes
        //        + layers * qk_rope_head_dim * context * batch * bytes
        uint64_t kv_lora_part = 2ULL * model.layers * model.kv_lora_rank * context_length * batch_size;
        uint64_t rope_part = model.layers * model.qk_rope_head_dim * context_length * batch_size;
        kv_bytes = static_cast<uint64_t>((kv_lora_part + rope_part) * bytes_per_kv_element);
    } else if (model.kv_heads > 0 && model.head_dim > 0) {
        // Standard attention: full KV cache per head
        // Formula: 2 * layers * kv_heads * head_dim * context * batch * bytes
        uint64_t kv_elements = 2ULL * model.layers * model.kv_heads * model.head_dim * context_length * batch_size;
        kv_bytes = static_cast<uint64_t>(kv_elements * bytes_per_kv_element);
    }
    
    return kv_bytes;
}

uint64_t predict_overhead_memory(const ModelSpec& model, uint32_t batch_size, bool use_gpu,
                                  bool is_unified_memory) {
    // Runtime overhead includes:
    // - GPU context initialization (varies by platform)
    // - ggml compute buffers (scales with batch size and context)
    // - Driver allocations
    // - Fragmentation waste
    //
    // Platform-specific overhead (Phase G):
    //   NVIDIA CUDA:  200-500 MB (large CUDA context)
    //   AMD ROCm:     150-400 MB (slightly smaller)
    //   Apple Metal:   50-200 MB (unified memory, smaller context)
    //   CPU-only:     128 MB (minimal)
    //
    // This constant is calibrated empirically, not derivable analytically.
    //
    uint64_t base_overhead = 0;
    
    if (use_gpu) {
        if (is_unified_memory) {
            // Apple Silicon: smaller overhead (unified memory, no VRAM context)
            base_overhead = 128ULL * 1024 * 1024;  // 128 MB
        } else {
            // Discrete GPU (NVIDIA/AMD): larger overhead (CUDA/HIP context)
            base_overhead = 512ULL * 1024 * 1024;  // 512 MB
        }
    } else {
        // CPU-only: minimal overhead
        base_overhead = 128ULL * 1024 * 1024;  // 128 MB
    }
    
    // Activation memory scales with batch_size * embedding_dim * layers
    // Rough estimate: 4 bytes per activation element
    uint64_t activation_bytes = 4ULL * batch_size * model.embedding_dim * model.layers;
    
    return base_overhead + activation_bytes;
}

double predict_kv_bytes_per_token(const ModelSpec& model, uint32_t kv_quant_bits) {
    if (model.layers == 0) return 0.0;
    
    double bytes_per_kv_element = kv_bytes_per_element(kv_quant_bits);
    
    // Check if MLA model
    bool is_mla = (model.architecture == "deepseek2" || model.architecture == "deepseek_v2");
    
    double kv_per_token = 0.0;
    
    if (is_mla && model.kv_lora_rank > 0) {
        // MLA: 2 * kv_lora_rank + qk_rope_head_dim per layer
        kv_per_token = (2.0 * model.kv_lora_rank + model.qk_rope_head_dim) * bytes_per_kv_element;
    } else if (model.kv_heads > 0 && model.head_dim > 0) {
        // Standard: 2 * kv_heads * head_dim per layer
        kv_per_token = 2.0 * model.kv_heads * model.head_dim * bytes_per_kv_element;
    }
    
    // Total across all layers
    return kv_per_token * model.layers;
}
