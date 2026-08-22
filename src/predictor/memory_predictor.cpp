#include "memory_predictor.h"

// =============================================================================
// Memory Prediction Functions
// =============================================================================

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
    
    // Convert kv_quant_bits to bytes per element
    double bytes_per_kv_element = 2.0;  // Default FP16
    if (kv_quant_bits == 8) bytes_per_kv_element = 1.0;
    else if (kv_quant_bits == 4) bytes_per_kv_element = 0.5;
    else if (kv_quant_bits == 16) bytes_per_kv_element = 2.0;
    
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

uint64_t predict_overhead_memory(const ModelSpec& model, uint32_t batch_size, bool use_gpu) {
    // Runtime overhead includes:
    // - CUDA context initialization (~200-500 MB on NVIDIA)
    // - ggml compute buffers (scales with batch size and context)
    // - Driver allocations
    // - Fragmentation waste
    //
    // This constant is calibrated empirically, not derivable analytically.
    // Starting estimate: 512 MB for CUDA backend, 128 MB for CPU-only.
    
    uint64_t base_overhead = use_gpu ? (512ULL * 1024 * 1024) : (128ULL * 1024 * 1024);
    
    // Activation memory scales with batch_size * embedding_dim * layers
    // Rough estimate: 4 bytes per activation element
    uint64_t activation_bytes = 4ULL * batch_size * model.embedding_dim * model.layers;
    
    return base_overhead + activation_bytes;
}

double predict_kv_bytes_per_token(const ModelSpec& model, uint32_t kv_quant_bits) {
    if (model.layers == 0) return 0.0;
    
    // Convert kv_quant_bits to bytes per element
    double bytes_per_kv_element = 2.0;  // Default FP16
    if (kv_quant_bits == 8) bytes_per_kv_element = 1.0;
    else if (kv_quant_bits == 4) bytes_per_kv_element = 0.5;
    else if (kv_quant_bits == 16) bytes_per_kv_element = 2.0;
    
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
