#include "context_analyzer.h"
#include "memory_predictor.h"
#include <algorithm>

// =============================================================================
// Context Analysis Functions
// =============================================================================

uint64_t calculate_memory_budget(const HardwareSpec& hw, const ModelSpec& model, 
                                 const StrategyConfig& strategy) {
    // Determine effective GPU layers
    uint32_t gpu_layers = strategy.gpu_layers;
    if (gpu_layers == 0 && strategy.placement == PlacementStrategy::FULL_GPU) {
        gpu_layers = model.layers;
    } else if (strategy.placement == PlacementStrategy::CPU_ONLY) {
        gpu_layers = 0;
    }
    
    uint64_t memory_budget = 0;
    
    if (gpu_layers >= model.layers) {
        // Full GPU: budget is VRAM
        memory_budget = hw.vram_free_bytes;
    } else if (gpu_layers == 0) {
        // CPU only: budget is RAM
        memory_budget = hw.ram_free_bytes;
    } else {
        // Split: budget is weighted average
        double gpu_ratio = static_cast<double>(gpu_layers) / model.layers;
        memory_budget = static_cast<uint64_t>(hw.vram_free_bytes * gpu_ratio + 
                                               hw.ram_free_bytes * (1.0 - gpu_ratio));
    }
    
    // Apply 90% safety margin
    memory_budget = static_cast<uint64_t>(memory_budget * 0.9);
    
    return memory_budget;
}

uint32_t calculate_max_safe_context(const HardwareSpec& hw, const ModelSpec& model, 
                                    const StrategyConfig& strategy) {
    // Determine effective GPU layers
    uint32_t gpu_layers = strategy.gpu_layers;
    if (gpu_layers == 0 && strategy.placement == PlacementStrategy::FULL_GPU) {
        gpu_layers = model.layers;
    } else if (strategy.placement == PlacementStrategy::CPU_ONLY) {
        gpu_layers = 0;
    }
    
    // Calculate weight memory
    uint64_t weight_bytes = predict_weight_memory(model);
    
    // Calculate overhead
    bool use_gpu = (gpu_layers > 0);
    uint64_t overhead_bytes = predict_overhead_memory(model, strategy.batch_size, use_gpu);
    
    // Get memory budget
    uint64_t memory_budget = calculate_memory_budget(hw, model, strategy);
    
    // Calculate available space for KV cache
    int64_t available_for_kv = static_cast<int64_t>(memory_budget) - weight_bytes - overhead_bytes;
    if (available_for_kv <= 0) return 0;  // No room for KV cache
    
    // Convert kv_quant_bits to bytes per element
    double bytes_per_kv_element = 2.0;  // Default FP16
    if (strategy.kv_quant_bits == 8) bytes_per_kv_element = 1.0;
    else if (strategy.kv_quant_bits == 4) bytes_per_kv_element = 0.5;
    else if (strategy.kv_quant_bits == 16) bytes_per_kv_element = 2.0;
    
    // Check if MLA model
    bool is_mla = (model.architecture == "deepseek2" || model.architecture == "deepseek_v2");
    
    // Calculate KV cache per token
    double kv_per_token = 0.0;
    if (is_mla && model.kv_lora_rank > 0) {
        // MLA: 2 * kv_lora_rank + qk_rope_head_dim per layer
        kv_per_token = (2.0 * model.kv_lora_rank + model.qk_rope_head_dim) * bytes_per_kv_element;
    } else if (model.kv_heads > 0 && model.head_dim > 0) {
        // Standard: 2 * kv_heads * head_dim per layer
        kv_per_token = 2.0 * model.kv_heads * model.head_dim * bytes_per_kv_element;
    }
    
    if (kv_per_token <= 0) return 0;
    
    // Total KV cache per token across all layers
    double kv_per_token_total = kv_per_token * model.layers * strategy.batch_size;
    
    // Max context = available KV space / KV per token
    uint32_t max_ctx = static_cast<uint32_t>(available_for_kv / kv_per_token_total);
    
    // Clamp to model's maximum context
    if (model.context_length > 0 && max_ctx > model.context_length) {
        max_ctx = model.context_length;
    }
    
    return max_ctx;
}

bool is_context_viable(const HardwareSpec& hw, const ModelSpec& model, 
                       const StrategyConfig& strategy, uint32_t context_length) {
    // Create a modified strategy with the specified context
    StrategyConfig modified = strategy;
    modified.context_length = context_length;
    
    // Calculate max safe context
    uint32_t max_safe = calculate_max_safe_context(hw, model, modified);
    
    return context_length <= max_safe;
}
