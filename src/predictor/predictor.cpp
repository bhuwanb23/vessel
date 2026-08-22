#include "predictor.h"
#include <sstream>
#include <iomanip>

// =============================================================================
// Main Prediction Function (Orchestrator)
// =============================================================================

Prediction predict(const HardwareSpec& hw, const ModelSpec& model, const StrategyConfig& strategy) {
    Prediction pred;
    
    // Determine effective context length
    uint32_t ctx_len = strategy.context_length > 0 ? strategy.context_length : model.context_length;
    if (ctx_len == 0) ctx_len = 4096;  // Default fallback
    
    // Determine effective GPU layers
    uint32_t gpu_layers = strategy.gpu_layers;
    if (gpu_layers == 0 && strategy.placement == PlacementStrategy::FULL_GPU) {
        gpu_layers = model.layers;
    } else if (strategy.placement == PlacementStrategy::CPU_ONLY) {
        gpu_layers = 0;
    }
    
    // Calculate memory components
    pred.memory_vram_bytes = 0;
    pred.memory_ram_bytes = 0;
    
    // 1. Weight memory
    uint64_t weight_bytes = predict_weight_memory(model);
    
    // 2. KV cache memory
    uint64_t kv_bytes = predict_kv_cache_memory(model, ctx_len, strategy.kv_quant_bits);
    
    // 3. Overhead memory
    bool use_gpu = (gpu_layers > 0);
    uint64_t overhead_bytes = predict_overhead_memory(model, strategy.batch_size, use_gpu);
    
    // 4. Distribute memory based on strategy
    if (gpu_layers >= model.layers) {
        // Full GPU: everything on VRAM
        pred.memory_vram_bytes = weight_bytes + kv_bytes + overhead_bytes;
        pred.memory_ram_bytes = 0;
    } else if (gpu_layers == 0) {
        // CPU only: everything on RAM
        pred.memory_vram_bytes = 0;
        pred.memory_ram_bytes = weight_bytes + kv_bytes + overhead_bytes;
    } else {
        // Split: proportionally distribute
        double gpu_ratio = static_cast<double>(gpu_layers) / model.layers;
        double cpu_ratio = 1.0 - gpu_ratio;
        
        // Weights split by layer count
        uint64_t gpu_weights = static_cast<uint64_t>(weight_bytes * gpu_ratio);
        uint64_t cpu_weights = weight_bytes - gpu_weights;
        
        // KV cache: each layer's KV stays with its layer
        uint64_t gpu_kv = static_cast<uint64_t>(kv_bytes * gpu_ratio);
        uint64_t cpu_kv = kv_bytes - gpu_kv;
        
        // Overhead: mostly on GPU (CUDA context)
        uint64_t gpu_overhead = static_cast<uint64_t>(overhead_bytes * 0.7);
        uint64_t cpu_overhead = overhead_bytes - gpu_overhead;
        
        pred.memory_vram_bytes = gpu_weights + gpu_kv + gpu_overhead;
        pred.memory_ram_bytes = cpu_weights + cpu_kv + cpu_overhead;
    }
    
    pred.memory_total_bytes = pred.memory_vram_bytes + pred.memory_ram_bytes;
    
    // Calculate max safe context
    pred.max_safe_context = calculate_max_safe_context(hw, model, strategy);
    
    // Calculate performance predictions
    pred.tokens_per_sec = predict_decode_speed(hw, model, gpu_layers, ctx_len, strategy.kv_quant_bits);
    pred.prompt_eval_tps = predict_prompt_eval_speed(hw, model, gpu_layers);
    pred.ttft_ms = predict_ttft_ms(hw, model, ctx_len, gpu_layers);
    
    // Check viability
    pred.viable = check_viability(hw, pred.memory_total_bytes);
    
    // Determine confidence
    if (hw.gpu_bandwidth_gbs > 0 && model.param_count > 0 && model.bits_per_weight > 0) {
        pred.confidence = PredictionConfidence::HIGH;
    } else if (model.param_count > 0 && model.bits_per_weight > 0) {
        pred.confidence = PredictionConfidence::MEDIUM;
    } else {
        pred.confidence = PredictionConfidence::LOW;
        if (model.bits_per_weight == 0) {
            pred.warnings += "Unknown quantization type - using default bpw estimate. ";
        }
    }
    
    // Add warnings
    if (!pred.viable) {
        pred.warnings = "Model does not fit in available memory with this strategy.";
    } else if (pred.memory_vram_bytes > hw.vram_free_bytes && hw.vram_free_bytes > 0) {
        pred.warnings = "VRAM usage exceeds free VRAM; some layers will be offloaded to CPU.";
    }
    
    // Context warning
    if (ctx_len > pred.max_safe_context && pred.max_safe_context > 0) {
        pred.warnings += "Context length exceeds safe limit; may cause OOM. ";
    }
    
    return pred;
}

// =============================================================================
// Viability Check
// =============================================================================

bool check_viability(const HardwareSpec& hw, uint64_t total_memory_needed) {
    // Check if total memory needed fits in available memory
    // Use 90% threshold to leave headroom for OS
    uint64_t available = hw.vram_free_bytes + hw.ram_free_bytes;
    uint64_t safe_available = static_cast<uint64_t>(available * 0.9);
    
    return total_memory_needed <= safe_available;
}

// =============================================================================
// Utility Functions
// =============================================================================

std::string format_bytes(uint64_t bytes) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);
    
    if (bytes >= 1024ULL * 1024 * 1024) {
        oss << (static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0)) << " GB";
    } else if (bytes >= 1024ULL * 1024) {
        oss << (static_cast<double>(bytes) / (1024.0 * 1024.0)) << " MB";
    } else if (bytes >= 1024ULL) {
        oss << (static_cast<double>(bytes) / 1024.0) << " KB";
    } else {
        oss << bytes << " B";
    }
    
    return oss.str();
}

std::string format_speed(double tokens_per_sec) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1) << tokens_per_sec << " t/s";
    return oss.str();
}

const char* get_placement_name(PlacementStrategy strategy) {
    switch (strategy) {
        case PlacementStrategy::FULL_GPU: return "Full GPU";
        case PlacementStrategy::GPU_CPU_SPLIT: return "GPU/CPU Split";
        case PlacementStrategy::CPU_ONLY: return "CPU Only";
        default: return "Unknown";
    }
}

const char* get_confidence_name(PredictionConfidence confidence) {
    switch (confidence) {
        case PredictionConfidence::HIGH: return "High";
        case PredictionConfidence::MEDIUM: return "Medium";
        case PredictionConfidence::LOW: return "Low";
        default: return "Unknown";
    }
}
