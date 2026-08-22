#include "predictor.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <iomanip>

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

uint64_t predict_kv_cache_memory(const ModelSpec& model, uint32_t context_length, uint32_t kv_quant_bits) {
    if (model.layers == 0 || model.kv_heads == 0 || model.head_dim == 0) return 0;

    // Formula: 2 * layers * context_length * kv_heads * head_dim * kv_quant_bits / 8
    // The "2" accounts for both K and V caches
    uint64_t kv_elements = 2ULL * model.layers * context_length * model.kv_heads * model.head_dim;
    uint64_t kv_bytes = (kv_elements * kv_quant_bits) / 8;
    return kv_bytes;
}

uint64_t predict_overhead_memory(const ModelSpec& model, uint32_t batch_size) {
    // Overhead includes:
    // - CUDA context: ~200-300MB on Windows
    // - Activation buffers: depends on batch size and model dimensions
    // - ggml internal buffers: ~100MB
    // - OS and driver overhead: ~50MB

    uint64_t base_overhead = 350ULL * 1024 * 1024;  // 350MB base

    // Activation memory scales with batch_size * embedding_dim * layers
    // Rough estimate: 4 bytes per activation element
    uint64_t activation_bytes = 4ULL * batch_size * model.embedding_dim * model.layers;

    return base_overhead + activation_bytes;
}

// =============================================================================
// Performance Prediction Functions
// =============================================================================

double predict_tokens_per_sec(const HardwareSpec& hw, const ModelSpec& model, uint32_t gpu_layers) {
    if (model.param_count == 0 || model.bits_per_weight == 0) return 0;

    // Decode speed is primarily memory-bandwidth bound
    // Formula: tokens_per_sec ≈ bandwidth / model_size_in_memory
    //
    // For GPU-only: use GPU bandwidth
    // For CPU-only: use RAM bandwidth
    // For split: weighted average based on layers on each

    uint64_t weight_bytes = predict_weight_memory(model);
    if (weight_bytes == 0) return 0;

    double effective_bandwidth_gbs = 0.0;

    if (gpu_layers == 0) {
        // CPU-only: use RAM bandwidth
        effective_bandwidth_gbs = hw.ram_bandwidth_gbs;
        if (effective_bandwidth_gbs <= 0) {
            // Estimate RAM bandwidth if not measured (DDR4-3200 is ~25 GB/s)
            effective_bandwidth_gbs = 25.0;
        }
    } else if (gpu_layers >= model.layers) {
        // Full GPU: use GPU bandwidth
        effective_bandwidth_gbs = hw.gpu_bandwidth_gbs;
        if (effective_bandwidth_gbs <= 0) {
            // Can't predict without GPU bandwidth
            return 0;
        }
    } else {
        // Split: weighted average
        double gpu_ratio = static_cast<double>(gpu_layers) / model.layers;
        double cpu_ratio_val = 1.0 - gpu_ratio;

        double gpu_bw = hw.gpu_bandwidth_gbs > 0 ? hw.gpu_bandwidth_gbs : 0;
        double cpu_bw = hw.ram_bandwidth_gbs > 0 ? hw.ram_bandwidth_gbs : 25.0;

        effective_bandwidth_gbs = (gpu_ratio * gpu_bw) + (cpu_ratio_val * cpu_bw);
    }

    // tokens_per_sec = bandwidth (GB/s) * 1e9 / weight_bytes
    // This gives the theoretical maximum decode speed
    double tps = (effective_bandwidth_gbs * 1e9) / static_cast<double>(weight_bytes);

    // Apply efficiency factor (real-world is ~15-25% of theoretical)
    // This accounts for:
    // - KV cache reads (not just weights)
    // - Attention computation overhead
    // - Memory access patterns
    // - CUDA kernel launch overhead
    // - Not all layers may be on GPU
    // Calibrated against RTX 5060 + Llama-3.2-3B baseline: 43.4 t/s actual vs 266 t/s theoretical = 16.3%
    double efficiency = 0.18;  // Calibrated estimate

    // Adjust efficiency based on model size relative to VRAM
    // Larger models that barely fit have worse cache behavior
    if (gpu_layers > 0 && hw.vram_total_bytes > 0) {
        double model_fit_ratio = static_cast<double>(weight_bytes) / hw.vram_total_bytes;
        if (model_fit_ratio > 0.8) {
            efficiency *= 0.8;  // Penalize models that barely fit
        }
    }

    return tps * efficiency;
}

double predict_prompt_eval_speed(const HardwareSpec& hw, const ModelSpec& model, uint32_t gpu_layers) {
    // Prompt evaluation (prefill) is compute-bound, not memory-bound
    // Formula: tokens_per_sec ≈ (GPU_TFLOPS * 1e12) / (2 * params_per_layer * layers)
    //
    // For CPU-only: much slower, depends on CPU FLOPS
    // For GPU: use TFLOPS

    if (model.param_count == 0 || model.layers == 0) return 0;

    // Compute-bound estimation
    // Each token requires ~2 * params FLOPs (forward pass)
    double total_flops = 2.0 * static_cast<double>(model.param_count);

    double tflops = 0.0;

    if (gpu_layers > 0 && hw.gpu_tflops_fp16 > 0) {
        // GPU-accelerated: use GPU TFLOPS
        // Scale by fraction of layers on GPU
        double gpu_ratio = static_cast<double>(gpu_layers) / model.layers;
        tflops = hw.gpu_tflops_fp16 * gpu_ratio;

        // Add CPU contribution for remaining layers
        if (gpu_ratio < 1.0) {
            // Assume ~1 TFLOPS for CPU (rough estimate)
            tflops += (1.0 - gpu_ratio) * 1.0;
        }
    } else {
        // CPU-only: assume ~1-2 TFLOPS for modern CPU
        tflops = 1.5;
    }

    if (tflops <= 0) return 0;

    // tokens_per_sec = TFLOPS * 1e12 / FLOPS_per_token
    double tps = (tflops * 1e12) / total_flops;

    // Apply efficiency factor for prompt processing (~70-85%)
    double efficiency = 0.75;
    return tps * efficiency;
}

double predict_ttft_ms(const HardwareSpec& hw, const ModelSpec& model, uint32_t prompt_tokens, uint32_t gpu_layers) {
    if (prompt_tokens == 0) return 0;

    // TTFT = prompt_tokens / prompt_eval_speed * 1000 (convert to ms)
    double prompt_tps = predict_prompt_eval_speed(hw, model, gpu_layers);
    if (prompt_tps <= 0) return 10000.0;  // 10s default if can't predict

    double ttft_seconds = static_cast<double>(prompt_tokens) / prompt_tps;
    return ttft_seconds * 1000.0;
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
// Main Prediction Function
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
    uint64_t overhead_bytes = predict_overhead_memory(model, strategy.batch_size);

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

    // Calculate performance predictions
    pred.tokens_per_sec = predict_tokens_per_sec(hw, model, gpu_layers);
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

    return pred;
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

// =============================================================================
// BPW Validation
// =============================================================================

double validate_bpw_from_file(uint64_t file_size_bytes, uint64_t param_count) {
    if (file_size_bytes == 0 || param_count == 0) return 0.0;
    
    // Formula: bpw = (file_size_bytes * 8) / param_count
    double bpw = (static_cast<double>(file_size_bytes) * 8.0) / static_cast<double>(param_count);
    return bpw;
}
