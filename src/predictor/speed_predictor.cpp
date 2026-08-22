#include "speed_predictor.h"
#include "memory_predictor.h"
#include <cmath>

// =============================================================================
// Helper Functions
// =============================================================================

double predict_bytes_per_token(const ModelSpec& model) {
    // Formula: param_count * bits_per_weight / 8
    // This is how many bytes need to be read for each token
    if (model.param_count == 0 || model.bits_per_weight == 0) return 0.0;
    
    double bits = static_cast<double>(model.param_count) * model.bits_per_weight;
    return bits / 8.0;
}

double calculate_effective_bandwidth(const HardwareSpec& hw, uint32_t gpu_layers, uint32_t total_layers) {
    // Returns the effective bandwidth in GB/s based on placement
    if (gpu_layers == 0) {
        // CPU only
        return hw.ram_bandwidth_gbs > 0 ? hw.ram_bandwidth_gbs : 25.0;
    } else if (gpu_layers >= total_layers) {
        // Full GPU
        return hw.gpu_bandwidth_gbs;
    } else {
        // Split - returns weighted average for compatibility
        // NOTE: This is NOT used for actual speed calculation in split mode
        double gpu_ratio = static_cast<double>(gpu_layers) / total_layers;
        double cpu_ratio = 1.0 - gpu_ratio;
        double gpu_bw = hw.gpu_bandwidth_gbs > 0 ? hw.gpu_bandwidth_gbs : 0;
        double cpu_bw = hw.ram_bandwidth_gbs > 0 ? hw.ram_bandwidth_gbs : 25.0;
        return (gpu_ratio * gpu_bw) + (cpu_ratio * cpu_bw);
    }
}

// =============================================================================
// Decode Speed Prediction (Phase D)
// =============================================================================

double predict_decode_speed(const HardwareSpec& hw, const ModelSpec& model, 
                           uint32_t gpu_layers, uint32_t context_length,
                           uint32_t kv_quant_bits) {
    if (model.param_count == 0 || model.bits_per_weight == 0) return 0.0;
    
    // Calculate weight bytes per token
    double weight_bytes_per_token = predict_bytes_per_token(model);
    if (weight_bytes_per_token <= 0) return 0.0;
    
    // Calculate KV cache bytes per token (for context-aware prediction)
    double kv_bytes_per_token = 0.0;
    if (context_length > 0) {
        kv_bytes_per_token = predict_kv_bytes_per_token(model, kv_quant_bits);
    }
    
    // Total bytes to read per token
    double total_bytes_per_token = weight_bytes_per_token + kv_bytes_per_token;
    
    double tokens_per_sec = 0.0;
    
    // =========================================================================
    // FULL_GPU: Simple case - all weights in VRAM
    // =========================================================================
    if (gpu_layers >= model.layers) {
        double gpu_bw = hw.gpu_bandwidth_gbs;
        if (gpu_bw <= 0) return 0.0;
        
        // tokens_per_sec = bandwidth / bytes_per_token
        tokens_per_sec = (gpu_bw * 1e9) / total_bytes_per_token;
    }
    // =========================================================================
    // CPU_ONLY: All weights in system RAM
    // =========================================================================
    else if (gpu_layers == 0) {
        double ram_bw = hw.ram_bandwidth_gbs;
        if (ram_bw <= 0) ram_bw = 25.0;  // DDR4-3200 estimate
        
        tokens_per_sec = (ram_bw * 1e9) / total_bytes_per_token;
    }
    // =========================================================================
    // GPU_CPU_SPLIT: Sequential dependency model
    // From spec: "Sequential dependency — slower side dominates total time."
    // =========================================================================
    else {
        double gpu_fraction = static_cast<double>(gpu_layers) / model.layers;
        double cpu_fraction = 1.0 - gpu_fraction;
        
        double gpu_bw = hw.gpu_bandwidth_gbs > 0 ? hw.gpu_bandwidth_gbs : 0;
        double ram_bw = hw.ram_bandwidth_gbs > 0 ? hw.ram_bandwidth_gbs : 25.0;
        
        // Weight bytes split by layer count
        double bytes_gpu = gpu_fraction * weight_bytes_per_token;
        double bytes_cpu = cpu_fraction * weight_bytes_per_token;
        
        // Time per token for each part
        double time_gpu_sec = (gpu_bw > 0) ? (bytes_gpu / (gpu_bw * 1e9)) : 1e6;
        double time_cpu_sec = bytes_cpu / (ram_bw * 1e9);
        
        // KV cache reads (split proportionally)
        if (kv_bytes_per_token > 0) {
            double kv_gpu = gpu_fraction * kv_bytes_per_token;
            double kv_cpu = cpu_fraction * kv_bytes_per_token;
            time_gpu_sec += (gpu_bw > 0) ? (kv_gpu / (gpu_bw * 1e9)) : 1e6;
            time_cpu_sec += kv_cpu / (ram_bw * 1e9);
        }
        
        // Total time is sum (sequential, not parallel)
        double total_time_sec = time_gpu_sec + time_cpu_sec;
        
        if (total_time_sec > 0) {
            tokens_per_sec = 1.0 / total_time_sec;
        }
    }
    
    // Apply efficiency factor (real-world is ~15-25% of theoretical)
    // This accounts for:
    // - KV cache reads (not just weights)
    // - Attention computation overhead
    // - Memory access patterns
    // - CUDA kernel launch overhead
    // Calibrated against RTX 5060 + Llama-3.2-3B baseline: 43.4 t/s actual vs 266 t/s theoretical = 16.3%
    double efficiency = 0.18;  // Calibrated estimate
    
    // Adjust efficiency based on model size relative to VRAM
    // Larger models that barely fit have worse cache behavior
    if (gpu_layers > 0 && hw.vram_total_bytes > 0) {
        double weight_bytes = static_cast<double>(predict_weight_memory(model));
        double model_fit_ratio = weight_bytes / hw.vram_total_bytes;
        if (model_fit_ratio > 0.8) {
            efficiency *= 0.8;  // Penalize models that barely fit
        }
    }
    
    return tokens_per_sec * efficiency;
}

// =============================================================================
// Prompt Evaluation Speed (Compute-bound)
// =============================================================================

double predict_prompt_eval_speed(const HardwareSpec& hw, const ModelSpec& model, uint32_t gpu_layers) {
    // Prompt evaluation (prefill) is compute-bound, not memory-bound
    // Formula: tokens_per_sec ≈ (TFLOPS * 1e12) / (2 * params)
    
    if (model.param_count == 0 || model.layers == 0) return 0.0;
    
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
    
    if (tflops <= 0) return 0.0;
    
    // tokens_per_sec = TFLOPS * 1e12 / FLOPS_per_token
    double tps = (tflops * 1e12) / total_flops;
    
    // Apply efficiency factor for prompt processing (~70-85%)
    double efficiency = 0.75;
    return tps * efficiency;
}

// =============================================================================
// Time to First Token (TTFT)
// =============================================================================

double predict_ttft_ms(const HardwareSpec& hw, const ModelSpec& model, 
                       uint32_t prompt_tokens, uint32_t gpu_layers) {
    if (prompt_tokens == 0) return 0.0;
    
    // TTFT = prompt_tokens / prompt_eval_speed * 1000 (convert to ms)
    double prompt_tps = predict_prompt_eval_speed(hw, model, gpu_layers);
    if (prompt_tps <= 0) return 10000.0;  // 10s default if can't predict
    
    double ttft_seconds = static_cast<double>(prompt_tokens) / prompt_tps;
    return ttft_seconds * 1000.0;
}
