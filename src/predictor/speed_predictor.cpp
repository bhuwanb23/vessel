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
        // CPU only - default to 40 GB/s (DDR5-5600 estimate for modern systems)
        return hw.ram_bandwidth_gbs > 0 ? hw.ram_bandwidth_gbs : 40.0;
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
        double ram_bw = hw.ram_bandwidth_gbs > 0 ? hw.ram_bandwidth_gbs : 40.0;
        
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
    
    // Apply efficiency factor (real-world is ~20-30% of theoretical)
    // This accounts for:
    // - KV cache reads (not just weights)
    // - Attention computation overhead
    // - Memory access patterns
    // - CUDA kernel launch overhead
    // Calibrated against RTX 5060 + Llama-3.2-3B: 61.3 t/s actual vs 230 t/s theoretical = 26.6%
    double efficiency = 0.27;  // Calibrated estimate
    
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
// Prompt Evaluation Speed (Compute-bound) - Phase E
// =============================================================================
// From spec: "Prefill is compute-bound (whole prompt processed in parallel),
//            not bandwidth-bound."
// =============================================================================

double predict_prompt_eval_speed(const HardwareSpec& hw, const ModelSpec& model, uint32_t gpu_layers) {
    // Prompt evaluation (prefill) is compute-bound, not memory-bound
    // Formula: tokens_per_sec ≈ device_compute_throughput / flops_per_token
    
    if (model.param_count == 0 || model.layers == 0) return 0.0;
    
    // FLOPS per token = 2 × active_params (multiply-accumulate)
    double flops_per_token = 2.0 * static_cast<double>(model.param_count);
    
    // Device compute throughput in TFLOPS
    double device_tflops = 0.0;
    
    if (gpu_layers > 0 && hw.gpu_tflops_fp16 > 0) {
        // GPU-accelerated: use GPU TFLOPS with efficiency factor
        // From spec: "Typical values: 0.2-0.6 of peak"
        // Small batch (batch=1) + small model: use lower end (0.3)
        double gpu_efficiency = 0.3;  // Conservative for batch=1
        
        // Scale by fraction of layers on GPU
        double gpu_ratio = static_cast<double>(gpu_layers) / model.layers;
        device_tflops = hw.gpu_tflops_fp16 * gpu_ratio * gpu_efficiency;
        
        // Add CPU contribution for remaining layers
        if (gpu_ratio < 1.0) {
            // CPU TFLOPS estimate: 0.8 TFLOPS for modern desktop CPU
            // From spec: "AVX2 (most modern CPUs): ~0.5-1.5 TFLOPS FP16 equivalent"
            double cpu_tflops = 0.8;
            device_tflops += (1.0 - gpu_ratio) * cpu_tflops;
        }
    } else {
        // CPU-only: use CPU TFLOPS estimate
        // From spec: "Starting estimate for MVP: 0.8 TFLOPS"
        device_tflops = 0.8;
    }
    
    if (device_tflops <= 0) return 0.0;
    
    // tokens_per_sec = (device_tflops × 1e12) / flops_per_token
    double tps = (device_tflops * 1e12) / flops_per_token;
    
    return tps;
}

// =============================================================================
// Time to First Token (TTFT) - Phase E
// =============================================================================
// From spec: "TTFT is harder to predict than tokens/sec" - report with wider
//            confidence band. Depends on prompt length, compute throughput,
//            and batch scheduling heuristics in llama.cpp.
// =============================================================================

double predict_ttft_ms(const HardwareSpec& hw, const ModelSpec& model, 
                       uint32_t prompt_tokens, uint32_t gpu_layers) {
    if (prompt_tokens == 0) return 0.0;
    
    // Method 1: Direct formula from spec
    // total_flops = flops_per_token × prompt_tokens
    // ttft_seconds = total_flops / (device_compute_throughput × 1e12)
    
    double flops_per_token = 2.0 * static_cast<double>(model.param_count);
    double total_flops = flops_per_token * prompt_tokens;
    
    // Device compute throughput in TFLOPS
    double device_tflops = 0.0;
    
    if (gpu_layers > 0 && hw.gpu_tflops_fp16 > 0) {
        // GPU efficiency: 0.2-0.6, use 0.3 for small batch
        double gpu_efficiency = 0.3;
        double gpu_ratio = static_cast<double>(gpu_layers) / model.layers;
        device_tflops = hw.gpu_tflops_fp16 * gpu_ratio * gpu_efficiency;
        
        if (gpu_ratio < 1.0) {
            double cpu_tflops = 0.8;
            device_tflops += (1.0 - gpu_ratio) * cpu_tflops;
        }
    } else {
        // CPU-only: 0.8 TFLOPS
        device_tflops = 0.8;
    }
    
    if (device_tflops <= 0) return 10000.0;  // 10s default
    
    // TTFT in seconds
    double ttft_seconds = total_flops / (device_tflops * 1e12);
    
    // Convert to milliseconds
    return ttft_seconds * 1000.0;
}

// Get TTFT confidence bounds (for reporting)
// Returns {lower_bound_ms, upper_bound_ms}
// TTFT has wider uncertainty than decode speed
void predict_ttft_bounds(const HardwareSpec& hw, const ModelSpec& model,
                         uint32_t prompt_tokens, uint32_t gpu_layers,
                         double& lower_ms, double& upper_ms) {
    // Center estimate
    double center = predict_ttft_ms(hw, model, prompt_tokens, gpu_layers);
    
    // TTFT uncertainty: ±40% due to:
    // - Compute throughput variability
    // - Batch scheduling heuristics
    // - Prompt processing optimizations
    lower_ms = center * 0.6;  // Best case: 60% of center
    upper_ms = center * 1.4;  // Worst case: 140% of center
}
