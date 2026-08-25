#include "speed_predictor.h"
#include "memory_predictor.h"
#include <cmath>
#include <algorithm>

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

// =============================================================================
// Decode Speed Prediction (Phase D) — FIXED: Split mode efficiency
// =============================================================================
// Key insight from Phase H evaluation:
// - Full GPU: memory-bandwidth-bound, efficiency ~27% of theoretical
// - Split mode: CPU layers are bottleneck, CPU operates near theoretical BW
// - CPU-only: near theoretical bandwidth (no GPU overhead)
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
    // Memory-bandwidth-bound: GPU has overhead from kernel launch, cache misses
    // Calibrated: RTX 5060 + Llama-3.2-3B: 61.3 t/s actual vs 261 t/s theoretical
    // =========================================================================
    if (gpu_layers >= model.layers) {
        double gpu_bw = hw.gpu_bandwidth_gbs;
        if (gpu_bw <= 0) return 0.0;
        
        // Theoretical: bandwidth / bytes_per_token
        double theoretical_tps = (gpu_bw * 1e9) / total_bytes_per_token;
        
        // Efficiency: 27% for full GPU (calibrated against real hardware)
        double efficiency = 0.27;
        
        // Penalize models that barely fit in VRAM (cache thrashing)
        if (hw.vram_total_bytes > 0) {
            double weight_bytes = static_cast<double>(predict_weight_memory(model));
            double model_fit_ratio = weight_bytes / hw.vram_total_bytes;
            if (model_fit_ratio > 0.8) {
                efficiency *= 0.8;
            }
        }
        
        tokens_per_sec = theoretical_tps * efficiency;
    }
    // =========================================================================
    // CPU_ONLY: All weights in system RAM
    // Near theoretical bandwidth (no GPU kernel overhead)
    // CPU SIMD operations are efficient for sequential reads
    // =========================================================================
    else if (gpu_layers == 0) {
        double ram_bw = hw.ram_bandwidth_gbs;
        if (ram_bw <= 0) ram_bw = 40.0;  // DDR5-5600 estimate
        
        // Theoretical: bandwidth / bytes_per_token
        double theoretical_tps = (ram_bw * 1e9) / total_bytes_per_token;
        
        // Efficiency: 80% for CPU-only (near theoretical, minimal overhead)
        double efficiency = 0.80;
        
        tokens_per_sec = theoretical_tps * efficiency;
    }
    // =========================================================================
    // GPU_CPU_SPLIT: Platform-dependent model
    //
    // Discrete GPU (NVIDIA/AMD):
    //   Sequential dependency — layers must be processed one after another.
    //   GPU and CPU can't overlap because data must cross PCIe.
    //   Total time = GPU time + CPU time
    //
    // Apple Silicon (Unified Memory):
    //   Parallel execution — no PCIe transfer needed.
    //   Metal GPU and CPU can process different layers simultaneously.
    //   Total time = max(GPU time, CPU time) — the faster side finishes first.
    //   This makes split strategies MUCH more attractive on Apple Silicon.
    // =========================================================================
    else {
        double gpu_fraction = static_cast<double>(gpu_layers) / model.layers;
        double cpu_fraction = 1.0 - gpu_fraction;
        
        double gpu_bw = hw.gpu_bandwidth_gbs > 0 ? hw.gpu_bandwidth_gbs : 0;
        double ram_bw = hw.ram_bandwidth_gbs > 0 ? hw.ram_bandwidth_gbs : 40.0;
        
        // Weight bytes split by layer count
        double bytes_gpu = gpu_fraction * weight_bytes_per_token;
        double bytes_cpu = cpu_fraction * weight_bytes_per_token;
        
        // GPU time: includes efficiency penalty (memory-bandwidth-bound)
        double gpu_efficiency = 0.27;
        double time_gpu_sec = (gpu_bw > 0) ? (bytes_gpu / (gpu_bw * 1e9 * gpu_efficiency)) : 1e6;
        
        // CPU time: near theoretical bandwidth (no GPU overhead)
        double cpu_efficiency = 0.80;
        double time_cpu_sec = bytes_cpu / (ram_bw * 1e9 * cpu_efficiency);
        
        // KV cache reads (split proportionally, same efficiency as weights)
        if (kv_bytes_per_token > 0) {
            double kv_gpu = gpu_fraction * kv_bytes_per_token;
            double kv_cpu = cpu_fraction * kv_bytes_per_token;
            time_gpu_sec += (gpu_bw > 0) ? (kv_gpu / (gpu_bw * 1e9 * gpu_efficiency)) : 1e6;
            time_cpu_sec += kv_cpu / (ram_bw * 1e9 * cpu_efficiency);
        }
        
        // Platform-specific timing model
        double total_time_sec = 0.0;
        
        if (hw.is_unified_memory) {
            // Apple Silicon: parallel execution, no PCIe penalty
            // Metal GPU and CPU can work on different layers simultaneously
            // Total time = max(GPU time, CPU time)
            total_time_sec = std::max(time_gpu_sec, time_cpu_sec);
            
            // Add a small synchronization overhead (~5%)
            // Metal and CPU still need to synchronize at layer boundaries
            double sync_overhead = 0.05 * std::min(time_gpu_sec, time_cpu_sec);
            total_time_sec += sync_overhead;
        } else {
            // Discrete GPU: sequential execution, PCIe transfer dominates
            // Total time = GPU time + CPU time
            total_time_sec = time_gpu_sec + time_cpu_sec;
        }
        
        if (total_time_sec > 0) {
            tokens_per_sec = 1.0 / total_time_sec;
        }
    }
    
    return tokens_per_sec;
}

// =============================================================================
// Prompt Evaluation Speed (Compute-bound) — FIXED: Efficiency reduced
// =============================================================================
// From spec: "Prefill is compute-bound (whole prompt processed in parallel),
//            not bandwidth-bound."
//
// Phase H finding: Predicted 933 t/s vs actual 730.6 t/s (28% over)
// Fix: Reduce GPU efficiency from 0.3 to 0.23
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
        // Phase H calibration: 0.3 was too optimistic, use 0.23
        double gpu_efficiency = 0.23;  // Calibrated against 730.6 t/s actual
        
        // Scale by fraction of layers on GPU
        double gpu_ratio = static_cast<double>(gpu_layers) / model.layers;
        device_tflops = hw.gpu_tflops_fp16 * gpu_ratio * gpu_efficiency;
        
        // Add CPU contribution for remaining layers
        if (gpu_ratio < 1.0) {
            // CPU TFLOPS estimate: 0.8 TFLOPS for modern desktop CPU
            double cpu_tflops = 0.8;
            device_tflops += (1.0 - gpu_ratio) * cpu_tflops;
        }
    } else {
        // CPU-only: use CPU TFLOPS estimate
        device_tflops = 0.8;
    }
    
    if (device_tflops <= 0) return 0.0;
    
    // tokens_per_sec = (device_tflops × 1e12) / flops_per_token
    double tps = (device_tflops * 1e12) / flops_per_token;
    
    return tps;
}

// =============================================================================
// Time to First Token (TTFT) — FIXED: Hybrid model
// =============================================================================
// Phase H finding: Predicted 3,874 ms vs actual ~55 ms (70x off!)
//
// Root cause: Formula uses compute-bound model, but llama.cpp's "prompt eval time"
// includes KV cache prefill (memory-bound) and batch processing optimizations.
//
// Fix: Use hybrid model:
// - Short prompts (<512 tokens): memory-bandwidth-bound (like decode)
// - Long prompts (>2K tokens): compute-bound
// - Smooth transition between the two
// =============================================================================

double predict_ttft_ms(const HardwareSpec& hw, const ModelSpec& model, 
                       uint32_t prompt_tokens, uint32_t gpu_layers) {
    if (prompt_tokens == 0) return 0.0;
    
    // Method 1: Memory-bandwidth-bound (like decode speed)
    // For short prompts, the bottleneck is reading weights, not compute
    double decode_tps = predict_decode_speed(hw, model, gpu_layers, 4096, 16);
    double ttft_memory_ms = (decode_tps > 0) ? 
        (static_cast<double>(prompt_tokens) / decode_tps * 1000.0) : 10000.0;
    
    // Method 2: Compute-bound (original formula)
    // For long prompts, the bottleneck is matrix multiplication
    double flops_per_token = 2.0 * static_cast<double>(model.param_count);
    double total_flops = flops_per_token * prompt_tokens;
    
    double device_tflops = 0.0;
    if (gpu_layers > 0 && hw.gpu_tflops_fp16 > 0) {
        double gpu_efficiency = 0.23;
        double gpu_ratio = static_cast<double>(gpu_layers) / model.layers;
        device_tflops = hw.gpu_tflops_fp16 * gpu_ratio * gpu_efficiency;
        if (gpu_ratio < 1.0) {
            device_tflops += (1.0 - gpu_ratio) * 0.8;
        }
    } else {
        device_tflops = 0.8;
    }
    
    double ttft_compute_ms = (device_tflops > 0) ? 
        (total_flops / (device_tflops * 1e12) * 1000.0) : 10000.0;
    
    // Hybrid: blend between memory-bound and compute-bound
    // Short prompts: mostly memory-bound
    // Long prompts: mostly compute-bound
    // Transition happens around 512-2048 tokens
    
    double alpha = 0.0;  // 0 = pure memory-bound, 1 = pure compute-bound
    if (prompt_tokens <= 256) {
        alpha = 0.0;  // Pure memory-bound for very short prompts
    } else if (prompt_tokens >= 2048) {
        alpha = 1.0;  // Pure compute-bound for long prompts
    } else {
        // Smooth transition: linear interpolation
        alpha = static_cast<double>(prompt_tokens - 256) / (2048.0 - 256.0);
    }
    
    double ttft_ms = (1.0 - alpha) * ttft_memory_ms + alpha * ttft_compute_ms;
    
    // Sanity bounds: TTFT should be at least prompt_tokens * 0.1ms (very fast)
    // and at most 60 seconds (extremely slow)
    ttft_ms = std::max(ttft_ms, static_cast<double>(prompt_tokens) * 0.1);
    ttft_ms = std::min(ttft_ms, 60000.0);
    
    return ttft_ms;
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

// =============================================================================
// Calibrated Overloads (Step 7)
// =============================================================================

double predict_prompt_eval_speed(const HardwareSpec& hw, const ModelSpec& model,
                                 uint32_t gpu_layers, double gpu_prefill_efficiency) {
    if (model.param_count == 0 || model.layers == 0) return 0.0;

    double flops_per_token = 2.0 * static_cast<double>(model.param_count);

    double device_tflops = 0.0;

    if (gpu_layers > 0 && hw.gpu_tflops_fp16 > 0) {
        // Use calibrated efficiency if > 0, otherwise default to 0.23
        double eff = (gpu_prefill_efficiency > 0) ? gpu_prefill_efficiency : 0.23;

        double gpu_ratio = static_cast<double>(gpu_layers) / model.layers;
        device_tflops = hw.gpu_tflops_fp16 * gpu_ratio * eff;

        if (gpu_ratio < 1.0) {
            double cpu_tflops = 0.8;
            device_tflops += (1.0 - gpu_ratio) * cpu_tflops;
        }
    } else {
        device_tflops = 0.8;
    }

    if (device_tflops <= 0) return 0.0;
    return (device_tflops * 1e12) / flops_per_token;
}

double predict_ttft_ms(const HardwareSpec& hw, const ModelSpec& model,
                       uint32_t prompt_tokens, uint32_t gpu_layers,
                       double gpu_prefill_efficiency) {
    if (prompt_tokens == 0) return 0.0;

    // Memory-bound component
    double decode_tps = predict_decode_speed(hw, model, gpu_layers, 4096, 16);
    double ttft_memory_ms = (decode_tps > 0)
        ? (static_cast<double>(prompt_tokens) / decode_tps * 1000.0) : 10000.0;

    // Compute-bound component (with calibrated efficiency)
    double flops_per_token = 2.0 * static_cast<double>(model.param_count);
    double total_flops = flops_per_token * prompt_tokens;

    double device_tflops = 0.0;
    if (gpu_layers > 0 && hw.gpu_tflops_fp16 > 0) {
        double eff = (gpu_prefill_efficiency > 0) ? gpu_prefill_efficiency : 0.23;
        double gpu_ratio = static_cast<double>(gpu_layers) / model.layers;
        device_tflops = hw.gpu_tflops_fp16 * gpu_ratio * eff;
        if (gpu_ratio < 1.0) {
            device_tflops += (1.0 - gpu_ratio) * 0.8;
        }
    } else {
        device_tflops = 0.8;
    }

    double ttft_compute_ms = (device_tflops > 0)
        ? (total_flops / (device_tflops * 1e12) * 1000.0) : 10000.0;

    // Hybrid blend
    double alpha = 0.0;
    if (prompt_tokens <= 256) alpha = 0.0;
    else if (prompt_tokens >= 2048) alpha = 1.0;
    else alpha = static_cast<double>(prompt_tokens - 256) / (2048.0 - 256.0);

    double ttft_ms = (1.0 - alpha) * ttft_memory_ms + alpha * ttft_compute_ms;
    ttft_ms = std::max(ttft_ms, static_cast<double>(prompt_tokens) * 0.1);
    ttft_ms = std::min(ttft_ms, 60000.0);

    return ttft_ms;
}
