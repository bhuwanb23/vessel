#include "predictor.h"
#include "moe_predictor.h"
#include "../../include/moe_placer.h"
#include "calibration_aggregator.h"
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
    
    // MoE range prediction (Step 9, Phase D)
    // NOTE: For matrix-generated MoE strategies, range is computed in matrix.cpp
    // This handles cases where predict() is called directly for MoE models
    if (is_moe_model(model)) {
        MoEPlacementPlan moe_plan = computeMoEPlacement(hw, model, strategy);
        if (moe_plan.viable && moe_plan.gpu_experts_per_layer > 0 
            && moe_plan.gpu_experts_per_layer < model.expert_count) {
            MoEPrediction moe_pred = predictMoERange(hw, model, moe_plan, strategy.kv_quant_bits);
            if (moe_pred.valid) {
                pred.is_moe_range = true;
                pred.tok_s_best = moe_pred.tok_s_best;
                pred.tok_s_worst = moe_pred.tok_s_worst;
                pred.tok_s_expected = moe_pred.tok_s_expected;
                pred.gpu_hit_probability = moe_pred.p_gpu;
                pred.tokens_per_sec = moe_pred.tok_s_expected;
            }
        }
    }
    
    // Check viability
    pred.viable = check_viability(hw, pred.memory_total_bytes);
    
    // Calculate confidence level (Phase F)
    // Check if hot/cold profiling mask exists
    bool has_profiling_mask = false;
    if (strategy.placement == PlacementStrategy::HOT_COLD_SPLIT) {
        // Check for mask file (would need to check filesystem, simplified here)
        has_profiling_mask = false;  // Will be set by executor if mask exists
    }
    ConfidenceResult conf = calculate_confidence(model, hw, ctx_len, 0,
                                                 strategy.placement, has_profiling_mask);
    pred.confidence = conf.level;
    pred.confidence_reason = conf.reason;
    
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
// Calibrated Prediction (Step 7)
// =============================================================================
// Same logic as predict() but uses adjusted constants from calibration log.
// =============================================================================

Prediction predict(const HardwareSpec& hw, const ModelSpec& model,
                   const StrategyConfig& strategy, const CalibrationData& cal) {
    Prediction pred;

    // Determine effective context length
    uint32_t ctx_len = strategy.context_length > 0 ? strategy.context_length : model.context_length;
    if (ctx_len == 0) ctx_len = 4096;

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

    uint64_t weight_bytes = predict_weight_memory(model);
    uint64_t kv_bytes = predict_kv_cache_memory(model, ctx_len, strategy.kv_quant_bits);

    // Use calibrated overhead if available, otherwise use defaults
    bool use_gpu = (gpu_layers > 0);
    uint64_t overhead_bytes;
    if (use_gpu && cal.adjusted_gpu_overhead_bytes > 0) {
        overhead_bytes = cal.adjusted_gpu_overhead_bytes;
    } else if (!use_gpu && cal.adjusted_cpu_overhead_bytes > 0) {
        overhead_bytes = cal.adjusted_cpu_overhead_bytes;
    } else {
        overhead_bytes = predict_overhead_memory(model, strategy.batch_size, use_gpu);
    }

    // Distribute memory based on strategy
    if (gpu_layers >= model.layers) {
        pred.memory_vram_bytes = weight_bytes + kv_bytes + overhead_bytes;
        pred.memory_ram_bytes = 0;
    } else if (gpu_layers == 0) {
        pred.memory_vram_bytes = 0;
        pred.memory_ram_bytes = weight_bytes + kv_bytes + overhead_bytes;
    } else {
        double gpu_ratio = static_cast<double>(gpu_layers) / model.layers;
        double cpu_ratio = 1.0 - gpu_ratio;
        uint64_t gpu_weights = static_cast<uint64_t>(weight_bytes * gpu_ratio);
        uint64_t cpu_weights = weight_bytes - gpu_weights;
        uint64_t gpu_kv = static_cast<uint64_t>(kv_bytes * gpu_ratio);
        uint64_t cpu_kv = kv_bytes - gpu_kv;
        uint64_t gpu_overhead = static_cast<uint64_t>(overhead_bytes * 0.7);
        uint64_t cpu_overhead = overhead_bytes - gpu_overhead;
        pred.memory_vram_bytes = gpu_weights + gpu_kv + gpu_overhead;
        pred.memory_ram_bytes = cpu_weights + cpu_kv + cpu_overhead;
    }

    pred.memory_total_bytes = pred.memory_vram_bytes + pred.memory_ram_bytes;
    pred.max_safe_context = calculate_max_safe_context(hw, model, strategy);

    // Calculate decode speed with calibrated efficiency
    pred.tokens_per_sec = 0.0;
    {
        double weight_bpt = predict_bytes_per_token(model);
        double kv_bpt = predict_kv_bytes_per_token(model, strategy.kv_quant_bits);
        double total_bpt = weight_bpt + kv_bpt;
        if (total_bpt > 0) {
            if (gpu_layers >= model.layers && hw.gpu_bandwidth_gbs > 0) {
                double eff = (cal.adjusted_gpu_decode_efficiency > 0)
                    ? cal.adjusted_gpu_decode_efficiency : 0.27;
                double theoretical = (hw.gpu_bandwidth_gbs * 1e9) / total_bpt;
                pred.tokens_per_sec = theoretical * eff;
            } else if (gpu_layers == 0) {
                double ram_bw = hw.ram_bandwidth_gbs > 0 ? hw.ram_bandwidth_gbs : 40.0;
                double eff = (cal.adjusted_cpu_decode_efficiency > 0)
                    ? cal.adjusted_cpu_decode_efficiency : 0.80;
                double theoretical = (ram_bw * 1e9) / total_bpt;
                pred.tokens_per_sec = theoretical * eff;
            } else {
                // Split: use GPU efficiency for GPU portion, CPU for CPU portion
                double gpu_frac = static_cast<double>(gpu_layers) / model.layers;
                double cpu_frac = 1.0 - gpu_frac;
                double gpu_bw = hw.gpu_bandwidth_gbs > 0 ? hw.gpu_bandwidth_gbs : 0;
                double ram_bw = hw.ram_bandwidth_gbs > 0 ? hw.ram_bandwidth_gbs : 40.0;
                double gpu_eff = (cal.adjusted_gpu_decode_efficiency > 0)
                    ? cal.adjusted_gpu_decode_efficiency : 0.27;
                double cpu_eff = (cal.adjusted_cpu_decode_efficiency > 0)
                    ? cal.adjusted_cpu_decode_efficiency : 0.80;
                double time_gpu = (gpu_bw > 0)
                    ? (gpu_frac * total_bpt / (gpu_bw * 1e9 * gpu_eff)) : 1e6;
                double time_cpu = cpu_frac * total_bpt / (ram_bw * 1e9 * cpu_eff);
                if (time_gpu + time_cpu > 0)
                    pred.tokens_per_sec = 1.0 / (time_gpu + time_cpu);
            }
        }
    }

    // Calculate prompt eval speed with calibrated prefill efficiency
    double prefill_eff = (cal.adjusted_gpu_prefill_efficiency > 0)
        ? cal.adjusted_gpu_prefill_efficiency : 0.0;  // 0 = use default
    pred.prompt_eval_tps = (prefill_eff > 0)
        ? predict_prompt_eval_speed(hw, model, gpu_layers, prefill_eff)
        : predict_prompt_eval_speed(hw, model, gpu_layers);

    // Calculate TTFT with calibrated prefill efficiency
    pred.ttft_ms = (prefill_eff > 0)
        ? predict_ttft_ms(hw, model, ctx_len, gpu_layers, prefill_eff)
        : predict_ttft_ms(hw, model, ctx_len, gpu_layers);

    // Viability and confidence
    pred.viable = check_viability(hw, pred.memory_total_bytes);
    int cal_count = cal.matching_record_count;
    ConfidenceResult conf = calculate_confidence(model, hw, ctx_len, cal_count);
    pred.confidence = conf.level;
    pred.confidence_reason = conf.reason;

    // Warnings
    if (!pred.viable) {
        pred.warnings = "Model does not fit in available memory with this strategy.";
    } else if (pred.memory_vram_bytes > hw.vram_free_bytes && hw.vram_free_bytes > 0) {
        pred.warnings = "VRAM usage exceeds free VRAM; some layers will be offloaded to CPU.";
    }
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
