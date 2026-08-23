#include "matrix.h"
#include "moe_placer.h"
#include "../predictor/predictor.h"
#include "../predictor/memory_predictor.h"
#include "../predictor/context_analyzer.h"
#include "calibration_aggregator.h"
#include <algorithm>
#include <sstream>
#include <set>

// =============================================================================
// Method Matrix Generator (Phase B)
// =============================================================================
// Three dimensions:
//   1. Placement: FULL_GPU, GPU_CPU_SPLIT (max-fit, half, minimal), CPU_ONLY
//   2. Context Length: 4K, max-safe
//   3. KV Cache Precision: FP16 (16 bits), Q8 (8 bits)
// =============================================================================

// Calculate maximum layers that fit in VRAM
static uint32_t calculate_max_fit_layers(const HardwareSpec& hw, const ModelSpec& model, uint32_t context_length) {
    if (model.layers == 0) return 0;
    
    // Calculate per-layer memory
    uint64_t weight_bytes = predict_weight_memory(model);
    uint64_t kv_bytes = predict_kv_cache_memory(model, context_length, 16, 1);  // FP16 KV
    
    double per_layer_weight = static_cast<double>(weight_bytes) / model.layers;
    double per_layer_kv = static_cast<double>(kv_bytes) / model.layers;
    double per_layer_total = per_layer_weight + per_layer_kv;
    
    if (per_layer_total <= 0) return 0;
    
    // Available VRAM for layers (after overhead)
    uint64_t gpu_overhead = 512ULL * 1024 * 1024;  // 512 MB CUDA context
    if (hw.vram_free_bytes <= gpu_overhead) return 0;
    
    double available_for_layers = static_cast<double>(hw.vram_free_bytes - gpu_overhead);
    
    // Calculate max layers
    uint32_t max_layers = static_cast<uint32_t>(available_for_layers / per_layer_total);
    
    // Clamp to model layers
    if (max_layers > model.layers) max_layers = model.layers;
    
    return max_layers;
}

// Calculate max-safe context for a given memory budget
static uint32_t calculate_max_safe_ctx(const HardwareSpec& hw, const ModelSpec& model, uint64_t memory_budget) {
    uint64_t weight_bytes = predict_weight_memory(model);
    uint64_t overhead_bytes = predict_overhead_memory(model, 1, true);
    
    // Available space for KV cache
    int64_t available_for_kv = static_cast<int64_t>(memory_budget) - weight_bytes - overhead_bytes;
    if (available_for_kv <= 0) return 0;
    
    // KV cache per token (FP16)
    double kv_per_token = predict_kv_bytes_per_token(model, 16);
    if (kv_per_token <= 0) return 0;
    
    // Max context
    uint32_t max_ctx = static_cast<uint32_t>(available_for_kv / kv_per_token);
    
    // Clamp to model's max context
    if (model.context_length > 0 && max_ctx > model.context_length) {
        max_ctx = model.context_length;
    }
    
    // Floor at 4096
    if (max_ctx < 4096) max_ctx = 4096;
    
    return max_ctx;
}

// Deduplicate strategies (remove identical entries)
static std::vector<StrategyResult> deduplicate(const std::vector<StrategyResult>& strategies) {
    std::vector<StrategyResult> unique;
    std::set<std::string> seen;
    
    for (const auto& strat : strategies) {
        // Create unique key from strategy parameters + MoE variant
        std::ostringstream oss;
        oss << strat.strategy.gpu_layers << "_"
            << strat.strategy.context_length << "_"
            << strat.strategy.kv_quant_bits;
        // Add MoE variant name if present (prevents dense/MoE dedup)
        if (!strat.moe_plan.variant_name.empty()) {
            oss << "_" << strat.moe_plan.variant_name;
        }
        std::string key = oss.str();
        
        if (seen.find(key) == seen.end()) {
            seen.insert(key);
            unique.push_back(strat);
        }
    }
    
    return unique;
}

// =============================================================================
// Main Matrix Generation
// =============================================================================

std::vector<StrategyResult> generate_matrix(const HardwareSpec& hw, const ModelSpec& model) {
    CalibrationData empty_cal;
    return generate_matrix(hw, model, empty_cal);
}

std::vector<StrategyResult> generate_matrix(const HardwareSpec& hw, const ModelSpec& model,
                                            const CalibrationData& cal) {
    std::vector<StrategyResult> all_strategies;
    
    // Calculate key values
    uint64_t weight_bytes = predict_weight_memory(model);
    
    // Calculate max-safe context for FULL_GPU (VRAM budget)
    uint64_t vram_budget = hw.vram_free_bytes;
    uint32_t max_safe_ctx_gpu = calculate_max_safe_ctx(hw, model, vram_budget);
    
    // Calculate max-safe context for CPU_ONLY (RAM budget)
    uint64_t ram_budget = hw.ram_free_bytes;
    uint32_t max_safe_ctx_cpu = calculate_max_safe_ctx(hw, model, ram_budget);
    
    // Use the larger of the two for the matrix
    uint32_t max_safe_ctx = std::max(max_safe_ctx_gpu, max_safe_ctx_cpu);
    
    // Context lengths to test: 4K and max-safe (deduplicate if same)
    std::vector<uint32_t> contexts = {4096};
    if (max_safe_ctx > 4096) {
        contexts.push_back(max_safe_ctx);
    }
    
    // KV cache precisions to test: FP16 (16 bits) and Q8 (8 bits)
    std::vector<uint32_t> kv_quants = {16, 8};
    
    // Calculate split points
    uint32_t max_gpu_layers = calculate_max_fit_layers(hw, model, max_safe_ctx);
    
    // Generate split points (deduplicate later)
    std::vector<uint32_t> split_points;
    
    bool has_gpu = (hw.vram_free_bytes > 512ULL * 1024 * 1024);  // >512MB free VRAM
    
    if (has_gpu) {
        // 1. Full GPU (all layers)
        split_points.push_back(model.layers);
        
        // 2. Max-fit split (maximum layers that fit in VRAM)
        if (max_gpu_layers > 0 && max_gpu_layers < model.layers) {
            split_points.push_back(max_gpu_layers);
        }
        
        // 3. Half split
        uint32_t half_layers = model.layers / 2;
        if (half_layers > 0 && half_layers < model.layers) {
            split_points.push_back(half_layers);
        }
        
        // 4. Minimal GPU (1-4 layers)
        uint32_t minimal_layers = std::min(4u, model.layers / 4);
        if (minimal_layers > 0 && minimal_layers < model.layers) {
            split_points.push_back(minimal_layers);
        }
    }
    
    // 5. CPU Only (0 layers) — always include
    split_points.push_back(0);
    
    // Sort and deduplicate split points
    std::sort(split_points.begin(), split_points.end());
    split_points.erase(std::unique(split_points.begin(), split_points.end()), split_points.end());
    
    // Generate all combinations
    for (uint32_t gpu_layers : split_points) {
        for (uint32_t ctx : contexts) {
            for (uint32_t kv_bits : kv_quants) {
                // Skip if context exceeds model max
                if (ctx > model.context_length) continue;
                
                // Determine placement type
                PlacementStrategy placement;
                if (gpu_layers >= model.layers) {
                    placement = PlacementStrategy::FULL_GPU;
                } else if (gpu_layers == 0) {
                    placement = PlacementStrategy::CPU_ONLY;
                } else {
                    placement = PlacementStrategy::GPU_CPU_SPLIT;
                }
                
                // Create strategy config
                StrategyConfig strat;
                strat.placement = placement;
                strat.gpu_layers = gpu_layers;
                strat.context_length = ctx;
                strat.batch_size = 1;
                strat.kv_quant_bits = kv_bits;
                
                // Get prediction (with calibration if available)
                Prediction pred = predict(hw, model, strat, cal);
                
                // Create result
                StrategyResult result;
                result.strategy = strat;
                result.prediction = pred;
                
                // Build description
                std::ostringstream oss;
                switch (placement) {
                    case PlacementStrategy::FULL_GPU:
                        oss << "Full GPU";
                        break;
                    case PlacementStrategy::GPU_CPU_SPLIT:
                        oss << "Split " << gpu_layers << "/" << (model.layers - gpu_layers);
                        break;
                    case PlacementStrategy::CPU_ONLY:
                        oss << "CPU Only";
                        break;
                }
                oss << " (ctx=" << ctx;
                if (kv_bits == 8) oss << ", KV=Q8";
                oss << ")";
                result.description = oss.str();
                
                all_strategies.push_back(result);
            }
        }
    }
    
    // =========================================================================
    // MoE-Specific Strategies (Step 9, Phase B)
    // =========================================================================
    // For MoE models, add three specific expert-offload variants:
    //   1. MoE-Full-VRAM: All shared + all routed experts in VRAM
    //   2. MoE-Expert-Offload: All shared + E_gpu experts in VRAM, rest in RAM
    //   3. MoE-CPU-Only: Everything in RAM
    // =========================================================================
    if (is_moe_model(model)) {
        for (uint32_t ctx : contexts) {
            for (uint32_t kv_bits : kv_quants) {
                if (ctx > model.context_length) continue;
                
                // Strategy 1: MoE Full VRAM (all experts on GPU)
                {
                    MoEPlacementPlan plan = computeMoEFullVRAM(hw, model, ctx, kv_bits);
                    
                    StrategyConfig strat;
                    strat.placement = plan.viable ? PlacementStrategy::FULL_GPU : PlacementStrategy::GPU_CPU_SPLIT;
                    strat.gpu_layers = plan.viable ? model.layers : 0;
                    strat.context_length = ctx;
                    strat.batch_size = 1;
                    strat.kv_quant_bits = kv_bits;
                    
                    Prediction pred = predict(hw, model, strat, cal);
                    // Override memory with MoE plan values
                    pred.memory_vram_bytes = plan.total_vram_bytes;
                    pred.memory_ram_bytes = plan.total_ram_bytes;
                    pred.memory_total_bytes = plan.total_vram_bytes + plan.total_ram_bytes;
                    pred.viable = plan.viable;
                    
                    StrategyResult result;
                    result.strategy = strat;
                    result.prediction = pred;
                    result.moe_plan = plan;
                    
                    std::ostringstream oss;
                    oss << "MoE Full VRAM";
                    if (!plan.viable) oss << " (" << plan.reason << ")";
                    oss << " (ctx=" << ctx;
                    if (kv_bits == 8) oss << ", KV=Q8";
                    oss << ")";
                    result.description = oss.str();
                    
                    all_strategies.push_back(result);
                }
                
                // Strategy 2: MoE Expert Offload (flagship — as many experts on GPU as fit)
                {
                    MoEPlacementPlan plan = computeMoEExpertOffload(hw, model, ctx, kv_bits);
                    
                    StrategyConfig strat;
                    strat.placement = plan.viable ? PlacementStrategy::GPU_CPU_SPLIT : PlacementStrategy::CPU_ONLY;
                    strat.gpu_layers = plan.viable ? model.layers : 0;  // All layers have GPU experts
                    strat.context_length = ctx;
                    strat.batch_size = 1;
                    strat.kv_quant_bits = kv_bits;
                    
                    Prediction pred = predict(hw, model, strat, cal);
                    pred.memory_vram_bytes = plan.total_vram_bytes;
                    pred.memory_ram_bytes = plan.total_ram_bytes;
                    pred.memory_total_bytes = plan.total_vram_bytes + plan.total_ram_bytes;
                    pred.viable = plan.viable;
                    
                    StrategyResult result;
                    result.strategy = strat;
                    result.prediction = pred;
                    result.moe_plan = plan;
                    
                    std::ostringstream oss;
                    oss << "MoE Expert Offload";
                    if (plan.viable) {
                        oss << " (" << plan.gpu_experts_per_layer << "/" << model.expert_count << " experts on GPU";
                    } else {
                        oss << " (" << plan.reason;
                    }
                    oss << ", ctx=" << ctx;
                    if (kv_bits == 8) oss << ", KV=Q8";
                    oss << ")";
                    result.description = oss.str();
                    
                    all_strategies.push_back(result);
                }
                
                // Strategy 3: MoE CPU Only (everything in RAM)
                {
                    MoEPlacementPlan plan = computeMoECPUOnly(hw, model, ctx, kv_bits);
                    
                    StrategyConfig strat;
                    strat.placement = PlacementStrategy::CPU_ONLY;
                    strat.gpu_layers = 0;
                    strat.context_length = ctx;
                    strat.batch_size = 1;
                    strat.kv_quant_bits = kv_bits;
                    
                    Prediction pred = predict(hw, model, strat, cal);
                    pred.memory_vram_bytes = plan.total_vram_bytes;
                    pred.memory_ram_bytes = plan.total_ram_bytes;
                    pred.memory_total_bytes = plan.total_vram_bytes + plan.total_ram_bytes;
                    pred.viable = plan.viable;
                    
                    StrategyResult result;
                    result.strategy = strat;
                    result.prediction = pred;
                    result.moe_plan = plan;
                    
                    std::ostringstream oss;
                    oss << "MoE CPU Only";
                    if (!plan.viable) oss << " (" << plan.reason << ")";
                    oss << " (ctx=" << ctx;
                    if (kv_bits == 8) oss << ", KV=Q8";
                    oss << ")";
                    result.description = oss.str();
                    
                    all_strategies.push_back(result);
                }
            }
        }
    }
    
    // Deduplicate
    all_strategies = deduplicate(all_strategies);
    
    // Sort by: viable first, then by speed (fastest first)
    std::sort(all_strategies.begin(), all_strategies.end(),
        [](const StrategyResult& a, const StrategyResult& b) {
            if (a.prediction.viable != b.prediction.viable) {
                return a.prediction.viable > b.prediction.viable;
            }
            return a.prediction.tokens_per_sec > b.prediction.tokens_per_sec;
        });
    
    return all_strategies;
}

std::string format_strategy_description(const StrategyConfig& strat, uint32_t total_layers) {
    std::ostringstream oss;
    
    switch (strat.placement) {
        case PlacementStrategy::FULL_GPU:
            oss << "Full GPU (" << total_layers << " layers)";
            break;
        case PlacementStrategy::GPU_CPU_SPLIT:
            oss << "Split " << strat.gpu_layers << "/" << (total_layers - strat.gpu_layers);
            break;
        case PlacementStrategy::CPU_ONLY:
            oss << "CPU Only";
            break;
        default:
            oss << "Unknown";
    }
    
    oss << " ctx=" << strat.context_length;
    if (strat.kv_quant_bits == 8) oss << " KV=Q8";
    
    return oss.str();
}
