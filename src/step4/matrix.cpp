#include "matrix.h"
#include "../predictor/predictor.h"
#include <algorithm>
#include <sstream>

// =============================================================================
// Method Matrix Generator
// =============================================================================

std::vector<uint32_t> get_context_lengths(uint32_t model_max_context) {
    std::vector<uint32_t> lengths;
    
    // Common context lengths to test
    uint32_t candidates[] = {512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072};
    
    for (uint32_t len : candidates) {
        if (len <= model_max_context) {
            lengths.push_back(len);
        }
    }
    
    // Always include model max if not already included
    if (lengths.empty() || lengths.back() != model_max_context) {
        lengths.push_back(model_max_context);
    }
    
    return lengths;
}

std::vector<StrategyResult> generate_gpu_strategies(const HardwareSpec& hw, const ModelSpec& model) {
    std::vector<StrategyResult> results;
    
    // Generate strategies for different context lengths (Full GPU)
    auto ctx_lengths = get_context_lengths(model.context_length);
    
    for (uint32_t ctx : ctx_lengths) {
        StrategyConfig strat;
        strat.placement = PlacementStrategy::FULL_GPU;
        strat.gpu_layers = model.layers;
        strat.context_length = ctx;
        strat.batch_size = 1;
        strat.kv_quant_bits = 16;
        
        Prediction pred = predict(hw, model, strat);
        
        StrategyResult result;
        result.strategy = strat;
        result.prediction = pred;
        result.description = "Full GPU";
        
        results.push_back(result);
    }
    
    return results;
}

std::vector<StrategyResult> generate_split_strategies(const HardwareSpec& hw, const ModelSpec& model) {
    std::vector<StrategyResult> results;
    
    // Generate strategies for different GPU layer counts
    // Try: 75%, 50%, 25% of layers on GPU
    uint32_t split_points[] = {
        static_cast<uint32_t>(model.layers * 3 / 4),   // 75%
        model.layers / 2,                                // 50%
        model.layers / 4                                 // 25%
    };
    
    auto ctx_lengths = get_context_lengths(model.context_length);
    
    for (uint32_t gpu_layers : split_points) {
        if (gpu_layers == 0 || gpu_layers >= model.layers) continue;
        
        // Test with a few context lengths
        for (uint32_t ctx : ctx_lengths) {
            StrategyConfig strat;
            strat.placement = PlacementStrategy::GPU_CPU_SPLIT;
            strat.gpu_layers = gpu_layers;
            strat.context_length = ctx;
            strat.batch_size = 1;
            strat.kv_quant_bits = 16;
            
            Prediction pred = predict(hw, model, strat);
            
            StrategyResult result;
            result.strategy = strat;
            result.prediction = pred;
            
            std::ostringstream oss;
            oss << "Split " << gpu_layers << "/" << (model.layers - gpu_layers);
            result.description = oss.str();
            
            results.push_back(result);
        }
    }
    
    return results;
}

std::vector<StrategyResult> generate_cpu_strategies(const HardwareSpec& hw, const ModelSpec& model) {
    std::vector<StrategyResult> results;
    
    auto ctx_lengths = get_context_lengths(model.context_length);
    
    for (uint32_t ctx : ctx_lengths) {
        StrategyConfig strat;
        strat.placement = PlacementStrategy::CPU_ONLY;
        strat.gpu_layers = 0;
        strat.context_length = ctx;
        strat.batch_size = 1;
        strat.kv_quant_bits = 16;
        
        Prediction pred = predict(hw, model, strat);
        
        StrategyResult result;
        result.strategy = strat;
        result.prediction = pred;
        result.description = "CPU Only";
        
        results.push_back(result);
    }
    
    return results;
}

std::vector<StrategyResult> generate_matrix(const HardwareSpec& hw, const ModelSpec& model) {
    std::vector<StrategyResult> all_results;
    
    // Generate all strategy types
    auto gpu_strats = generate_gpu_strategies(hw, model);
    auto split_strats = generate_split_strategies(hw, model);
    auto cpu_strats = generate_cpu_strategies(hw, model);
    
    // Combine all results
    all_results.insert(all_results.end(), gpu_strats.begin(), gpu_strats.end());
    all_results.insert(all_results.end(), split_strats.begin(), split_strats.end());
    all_results.insert(all_results.end(), cpu_strats.begin(), cpu_strats.end());
    
    // Sort by speed (fastest first)
    std::sort(all_results.begin(), all_results.end(), 
        [](const StrategyResult& a, const StrategyResult& b) {
            if (a.prediction.viable != b.prediction.viable) {
                return a.prediction.viable > b.prediction.viable;  // Viable first
            }
            return a.prediction.tokens_per_sec > b.prediction.tokens_per_sec;
        });
    
    return all_results;
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
    
    return oss.str();
}
