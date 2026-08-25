#include "recommend/recommendation_engine.h"
#include "recommend/catalog_types.h"
#include "predictor.h"
#include "memory_predictor.h"
#include "speed_predictor.h"
#include "context_analyzer.h"
#include "moe_placer.h"
#include <algorithm>
#include <cmath>
#include <cstdio>

// =============================================================================
// Catalog to ModelSpec Conversion
// =============================================================================

ModelSpec catalog_to_model_spec(const CatalogModel& model, const GgufVariant& variant) {
    ModelSpec spec;
    
    // Identity
    spec.name = model.name;
    spec.architecture = model.architecture;
    spec.quant_type = variant.quant;
    spec.source = MetadataSource::GGUF_HEADER;  // We have pre-computed metadata
    spec.model_type = model.is_moe ? ModelType::MOE : ModelType::DENSE;
    
    // Dimensions
    spec.param_count = static_cast<uint64_t>(model.params_billions * 1e9);
    spec.layers = model.dimensions.layers;
    spec.embedding_dim = model.dimensions.embedding_dim;
    spec.attention_heads = model.dimensions.attention_heads;
    spec.kv_heads = model.dimensions.kv_heads;
    spec.head_dim = model.dimensions.head_dim;
    spec.ffn_dim = model.dimensions.ffn_dim;
    spec.context_length = model.max_context;
    spec.bits_per_weight = variant.bpw;
    
    // MoE fields
    spec.is_moe = model.is_moe;
    spec.expert_count = model.expert_count;
    spec.expert_used_count = model.expert_used_count;
    spec.expert_ffn_dim = model.expert_ffn_dim;
    
    // Calculate MoE parameters if needed
    if (model.is_moe) {
        spec.calculate_moe_parameters();
    } else {
        spec.estimate_parameters();
    }
    
    return spec;
}

// =============================================================================
// Pre-Filter: Viability Check
// =============================================================================

bool passes_viability_check(const HardwareSpec& hw, const ModelSpec& model) {
    // Quick check: can the smallest viable strategy fit in available memory?
    // Smallest strategy: CPU-only at 4K context
    
    uint64_t weight_bytes = predict_weight_memory(model);
    uint64_t kv_bytes = predict_kv_cache_memory(model, 4096, 16);  // FP16 KV
    uint64_t overhead_bytes = predict_overhead_memory(model, 1, false, hw.is_unified_memory);
    
    uint64_t total_memory = weight_bytes + kv_bytes + overhead_bytes;
    
    // Check if it fits in available memory (VRAM or RAM)
    uint64_t available = std::max(hw.vram_free_bytes, hw.ram_free_bytes);
    
    // Add 10% safety margin
    uint64_t safe_available = static_cast<uint64_t>(available * 0.9);
    
    return total_memory <= safe_available;
}

// =============================================================================
// Mini Method Matrix
// =============================================================================

std::vector<StrategyConfig> generate_mini_matrix(
    const HardwareSpec& hw,
    const ModelSpec& model
) {
    std::vector<StrategyConfig> strategies;
    
    uint64_t weight_bytes = predict_weight_memory(model);
    bool fits_in_vram = (weight_bytes + 512ULL * 1024 * 1024) <= hw.vram_free_bytes;
    bool fits_in_ram = (weight_bytes + 128ULL * 1024 * 1024) <= hw.ram_free_bytes;
    
    // Strategy 1: Full GPU at 4K (if viable)
    if (fits_in_vram) {
        StrategyConfig strat;
        strat.placement = PlacementStrategy::FULL_GPU;
        strat.gpu_layers = model.layers;
        strat.context_length = 4096;
        strat.batch_size = 1;
        strat.kv_quant_bits = 16;
        strategies.push_back(strat);
    }
    
    // Strategy 2: Best split at 4K (if full GPU doesn't fit)
    if (!fits_in_vram && fits_in_ram) {
        // Calculate best split point
        uint32_t max_gpu_layers = 0;
        uint64_t gpu_budget = hw.vram_free_bytes - 512ULL * 1024 * 1024;
        if (gpu_budget > 0 && model.layers > 0) {
            double per_layer_weight = static_cast<double>(weight_bytes) / model.layers;
            max_gpu_layers = static_cast<uint32_t>(gpu_budget / per_layer_weight);
            if (max_gpu_layers > model.layers) max_gpu_layers = model.layers;
        }
        
        if (max_gpu_layers > 0) {
            StrategyConfig strat;
            strat.placement = PlacementStrategy::GPU_CPU_SPLIT;
            strat.gpu_layers = max_gpu_layers;
            strat.context_length = 4096;
            strat.batch_size = 1;
            strat.kv_quant_bits = 16;
            strategies.push_back(strat);
        }
    }
    
    // Strategy 3: CPU-only at 4K (always include if fits)
    if (fits_in_ram) {
        StrategyConfig strat;
        strat.placement = PlacementStrategy::CPU_ONLY;
        strat.gpu_layers = 0;
        strat.context_length = 4096;
        strat.batch_size = 1;
        strat.kv_quant_bits = 16;
        strategies.push_back(strat);
    }
    
    // If no strategies are viable, return empty
    return strategies;
}

// =============================================================================
// Scoring Function for Cross-Model Ranking
// =============================================================================

double score_for_priority(
    const Prediction& pred,
    double quality_score,
    const std::string& priority
) {
    if (priority == "speed") {
        // Speed priority: tok/s is primary
        return pred.tokens_per_sec * 0.7
             + (pred.tokens_per_sec / std::max(pred.ttft_ms, 1.0)) * 0.2
             + quality_score * 0.1;
    } else if (priority == "quality") {
        // Quality priority: quality score is primary
        return quality_score * 0.6
             + std::min(pred.tokens_per_sec / 20.0, 1.0) * 0.3
             + 0.1;  // slight safety bonus
    } else {
        // Balanced: weighted combination
        double speed_norm = std::min(pred.tokens_per_sec / 50.0, 1.0);
        double quality_norm = quality_score / 10.0;
        double safety_norm = 0.5;  // Default safety
        
        // Calculate VRAM headroom if possible
        if (pred.memory_vram_bytes > 0) {
            safety_norm = 0.5;  // Assume moderate safety for recommendations
        }
        
        return speed_norm * 0.4 + quality_norm * 0.4 + safety_norm * 0.2;
    }
}

// =============================================================================
// Use Case Filter
// =============================================================================

bool matches_use_case(const CatalogModel& model, const std::string& use_case) {
    if (use_case == "all") return true;
    
    for (const auto& uc : model.use_cases) {
        if (uc == use_case) return true;
    }
    
    return false;
}

// =============================================================================
// Label Assignment
// =============================================================================

void assign_labels(std::vector<RecommendationResult>& recs) {
    if (recs.empty()) return;
    
    // First recommendation is always "Best Overall"
    recs[0].label = "🏆 Best Overall";
    
    // Find extremes
    auto fastest = std::max_element(recs.begin(), recs.end(),
        [](const auto& a, const auto& b) { return a.predicted_tok_s < b.predicted_tok_s; });
    
    auto best_quality = std::max_element(recs.begin(), recs.end(),
        [](const auto& a, const auto& b) { return a.model.quality_score < b.model.quality_score; });
    
    auto smallest = std::min_element(recs.begin(), recs.end(),
        [](const auto& a, const auto& b) { return a.variant.file_size_gb < b.variant.file_size_gb; });
    
    auto longest_ctx = std::max_element(recs.begin(), recs.end(),
        [](const auto& a, const auto& b) { return a.predicted_max_ctx < b.predicted_max_ctx; });
    
    // Assign labels (may overlap with Best Overall)
    if (fastest != recs.end() && fastest != recs.begin()) {
        fastest->label += (fastest->label.empty() ? "" : " ") + std::string("⚡ Fastest");
    }
    if (best_quality != recs.end() && best_quality != recs.begin()) {
        best_quality->label += (best_quality->label.empty() ? "" : " ") + std::string("🧠 Highest Quality");
    }
    if (smallest != recs.end() && smallest != recs.begin()) {
        smallest->label += (smallest->label.empty() ? "" : " ") + std::string("💾 Smallest");
    }
    if (longest_ctx != recs.end() && longest_ctx != recs.begin()) {
        longest_ctx->label += (longest_ctx->label.empty() ? "" : " ") + std::string("📄 Longest Context");
    }
    
    // Add "⚡ Fastest" to first if it's also the fastest
    if (fastest == recs.begin()) {
        recs[0].label = "🏆⚡ Best Overall + Fastest";
    }
}

// =============================================================================
// Main Recommendation Generation
// =============================================================================

std::vector<RecommendationResult> generate_recommendations(
    const HardwareSpec& hw,
    const ModelCatalog& catalog,
    const RecommendationRequest& request
) {
    std::vector<RecommendationResult> candidates;
    
    for (const auto& model : catalog.models) {
        // Filter by use case
        if (!matches_use_case(model, request.use_case)) {
            continue;
        }
        
        for (const auto& variant : model.variants) {
            // Filter by download size
            if (request.max_download_gb > 0 && variant.file_size_gb > request.max_download_gb) {
                continue;
            }
            
            // Convert to ModelSpec
            ModelSpec meta = catalog_to_model_spec(model, variant);
            
            // Quick viability check
            if (!passes_viability_check(hw, meta)) {
                continue;
            }
            
            // Generate mini matrix
            auto strategies = generate_mini_matrix(hw, meta);
            if (strategies.empty()) {
                continue;
            }
            
            // Predict for each strategy and find the best
            Prediction best_pred;
            StrategyConfig best_config;
            std::string best_desc;
            double best_score = -1;
            
            for (const auto& strategy : strategies) {
                Prediction pred = predict(hw, meta, strategy);
                if (!pred.viable) continue;
                
                double score = score_for_priority(pred, model.quality_score, request.priority);
                if (score > best_score) {
                    best_score = score;
                    best_pred = pred;
                    best_config = strategy;
                    
                    // Build description
                    switch (strategy.placement) {
                        case PlacementStrategy::FULL_GPU:
                            best_desc = "Full GPU, 4K";
                            break;
                        case PlacementStrategy::GPU_CPU_SPLIT:
                            best_desc = "Split " + std::to_string(strategy.gpu_layers) + "/" + 
                                       std::to_string(meta.layers) + ", 4K";
                            break;
                        case PlacementStrategy::CPU_ONLY:
                            best_desc = "CPU Only, 4K";
                            break;
                        default:
                            best_desc = "Strategy";
                            break;
                    }
                }
            }
            
            if (best_score > 0) {
                RecommendationResult rec;
                rec.model = model;
                rec.variant = variant;
                rec.best_strategy_desc = best_desc;
                rec.predicted_tok_s = best_pred.tokens_per_sec;
                rec.predicted_vram_gb = best_pred.memory_vram_bytes / 1e9;
                rec.predicted_ram_gb = best_pred.memory_ram_bytes / 1e9;
                rec.predicted_max_ctx = best_pred.max_safe_context;
                rec.rank_score = best_score;
                
                candidates.push_back(rec);
            }
        }
    }
    
    // Sort by rank_score descending
    std::sort(candidates.begin(), candidates.end(),
        [](const RecommendationResult& a, const RecommendationResult& b) {
            return a.rank_score > b.rank_score;
        });
    
    // Assign labels
    assign_labels(candidates);
    
    // Return top N
    if (static_cast<int>(candidates.size()) > request.top_n) {
        candidates.resize(request.top_n);
    }
    
    return candidates;
}
