#pragma once

#include "catalog_types.h"
#include "../types.h"
#include <string>
#include <vector>

// =============================================================================
// Recommendation Engine (Step 12, Phase B)
// =============================================================================
// Generates model recommendations based on hardware profile and user preferences
// =============================================================================

// =============================================================================
// Recommendation Request
// =============================================================================

struct RecommendationRequest {
    std::string priority = "balanced";   // "speed", "quality", "balanced"
    std::string use_case = "all";        // "chat", "coding", "reasoning", "all"
    double max_download_gb = 0.0;        // 0 = no limit
    int top_n = 8;                       // Number of recommendations to show
};

// =============================================================================
// Recommendation Result
// =============================================================================

struct RecommendationResult {
    CatalogModel model;
    GgufVariant variant;
    std::string best_strategy_desc;
    double predicted_tok_s = 0.0;
    double predicted_vram_gb = 0.0;
    double predicted_ram_gb = 0.0;
    uint32_t predicted_max_ctx = 0;
    std::string label;
    double rank_score = 0.0;
};

// =============================================================================
// Core Functions
// =============================================================================

// Generate recommendations based on hardware and request
std::vector<RecommendationResult> generate_recommendations(
    const HardwareSpec& hw,
    const ModelCatalog& catalog,
    const RecommendationRequest& request
);

// Convert catalog model to ModelSpec for prediction
ModelSpec catalog_to_model_spec(const CatalogModel& model, const GgufVariant& variant);

// Pre-filter: check if a model variant can fit on the hardware
bool passes_viability_check(const HardwareSpec& hw, const ModelSpec& model);

// Generate mini method matrix for a single model
std::vector<StrategyConfig> generate_mini_matrix(
    const HardwareSpec& hw,
    const ModelSpec& model
);

// Score a prediction for cross-model ranking
double score_for_priority(
    const Prediction& pred,
    double quality_score,
    const std::string& priority
);

// Assign labels to ranked recommendations
void assign_labels(std::vector<RecommendationResult>& recs);

// Get use case filter predicate
bool matches_use_case(const CatalogModel& model, const std::string& use_case);
