#pragma once

#include <string>
#include <vector>
#include <cstdint>

// =============================================================================
// Model Catalog Data Structures (Step 12, Phase A/E)
// =============================================================================
// Defines the data structures for the model catalog used by the
// auto-recommendation engine.
// =============================================================================

// =============================================================================
// GGUF Variant
// =============================================================================
// A specific quantization of a model available for download
// =============================================================================

struct GgufVariant {
    std::string quant;           // "Q4_K_M", "Q8_0", etc.
    double file_size_gb = 0.0;   // Download size in GB
    double bpw = 0.0;            // Bits per weight
    std::string hf_repo;         // Hugging Face repository (e.g., "bartowski/Llama-3.2-3B-Instruct-GGUF")
    std::string hf_file;         // Filename in the repository
    std::string hf_url;          // Full download URL
};

// =============================================================================
// Model Dimensions
// =============================================================================
// Pre-extracted model architecture dimensions (from GGUF header)
// =============================================================================

struct ModelDimensions {
    uint32_t layers = 0;
    uint32_t embedding_dim = 0;
    uint32_t attention_heads = 0;
    uint32_t kv_heads = 0;
    uint32_t head_dim = 0;
    uint32_t ffn_dim = 0;
};

// =============================================================================
// Catalog Model
// =============================================================================
// A single model entry in the catalog with all metadata needed for
// recommendation and prediction
// =============================================================================

struct CatalogModel {
    std::string id;              // Unique identifier (e.g., "llama-3.2-3b-instruct")
    std::string name;            // Human-readable name (e.g., "Llama 3.2 3B Instruct")
    std::string family;          // Model family (e.g., "llama", "qwen", "mistral")
    std::vector<std::string> use_cases;  // Tags: "chat", "coding", "reasoning", etc.
    std::string description;     // One-line summary
    
    // Model specs
    double params_billions = 0.0;   // Parameter count in billions
    std::string architecture;       // GGUF architecture string
    uint32_t max_context = 0;       // Maximum context length
    bool is_moe = false;            // Whether model uses Mixture of Experts
    
    // MoE-specific fields (only if is_moe = true)
    uint32_t expert_count = 0;      // Total routed experts per layer
    uint32_t expert_used_count = 0; // Active experts per token
    uint32_t expert_ffn_dim = 0;    // FFN dimension per expert
    
    // Quality assessment
    double quality_score = 0.0;     // 0-10 quality rating
    std::string quality_source;     // Citation for the quality score
    
    // Available variants
    std::vector<GgufVariant> variants;
    
    // Pre-computed dimensions (for prediction without GGUF fetch)
    ModelDimensions dimensions;
};

// =============================================================================
// Model Catalog
// =============================================================================
// The complete catalog of models available for recommendation
// =============================================================================

struct ModelCatalog {
    std::string version;             // Catalog version (e.g., "2026.08.15")
    std::vector<CatalogModel> models;  // All models in the catalog
};

// =============================================================================
// Recommendation Result
// =============================================================================
// A single recommendation from the engine
// =============================================================================

struct Recommendation {
    CatalogModel model;              // The recommended model
    GgufVariant variant;             // The specific variant to download
    std::string best_strategy_desc;  // Description of the best strategy
    double predicted_tok_s = 0.0;    // Predicted tokens per second
    double predicted_vram_gb = 0.0;  // Predicted VRAM usage in GB
    double predicted_ram_gb = 0.0;   // Predicted RAM usage in GB
    uint32_t predicted_max_ctx = 0;  // Predicted max safe context
    std::string label;               // "🏆 Best Overall", "⚡ Fastest", etc.
    double rank_score = 0.0;         // Ranking score
};

// =============================================================================
// Quality Score to Star Rating
// =============================================================================

inline const char* quality_to_stars(double score) {
    if (score >= 8.0) return "★★★★★";
    if (score >= 6.0) return "★★★★☆";
    if (score >= 4.0) return "★★★☆☆";
    if (score >= 2.0) return "★★☆☆☆";
    return "★☆☆☆☆";
}

inline const char* quality_to_label(double score) {
    if (score >= 8.0) return "State-of-the-art";
    if (score >= 6.0) return "Strong performer";
    if (score >= 4.0) return "Competent";
    if (score >= 2.0) return "Basic";
    return "Limited";
}
