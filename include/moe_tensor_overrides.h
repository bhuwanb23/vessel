#pragma once

#include "types.h"
#include "moe_placer.h"
#include <vector>

// =============================================================================
// MoE Tensor Override Generation (Step 9, Phase C)
// =============================================================================
// Generates tensor placement overrides for llama.cpp to split MoE expert
// tensors between GPU and CPU based on the placement plan.
// =============================================================================

// Tensor override rule — specifies which tensors go to which backend
struct TensorOverride {
    std::string name_pattern;   // Regex or exact tensor name
    int backend_id;             // 0 = GPU, 1 = CPU
    std::string description;    // Human-readable (for debugging)
};

// Result of tensor override generation
struct MoETensorOverrides {
    bool needed = false;                        // false for dense models
    std::vector<TensorOverride> overrides;      // Per-tensor rules
    float tensor_split[2] = {0.0f, 0.0f};      // GPU/CPU split ratio (fallback)
    uint64_t gpu_weight_bytes = 0;              // Total weights on GPU
    uint64_t cpu_weight_bytes = 0;              // Total weights on CPU
    std::string summary;                        // Human-readable summary
};

// Generate tensor overrides for a MoE model based on placement plan.
// Returns overrides that can be applied to llama_model_params.
MoETensorOverrides generateMoETensorOverrides(const ModelSpec& model,
                                               const MoEPlacementPlan& plan);

// Generate overrides from strategy config (convenience wrapper).
// Returns empty overrides for dense models.
MoETensorOverrides generateMoETensorOverrides(const ModelSpec& model,
                                               const StrategyConfig& strategy,
                                               const HardwareSpec& hw);
