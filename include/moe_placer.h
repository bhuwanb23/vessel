#pragma once

#include "types.h"

// =============================================================================
// MoE Expert Placement Engine (Step 9, Phase B)
// =============================================================================
// Given hardware profile and MoE model metadata, compute the optimal
// tensor-by-tensor placement configuration for expert offloading.
// =============================================================================

// Compute the MoE expert placement plan for a given strategy.
// Returns a PlacementPlan with VRAM/RAM distribution and expert counts.
MoEPlacementPlan computeMoEPlacement(const HardwareSpec& hw, const ModelSpec& model,
                                      const StrategyConfig& strategy);

// Compute placement for specific MoE variants
MoEPlacementPlan computeMoEFullVRAM(const HardwareSpec& hw, const ModelSpec& model,
                                     uint32_t context_length, uint32_t kv_quant_bits);

MoEPlacementPlan computeMoEExpertOffload(const HardwareSpec& hw, const ModelSpec& model,
                                          uint32_t context_length, uint32_t kv_quant_bits);

MoEPlacementPlan computeMoECPUOnly(const HardwareSpec& hw, const ModelSpec& model,
                                    uint32_t context_length, uint32_t kv_quant_bits);

// Check if a model is MoE (has expert_count > 0)
inline bool is_moe_model(const ModelSpec& model) {
    return model.is_moe && model.expert_count > 0;
}
