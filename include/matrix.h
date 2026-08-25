#pragma once

#include "types.h"
#include <vector>

// Forward declare
struct CalibrationData;

// =============================================================================
// Method Matrix Generator (Phase B)
// =============================================================================
// Generates all viable deployment strategies for a given hardware + model combo.
// Three dimensions:
//   1. Placement: FULL_GPU, GPU_CPU_SPLIT (max-fit, half, minimal), CPU_ONLY
//   2. Context Length: 4K, max-safe
//   3. KV Cache Precision: FP16 (16 bits), Q8 (8 bits)
// =============================================================================

// Generate all viable strategies (no calibration)
std::vector<StrategyResult> generate_matrix(const HardwareSpec& hw, const ModelSpec& model);

// Generate with calibrated predictions (Step 7)
std::vector<StrategyResult> generate_matrix(const HardwareSpec& hw, const ModelSpec& model,
                                            const CalibrationData& cal);

// Format strategy for display
std::string format_strategy_description(const StrategyConfig& strat, uint32_t total_layers,
                                         bool is_unified_memory = false);
