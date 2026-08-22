#pragma once

#include "types.h"
#include <vector>

// =============================================================================
// Method Matrix Generator
// =============================================================================
// Generates all viable deployment strategies for a given hardware + model combo.
// This is what Step 4 adds that Steps 1-3 didn't have.
// =============================================================================

// Generate all viable strategies
std::vector<StrategyResult> generate_matrix(const HardwareSpec& hw, const ModelSpec& model);

// Generate strategies for a specific placement type
std::vector<StrategyResult> generate_gpu_strategies(const HardwareSpec& hw, const ModelSpec& model);
std::vector<StrategyResult> generate_split_strategies(const HardwareSpec& hw, const ModelSpec& model);
std::vector<StrategyResult> generate_cpu_strategies(const HardwareSpec& hw, const ModelSpec& model);

// Get common context lengths to test
std::vector<uint32_t> get_context_lengths(uint32_t model_max_context);

// Format strategy for display
std::string format_strategy_description(const StrategyConfig& strat, uint32_t total_layers);
