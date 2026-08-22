#pragma once

#include "predictor_types.h"

// =============================================================================
// Context Analysis Functions
// =============================================================================

// Calculate maximum safe context length given memory budget
// Reverses the KV cache formula to find max context that fits
uint32_t calculate_max_safe_context(const HardwareSpec& hw, const ModelSpec& model, 
                                    const StrategyConfig& strategy);

// Calculate memory budget for a given strategy
uint64_t calculate_memory_budget(const HardwareSpec& hw, const ModelSpec& model, 
                                 const StrategyConfig& strategy);

// Check if a specific context length is viable
bool is_context_viable(const HardwareSpec& hw, const ModelSpec& model, 
                       const StrategyConfig& strategy, uint32_t context_length);
