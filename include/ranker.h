#pragma once

#include "types.h"
#include <vector>

// =============================================================================
// Step 5 — Ranker: Scoring Function + Three Priority Axes
// =============================================================================
// From spec: "A scoring function converts each prediction into a single double
//            score, then you just sort by that score descending."
//
// Score formula:
//   score = (primary × 0.7 + secondary × 0.2 + tertiary × 0.1) × confidence_mult
//
// Non-viable strategies always get score = -1.0 (bottom of list).
// =============================================================================

// Priority modes (reused by CLI and output)
enum class PriorityMode { SPEED, QUALITY, SAFETY };

// Parse priority string from CLI
PriorityMode parse_priority(const std::string& str);

// Get human-readable priority name
const char* get_priority_name(PriorityMode mode);

// Calculate memory headroom (0.0 to 1.0)
double calculate_memory_headroom(const HardwareSpec& hw, const Prediction& pred);

// Calculate composite score for a strategy (Phase B)
// Returns -1.0 for non-viable strategies, otherwise 0.0 to 1.0
double calculate_score(const StrategyResult& result, const HardwareSpec& hw,
                       PriorityMode priority,
                       double min_primary, double max_primary,
                       double min_secondary, double max_secondary,
                       double min_tertiary, double max_tertiary);

// Sort strategies by priority using scoring function (main entry point)
void sort_by_priority(std::vector<StrategyResult>& results, PriorityMode priority,
                      const HardwareSpec& hw);
