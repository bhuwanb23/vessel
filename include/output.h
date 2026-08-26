#pragma once

#include "types.h"
#include "ranker.h"
#include <vector>
#include <string>

// Forward declaration for Step 12 recommendation results
struct RecommendationResult;

// =============================================================================
// Output Module — Table Formatting & Printing
// =============================================================================
// Separated from main.cpp for maintainability.
// Handles: status determination, number formatting, table layout,
//          recommendations, and warnings.
// =============================================================================

// Context filter mode
enum class ContextMode { FOUR_K, MAX, BOTH };

// Parse context string from CLI
ContextMode parse_context(const std::string& str);

// Get human-readable context mode name
const char* get_context_name(ContextMode mode);

// Strategy status (viable / tight / no fit / low confidence)
enum class StrategyStatus {
    VIABLE,
    TIGHT,
    NO_FIT,
    LOW_CONF
};

// Determine if a strategy fits in memory
StrategyStatus determine_status(const HardwareSpec& hw, const Prediction& pred,
                                const StrategyConfig& strat);

// Get status icon (emoji) and text
const char* get_status_icon(StrategyStatus status);
const char* get_status_text(StrategyStatus status);

// Filter strategies by context mode
std::vector<StrategyResult> filter_by_context(const std::vector<StrategyResult>& results,
                                              ContextMode mode, uint32_t model_max_ctx);

// Print the full prediction table
void print_prediction_table(const std::vector<StrategyResult>& results,
                            const HardwareSpec& hw, PriorityMode priority);

// Print brief hardware/model summaries (one line each)
void print_hardware_brief(const HardwareSpec& hw);
void print_model_brief(const ModelSpec& model);

// Print full hardware/model reports (--verbose mode)
void print_hardware_full(const HardwareSpec& hw);
void print_model_full(const ModelSpec& model);

// Print warnings (low VRAM, slow disk)
void print_warnings(const HardwareSpec& hw);

// Print post-table warnings (no viable strategies, CPU-only only)
void print_post_table_warnings(const std::vector<StrategyResult>& results,
                               const HardwareSpec& hw);

// =============================================================================
// Recommendation Table (Step 12, Phase C)
// =============================================================================
// Prints the model recommendation table with labels, star ratings,
// download URLs, and copy-pasteable run commands.
// =============================================================================

void print_recommendation_table(
    const std::vector<RecommendationResult>& recs,
    const HardwareSpec& hw,
    const std::string& priority,
    const std::string& use_case
);

// Print usage help
void print_usage();

// Resolve model file path from URL and optional local path
// Returns resolved path if file exists, empty string otherwise
std::string resolve_model_path(const std::string& url, const std::string& local_path);
