#pragma once

#include "types.h"
#include "executor.h"
#include <string>

// =============================================================================
// Comparison Report — Predicted vs Actual
// =============================================================================
// Prints a side-by-side comparison after running inference.
// Used by the executor test and eventually the full pipeline.
// =============================================================================

// Delta accuracy level
enum class DeltaLevel {
    CLOSE,  // < 10% — prediction is accurate
    OFF,    // 10-25% — prediction needs calibration
    WRONG   // > 25% — formula or constant is significantly wrong
};

// Get human-readable delta level label
const char* get_delta_level_label(DeltaLevel level);

// Get delta level from a percentage delta (absolute value)
DeltaLevel classify_delta(double delta_pct);

// Print the predicted-vs-actual comparison report
// predicted: the Prediction struct from Step 3's predictor
// actual: the ExecutionResult from Step 6's executor
// strategy: the strategy that was executed
void print_comparison_report(const Prediction& predicted,
                             const ExecutionResult& actual,
                             const StrategyConfig& strategy);
