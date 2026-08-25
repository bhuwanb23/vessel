#pragma once

#include "predictor_types.h"

// =============================================================================
// Confidence Band Logic (Phase F)
// =============================================================================
// From spec §8.4: Three levels based on metadata source, calibration records,
// model type, and context length.
// =============================================================================

// Calculate confidence level based on multiple factors
// Returns: confidence level and reason string
struct ConfidenceResult {
    PredictionConfidence level;
    std::string reason;
};

// Main confidence calculation function
ConfidenceResult calculate_confidence(
    const ModelSpec& model,
    const HardwareSpec& hw,
    uint32_t context_length,
    uint32_t calibration_records = 0,  // Number of calibration records for this hardware
    PlacementStrategy placement = PlacementStrategy::FULL_GPU,  // Strategy placement
    bool has_profiling_mask = false  // Whether hot/cold profiling mask exists
);

// Helper: Check if hardware profile is complete
bool is_hardware_complete(const HardwareSpec& hw);

// Helper: Check if context is in validated range
bool is_context_validated(uint32_t context_length);

// Helper: Get confidence level name
const char* confidence_to_string(PredictionConfidence level);
