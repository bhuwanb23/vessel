#pragma once

#include "predictor_types.h"
#include "memory_predictor.h"
#include "speed_predictor.h"
#include "context_analyzer.h"
#include "predictor_validation.h"
#include "confidence_calculator.h"

// =============================================================================
// Main Prediction Function
// =============================================================================

// Main orchestrator - combines all prediction modules
Prediction predict(const HardwareSpec& hw, const ModelSpec& model, const StrategyConfig& strategy);

// =============================================================================
// Viability Check
// =============================================================================

// Check if a strategy is viable (fits in available memory)
bool check_viability(const HardwareSpec& hw, uint64_t total_memory_needed);

// =============================================================================
// Utility Functions
// =============================================================================

// Convert bytes to human-readable string (e.g., "4.50 GB")
std::string format_bytes(uint64_t bytes);

// Convert tokens/sec to human-readable string
std::string format_speed(double tokens_per_sec);

// Get placement strategy name
const char* get_placement_name(PlacementStrategy strategy);

// Get confidence level name
const char* get_confidence_name(PredictionConfidence confidence);
