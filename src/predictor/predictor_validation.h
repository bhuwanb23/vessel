#pragma once

#include <cstdint>

// =============================================================================
// Validation Functions
// =============================================================================

// Validate bits_per_weight by comparing file size to parameter count
// Returns: effective bpw calculated from actual file, or 0.0 if validation fails
double validate_bpw_from_file(uint64_t file_size_bytes, uint64_t param_count);

// Validate prediction against actual measurement
// Returns: error percentage (0.0 = perfect match)
double validate_prediction(double predicted, double actual);

// Check if prediction is within acceptable tolerance
bool is_prediction_accurate(double predicted, double actual, double tolerance_pct = 20.0);
