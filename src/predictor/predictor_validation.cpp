#include "predictor_validation.h"
#include <cmath>

// =============================================================================
// Validation Functions
// =============================================================================

double validate_bpw_from_file(uint64_t file_size_bytes, uint64_t param_count) {
    if (file_size_bytes == 0 || param_count == 0) return 0.0;
    
    // Formula: bpw = (file_size_bytes * 8) / param_count
    double bpw = (static_cast<double>(file_size_bytes) * 8.0) / static_cast<double>(param_count);
    return bpw;
}

double validate_prediction(double predicted, double actual) {
    if (actual == 0) return 0.0;
    
    // Calculate error percentage
    double error = std::abs(predicted - actual) / actual * 100.0;
    return error;
}

bool is_prediction_accurate(double predicted, double actual, double tolerance_pct) {
    double error = validate_prediction(predicted, actual);
    return error <= tolerance_pct;
}
