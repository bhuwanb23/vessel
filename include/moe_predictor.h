#pragma once

#include "types.h"

// =============================================================================
// MoE Predictor — Range-Based Speed Prediction (Step 9, Phase D)
// =============================================================================
// For MoE models, token generation speed varies based on routing.
// The predictor outputs a [worst, best] range instead of a point estimate.
// =============================================================================

// MoE-specific prediction result
struct MoEPrediction {
    bool valid = false;
    
    // Active bytes per token (weight bytes that must be read for one token)
    double active_bytes_per_token = 0.0;
    
    // Best-case: all k active experts hit GPU
    double tok_s_best = 0.0;
    double time_per_token_best_ms = 0.0;
    
    // Worst-case: all k active experts hit CPU
    double tok_s_worst = 0.0;
    double time_per_token_worst_ms = 0.0;
    
    // Expected: uniform routing probability
    double tok_s_expected = 0.0;
    double time_per_token_expected_ms = 0.0;
    
    // Routing probabilities
    double p_gpu = 0.0;          // P(single token hits GPU expert) = E_gpu / N
    double k_gpu_expected = 0.0; // Expected active GPU experts = k × p_gpu
    double k_cpu_expected = 0.0; // Expected active CPU experts = k × (1 - p_gpu)
    
    // Memory breakdown
    double gpu_part_bytes = 0.0; // Bytes read from GPU per token
    double cpu_part_bytes = 0.0; // Bytes read from CPU per token
};

// Compute MoE range prediction for a given hardware + model + placement plan
MoEPrediction predictMoERange(const HardwareSpec& hw, const ModelSpec& model,
                               const MoEPlacementPlan& plan, uint32_t kv_quant_bits = 16);
