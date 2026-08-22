#pragma once

#include "predictor_types.h"

// =============================================================================
// LLM Predictor — Pure Math Functions
// =============================================================================
// Contract: Given hardware specs, model metadata, and strategy config,
//           predict memory usage, tokens/sec, and time-to-first-token.
//
// No side effects. No global state. No I/O.
// Same inputs always produce same outputs.
// =============================================================================

// Main prediction function
// Takes: HardwareSpec (from Step 1), ModelSpec (from Step 2), StrategyConfig
// Returns: Prediction with memory, performance, and viability estimates
Prediction predict(const HardwareSpec& hw, const ModelSpec& model, const StrategyConfig& strategy);

// =============================================================================
// Individual Prediction Functions (used by predict(), also testable separately)
// =============================================================================

// Predict model weight memory usage (just the weights, not KV cache)
// Formula: param_count * bits_per_weight / 8
uint64_t predict_weight_memory(const ModelSpec& model);

// Predict KV cache memory usage
// Formula: 2 * layers * context_length * kv_heads * head_dim * kv_quant_bits / 8
uint64_t predict_kv_cache_memory(const ModelSpec& model, uint32_t context_length, uint32_t kv_quant_bits);

// Predict total memory overhead (activations, buffers, CUDA context)
// Typically 200-500MB depending on batch size and model
uint64_t predict_overhead_memory(const ModelSpec& model, uint32_t batch_size);

// Predict decode speed (tokens/sec)
// Based on memory bandwidth and model size
double predict_tokens_per_sec(const HardwareSpec& hw, const ModelSpec& model, uint32_t gpu_layers);

// Predict time to first token (prompt processing)
// Based on prompt length and prompt processing speed
double predict_ttft_ms(const HardwareSpec& hw, const ModelSpec& model, uint32_t prompt_tokens, uint32_t gpu_layers);

// Predict prompt evaluation speed (tokens/sec)
double predict_prompt_eval_speed(const HardwareSpec& hw, const ModelSpec& model, uint32_t gpu_layers);

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
