#pragma once

#include "predictor_types.h"

// =============================================================================
// Speed Prediction Functions
// =============================================================================

// Predict decode speed (tokens/sec) - memory bandwidth bound
// FULL_GPU: tokens_per_sec = gpu_bandwidth / bytes_per_token
// CPU_ONLY: tokens_per_sec = ram_bandwidth / bytes_per_token
// GPU_CPU_SPLIT: sequential model - time = time_gpu + time_cpu
double predict_decode_speed(const HardwareSpec& hw, const ModelSpec& model, 
                           uint32_t gpu_layers, uint32_t context_length = 0,
                           uint32_t kv_quant_bits = 16);

// Predict prompt evaluation speed (tokens/sec) - compute bound
// Formula: TFLOPS * 1e12 / (2 * params)
double predict_prompt_eval_speed(const HardwareSpec& hw, const ModelSpec& model, uint32_t gpu_layers);

// Predict time to first token (TTFT) in milliseconds
// Formula: total_flops / (device_compute_throughput × 1e12) × 1000
double predict_ttft_ms(const HardwareSpec& hw, const ModelSpec& model, 
                       uint32_t prompt_tokens, uint32_t gpu_layers);

// Calibrated overloads (Step 7) — use adjusted efficiency from calibration log
double predict_prompt_eval_speed(const HardwareSpec& hw, const ModelSpec& model,
                                 uint32_t gpu_layers, double gpu_prefill_efficiency);
double predict_ttft_ms(const HardwareSpec& hw, const ModelSpec& model,
                       uint32_t prompt_tokens, uint32_t gpu_layers,
                       double gpu_prefill_efficiency);

// Get TTFT confidence bounds (±40% due to compute variability)
void predict_ttft_bounds(const HardwareSpec& hw, const ModelSpec& model,
                         uint32_t prompt_tokens, uint32_t gpu_layers,
                         double& lower_ms, double& upper_ms);

// Calculate bytes per token for decode (weights only)
double predict_bytes_per_token(const ModelSpec& model);

// Calculate effective bandwidth based on placement strategy
double calculate_effective_bandwidth(const HardwareSpec& hw, uint32_t gpu_layers, uint32_t total_layers);
