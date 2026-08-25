#pragma once

#include "predictor_types.h"

// =============================================================================
// Memory Prediction Functions
// =============================================================================

// Predict model weight memory usage
// Formula: param_count * bits_per_weight / 8
uint64_t predict_weight_memory(const ModelSpec& model);

// Predict KV cache memory usage
// Standard: 2 * layers * kv_heads * head_dim * context * batch * bytes_per_element
// MLA: 2 * layers * kv_lora_rank * context * batch * bytes + layers * qk_rope_head_dim * context * batch * bytes
uint64_t predict_kv_cache_memory(const ModelSpec& model, uint32_t context_length, 
                                  uint32_t kv_quant_bits, uint32_t batch_size = 1);

// Predict total memory overhead (activations, buffers, GPU context)
// Platform-specific: NVIDIA 512 MB, Apple 128 MB, CPU 128 MB (calibrated empirically)
uint64_t predict_overhead_memory(const ModelSpec& model, uint32_t batch_size, bool use_gpu = true,
                                  bool is_unified_memory = false);

// Predict KV cache bytes per token (for speed calculations)
double predict_kv_bytes_per_token(const ModelSpec& model, uint32_t kv_quant_bits);
