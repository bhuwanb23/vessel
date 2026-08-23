#pragma once

#include "hotcold_types.h"
#include <vector>
#include <cstdint>

// =============================================================================
// Step 10 — Sparse FFN Operator
// =============================================================================
// Implements the hot/cold split FFN computation:
//   output = down_proj( activation( up_proj(x) ) * gate_proj(x) )
//
// Split into:
//   GPU: Compute hot neurons (smaller GEMM)
//   CPU: Compute cold neurons (sparse, only activated ones)
//   Combine: Sum partial results
// =============================================================================

// Sparse FFN computation context
struct SparseFFNContext {
    // Weight pointers (set before computation)
    const float* up_proj_hot = nullptr;    // [hidden_dim, n_hot] on GPU
    const float* up_proj_cold = nullptr;   // [hidden_dim, n_cold] on CPU
    const float* gate_proj_hot = nullptr;  // [hidden_dim, n_hot] on GPU
    const float* gate_proj_cold = nullptr; // [hidden_dim, n_cold] on CPU
    const float* down_proj_hot = nullptr;  // [n_hot, hidden_dim] on GPU
    const float* down_proj_cold = nullptr; // [n_cold, hidden_dim] on CPU
    
    // Dimensions
    uint32_t hidden_dim = 0;
    uint32_t ffn_dim = 0;
    uint32_t n_hot = 0;
    uint32_t n_cold = 0;
    
    // Activation function
    ActivationType activation = ActivationType::SILU;
    float silu_threshold = -2.0f;  // For SiLU sparsity detection
    
    // Hot neuron indices
    const uint32_t* hot_indices = nullptr;
    const uint32_t* cold_indices = nullptr;
};

// Compute sparse FFN for a single token
// Input: x [hidden_dim]
// Output: output [hidden_dim]
SparseFFNResult sparse_ffn_forward(
    const SparseFFNContext& ctx,
    const float* input,           // [hidden_dim]
    float* output                 // [hidden_dim] (pre-allocated)
);

// GPU path: compute hot neuron contribution
void sparse_ffn_gpu_hot(
    const SparseFFNContext& ctx,
    const float* input,           // [hidden_dim]
    float* gpu_partial            // [hidden_dim] (pre-allocated)
);

// CPU path: compute cold neuron contribution (sparse)
void sparse_ffn_cpu_cold(
    const SparseFFNContext& ctx,
    const float* input,           // [hidden_dim]
    float* cpu_partial            // [hidden_dim] (pre-allocated)
);

// Combine GPU and CPU partial results
void sparse_ffn_combine(
    const float* gpu_partial,     // [hidden_dim]
    const float* cpu_partial,     // [hidden_dim]
    float* output,                // [hidden_dim]
    uint32_t hidden_dim
);

// Determine which cold neurons are activated for this token
// Returns indices of cold neurons where activation > threshold
std::vector<uint32_t> detect_activated_cold_neurons(
    const float* pre_activation,  // [n_cold] pre-activation values
    uint32_t n_cold,
    ActivationType activation,
    float threshold
);

// Layer-streaming fallback: load and compute one layer at a time
SparseFFNResult layer_streaming_forward(
    const std::string& model_path,
    uint32_t layer_index,
    const float* input,           // [hidden_dim]
    float* output,                // [hidden_dim]
    uint64_t disk_read_speed_mbs
);
