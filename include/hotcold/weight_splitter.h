#pragma once

#include "hotcold_types.h"
#include <string>
#include <vector>
#include <cstdint>

// =============================================================================
// Step 10 — Hot/Cold Weight Splitter
// =============================================================================
// Splits model weight matrices into hot (GPU) and cold (CPU) subsets
// based on the neuron profile.
// =============================================================================

// Split a weight matrix by neuron indices
// input: weight matrix of shape [rows, cols] (row-major)
// hot_indices: which columns (neurons) go to GPU
// cold_indices: which columns (neurons) go to CPU
struct WeightSplit {
    std::vector<float> hot_weights;  // [rows, n_hot] on GPU
    std::vector<float> cold_weights; // [rows, n_cold] on CPU
    uint32_t rows = 0;
    uint32_t n_hot = 0;
    uint32_t n_cold = 0;
};

WeightSplit split_weight_matrix(
    const float* weights,       // Input weight matrix [rows, cols]
    uint32_t rows,
    uint32_t cols,
    const std::vector<uint32_t>& hot_indices,
    const std::vector<uint32_t>& cold_indices
);

// Split all FFN weight matrices for a layer
struct LayerWeightSplit {
    WeightSplit up_proj;     // [hidden_dim, ffn_dim] → hot + cold
    WeightSplit gate_proj;   // [hidden_dim, ffn_dim] → hot + cold
    WeightSplit down_proj;   // [ffn_dim, hidden_dim] → hot + cold (transposed)
};

LayerWeightSplit split_layer_weights(
    const float* up_proj_weights,
    const float* gate_proj_weights,
    const float* down_proj_weights,
    uint32_t hidden_dim,
    uint32_t ffn_dim,
    const LayerHotSet& hot_set
);

// Save split weights to disk (for pre-computed hot/cold sets)
bool save_split_weights(const LayerWeightSplit& split, const std::string& path);

// Load split weights from disk
LayerWeightSplit load_split_weights(const std::string& path);

// Calculate memory requirements for hot/cold split
struct SplitMemoryEstimate {
    uint64_t hot_weights_bytes = 0;   // Total hot weights across all layers
    uint64_t cold_weights_bytes = 0;  // Total cold weights across all layers
    uint64_t vram_required_bytes = 0; // Hot weights + CUDA overhead
    uint64_t ram_required_bytes = 0;  // Cold weights
    bool fits_in_budget = false;
};

SplitMemoryEstimate estimate_split_memory(
    const HotNeuronProfile& profile,
    uint64_t vram_budget_bytes,
    uint64_t ram_budget_bytes,
    double bytes_per_param
);
