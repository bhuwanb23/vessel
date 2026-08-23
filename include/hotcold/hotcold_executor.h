#pragma once

#include "hotcold_types.h"
#include "hotcold/mask_file.h"
#include "../types.h"
#include <string>

// =============================================================================
// Step 10 Phase C — Hot/Cold Executor
// =============================================================================
// Loads a model with hot/cold neuron splitting using llama.cpp's tensor_split.
// The mask file informs the GPU/CPU weight distribution ratio.
//
// How it works:
// 1. Load the hot neuron mask file for the model
// 2. Calculate what fraction of weights are "hot" (should be on GPU)
// 3. Compute tensor_split ratio: [gpu_fraction, cpu_fraction]
// 4. Pass tensor_split to llama_model_params during loading
// 5. llama.cpp distributes weights according to the ratio
//
// This is an approximation — llama.cpp splits by tensor, not by neuron.
// But it achieves the same goal: hot weights on GPU, cold on CPU.
// =============================================================================

// Hot/cold execution configuration
struct HotColdExecConfig {
    bool enabled = false;               // Is hot/cold splitting active?
    std::string mask_file_path;         // Path to .hot_neurons.bin
    HotNeuronProfile profile;           // Loaded profile
    
    // Split ratios
    float tensor_split_gpu = 0.0f;     // Fraction of weights on GPU [0.0, 1.0]
    float tensor_split_cpu = 1.0f;     // Fraction of weights on CPU [0.0, 1.0]
    
    // Memory estimates
    uint64_t hot_weights_bytes = 0;    // Weights that should be on GPU
    uint64_t cold_weights_bytes = 0;   // Weights that should be on CPU
    uint64_t attention_weights_bytes = 0;  // Attention (always on GPU if possible)
    
    // Metadata
    uint32_t hot_neurons_per_layer = 0;
    uint32_t total_neurons_per_layer = 0;
    double hot_ratio = 0.0;
};

// Create hot/cold execution config from a model path
// Checks for .hot_neurons.bin alongside the model file
// Returns config with enabled=false if no mask file exists
HotColdExecConfig create_hotcold_config(const std::string& model_path);

// Create hot/cold config from explicit mask file path
HotColdExecConfig create_hotcold_config_from_mask(
    const std::string& model_path,
    const std::string& mask_path
);

// Calculate tensor_split ratio from hot/cold profile
// Returns [gpu_fraction, cpu_fraction] for llama_model_params.tensor_split
void calculate_tensor_split(
    const HotNeuronProfile& profile,
    uint32_t hidden_dim,
    uint32_t ffn_dim,
    uint32_t n_layers,
    double bytes_per_param,
    float& out_gpu_fraction,
    float& out_cpu_fraction
);

// Print hot/cold execution info
void print_hotcold_exec_info(const HotColdExecConfig& config);
