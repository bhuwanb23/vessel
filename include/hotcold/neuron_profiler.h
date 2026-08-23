#pragma once

#include "hotcold_types.h"
#include "../types.h"
#include <string>
#include <vector>
#include <functional>

// =============================================================================
// Step 10 Phase B — Neuron Activation Profiler Engine
// =============================================================================
// Profiles which neurons activate frequently across diverse inputs by
// running the model through llama.cpp and capturing post-activation values
// from each FFN layer.
//
// The profiler uses a layer-by-layer approach:
// 1. Load model via llama.cpp API
// 2. For each prompt, tokenize and run inference
// 3. At each FFN layer, capture the post-activation output
// 4. Record which neurons activated (value > threshold)
// 5. Aggregate across all prompts and tokens
// 6. Rank neurons by activation frequency
// 7. Select top N as "hot" set
// =============================================================================

// Progress callback for profiling
struct ProfilingProgress {
    uint32_t current_prompt = 0;
    uint32_t total_prompts = 0;
    uint32_t current_layer = 0;
    uint32_t total_layers = 0;
    uint64_t tokens_processed = 0;
    double elapsed_seconds = 0.0;
    double estimated_remaining_seconds = 0.0;
};

using ProfilingProgressCallback = std::function<void(const ProfilingProgress&)>;

// Configuration for the profiler
struct ProfilingConfig {
    std::string model_path;         // Path to GGUF file
    std::string prompts_path;       // Path to prompts file (empty = bundled)
    std::string output_path;        // Output mask file path (empty = auto)
    
    uint32_t max_prompts = 1000;    // Maximum prompts to process
    uint32_t max_tokens_per_prompt = 200;  // Max tokens per prompt
    
    double hot_ratio = 0.15;        // Target hot neuron ratio (15% default)
    uint64_t vram_budget_bytes = 0; // VRAM budget for hot neurons (0 = auto-detect)
    
    ActivationType activation = ActivationType::SILU;  // Activation function type
    float activation_threshold = 0.01f;  // Threshold for SiLU/GELU sparsity
    
    bool validate_stability = false;  // Run stability validation after profiling
    
    int n_gpu_layers = 0;  // GPU layers for model loading (0 = CPU only for profiling)
};

// Result of profiling a single layer
struct LayerProfilingResult {
    uint32_t layer_index = 0;
    uint32_t ffn_dim = 0;
    std::vector<double> activation_frequency;  // Per-neuron frequency [0.0, 1.0]
    uint32_t total_tokens_profiled = 0;
    double avg_neurons_activated = 0.0;  // Average neurons activated per token
    uint32_t n_hot = 0;                  // Number of hot neurons selected
    
    // Ranked neuron indices (sorted by activation frequency, descending)
    std::vector<uint32_t> ranked_indices;
};

// Full profiling result
struct ProfilingResult {
    bool success = false;
    std::string error_message;
    
    HotNeuronProfile profile;
    
    // Per-layer results (before hot/cold selection)
    std::vector<LayerProfilingResult> layer_results;
    
    // Stability validation results
    double stability_overlap = 0.0;  // Overlap between two profile runs
    bool stability_passed = false;
    
    // Timing
    double total_seconds = 0.0;
    uint64_t total_tokens = 0;
};

// =============================================================================
// Core profiling functions
// =============================================================================

// Run the full profiling pipeline
// This is the main entry point for profiling
ProfilingResult run_neuron_profiling(
    const ProfilingConfig& config,
    ProfilingProgressCallback progress_cb = nullptr
);

// Profile activations for all layers using a set of prompts
// Returns per-layer activation frequency arrays
std::vector<LayerProfilingResult> profile_all_layers(
    const std::string& model_path,
    const std::vector<std::string>& prompts,
    uint32_t max_tokens_per_prompt,
    ActivationType activation_type,
    float activation_threshold,
    int n_gpu_layers,
    ProfilingProgressCallback progress_cb = nullptr
);

// Select hot neurons from activation frequencies
// Given per-layer frequencies and a VRAM budget, select the top N neurons
std::vector<LayerHotSet> select_hot_neurons(
    const std::vector<LayerProfilingResult>& layer_results,
    uint32_t num_layers,
    uint32_t ffn_dim,
    uint64_t vram_budget_bytes,
    double bytes_per_param,
    double hot_ratio
);

// Validate hot set stability by comparing two profile runs
// Returns overlap ratio [0.0, 1.0]
double validate_hot_set_stability(
    const std::vector<LayerHotSet>& set_a,
    const std::vector<LayerHotSet>& set_b
);

// =============================================================================
// Low-level profiling helpers
// =============================================================================

// Capture activations from a single forward pass through the model
// Returns post-activation values for each FFN layer
// Each layer's result is a vector of float values [ffn_dim]
struct ForwardPassActivations {
    std::vector<std::vector<float>> layer_activations;  // [layer][neuron]
    uint32_t tokens_processed = 0;
};

// Run a single prompt through the model and capture FFN activations
ForwardPassActivations capture_activations(
    const std::string& model_path,
    const std::string& prompt,
    uint32_t max_tokens,
    int n_gpu_layers
);

// Check if a neuron is activated given its post-activation value
inline bool is_neuron_activated(float value, ActivationType activation, float threshold) {
    switch (activation) {
        case ActivationType::RELU:  return value > 0.0f;
        case ActivationType::SILU:  return value > threshold;
        case ActivationType::GELU:  return value > threshold;
        default:                    return value > threshold;
    }
}
