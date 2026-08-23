#pragma once

#include "../types.h"
#include <vector>
#include <string>
#include <cstdint>

// =============================================================================
// Step 10 — Hot/Cold CPU-GPU Offload Types
// =============================================================================

// Activation function type
enum class ActivationType {
    RELU,       // Exact sparsity (output ≤ 0 = inactive)
    SILU,       // Threshold-based sparsity (pre_activation < threshold = inactive)
    GELU,       // Threshold-based (similar to SiLU)
    UNKNOWN
};

// Hot neuron set for a single layer
struct LayerHotSet {
    uint32_t layer_index = 0;
    uint32_t ffn_dim = 0;           // Total neurons in this layer
    uint32_t n_hot = 0;             // Number of hot neurons
    uint32_t n_cold = 0;            // Number of cold neurons (= ffn_dim - n_hot)
    std::vector<uint32_t> hot_indices;  // Indices of hot neurons (sorted)
    std::vector<uint32_t> cold_indices; // Indices of cold neurons (sorted)
};

// Complete hot neuron profile for a model
struct HotNeuronProfile {
    std::string model_name;
    ActivationType activation = ActivationType::SILU;
    uint32_t num_layers = 0;
    uint32_t ffn_dim = 0;           // FFN dimension (same for all layers)
    uint32_t hidden_dim = 0;        // Hidden dimension
    double hot_ratio = 0.15;        // Target hot ratio (15% default)
    
    std::vector<LayerHotSet> layers;  // Per-layer hot sets
    
    // Profiling metadata
    uint32_t num_prompts_used = 0;
    double avg_activation_rate = 0.0;  // Average % of neurons activated per token
};

// Hot/Cold placement strategy
struct HotColdStrategy {
    bool viable = false;
    std::string reason;
    
    // Memory budget
    uint64_t vram_budget_bytes = 0;
    uint64_t ram_budget_bytes = 0;
    
    // Hot set sizing
    uint32_t hot_neurons_per_layer = 0;  // How many hot neurons per layer
    uint32_t total_hot_neurons = 0;      // Across all layers
    uint64_t hot_weights_vram_bytes = 0; // VRAM used by hot weights
    uint64_t cold_weights_ram_bytes = 0; // RAM used by cold weights
    
    // Performance estimates
    double estimated_tok_s_best = 0.0;   // All hot neurons on GPU
    double estimated_tok_s_worst = 0.0;  // All cold neurons activated
    double estimated_tok_s_expected = 0.0; // Average case
    
    // Layer streaming fallback
    bool use_layer_streaming = false;
    uint64_t layer_streaming_disk_speed_mbs = 0;
    double estimated_layer_streaming_tok_s = 0.0;
};

// Sparse FFN computation result
struct SparseFFNResult {
    std::vector<float> output;      // Final FFN output
    uint32_t hot_neurons_activated = 0;
    uint32_t cold_neurons_activated = 0;
    double gpu_compute_ms = 0.0;
    double cpu_compute_ms = 0.0;
    double pcie_transfer_ms = 0.0;
};

// Activation profiling result for a single layer
struct ActivationProfile {
    uint32_t layer_index = 0;
    std::vector<double> activation_frequency;  // Per-neuron activation frequency
    uint32_t total_tokens_profiled = 0;
    double avg_neurons_activated = 0.0;  // Average neurons activated per token
};

// Layer streaming configuration
struct LayerStreamingConfig {
    bool enabled = false;
    std::string model_path;         // Path to GGUF file
    uint32_t layers_to_stream = 0;  // How many layers to stream (rest on GPU)
    uint64_t disk_read_speed_mbs = 0;  // From Step 1 profiler
    double estimated_time_per_layer_ms = 0.0;
};

// Placement strategy for hot/cold
enum class HotColdPlacement {
    FULL_GPU_HOT,           // All hot neurons on GPU (fastest)
    HOT_COLD_SPLIT,         // Hot on GPU, cold on CPU (main strategy)
    CPU_ONLY,               // All neurons on CPU (fallback)
    LAYER_STREAMING,        // Stream layers from disk (extreme fallback)
    NAIVE_LAYER_SPLIT       // Standard layer split (baseline comparison)
};
