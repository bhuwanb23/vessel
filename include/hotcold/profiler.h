#pragma once

#include "hotcold_types.h"
#include <string>
#include <vector>

// =============================================================================
// Step 10 — Activation Profiler
// =============================================================================
// Profiles which neurons activate frequently across diverse inputs.
// This is a one-time offline pass that produces the hot neuron set.
// =============================================================================

// Profile a model's activation patterns
// Returns a complete HotNeuronProfile with per-layer hot sets
HotNeuronProfile profile_model_activations(
    const std::string& model_path,
    const std::string& prompts_path = "",  // Empty = use bundled prompts
    uint32_t max_prompts = 1000,
    double hot_ratio = 0.15                // Target: 15% hot neurons
);

// Profile a single layer's activations
ActivationProfile profile_layer_activations(
    const std::string& model_path,
    uint32_t layer_index,
    const std::vector<std::string>& prompts,
    ActivationType activation = ActivationType::SILU
);

// Load default bundled prompts
std::vector<std::string> load_default_prompts();

// Load prompts from file (one prompt per line)
std::vector<std::string> load_prompts_from_file(const std::string& path);

// Save hot neuron profile to file
bool save_hot_neuron_profile(const HotNeuronProfile& profile, const std::string& path);

// Load hot neuron profile from file
HotNeuronProfile load_hot_neuron_profile(const std::string& path);

// Generate hot neuron profile path from model path
// e.g., "models/Llama-7B.gguf" → "models/Llama-7B.hot_neurons.bin"
std::string get_hot_neuron_path(const std::string& model_path);
