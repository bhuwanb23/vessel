#include "hotcold/profiler.h"
#include "hotcold/neuron_profiler.h"
#include "hotcold/mask_file.h"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <random>

// =============================================================================
// Default bundled prompts for activation profiling
// =============================================================================
// These are diverse, general-purpose prompts that exercise different neuron
// activation patterns. The hot set is stable across prompts, so a moderate
// set of diverse inputs is sufficient.

static const char* DEFAULT_PROMPTS[] = {
    // General knowledge
    "Explain the process of photosynthesis in simple terms.",
    "What are the main causes of climate change?",
    "How does the human immune system work?",
    "Describe the history of the Roman Empire.",
    "What is quantum computing and how does it differ from classical computing?",
    // Code
    "Write a Python function to sort a list of integers using quicksort.",
    "Explain the difference between a stack and a queue data structure.",
    "What is a hash table and how does it handle collisions?",
    "Write a SQL query to find the top 10 customers by total purchase amount.",
    "How does garbage collection work in Java?",
    // Math
    "Prove that the square root of 2 is irrational.",
    "Explain the concept of eigenvalues and eigenvectors.",
    "What is the fundamental theorem of calculus?",
    "Describe the Euclidean algorithm for finding GCD.",
    "What is a Fourier transform and what is it used for?",
    // Creative
    "Write a short poem about the ocean at sunset.",
    "Describe a futuristic city in the year 2150.",
    "Create a story about a robot learning to cook.",
    "What would life be like on Mars?",
    "Describe the feeling of flying for the first time.",
    // Analysis
    "Compare the economic systems of capitalism and socialism.",
    "What are the pros and cons of remote work?",
    "Analyze the impact of social media on mental health.",
    "Discuss the ethical implications of artificial intelligence.",
    "What are the key factors in successful project management?",
    // Science
    "Explain the theory of relativity in plain language.",
    "How do vaccines work at a molecular level?",
    "Describe the process of DNA replication.",
    "What causes earthquakes and how are they measured?",
    "Explain the water cycle and its importance to ecosystems.",
    // Language
    "What are the differences between Spanish and Portuguese?",
    "How does language evolution work over centuries?",
    "Explain the concept of linguistic relativity.",
    "What makes a language a dialect vs a separate language?",
    "How do tonal languages like Mandarin work?",
    // Philosophy
    "What is the trolley problem in ethics?",
    "Explain Plato's allegory of the cave.",
    "What is existentialism and who are its key thinkers?",
    "Discuss the mind-body problem in philosophy.",
    "What is the difference between morality and ethics?",
    // Practical
    "How do you make sourdough bread from scratch?",
    "What are the best practices for home networking?",
    "Explain how to change a car tire step by step.",
    "What should you consider when buying a house?",
    "How does compound interest work in savings accounts?",
    // Technical
    "Explain how TCP/IP networking works at a high level.",
    "What is a microservice architecture and when should you use it?",
    "How does encryption protect data in transit?",
    "Describe the CAP theorem in distributed systems.",
    "What are the differences between SQL and NoSQL databases?",
};

static const size_t DEFAULT_PROMPT_COUNT = sizeof(DEFAULT_PROMPTS) / sizeof(DEFAULT_PROMPTS[0]);

// =============================================================================
// Default prompts
// =============================================================================

std::vector<std::string> load_default_prompts() {
    std::vector<std::string> prompts;
    prompts.reserve(DEFAULT_PROMPT_COUNT);
    for (size_t i = 0; i < DEFAULT_PROMPT_COUNT; i++) {
        prompts.push_back(DEFAULT_PROMPTS[i]);
    }
    return prompts;
}

std::vector<std::string> load_prompts_from_file(const std::string& path) {
    std::vector<std::string> prompts;
    std::ifstream file(path);
    if (!file.is_open()) {
        fprintf(stderr, "Warning: Could not open prompts file: %s\n", path.c_str());
        return load_default_prompts();
    }
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty() && line[0] != '#') {
            prompts.push_back(line);
        }
    }
    if (prompts.empty()) {
        return load_default_prompts();
    }
    return prompts;
}

// =============================================================================
// Activation profiling
// =============================================================================

ActivationProfile profile_layer_activations(
    const std::string& model_path,
    uint32_t layer_index,
    const std::vector<std::string>& prompts,
    ActivationType activation)
{
    // NOTE: This is the skeleton for the profiling infrastructure.
    // Full implementation requires:
    // 1. Loading the model via llama.cpp C API
    // 2. Running each prompt through the model
    // 3. Hooking into the FFN computation to capture activation values
    // 4. Recording which neurons activate (output > threshold)
    //
    // The hook mechanism depends on ggml's compute graph, which requires
    // custom modifications to llama.cpp (similar to PowerInfer's approach).
    //
    // For now, this returns a placeholder profile that can be used for
    // testing the rest of the pipeline. The actual profiling will be
    // implemented when integrating with llama.cpp's compute graph.

    ActivationProfile profile;
    profile.layer_index = layer_index;
    profile.total_tokens_profiled = 0;
    profile.avg_neurons_activated = 0.0;

    // Placeholder: In full implementation, this would:
    // for (const auto& prompt : prompts) {
    //     auto tokens = tokenize(prompt);
    //     for (auto token : tokens) {
    //         auto activations = run_ffn_layer(model, layer_index, token);
    //         for (uint32_t n = 0; n < ffn_dim; n++) {
    //             if (is_activated(activations[n], activation)) {
    //                 profile.activation_frequency[n] += 1.0;
    //             }
    //         }
    //         profile.total_tokens_profiled++;
    //     }
    // }
    //
    // // Normalize frequencies
    // for (auto& freq : profile.activation_frequency) {
    //     freq /= profile.total_tokens_profiled;
    // }

    return profile;
}

// =============================================================================
// Full model profiling
// =============================================================================

HotNeuronProfile profile_model_activations(
    const std::string& model_path,
    const std::string& prompts_path,
    uint32_t max_prompts,
    double hot_ratio)
{
    // Use the neuron profiler engine for full implementation
    ProfilingConfig config;
    config.model_path = model_path;
    config.prompts_path = prompts_path;
    config.max_prompts = max_prompts;
    config.hot_ratio = hot_ratio;
    config.activation = ActivationType::SILU;
    config.activation_threshold = 0.01f;
    config.n_gpu_layers = 0;  // CPU-only for profiling
    
    fprintf(stderr, "[HotCold Profiler] Starting full profiling pipeline...\n");
    
    ProfilingResult result = run_neuron_profiling(config);
    
    if (result.success) {
        fprintf(stderr, "[HotCold Profiler] Profiling complete.\n");
        return result.profile;
    } else {
        fprintf(stderr, "[HotCold Profiler] Profiling failed: %s\n", result.error_message.c_str());
        // Return empty profile
        HotNeuronProfile profile;
        profile.model_name = model_path;
        profile.hot_ratio = hot_ratio;
        return profile;
    }
}

// =============================================================================
// Profile serialization
// =============================================================================

bool save_hot_neuron_profile(const HotNeuronProfile& profile, const std::string& path) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "Error: Could not save hot neuron profile to %s\n", path.c_str());
        return false;
    }

    // Header
    const char magic[] = "HOTN";
    uint32_t version = 1;
    fwrite(magic, 4, 1, f);
    fwrite(&version, sizeof(version), 1, f);

    // Model info
    uint32_t name_len = static_cast<uint32_t>(profile.model_name.size());
    fwrite(&name_len, sizeof(name_len), 1, f);
    fwrite(profile.model_name.c_str(), name_len, 1, f);

    uint32_t activation = static_cast<uint32_t>(profile.activation);
    fwrite(&activation, sizeof(activation), 1, f);
    fwrite(&profile.num_layers, sizeof(profile.num_layers), 1, f);
    fwrite(&profile.ffn_dim, sizeof(profile.ffn_dim), 1, f);
    fwrite(&profile.hidden_dim, sizeof(profile.hidden_dim), 1, f);

    double hot_ratio = profile.hot_ratio;
    fwrite(&hot_ratio, sizeof(hot_ratio), 1, f);
    fwrite(&profile.num_prompts_used, sizeof(profile.num_prompts_used), 1, f);

    double avg_activation = profile.avg_activation_rate;
    fwrite(&avg_activation, sizeof(avg_activation), 1, f);

    // Per-layer hot sets
    for (uint32_t i = 0; i < profile.num_layers; i++) {
        const LayerHotSet& layer = profile.layers[i];
        fwrite(&layer.layer_index, sizeof(layer.layer_index), 1, f);
        fwrite(&layer.ffn_dim, sizeof(layer.ffn_dim), 1, f);
        fwrite(&layer.n_hot, sizeof(layer.n_hot), 1, f);
        fwrite(&layer.n_cold, sizeof(layer.n_cold), 1, f);

        if (layer.n_hot > 0) {
            fwrite(layer.hot_indices.data(), sizeof(uint32_t), layer.n_hot, f);
        }
        if (layer.n_cold > 0) {
            fwrite(layer.cold_indices.data(), sizeof(uint32_t), layer.n_cold, f);
        }
    }

    fclose(f);
    fprintf(stderr, "[HotCold Profiler] Saved profile to %s\n", path.c_str());
    return true;
}

HotNeuronProfile load_hot_neuron_profile(const std::string& path) {
    HotNeuronProfile profile;

    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "[HotCold Profiler] No existing profile at %s\n", path.c_str());
        return profile;
    }

    // Header
    char magic[4];
    if (fread(magic, 4, 1, f) != 1 || memcmp(magic, "HOTN", 4) != 0) {
        fprintf(stderr, "[HotCold Profiler] Invalid profile file: %s\n", path.c_str());
        fclose(f);
        return profile;
    }

    uint32_t version;
    fread(&version, sizeof(version), 1, f);
    if (version != 1) {
        fprintf(stderr, "[HotCold Profiler] Unsupported profile version: %u\n", version);
        fclose(f);
        return profile;
    }

    // Model info
    uint32_t name_len;
    fread(&name_len, sizeof(name_len), 1, f);
    profile.model_name.resize(name_len);
    fread(&profile.model_name[0], name_len, 1, f);

    uint32_t activation;
    fread(&activation, sizeof(activation), 1, f);
    profile.activation = static_cast<ActivationType>(activation);
    fread(&profile.num_layers, sizeof(profile.num_layers), 1, f);
    fread(&profile.ffn_dim, sizeof(profile.ffn_dim), 1, f);
    fread(&profile.hidden_dim, sizeof(profile.hidden_dim), 1, f);

    fread(&profile.hot_ratio, sizeof(profile.hot_ratio), 1, f);
    fread(&profile.num_prompts_used, sizeof(profile.num_prompts_used), 1, f);
    fread(&profile.avg_activation_rate, sizeof(profile.avg_activation_rate), 1, f);

    // Per-layer hot sets
    profile.layers.resize(profile.num_layers);
    for (uint32_t i = 0; i < profile.num_layers; i++) {
        LayerHotSet& layer = profile.layers[i];
        fread(&layer.layer_index, sizeof(layer.layer_index), 1, f);
        fread(&layer.ffn_dim, sizeof(layer.ffn_dim), 1, f);
        fread(&layer.n_hot, sizeof(layer.n_hot), 1, f);
        fread(&layer.n_cold, sizeof(layer.n_cold), 1, f);

        layer.hot_indices.resize(layer.n_hot);
        if (layer.n_hot > 0) {
            fread(layer.hot_indices.data(), sizeof(uint32_t), layer.n_hot, f);
        }

        layer.cold_indices.resize(layer.n_cold);
        if (layer.n_cold > 0) {
            fread(layer.cold_indices.data(), sizeof(uint32_t), layer.n_cold, f);
        }
    }

    fclose(f);
    fprintf(stderr, "[HotCold Profiler] Loaded profile from %s (%u layers, %.0f%% hot)\n",
            path.c_str(), profile.num_layers, profile.hot_ratio * 100.0);
    return profile;
}

std::string get_hot_neuron_path(const std::string& model_path) {
    // Replace .gguf extension with .hot_neurons.bin
    std::string path = model_path;
    size_t gguf_pos = path.rfind(".gguf");
    if (gguf_pos != std::string::npos) {
        path = path.substr(0, gguf_pos);
    }
    return path + ".hot_neurons.bin";
}
