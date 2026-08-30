#pragma once

#include <llama.h>
#include <string>
#include <vector>

// =============================================================================
// Step 14 — Configurable Sampler Chain
// =============================================================================
// Maps OpenAI-compatible sampling parameters to llama.cpp sampler chains.
// Used by the API server for /v1/chat/completions and /v1/completions.
// =============================================================================

struct SamplingConfig {
    float temperature = 1.0f;
    float top_p = 1.0f;
    int32_t top_k = -1;            // -1 = disabled
    float min_p = 0.0f;            // 0 = disabled
    int32_t seed = -1;             // -1 = random
    float repeat_penalty = 1.0f;
    int32_t frequency_penalty = 0; // OpenAI-style (mapped to llama penalties)
    int32_t presence_penalty = 0;  // OpenAI-style (mapped to llama penalties)
    std::vector<std::string> stop_sequences;
};

// Build a llama_sampler chain from OpenAI-compatible params.
// Caller must llama_sampler_free() the returned pointer.
// For temperature == 0, returns a greedy sampler.
llama_sampler* build_sampler_chain(const SamplingConfig& config);

// Build a greedy sampler (convenience)
llama_sampler* build_greedy_sampler();
