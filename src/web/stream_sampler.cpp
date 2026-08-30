#include "stream_sampler.h"
#include <cstdio>

// =============================================================================
// Sampler Chain Builder
// =============================================================================

llama_sampler* build_greedy_sampler() {
    llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    llama_sampler* smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    return smpl;
}

llama_sampler* build_sampler_chain(const SamplingConfig& config) {
    // Temperature 0 or negative → greedy
    if (config.temperature <= 0.0f) {
        return build_greedy_sampler();
    }

    llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    llama_sampler* smpl = llama_sampler_chain_init(sparams);

    // Penalties (repeat/frequency/presence) — applied first
    if (config.repeat_penalty != 1.0f || config.frequency_penalty != 0 || config.presence_penalty != 0) {
        llama_sampler_chain_add(smpl, llama_sampler_init_penalties(
            64,                     // last_n
            64,                     // penalty_last_n
            config.repeat_penalty,
            config.frequency_penalty,
            config.presence_penalty
        ));
    }

    // Top-K (if specified)
    if (config.top_k > 0) {
        llama_sampler_chain_add(smpl, llama_sampler_init_top_k(config.top_k));
    }

    // Top-P (nucleus sampling)
    if (config.top_p > 0.0f && config.top_p < 1.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_top_p(config.top_p, 1));
    }

    // Min-P
    if (config.min_p > 0.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_min_p(config.min_p, 1));
    }

    // Temperature
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(config.temperature));

    // Distribution sampling (with seed)
    uint32_t seed = (config.seed >= 0) ? static_cast<uint32_t>(config.seed) : 42;
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(seed));

    return smpl;
}
