#pragma once

#include "hotcold_types.h"
#include "../types.h"

// =============================================================================
// Step 10 — Hot/Cold Predictor
// =============================================================================
// Predicts performance for hot/cold neuron offload strategies.
// Unlike dense predictions (single number), these output ranges because
// actual speed depends on which cold neurons activate per token.
// =============================================================================

// Calculate hot/cold placement strategy
// Returns viable strategy with memory estimates and speed ranges
HotColdStrategy compute_hotcold_strategy(
    const HardwareSpec& hw,
    const ModelSpec& model,
    const HotNeuronProfile& profile,
    uint32_t context_length,
    uint32_t kv_quant_bits,
    double hot_ratio = 0.15    // Default: 15% hot neurons
);

// Calculate decode speed range for hot/cold split
// Returns [worst_case, best_case, expected] tok/s
struct HotColdSpeedResult {
    double tok_s_best = 0.0;       // All hot neurons on GPU, few cold activated
    double tok_s_worst = 0.0;      // Many cold neurons activated, CPU bottleneck
    double tok_s_expected = 0.0;   // Average case (uniform activation probability)
    double gpu_time_ms = 0.0;      // Time for hot neuron computation
    double cpu_time_ms = 0.0;      // Time for cold neuron computation
    double pcie_transfer_ms = 0.0; // Time for CPU→GPU result transfer
};

HotColdSpeedResult predict_hotcold_speed(
    const HardwareSpec& hw,
    const ModelSpec& model,
    const HotColdStrategy& strategy,
    uint32_t kv_quant_bits
);

// Calculate memory usage for hot/cold strategy
struct HotColdMemoryResult {
    uint64_t vram_bytes = 0;       // Hot weights + CUDA overhead + KV cache
    uint64_t ram_bytes = 0;        // Cold weights
    bool viable = false;
    std::string reason;
};

HotColdMemoryResult predict_hotcold_memory(
    const HardwareSpec& hw,
    const ModelSpec& model,
    const HotColdStrategy& strategy,
    uint32_t context_length,
    uint32_t kv_quant_bits
);
