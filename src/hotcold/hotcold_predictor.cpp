#include "hotcold/hotcold_predictor.h"
#include "hotcold/weight_splitter.h"
#include <cstdio>
#include <cmath>
#include <algorithm>

// =============================================================================
// KV cache size calculation (same as Step 3)
// =============================================================================

static uint64_t calculate_kv_cache_bytes(
    uint32_t layers,
    uint32_t kv_heads,
    uint32_t head_dim,
    uint32_t context_length,
    uint32_t kv_quant_bits)
{
    // KV cache = 2 (K + V) × layers × kv_heads × head_dim × context × bytes_per_element
    double bytes_per_element;
    switch (kv_quant_bits) {
        case 4:  bytes_per_element = 0.5; break;   // Q4_0
        case 8:  bytes_per_element = 1.0; break;   // Q8_0
        default: bytes_per_element = 2.0; break;    // FP16
    }

    return 2ULL * layers * kv_heads * head_dim * context_length *
           static_cast<uint64_t>(bytes_per_element);
}

// =============================================================================
// Hot/Cold strategy computation
// =============================================================================

HotColdStrategy compute_hotcold_strategy(
    const HardwareSpec& hw,
    const ModelSpec& model,
    const HotNeuronProfile& profile,
    uint32_t context_length,
    uint32_t kv_quant_bits,
    double hot_ratio)
{
    HotColdStrategy strategy;

    if (model.layers == 0 || model.embedding_dim == 0 || model.ffn_dim == 0) {
        strategy.reason = "Missing model dimensions";
        return strategy;
    }

    strategy.vram_budget_bytes = hw.vram_free_bytes;
    strategy.ram_budget_bytes = hw.ram_free_bytes;

    // Use profile's hot_ratio if available, otherwise use parameter
    double actual_hot_ratio = (profile.hot_ratio > 0.0) ? profile.hot_ratio : hot_ratio;
    strategy.hot_neurons_per_layer = static_cast<uint32_t>(model.ffn_dim * actual_hot_ratio);
    if (strategy.hot_neurons_per_layer < 1) strategy.hot_neurons_per_layer = 1;

    strategy.total_hot_neurons = strategy.hot_neurons_per_layer * model.layers;

    // Calculate memory requirements for hot/cold split
    // Each FFN layer has 3 weight matrices: up_proj, gate_proj, down_proj
    // up_proj: [hidden_dim, ffn_dim], gate_proj: [hidden_dim, ffn_dim], down_proj: [ffn_dim, hidden_dim]
    uint32_t hot = strategy.hot_neurons_per_layer;
    uint32_t cold = model.ffn_dim - hot;

    // Hot weights per layer: up_hot + gate_hot + down_hot
    uint64_t hot_params_per_layer =
        static_cast<uint64_t>(model.embedding_dim) * hot +   // up_proj hot columns
        static_cast<uint64_t>(model.embedding_dim) * hot +   // gate_proj hot columns
        static_cast<uint64_t>(hot) * model.embedding_dim;    // down_proj hot rows

    uint64_t cold_params_per_layer =
        static_cast<uint64_t>(model.embedding_dim) * cold +  // up_proj cold columns
        static_cast<uint64_t>(model.embedding_dim) * cold +  // gate_proj cold columns
        static_cast<uint64_t>(cold) * model.embedding_dim;   // down_proj cold rows

    double bytes_per_param = model.bits_per_weight / 8.0;

    strategy.hot_weights_vram_bytes = hot_params_per_layer * model.layers *
                                      static_cast<uint64_t>(bytes_per_param);
    strategy.cold_weights_ram_bytes = cold_params_per_layer * model.layers *
                                      static_cast<uint64_t>(bytes_per_param);

    // CUDA overhead: 512 MB
    uint64_t cuda_overhead = 512ULL * 1024 * 1024;

    // KV cache (assume full GPU for simplicity — can be split later)
    uint32_t kv_heads = (model.kv_heads > 0) ? model.kv_heads : model.attention_heads;
    uint32_t head_dim = (model.head_dim > 0) ? model.head_dim :
                        (model.attention_heads > 0 ? model.embedding_dim / model.attention_heads : 128);
    uint64_t kv_cache = calculate_kv_cache_bytes(
        model.layers, kv_heads, head_dim, context_length, kv_quant_bits);

    // Total VRAM: hot weights + CUDA overhead + KV cache
    uint64_t total_vram = strategy.hot_weights_vram_bytes + cuda_overhead + kv_cache;

    // Check viability
    if (total_vram > hw.vram_free_bytes) {
        strategy.viable = false;
        double needed_gb = total_vram / (1024.0 * 1024.0 * 1024.0);
        double free_gb = hw.vram_free_bytes / (1024.0 * 1024.0 * 1024.0);
        char buf[256];
        snprintf(buf, sizeof(buf),
                 "Hot weights + KV cache (%.1f GB) exceed VRAM (%.1f GB free)",
                 needed_gb, free_gb);
        strategy.reason = buf;
        return strategy;
    }

    // Check if cold weights fit in RAM
    if (strategy.cold_weights_ram_bytes > hw.ram_free_bytes) {
        strategy.viable = false;
        double needed_gb = strategy.cold_weights_ram_bytes / (1024.0 * 1024.0 * 1024.0);
        double free_gb = hw.ram_free_bytes / (1024.0 * 1024.0 * 1024.0);
        char buf[256];
        snprintf(buf, sizeof(buf),
                 "Cold weights (%.1f GB) exceed RAM (%.1f GB free)",
                 needed_gb, free_gb);
        strategy.reason = buf;
        return strategy;
    }

    strategy.viable = true;

    // Calculate speed estimates
    auto speed = predict_hotcold_speed(hw, model, strategy, kv_quant_bits);
    strategy.estimated_tok_s_best = speed.tok_s_best;
    strategy.estimated_tok_s_worst = speed.tok_s_worst;
    strategy.estimated_tok_s_expected = speed.tok_s_expected;

    return strategy;
}

// =============================================================================
// Speed prediction for hot/cold split
// =============================================================================

HotColdSpeedResult predict_hotcold_speed(
    const HardwareSpec& hw,
    const ModelSpec& model,
    const HotColdStrategy& strategy,
    uint32_t kv_quant_bits)
{
    HotColdSpeedResult result;

    if (!strategy.viable || model.layers == 0) return result;

    double bytes_per_param = model.bits_per_weight / 8.0;

    // Active bytes per token for hot/cold split
    // Hot part: shared params are always on GPU, but for dense models
    // "shared" doesn't apply — instead, hot neurons are the ones on GPU
    uint32_t hot = strategy.hot_neurons_per_layer;
    uint32_t cold = model.ffn_dim - hot;

    // Best case: All hot neurons on GPU, very few cold neurons activate
    // Assume ~10% of cold neurons activate (typical for power-law distribution)
    uint32_t cold_activated_best = cold / 10;
    if (cold_activated_best < 1) cold_activated_best = 1;

    // GPU bytes = hot weights + attention weights (always on GPU)
    // For hot/cold split, attention layers stay on GPU
    uint32_t kv_heads = (model.kv_heads > 0) ? model.kv_heads : model.attention_heads;
    uint32_t head_dim = (model.head_dim > 0) ? model.head_dim :
                        (model.attention_heads > 0 ? model.embedding_dim / model.attention_heads : 128);

    uint64_t attention_params_per_layer =
        2ULL * model.embedding_dim * model.embedding_dim +  // Q, O
        2ULL * model.embedding_dim * kv_heads * head_dim;   // K, V

    double gpu_bytes_best =
        (strategy.hot_weights_vram_bytes / model.layers +  // Hot FFN weights per layer
         attention_params_per_layer * bytes_per_param) *    // Attention weights per layer
        model.layers;                                        // All layers

    // CPU bytes = activated cold neurons
    uint64_t cold_activated_params =
        static_cast<uint64_t>(cold_activated_best) * model.embedding_dim * 3;  // up + gate + down
    double cpu_bytes_best = cold_activated_params * bytes_per_param * model.layers;

    // Best case speed: GPU-bound (hot neurons dominate)
    // tok/s = 1 / (gpu_bytes / (BW_vram * efficiency))
    double gpu_efficiency = 0.27;  // Conservative efficiency factor
    double gpu_time_best = gpu_bytes_best / (hw.gpu_bandwidth_gbs * 1e9 * gpu_efficiency);
    double cpu_time_best = cpu_bytes_best / (hw.ram_bandwidth_gbs * 1e9);
    // PCIe transfer for cold results: hidden_dim * 4 bytes * 2 (complex number)
    double pcie_time_best = (model.embedding_dim * 4.0 * 2) / (12.0e9);  // PCIe 4.0 x16 ≈ 12 GB/s

    double total_time_best = gpu_time_best + cpu_time_best + pcie_time_best;
    result.tok_s_best = (total_time_best > 0) ? 1.0 / total_time_best : 0.0;
    result.gpu_time_ms = gpu_time_best * 1000.0;
    result.cpu_time_ms = cpu_time_best * 1000.0;
    result.pcie_transfer_ms = pcie_time_best * 1000.0;

    // Worst case: Many cold neurons activate (50% of cold)
    uint32_t cold_activated_worst = cold / 2;
    uint64_t cold_activated_params_worst =
        static_cast<uint64_t>(cold_activated_worst) * model.embedding_dim * 3;
    double cpu_bytes_worst = cold_activated_params_worst * bytes_per_param * model.layers;

    double cpu_time_worst = cpu_bytes_worst / (hw.ram_bandwidth_gbs * 1e9);
    double pcie_time_worst = (model.embedding_dim * 4.0 * 2) / (12.0e9);
    double total_time_worst = gpu_time_best + cpu_time_worst + pcie_time_worst;
    result.tok_s_worst = (total_time_worst > 0) ? 1.0 / total_time_worst : 0.0;

    // Expected case: ~25% of cold neurons activate (average across prompts)
    uint32_t cold_activated_expected = cold / 4;
    uint64_t cold_activated_params_expected =
        static_cast<uint64_t>(cold_activated_expected) * model.embedding_dim * 3;
    double cpu_bytes_expected = cold_activated_params_expected * bytes_per_param * model.layers;

    double cpu_time_expected = cpu_bytes_expected / (hw.ram_bandwidth_gbs * 1e9);
    double pcie_time_expected = (model.embedding_dim * 4.0 * 2) / (12.0e9);
    double total_time_expected = gpu_time_best + cpu_time_expected + pcie_time_expected;
    result.tok_s_expected = (total_time_expected > 0) ? 1.0 / total_time_expected : 0.0;

    return result;
}

// =============================================================================
// Memory prediction for hot/cold split
// =============================================================================

HotColdMemoryResult predict_hotcold_memory(
    const HardwareSpec& hw,
    const ModelSpec& model,
    const HotColdStrategy& strategy,
    uint32_t context_length,
    uint32_t kv_quant_bits)
{
    HotColdMemoryResult result;

    if (!strategy.viable) {
        result.reason = strategy.reason;
        return result;
    }

    // CUDA overhead
    uint64_t cuda_overhead = 512ULL * 1024 * 1024;

    // KV cache
    uint32_t kv_heads = (model.kv_heads > 0) ? model.kv_heads : model.attention_heads;
    uint32_t head_dim = (model.head_dim > 0) ? model.head_dim :
                        (model.attention_heads > 0 ? model.embedding_dim / model.attention_heads : 128);
    uint64_t kv_cache = calculate_kv_cache_bytes(
        model.layers, kv_heads, head_dim, context_length, kv_quant_bits);

    result.vram_bytes = strategy.hot_weights_vram_bytes + cuda_overhead + kv_cache;
    result.ram_bytes = strategy.cold_weights_ram_bytes;

    result.viable =
        (result.vram_bytes <= hw.vram_free_bytes) &&
        (result.ram_bytes <= hw.ram_free_bytes);

    if (!result.viable) {
        if (result.vram_bytes > hw.vram_free_bytes) {
            result.reason = "VRAM exceeded";
        } else {
            result.reason = "RAM exceeded";
        }
    }

    return result;
}
