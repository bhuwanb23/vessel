#include "hotcold/hotcold_types.h"
#include <cstdio>
#include <cstring>
#include <cmath>

// =============================================================================
// Layer-Streaming Fallback
// =============================================================================
// When a model is too large for VRAM + RAM (e.g., 70B on 8GB VRAM + 32GB RAM),
// stream one transformer layer at a time from disk:
//   1. Read layer N weights from NVMe SSD
//   2. Compute attention + FFN for layer N
//   3. Discard layer N weights
//   4. Read layer N+1
//   5. Repeat until all layers processed
//
// This makes impossible models technically runnable, at extreme speed cost.
// =============================================================================

// =============================================================================
// Timing model for layer streaming
// =============================================================================

struct LayerStreamTiming {
    double disk_read_ms = 0.0;      // Time to read one layer from NVMe
    double gpu_compute_ms = 0.0;    // Time to compute attention + FFN on GPU
    double total_per_layer_ms = 0.0; // disk_read + gpu_compute
    double total_model_ms = 0.0;    // total_per_layer * num_layers
    double estimated_tok_s = 0.0;   // 1000 / total_per_layer_ms (tokens per second)
};

// Calculate layer weight size in bytes
static uint64_t estimate_layer_weight_bytes(
    uint32_t hidden_dim,
    uint32_t ffn_dim,
    uint32_t attention_heads,
    uint32_t kv_heads,
    double bytes_per_param)
{
    // Attention: Q, K, V, O projections
    // Q: [hidden_dim, hidden_dim]
    // K: [hidden_dim, kv_dim] where kv_dim = kv_heads * head_dim
    // V: [hidden_dim, kv_dim]
    // O: [hidden_dim, hidden_dim]
    uint32_t kv_dim = kv_heads * (hidden_dim / attention_heads);
    uint64_t attention_params =
        static_cast<uint64_t>(hidden_dim) * hidden_dim +   // Q
        static_cast<uint64_t>(hidden_dim) * kv_dim +       // K
        static_cast<uint64_t>(hidden_dim) * kv_dim +       // V
        static_cast<uint64_t>(hidden_dim) * hidden_dim;    // O

    // FFN: up_proj, gate_proj, down_proj
    uint64_t ffn_params =
        3ULL * static_cast<uint64_t>(hidden_dim) * ffn_dim;

    // Norms: 2 * hidden_dim (negligible)
    uint64_t norm_params = 2 * hidden_dim;

    uint64_t total_params = attention_params + ffn_params + norm_params;
    return static_cast<uint64_t>(total_params * bytes_per_param);
}

// =============================================================================
// Layer streaming timing model
// =============================================================================

LayerStreamTiming calculate_layer_stream_timing(
    uint32_t num_layers,
    uint32_t hidden_dim,
    uint32_t ffn_dim,
    uint32_t attention_heads,
    uint32_t kv_heads,
    double bytes_per_param,
    double nvme_sequential_mbs,
    double gpu_tflops_fp16)
{
    LayerStreamTiming timing;

    // Layer weight size
    uint64_t layer_bytes = estimate_layer_weight_bytes(
        hidden_dim, ffn_dim, attention_heads, kv_heads, bytes_per_param);

    // Disk read time (NVMe sequential read)
    // nvme_sequential_mbs is in MB/s, layer_bytes is in bytes
    double layer_mb = static_cast<double>(layer_bytes) / (1024.0 * 1024.0);
    timing.disk_read_ms = (layer_mb / nvme_sequential_mbs) * 1000.0;

    // GPU compute time for one layer
    // Approximate: attention is O(hidden_dim^2) + O(hidden_dim * kv_dim)
    // FFN is O(3 * hidden_dim * ffn_dim)
    // Total FLOPs per token per layer ≈ 2 * (attention_params + ffn_params)
    uint32_t kv_dim = kv_heads * (hidden_dim / attention_heads);
    uint64_t attention_params =
        static_cast<uint64_t>(hidden_dim) * hidden_dim * 2 +   // Q, O
        static_cast<uint64_t>(hidden_dim) * kv_dim * 2;        // K, V
    uint64_t ffn_params = 3ULL * static_cast<uint64_t>(hidden_dim) * ffn_dim;

    // FLOPs per token = 2 * (attention_params + ffn_params)
    // (multiply by 2 for multiply-accumulate)
    double flops_per_token = 2.0 * static_cast<double>(attention_params + ffn_params);

    // GPU time = FLOPs / (TFLOPS * 1e12)
    timing.gpu_compute_ms = (flops_per_token / (gpu_tflops_fp16 * 1e12)) * 1000.0;

    // Total per layer
    timing.total_per_layer_ms = timing.disk_read_ms + timing.gpu_compute_ms;

    // Total for all layers
    timing.total_model_ms = timing.total_per_layer_ms * num_layers;

    // Tokens per second (1 token at a time, sequential)
    timing.estimated_tok_s = (timing.total_per_layer_ms > 0.0) ?
        1000.0 / timing.total_per_layer_ms : 0.0;

    return timing;
}

// =============================================================================
// Layer streaming placement strategy
// =============================================================================

LayerStreamingConfig compute_layer_streaming_config(
    uint32_t num_layers,
    uint32_t hidden_dim,
    uint32_t ffn_dim,
    uint32_t attention_heads,
    uint32_t kv_heads,
    double bytes_per_param,
    uint64_t vram_budget_bytes,
    double nvme_sequential_mbs,
    double gpu_tflops_fp16)
{
    LayerStreamingConfig config;
    config.enabled = true;
    config.disk_read_speed_mbs = static_cast<uint64_t>(nvme_sequential_mbs);

    // Calculate how many layers fit in VRAM (at minimum, we need 1 layer)
    uint64_t layer_bytes = estimate_layer_weight_bytes(
        hidden_dim, ffn_dim, attention_heads, kv_heads, bytes_per_param);

    // CUDA overhead: 512 MB
    uint64_t cuda_overhead = 512ULL * 1024 * 1024;
    uint64_t available_for_layers = (vram_budget_bytes > cuda_overhead) ?
        vram_budget_bytes - cuda_overhead : 0;

    uint32_t layers_in_vram = static_cast<uint32_t>(available_for_layers / layer_bytes);
    if (layers_in_vram > num_layers) layers_in_vram = num_layers;
    if (layers_in_vram < 1) layers_in_vram = 1;

    config.layers_to_stream = num_layers - layers_in_vram;

    // Calculate timing
    auto timing = calculate_layer_stream_timing(
        config.layers_to_stream, hidden_dim, ffn_dim,
        attention_heads, kv_heads, bytes_per_param,
        nvme_sequential_mbs, gpu_tflops_fp16);

    config.estimated_time_per_layer_ms = timing.total_per_layer_ms;

    fprintf(stderr, "[LayerStreamer] Config: %u layers in VRAM, %u layers streamed\n",
            layers_in_vram, config.layers_to_stream);
    fprintf(stderr, "[LayerStreamer] Estimated: %.1f ms/layer, ~%.2f tok/s\n",
            timing.total_per_layer_ms, timing.estimated_tok_s);

    return config;
}

// =============================================================================
// Check if layer streaming is needed
// =============================================================================

bool needs_layer_streaming(
    uint64_t model_weight_bytes,
    uint64_t vram_budget_bytes,
    uint64_t ram_budget_bytes)
{
    // Layer streaming is needed when the model doesn't fit in VRAM+RAM
    uint64_t total_budget = vram_budget_bytes + ram_budget_bytes;
    // Add 1 GB overhead for OS, CUDA, etc.
    uint64_t overhead = 1024ULL * 1024 * 1024;
    return model_weight_bytes > (total_budget > overhead ? total_budget - overhead : 0);
}
