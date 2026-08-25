#include "hotcold/hotcold_types.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>

// =============================================================================
// Layer-Streaming Fallback (AirLLM-Style)
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
// Minimum acceptable speed threshold
// =============================================================================
// If predicted speed is below this, warn user it's not practical
static const double MIN_USEFUL_TOK_S = 0.01;  // One token per 100 seconds

// =============================================================================
// Calculate layer weight size in bytes
// =============================================================================
static uint64_t estimate_layer_weight_bytes(
    uint32_t hidden_dim,
    uint32_t ffn_dim,
    uint32_t attention_heads,
    uint32_t kv_heads,
    double bytes_per_param)
{
    // Attention: Q, K, V, O projections
    uint32_t kv_dim = kv_heads * (hidden_dim / attention_heads);
    uint64_t attention_params =
        static_cast<uint64_t>(hidden_dim) * hidden_dim +   // Q
        static_cast<uint64_t>(hidden_dim) * kv_dim +       // K
        static_cast<uint64_t>(hidden_dim) * kv_dim +       // V
        static_cast<uint64_t>(hidden_dim) * hidden_dim;    // O

    // FFN: up_proj, gate_proj, down_proj (SwiGLU)
    uint64_t ffn_params =
        3ULL * static_cast<uint64_t>(hidden_dim) * ffn_dim;

    // Norms: 2 * hidden_dim (negligible)
    uint64_t norm_params = 2 * hidden_dim;

    uint64_t total_params = attention_params + ffn_params + norm_params;
    return static_cast<uint64_t>(total_params * bytes_per_param);
}

// =============================================================================
// Calculate resident memory footprint (always in memory)
// =============================================================================
struct ResidentMemory {
    uint64_t embedding_bytes;      // Token embedding table
    uint64_t output_head_bytes;    // LM head / output projection
    uint64_t kv_cache_bytes;       // KV cache (persists across layers)
    uint64_t compute_buffer_bytes; // Intermediate buffers for one layer
    uint64_t cuda_overhead_bytes;  // CUDA runtime overhead
    uint64_t total_bytes;
};

static ResidentMemory calculate_resident_memory(
    uint32_t vocab_size,
    uint32_t hidden_dim,
    uint32_t num_layers,
    uint32_t kv_heads,
    uint32_t head_dim,
    uint32_t context_length,
    uint32_t kv_quant_bits,
    double bytes_per_param)
{
    ResidentMemory mem;

    // Embedding table: vocab_size × hidden_dim
    mem.embedding_bytes = static_cast<uint64_t>(vocab_size) * hidden_dim *
                          static_cast<uint64_t>(bytes_per_param);

    // Output head: same size as embedding
    mem.output_head_bytes = mem.embedding_bytes;

    // KV cache: 2 (K + V) × layers × kv_heads × head_dim × context × bytes_per_element
    double bytes_per_kv_element;
    switch (kv_quant_bits) {
        case 4:  bytes_per_kv_element = 0.5; break;   // Q4_0
        case 8:  bytes_per_kv_element = 1.0; break;   // Q8_0
        default: bytes_per_kv_element = 2.0; break;    // FP16
    }
    mem.kv_cache_bytes = 2ULL * num_layers * kv_heads * head_dim *
                         context_length * static_cast<uint64_t>(bytes_per_kv_element);

    // Compute buffer: ~4 × hidden_dim × 4 bytes (activations, intermediate results)
    mem.compute_buffer_bytes = 4ULL * hidden_dim * 4;

    // CUDA overhead: 512 MB
    mem.cuda_overhead_bytes = 512ULL * 1024 * 1024;

    mem.total_bytes = mem.embedding_bytes + mem.output_head_bytes +
                      mem.kv_cache_bytes + mem.compute_buffer_bytes +
                      mem.cuda_overhead_bytes;

    return mem;
}

// =============================================================================
// Layer streaming timing model
// =============================================================================

struct LayerStreamTiming {
    double disk_read_ms = 0.0;      // Time to read one layer from NVMe
    double gpu_compute_ms = 0.0;    // Time to compute attention + FFN on GPU
    double total_per_layer_ms = 0.0; // disk_read + gpu_compute
    double total_per_token_ms = 0.0; // total_per_layer * num_layers (for streaming)
    double estimated_tok_s = 0.0;   // 1000 / total_per_token_ms
    bool is_worthwhile = false;     // Is this speed usable?
    std::string worthit_reason;     // Why/why not
};

LayerStreamTiming calculate_layer_stream_timing(
    uint32_t num_layers_to_stream,
    uint32_t hidden_dim,
    uint32_t ffn_dim,
    uint32_t attention_heads,
    uint32_t kv_heads,
    double bytes_per_param,
    double nvme_sequential_mbs,
    double gpu_tflops_fp16)
{
    LayerStreamTiming timing;

    if (num_layers_to_stream == 0 || hidden_dim == 0 || ffn_dim == 0) {
        timing.worthit_reason = "Invalid parameters";
        return timing;
    }

    // Layer weight size
    uint64_t layer_bytes = estimate_layer_weight_bytes(
        hidden_dim, ffn_dim, attention_heads, kv_heads, bytes_per_param);

    // Disk read time (NVMe sequential read)
    double layer_mb = static_cast<double>(layer_bytes) / (1024.0 * 1024.0);
    if (nvme_sequential_mbs > 0) {
        timing.disk_read_ms = (layer_mb / nvme_sequential_mbs) * 1000.0;
    } else {
        // Fallback: assume 1 GB/s if no NVMe data
        timing.disk_read_ms = (layer_mb / 1000.0) * 1000.0;
    }

    // GPU compute time for one layer
    // FLOPs per token per layer ≈ 2 * (attention_params + ffn_params)
    uint32_t kv_dim = kv_heads * (hidden_dim / attention_heads);
    uint64_t attention_params =
        static_cast<uint64_t>(hidden_dim) * hidden_dim * 2 +   // Q, O
        static_cast<uint64_t>(hidden_dim) * kv_dim * 2;        // K, V
    uint64_t ffn_params = 3ULL * static_cast<uint64_t>(hidden_dim) * ffn_dim;

    double flops_per_token = 2.0 * static_cast<double>(attention_params + ffn_params);

    if (gpu_tflops_fp16 > 0) {
        timing.gpu_compute_ms = (flops_per_token / (gpu_tflops_fp16 * 1e12)) * 1000.0;
    } else {
        timing.gpu_compute_ms = 0.01;  // Assume negligible if no GPU data
    }

    // Total per layer (disk read is the bottleneck)
    timing.total_per_layer_ms = timing.disk_read_ms + timing.gpu_compute_ms;

    // Total for all streamed layers (per token)
    timing.total_per_token_ms = timing.total_per_layer_ms * num_layers_to_stream;

    // Tokens per second
    timing.estimated_tok_s = (timing.total_per_token_ms > 0.0) ?
        1000.0 / timing.total_per_token_ms : 0.0;

    // Check if this is worthwhile
    if (timing.estimated_tok_s >= MIN_USEFUL_TOK_S) {
        timing.is_worthwhile = true;
        timing.worthit_reason = "Speed is usable for interactive chat";
    } else if (timing.estimated_tok_s >= 0.001) {
        timing.is_worthwhile = false;
        double seconds_per_token = 1.0 / timing.estimated_tok_s;
        char buf[256];
        snprintf(buf, sizeof(buf),
                 "Very slow: ~%.1f seconds per token. "
                 "Usable for batch jobs, not interactive.",
                 seconds_per_token);
        timing.worthit_reason = buf;
    } else {
        timing.is_worthwhile = false;
        double minutes_per_token = 1.0 / (timing.estimated_tok_s * 60.0);
        char buf[256];
        snprintf(buf, sizeof(buf),
                 "Extremely slow: ~%.1f minutes per token. "
                 "Not practically useful.",
                 minutes_per_token);
        timing.worthit_reason = buf;
    }

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
    fprintf(stderr, "[LayerStreamer] Estimated: %.1f ms/layer, ~%.4f tok/s\n",
            timing.total_per_layer_ms, timing.estimated_tok_s);
    if (!timing.is_worthwhile) {
        fprintf(stderr, "[LayerStreamer] WARNING: %s\n", timing.worthit_reason.c_str());
    }

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

// =============================================================================
// Get worth-it threshold
// =============================================================================

double get_layer_streaming_min_useful_speed() {
    return MIN_USEFUL_TOK_S;
}

// =============================================================================
// Format layer-streaming prediction for display
// =============================================================================

std::string format_layer_streaming_prediction(
    double tok_s,
    uint32_t layers_streamed,
    uint32_t total_layers,
    double nvme_mbs)
{
    char buf[256];

    if (tok_s <= 0) {
        snprintf(buf, sizeof(buf), "N/A");
    } else if (tok_s < 0.001) {
        double min_per_token = 1.0 / (tok_s * 60.0);
        snprintf(buf, sizeof(buf), "~%.1f min/tok", min_per_token);
    } else if (tok_s < MIN_USEFUL_TOK_S) {
        double sec_per_token = 1.0 / tok_s;
        snprintf(buf, sizeof(buf), "~%.1f s/tok", sec_per_token);
    } else {
        snprintf(buf, sizeof(buf), "~%.2f tok/s", tok_s);
    }

    return std::string(buf);
}
