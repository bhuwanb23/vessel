#include "hotcold/sparse_ffn.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <chrono>

// =============================================================================
// Activation functions
// =============================================================================

static inline float silu(float x) {
    return x / (1.0f + std::exp(-x));
}

static inline float gelu(float x) {
    return 0.5f * x * (1.0f + std::erf(x / std::sqrt(2.0f)));
}

static inline float apply_activation(float x, ActivationType act) {
    switch (act) {
        case ActivationType::RELU:  return std::max(0.0f, x);
        case ActivationType::SILU:  return silu(x);
        case ActivationType::GELU:  return gelu(x);
        default:                    return silu(x);
    }
}

static inline bool is_inactive(float x, ActivationType act, float threshold) {
    switch (act) {
        case ActivationType::RELU:  return x <= 0.0f;
        case ActivationType::SILU:  return x < threshold;
        case ActivationType::GELU:  return x < threshold;
        default:                    return x < threshold;
    }
}

// =============================================================================
// Cold neuron detection
// =============================================================================

std::vector<uint32_t> detect_activated_cold_neurons(
    const float* pre_activation,  // [n_cold] pre-activation values (before activation fn)
    uint32_t n_cold,
    ActivationType activation,
    float threshold)
{
    std::vector<uint32_t> activated;
    activated.reserve(n_cold / 4);  // Estimate ~25% activation rate

    for (uint32_t i = 0; i < n_cold; i++) {
        if (!is_inactive(pre_activation[i], activation, threshold)) {
            activated.push_back(i);
        }
    }

    return activated;
}

// =============================================================================
// Sparse FFN — GPU path (hot neurons)
// =============================================================================

void sparse_ffn_gpu_hot(
    const SparseFFNContext& ctx,
    const float* input,           // [hidden_dim]
    float* gpu_partial            // [hidden_dim] (pre-allocated, zeroed)
) {
    // Step 1: Compute up_proj_hot(x) -> [n_hot]
    // up_proj_hot is [hidden_dim, n_hot], input is [hidden_dim]
    // Result: pre_activation_hot[i] = sum_j(input[j] * up_proj_hot[j * n_hot + i])
    std::vector<float> up_result(ctx.n_hot, 0.0f);
    for (uint32_t h = 0; h < ctx.n_hot; h++) {
        float sum = 0.0f;
        for (uint32_t j = 0; j < ctx.hidden_dim; j++) {
            sum += input[j] * ctx.up_proj_hot[j * ctx.n_hot + h];
        }
        up_result[h] = sum;
    }

    // Step 2: Compute gate_proj_hot(x) -> [n_hot]
    std::vector<float> gate_result(ctx.n_hot, 0.0f);
    for (uint32_t h = 0; h < ctx.n_hot; h++) {
        float sum = 0.0f;
        for (uint32_t j = 0; j < ctx.hidden_dim; j++) {
            sum += input[j] * ctx.gate_proj_hot[j * ctx.n_hot + h];
        }
        gate_result[h] = sum;
    }

    // Step 3: Apply activation and gating
    // gated[h] = activation(gate_result[h]) * up_result[h]
    // Note: Standard Llama FFN: y = SiLU(gate) * up, not SiLU(up) * gate
    std::vector<float> gated(ctx.n_hot, 0.0f);
    for (uint32_t h = 0; h < ctx.n_hot; h++) {
        gated[h] = apply_activation(gate_result[h], ctx.activation) * up_result[h];
    }

    // Step 4: down_proj_hot(gated) -> [hidden_dim]
    // down_proj_hot is [n_hot, hidden_dim], gated is [n_hot]
    // Result: gpu_partial[j] = sum_h(gated[h] * down_proj_hot[h * hidden_dim + j])
    memset(gpu_partial, 0, sizeof(float) * ctx.hidden_dim);
    for (uint32_t j = 0; j < ctx.hidden_dim; j++) {
        float sum = 0.0f;
        for (uint32_t h = 0; h < ctx.n_hot; h++) {
            sum += gated[h] * ctx.down_proj_hot[h * ctx.hidden_dim + j];
        }
        gpu_partial[j] = sum;
    }
}

// =============================================================================
// Sparse FFN — CPU path (cold neurons, sparse)
// =============================================================================

void sparse_ffn_cpu_cold(
    const SparseFFNContext& ctx,
    const float* input,           // [hidden_dim]
    float* cpu_partial            // [hidden_dim] (pre-allocated, zeroed)
) {
    // Step 1: Compute gate pre-activation for all cold neurons
    // Standard FFN: y = SiLU(gate) * up, so sparsity is determined by gate pre-activation
    // gate_proj_cold is [hidden_dim, n_cold]
    std::vector<float> gate_pre(ctx.n_cold, 0.0f);
    std::vector<float> up_pre(ctx.n_cold, 0.0f);
    for (uint32_t c = 0; c < ctx.n_cold; c++) {
        float g_sum = 0.0f;
        float u_sum = 0.0f;
        for (uint32_t j = 0; j < ctx.hidden_dim; j++) {
            g_sum += input[j] * ctx.gate_proj_cold[j * ctx.n_cold + c];
            u_sum += input[j] * ctx.up_proj_cold[j * ctx.n_cold + c];
        }
        gate_pre[c] = g_sum;
        up_pre[c] = u_sum;
    }

    // Step 2: Detect which cold neurons actually activate
    // For ReLU: neurons with gate_pre_activation <= 0 are exactly zero
    // For SiLU/GELU: use threshold on gate pre-activation
    auto activated_cold = detect_activated_cold_neurons(
        gate_pre.data(), ctx.n_cold, ctx.activation, ctx.silu_threshold);

    if (activated_cold.empty()) {
        // No cold neurons activated — cpu_partial stays zero
        return;
    }

    // Step 3: Compute gated values for activated cold neurons only
    // Same convention: SiLU(gate) * up
    std::vector<float> gated_cold(activated_cold.size(), 0.0f);
    for (size_t i = 0; i < activated_cold.size(); i++) {
        uint32_t c = activated_cold[i];
        gated_cold[i] = apply_activation(gate_pre[c], ctx.activation) * up_pre[c];
    }

    // Step 5: down_proj for activated cold neurons
    // Only compute rows corresponding to activated cold neurons
    memset(cpu_partial, 0, sizeof(float) * ctx.hidden_dim);
    for (size_t i = 0; i < activated_cold.size(); i++) {
        uint32_t c = activated_cold[i];
        for (uint32_t j = 0; j < ctx.hidden_dim; j++) {
            cpu_partial[j] += gated_cold[i] * ctx.down_proj_cold[c * ctx.hidden_dim + j];
        }
    }
}

// =============================================================================
// Combine GPU and CPU partial results
// =============================================================================

void sparse_ffn_combine(
    const float* gpu_partial,     // [hidden_dim]
    const float* cpu_partial,     // [hidden_dim]
    float* output,                // [hidden_dim]
    uint32_t hidden_dim)
{
    for (uint32_t j = 0; j < hidden_dim; j++) {
        output[j] = gpu_partial[j] + cpu_partial[j];
    }
}

// =============================================================================
// Full sparse FFN forward pass
// =============================================================================

SparseFFNResult sparse_ffn_forward(
    const SparseFFNContext& ctx,
    const float* input,           // [hidden_dim]
    float* output)                // [hidden_dim] (pre-allocated)
{
    SparseFFNResult result;

    std::vector<float> gpu_partial(ctx.hidden_dim, 0.0f);
    std::vector<float> cpu_partial(ctx.hidden_dim, 0.0f);

    // GPU path: hot neurons
    auto t0 = std::chrono::high_resolution_clock::now();
    sparse_ffn_gpu_hot(ctx, input, gpu_partial.data());
    auto t1 = std::chrono::high_resolution_clock::now();
    result.gpu_compute_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    result.hot_neurons_activated = ctx.n_hot;  // All hot neurons always activate

    // CPU path: cold neurons (sparse)
    auto t2 = std::chrono::high_resolution_clock::now();
    sparse_ffn_cpu_cold(ctx, input, cpu_partial.data());
    auto t3 = std::chrono::high_resolution_clock::now();
    result.cpu_compute_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    // Count activated cold neurons (exact count from the detection step)
    // We recompute here to get the actual count for reporting
    {
        std::vector<float> gate_pre(ctx.n_cold, 0.0f);
        for (uint32_t c = 0; c < ctx.n_cold; c++) {
            float sum = 0.0f;
            for (uint32_t j = 0; j < ctx.hidden_dim; j++) {
                sum += input[j] * ctx.gate_proj_cold[j * ctx.n_cold + c];
            }
            gate_pre[c] = sum;
        }
        auto activated = detect_activated_cold_neurons(
            gate_pre.data(), ctx.n_cold, ctx.activation, ctx.silu_threshold);
        result.cold_neurons_activated = static_cast<uint32_t>(activated.size());
    }

    // Combine
    sparse_ffn_combine(gpu_partial.data(), cpu_partial.data(),
                       output, ctx.hidden_dim);

    result.output.assign(output, output + ctx.hidden_dim);

    return result;
}

// =============================================================================
// Layer-streaming fallback
// =============================================================================

SparseFFNResult layer_streaming_forward(
    const std::string& model_path,
    uint32_t layer_index,
    const float* input,           // [hidden_dim]
    float* output,                // [hidden_dim]
    uint64_t disk_read_speed_mbs)
{
    // NOTE: This is the skeleton for layer-streaming fallback.
    // Full implementation requires:
    // 1. Memory-map the GGUF file
    // 2. Locate the layer's weight tensors by offset
    // 3. Read the layer weights from disk (streaming)
    // 4. Compute the FFN using the streamed weights
    // 5. Discard the weights
    // 6. Repeat for attention layer
    // 7. Return the output
    //
    // The key constraint is that only ONE layer's weights are resident
    // in memory at any time, enabling models much larger than VRAM+RAM.

    SparseFFNResult result;

    fprintf(stderr, "[LayerStreamer] Streaming layer %u from %s (disk: %lu MB/s)\n",
            layer_index, model_path.c_str(), disk_read_speed_mbs);

    // Placeholder: In full implementation, this would read the layer
    // from disk and compute the FFN. For now, return zeros.
    memset(output, 0, sizeof(float) * 4096);  // Assume hidden_dim=4096

    result.output.assign(output, output + 4096);
    result.gpu_compute_ms = 0.0;
    result.cpu_compute_ms = 0.0;

    return result;
}
