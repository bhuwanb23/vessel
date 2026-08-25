// =============================================================================
// Sparse FFN Validation Test (Step 10 Phase D)
// =============================================================================
// Validates that the sparse hot/cold FFN produces the same output as the
// standard dense FFN, and measures the speedup from sparsity optimization.
//
// Test approach:
// 1. Load model via llama.cpp
// 2. Extract FFN weights for a single layer
// 3. Split weights using the hot neuron mask
// 4. Run dense FFN (standard computation)
// 5. Run sparse FFN (hot/cold split with activation pruning)
// 6. Compare outputs (must match within floating-point tolerance)
// 7. Measure and report speedup
// =============================================================================

#include "hotcold/sparse_ffn.h"
#include "hotcold/mask_file.h"
#include "hotcold/hotcold_types.h"
#include "types.h"

#include <llama.h>
#include <ggml.h>

#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <vector>
#include <random>

// =============================================================================
// Weight extraction from ggml tensors
// =============================================================================

// Extract float weights from a ggml tensor
// Returns true on success
static bool extract_tensor_weights(
    const struct ggml_tensor* tensor,
    std::vector<float>& out_weights)
{
    if (!tensor) return false;
    
    // Check if tensor is contiguous and f32
    // For quantized tensors, we need to dequantize
    if (tensor->type == GGML_TYPE_F32) {
        size_t n = ggml_nelements(tensor);
        out_weights.resize(n);
        memcpy(out_weights.data(), tensor->data, n * sizeof(float));
        return true;
    }
    
    // For quantized types, use ggml's dequantization
    // Create a f32 copy
    size_t n = ggml_nelements(tensor);
    out_weights.resize(n);
    
    // Use ggml_backend_tensor_get to read data
    // For now, handle common quantized types
    if (tensor->type == GGML_TYPE_F16) {
        const ggml_fp16_t* data = (const ggml_fp16_t*)tensor->data;
        for (size_t i = 0; i < n; i++) {
            out_weights[i] = ggml_fp16_to_fp32(data[i]);
        }
        return true;
    }
    
    // For other quantized types (Q4_K, Q8_0, etc.), we need ggml's dequantize
    // This is complex — for the test, we'll use F16 or F32 models
    fprintf(stderr, "[SparseFFN Test] Warning: Cannot extract weights from type %d\n", tensor->type);
    return false;
}

// =============================================================================
// Dense FFN computation (baseline)
// =============================================================================

static void dense_ffn_forward(
    const float* input,            // [hidden_dim]
    const float* up_proj,          // [hidden_dim, ffn_dim] — columns are neurons
    const float* gate_proj,        // [hidden_dim, ffn_dim] — columns are neurons
    const float* down_proj,        // [ffn_dim, hidden_dim] — rows are neurons
    float* output,                 // [hidden_dim]
    uint32_t hidden_dim,
    uint32_t ffn_dim,
    ActivationType activation)
{
    // g = x @ W_gate -> [ffn_dim]  (gate_proj is [hidden_dim, ffn_dim])
    // u = x @ W_up -> [ffn_dim]    (up_proj is [hidden_dim, ffn_dim])
    // a = SiLU(g) * u -> [ffn_dim]
    // y = a @ W_down -> [hidden_dim]  (down_proj is [ffn_dim, hidden_dim])
    
    std::vector<float> gate(ffn_dim, 0.0f);
    std::vector<float> up(ffn_dim, 0.0f);
    
    // Gate and up projections: x @ W -> [ffn_dim]
    for (uint32_t f = 0; f < ffn_dim; f++) {
        float g_sum = 0.0f;
        float u_sum = 0.0f;
        for (uint32_t h = 0; h < hidden_dim; h++) {
            g_sum += input[h] * gate_proj[h * ffn_dim + f];
            u_sum += input[h] * up_proj[h * ffn_dim + f];
        }
        gate[f] = g_sum;
        up[f] = u_sum;
    }
    
    // Activation and gating
    std::vector<float> activated(ffn_dim, 0.0f);
    for (uint32_t f = 0; f < ffn_dim; f++) {
        float a;
        switch (activation) {
            case ActivationType::RELU:  a = std::max(0.0f, gate[f]); break;
            case ActivationType::SILU:  a = gate[f] / (1.0f + std::exp(-gate[f])); break;
            case ActivationType::GELU:  a = 0.5f * gate[f] * (1.0f + std::erf(gate[f] / std::sqrt(2.0f))); break;
            default:                    a = gate[f] / (1.0f + std::exp(-gate[f])); break;
        }
        activated[f] = a * up[f];
    }
    
    // Down projection: a @ W_down -> [hidden_dim]
    memset(output, 0, sizeof(float) * hidden_dim);
    for (uint32_t h = 0; h < hidden_dim; h++) {
        float sum = 0.0f;
        for (uint32_t f = 0; f < ffn_dim; f++) {
            sum += activated[f] * down_proj[f * hidden_dim + h];
        }
        output[h] = sum;
    }
}

// =============================================================================
// Compute L2 error between two vectors
// =============================================================================

static float compute_l2_error(const float* a, const float* b, uint32_t n) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < n; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum / n);
}

// =============================================================================
// Main test
// =============================================================================

int main(int argc, char* argv[]) {
    printf("=== Sparse FFN Validation Test (Step 10 Phase D) ===\n\n");
    
    // Parse arguments
    std::string model_path = "models/Llama-3.2-3B-Instruct-Q4_K_M.gguf";
    std::string mask_path = "";
    int layer_to_test = 0;
    int num_iterations = 100;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) model_path = argv[++i];
        else if (arg == "--mask" && i + 1 < argc) mask_path = argv[++i];
        else if (arg == "--layer" && i + 1 < argc) layer_to_test = atoi(argv[++i]);
        else if (arg == "--iterations" && i + 1 < argc) num_iterations = atoi(argv[++i]);
    }
    
    // Auto-detect mask path
    if (mask_path.empty()) {
        mask_path = get_mask_file_path(model_path);
    }
    
    printf("Model: %s\n", model_path.c_str());
    printf("Mask:  %s\n", mask_path.c_str());
    printf("Layer: %d\n", layer_to_test);
    printf("Iterations: %d\n\n", num_iterations);
    
    // =========================================================================
    // Step 1: Load hot neuron mask
    // =========================================================================
    printf("Step 1: Loading hot neuron mask...\n");
    
    HotNeuronProfile profile = load_mask_file(mask_path);
    if (profile.num_layers == 0) {
        fprintf(stderr, "Error: Could not load mask file: %s\n", mask_path.c_str());
        fprintf(stderr, "  Run: llm-planner --model <path> --profile-neurons\n");
        return 1;
    }
    
    printf("  Layers: %u, FFN dim: %u, Hidden dim: %u\n",
           profile.num_layers, profile.ffn_dim, profile.hidden_dim);
    printf("  Hot ratio: %.0f%%\n", profile.hot_ratio * 100.0);
    
    if (layer_to_test >= static_cast<int>(profile.num_layers)) {
        fprintf(stderr, "Error: Layer %d >= num_layers %u\n", layer_to_test, profile.num_layers);
        return 1;
    }
    
    const LayerHotSet& hot_set = profile.layers[layer_to_test];
    printf("  Layer %d: %u hot / %u cold\n",
           layer_to_test, hot_set.n_hot, hot_set.n_cold);
    
    // =========================================================================
    // Step 2: Load model and extract weights
    // =========================================================================
    printf("\nStep 2: Loading model and extracting weights...\n");
    
    llama_backend_init();
    
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;  // CPU only for weight extraction
    model_params.vocab_only = false;
    
    struct llama_model* model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model) {
        fprintf(stderr, "Error: Could not load model: %s\n", model_path.c_str());
        llama_backend_free();
        return 1;
    }
    
    int32_t n_embd = llama_model_n_embd(model);
    int32_t n_layer = llama_model_n_layer(model);
    
    // Estimate FFN dim (not directly available via API)
    uint32_t ffn_dim = profile.ffn_dim;
    if (ffn_dim == 0) ffn_dim = n_embd * 4;  // Fallback
    
    printf("  Model: %d layers, %d embd, %u ffn\n", n_layer, n_embd, ffn_dim);
    printf("  Extracting weights for layer %d...\n", layer_to_test);
    
    // NOTE: Extracting individual layer weights from ggml is complex.
    // For this test, we create synthetic weights that match the model's
    // dimensions. The test validates the sparse FFN LOGIC, not the actual
    // model weights. In production, the weights come from ggml's tensors.
    
    // Create synthetic weights for testing
    uint32_t hidden_dim = static_cast<uint32_t>(n_embd);
    uint32_t test_ffn_dim = ffn_dim;
    
    printf("  Using synthetic weights for validation (hidden=%u, ffn=%u)\n",
           hidden_dim, test_ffn_dim);
    
    // Generate random weights
    // Layout convention (matching sparse_ffn context):
    //   gate_proj: [hidden_dim, ffn_dim] — columns are neurons
    //   up_proj:   [hidden_dim, ffn_dim] — columns are neurons
    //   down_proj: [ffn_dim, hidden_dim] — rows are neurons
    std::mt19937 rng(42);  // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    
    std::vector<float> up_proj(hidden_dim * test_ffn_dim);    // [hidden_dim, ffn_dim]
    std::vector<float> gate_proj(hidden_dim * test_ffn_dim);  // [hidden_dim, ffn_dim]
    std::vector<float> down_proj(test_ffn_dim * hidden_dim);  // [ffn_dim, hidden_dim]
    
    for (auto& w : up_proj) w = dist(rng);
    for (auto& w : gate_proj) w = dist(rng);
    for (auto& w : down_proj) w = dist(rng);
    
    // Generate random input
    std::vector<float> input(hidden_dim);
    for (auto& v : input) v = dist(rng);
    
    printf("  Weights generated: up=%zu, gate=%zu, down=%zu floats\n",
           up_proj.size(), gate_proj.size(), down_proj.size());
    
    // =========================================================================
    // Step 3: Split weights using hot/cold mask
    // =========================================================================
    printf("\nStep 3: Splitting weights (hot=%u, cold=%u)...\n",
           hot_set.n_hot, hot_set.n_cold);
    
    // Split up_proj: [hidden_dim, ffn_dim] → hot cols + cold cols
    // up_proj[h * ffn_dim + f] → up_hot[h * n_hot + i]
    std::vector<float> up_hot(hidden_dim * hot_set.n_hot);
    std::vector<float> up_cold(hidden_dim * hot_set.n_cold);
    
    for (uint32_t h = 0; h < hidden_dim; h++) {
        for (uint32_t i = 0; i < hot_set.n_hot; i++) {
            uint32_t col = hot_set.hot_indices[i];
            up_hot[h * hot_set.n_hot + i] = up_proj[h * test_ffn_dim + col];
        }
        for (uint32_t i = 0; i < hot_set.n_cold; i++) {
            uint32_t col = hot_set.cold_indices[i];
            up_cold[h * hot_set.n_cold + i] = up_proj[h * test_ffn_dim + col];
        }
    }
    
    // Split gate_proj: [hidden_dim, ffn_dim] → hot cols + cold cols
    // gate_proj[h * ffn_dim + f] → gate_hot[h * n_hot + i]
    std::vector<float> gate_hot(hidden_dim * hot_set.n_hot);
    std::vector<float> gate_cold(hidden_dim * hot_set.n_cold);
    
    for (uint32_t h = 0; h < hidden_dim; h++) {
        for (uint32_t i = 0; i < hot_set.n_hot; i++) {
            uint32_t col = hot_set.hot_indices[i];
            gate_hot[h * hot_set.n_hot + i] = gate_proj[h * test_ffn_dim + col];
        }
        for (uint32_t i = 0; i < hot_set.n_cold; i++) {
            uint32_t col = hot_set.cold_indices[i];
            gate_cold[h * hot_set.n_cold + i] = gate_proj[h * test_ffn_dim + col];
        }
    }
    
    // Split down_proj: [ffn_dim, hidden_dim] → hot rows + cold rows
    // down_proj[f * hidden_dim + h] → down_hot[i * hidden_dim + h]
    std::vector<float> down_hot(hot_set.n_hot * hidden_dim);
    std::vector<float> down_cold(hot_set.n_cold * hidden_dim);
    
    for (uint32_t i = 0; i < hot_set.n_hot; i++) {
        uint32_t row = hot_set.hot_indices[i];
        memcpy(&down_hot[i * hidden_dim], &down_proj[row * hidden_dim],
               sizeof(float) * hidden_dim);
    }
    for (uint32_t i = 0; i < hot_set.n_cold; i++) {
        uint32_t row = hot_set.cold_indices[i];
        memcpy(&down_cold[i * hidden_dim], &down_proj[row * hidden_dim],
               sizeof(float) * hidden_dim);
    }
    
    printf("  Split complete: up_hot=%zu, up_cold=%zu\n", up_hot.size(), up_cold.size());
    
    // =========================================================================
    // Step 4: Run dense FFN (baseline)
    // =========================================================================
    printf("\nStep 4: Running dense FFN (baseline)...\n");
    
    std::vector<float> dense_output(hidden_dim);
    
    auto t_dense_start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < num_iterations; iter++) {
        dense_ffn_forward(
            input.data(), up_proj.data(), gate_proj.data(), down_proj.data(),
            dense_output.data(), hidden_dim, test_ffn_dim, ActivationType::SILU);
    }
    auto t_dense_end = std::chrono::high_resolution_clock::now();
    double dense_ms = std::chrono::duration<double, std::milli>(t_dense_end - t_dense_start).count();
    
    printf("  Dense FFN: %.2f ms total, %.4f ms/iter\n",
           dense_ms, dense_ms / num_iterations);
    
    // =========================================================================
    // Step 5a: Run sparse FFN WITHOUT pruning (threshold = -inf) for correctness
    // =========================================================================
    printf("\nStep 5a: Running sparse FFN (no pruning, for correctness)...\n");
    
    SparseFFNContext ctx_correctness;
    ctx_correctness.up_proj_hot = up_hot.data();
    ctx_correctness.up_proj_cold = up_cold.data();
    ctx_correctness.gate_proj_hot = gate_hot.data();
    ctx_correctness.gate_proj_cold = gate_cold.data();
    ctx_correctness.down_proj_hot = down_hot.data();
    ctx_correctness.down_proj_cold = down_cold.data();
    ctx_correctness.hidden_dim = hidden_dim;
    ctx_correctness.ffn_dim = test_ffn_dim;
    ctx_correctness.n_hot = hot_set.n_hot;
    ctx_correctness.n_cold = hot_set.n_cold;
    ctx_correctness.activation = ActivationType::SILU;
    ctx_correctness.silu_threshold = -1e30f;  // No pruning — all cold neurons computed
    
    std::vector<float> sparse_output(hidden_dim);
    SparseFFNResult last_result;
    
    auto t_sparse_start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < num_iterations; iter++) {
        last_result = sparse_ffn_forward(ctx_correctness, input.data(), sparse_output.data());
    }
    auto t_sparse_end = std::chrono::high_resolution_clock::now();
    double sparse_ms = std::chrono::duration<double, std::milli>(t_sparse_end - t_sparse_start).count();
    
    printf("  Sparse FFN (no prune): %.2f ms total, %.4f ms/iter\n",
           sparse_ms, sparse_ms / num_iterations);
    printf("  Last run: GPU=%.4f ms, CPU=%.4f ms\n",
           last_result.gpu_compute_ms, last_result.cpu_compute_ms);
    
    // =========================================================================
    // Step 5b: Run sparse FFN WITH pruning for speedup measurement
    // =========================================================================
    printf("\nStep 5b: Running sparse FFN (with pruning, for speedup)...\n");
    
    SparseFFNContext ctx_sparse;
    ctx_sparse.up_proj_hot = up_hot.data();
    ctx_sparse.up_proj_cold = up_cold.data();
    ctx_sparse.gate_proj_hot = gate_hot.data();
    ctx_sparse.gate_proj_cold = gate_cold.data();
    ctx_sparse.down_proj_hot = down_hot.data();
    ctx_sparse.down_proj_cold = down_cold.data();
    ctx_sparse.hidden_dim = hidden_dim;
    ctx_sparse.ffn_dim = test_ffn_dim;
    ctx_sparse.n_hot = hot_set.n_hot;
    ctx_sparse.n_cold = hot_set.n_cold;
    ctx_sparse.activation = ActivationType::SILU;
    ctx_sparse.silu_threshold = 0.01f;  // Prune neurons where SiLU(gate) < 0.01
    
    std::vector<float> sparse_pruned_output(hidden_dim);
    SparseFFNResult last_pruned_result;
    
    auto t_sparse_pruned_start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < num_iterations; iter++) {
        last_pruned_result = sparse_ffn_forward(ctx_sparse, input.data(), sparse_pruned_output.data());
    }
    auto t_sparse_pruned_end = std::chrono::high_resolution_clock::now();
    double sparse_pruned_ms = std::chrono::duration<double, std::milli>(t_sparse_pruned_end - t_sparse_pruned_start).count();
    
    printf("  Sparse FFN (pruned):  %.2f ms total, %.4f ms/iter\n",
           sparse_pruned_ms, sparse_pruned_ms / num_iterations);
    printf("  Last run: GPU=%.4f ms, CPU=%.4f ms\n",
           last_pruned_result.gpu_compute_ms, last_pruned_result.cpu_compute_ms);
    
    // =========================================================================
    // Step 6: Compare outputs (using no-pruning result for correctness)
    // =========================================================================
    printf("\nStep 6: Comparing outputs (no-pruning sparse vs dense)...\n");
    
    float l2_error = compute_l2_error(dense_output.data(), sparse_output.data(), hidden_dim);
    float max_error = 0.0f;
    for (uint32_t i = 0; i < hidden_dim; i++) {
        float err = std::abs(dense_output[i] - sparse_output[i]);
        if (err > max_error) max_error = err;
    }
    
    // Compute reference magnitude for relative error
    float ref_magnitude = 0.0f;
    for (uint32_t i = 0; i < hidden_dim; i++) {
        ref_magnitude += dense_output[i] * dense_output[i];
    }
    ref_magnitude = std::sqrt(ref_magnitude / hidden_dim);
    
    printf("  L2 error (RMS):     %.6f\n", l2_error);
    printf("  Max absolute error: %.6f\n", max_error);
    printf("  Reference magnitude: %.6f\n", ref_magnitude);
    printf("  Relative error:     %.4f%%\n",
           (ref_magnitude > 0) ? (l2_error / ref_magnitude * 100.0f) : 0.0f);
    
    // =========================================================================
    // Step 7: Report results
    // =========================================================================
    printf("\n=== Results ===\n");
    
    float speedup_no_prune = (sparse_ms > 0) ? dense_ms / sparse_ms : 0.0f;
    float speedup_pruned = (sparse_pruned_ms > 0) ? dense_ms / sparse_pruned_ms : 0.0f;
    float sparsity_achieved = (hot_set.n_cold > 0) ?
        (1.0f - static_cast<float>(last_pruned_result.cold_neurons_activated) / hot_set.n_cold) * 100.0f : 0.0f;
    
    printf("Speedup (no prune): %.2fx\n", speedup_no_prune);
    printf("Speedup (pruned):   %.2fx\n", speedup_pruned);
    printf("Sparsity:         %.1f%% of cold neurons skipped\n", sparsity_achieved);
    printf("Hot neurons:      %u / %u (%.0f%%)\n",
           hot_set.n_hot, test_ffn_dim, 100.0f * hot_set.n_hot / test_ffn_dim);
    printf("Cold neurons:     %u / %u (%.0f%%)\n",
           hot_set.n_cold, test_ffn_dim, 100.0f * hot_set.n_cold / test_ffn_dim);
    
    // Validation criteria
    bool pass_l2 = (l2_error < 0.01f);  // L2 error < 0.01
    bool pass_max = (max_error < 0.1f);  // Max error < 0.1
    bool pass_relative = (ref_magnitude > 0) ? (l2_error / ref_magnitude < 0.05f) : true;
    
    printf("\nValidation:\n");
    printf("  L2 error < 0.01:        %s (%.6f)\n", pass_l2 ? "PASS" : "FAIL", l2_error);
    printf("  Max error < 0.1:        %s (%.6f)\n", pass_max ? "PASS" : "FAIL", max_error);
    printf("  Relative error < 5%%:    %s\n", pass_relative ? "PASS" : "FAIL");
    
    if (pass_l2 && pass_max && pass_relative) {
        printf("\n✅ VALIDATION PASSED: Sparse FFN matches dense FFN\n");
    } else {
        printf("\n❌ VALIDATION FAILED: Output mismatch detected\n");
    }
    
    // Note about synthetic weights
    printf("\nNote: This test uses synthetic weights to validate the sparse FFN\n");
    printf("logic. Real model weights are extracted from GGUF tensors during\n");
    printf("actual inference (Step 10 Phase E integration).\n");
    
    // Cleanup
    llama_model_free(model);
    llama_backend_free();
    
    return (pass_l2 && pass_max && pass_relative) ? 0 : 1;
}
