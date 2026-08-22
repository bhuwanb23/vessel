#include "predictor.h"
#include <cstdio>
#include <cmath>

// =============================================================================
// Test Program: Predictor Math Validation
// =============================================================================
// This program tests the predictor formulas against real hardware and model data.
// All inputs are hardcoded from Step 1 (hardware profiler) and Step 2 (metadata fetcher).
// =============================================================================

void print_separator() {
    printf("=================================================\n");
}

void test_prediction(const HardwareSpec& hw, const ModelSpec& model, const StrategyConfig& strategy, const char* test_name) {
    printf("\n");
    print_separator();
    printf("TEST: %s\n", test_name);
    print_separator();

    // Run prediction
    Prediction pred = predict(hw, model, strategy);

    // Print inputs
    printf("\n--- Inputs ---\n");
    printf("GPU:           %s\n", hw.gpu_name.c_str());
    printf("VRAM:          %s total, %s free\n",
           format_bytes(hw.vram_total_bytes).c_str(),
           format_bytes(hw.vram_free_bytes).c_str());
    printf("GPU Bandwidth: %.1f GB/s\n", hw.gpu_bandwidth_gbs);
    printf("RAM:           %s total, %s free\n",
           format_bytes(hw.ram_total_bytes).c_str(),
           format_bytes(hw.ram_free_bytes).c_str());
    printf("Model:         %s (%s)\n", model.name.c_str(), model.architecture.c_str());
    printf("Layers:        %u\n", model.layers);
    printf("Quantization:  %s (%.1f bpw)\n", model.quant_type.c_str(), model.bits_per_weight);
    printf("Context:       %u tokens\n", strategy.context_length > 0 ? strategy.context_length : model.context_length);
    printf("Strategy:      %s (%u GPU layers)\n", get_placement_name(strategy.placement), strategy.gpu_layers);

    // Print predictions
    printf("\n--- Predictions ---\n");
    printf("Total Memory:  %s\n", format_bytes(pred.memory_total_bytes).c_str());
    printf("  VRAM:        %s\n", format_bytes(pred.memory_vram_bytes).c_str());
    printf("  RAM:         %s\n", format_bytes(pred.memory_ram_bytes).c_str());
    printf("Speed:         %s\n", format_speed(pred.tokens_per_sec).c_str());
    printf("TTFT:          %.1f ms\n", pred.ttft_ms);
    printf("Prompt Speed:  %s\n", format_speed(pred.prompt_eval_tps).c_str());
    printf("Viable:        %s\n", pred.viable ? "YES" : "NO");
    printf("Confidence:    %s\n", get_confidence_name(pred.confidence));

    if (!pred.warnings.empty()) {
        printf("Warning:       %s\n", pred.warnings.c_str());
    }

    printf("\n--- Validation ---\n");
    printf("Baseline: 43.4 tokens/sec (from llama-cli)\n");
    printf("Predicted: %.1f tokens/sec\n", pred.tokens_per_sec);

    double error_pct = 0.0;
    if (pred.tokens_per_sec > 0) {
        error_pct = std::abs(pred.tokens_per_sec - 43.4) / 43.4 * 100.0;
    }
    printf("Error:     %.1f%%\n", error_pct);

    if (error_pct < 20.0) {
        printf("Status:    PASS (<20%% error)\n");
    } else if (error_pct < 50.0) {
        printf("Status:    ACCEPTABLE (20-50%% error)\n");
    } else {
        printf("Status:    NEEDS CALIBRATION (>50%% error)\n");
    }
}

int main() {
    printf("\n");
    print_separator();
    printf("LLM Deployment Planner - Predictor Math Test\n");
    print_separator();
    printf("\nAll inputs are hardcoded from Step 1 (hardware) and Step 2 (model metadata).\n");
    printf("Baseline from Step 0: 43.4 tokens/sec (Llama-3.2-3B Q4_K_M on RTX 5060)\n");

    // =========================================================================
    // Hardware Specs (from Step 1 - RTX 5060 system)
    // =========================================================================
    HardwareSpec hw;
    hw.ram_total_bytes    = 31.46 * 1024 * 1024 * 1024;    // 31.46 GB
    hw.ram_free_bytes     = 12.0 * 1024 * 1024 * 1024;      // ~12 GB available
    hw.ram_bandwidth_gbs  = 25.0;                            // DDR5 estimate (not measured)
    hw.vram_total_bytes   = 8546942976ULL;                   // 7.96 GB (8151 MiB)
    hw.vram_free_bytes    = 804 * 1024 * 1024;               // ~804 MiB free
    hw.gpu_bandwidth_gbs  = 448.0;                           // Derived from NVML (128-bit, 14001 MHz)
    hw.gpu_tflops_fp16    = 20.0;                            // RTX 5060 estimate (~20 TFLOPS)
    hw.nvme_sequential_mbs = 3019.0;                         // From disk benchmark
    hw.nvme_random_4k_mbs  = 38.0;                           // From disk benchmark
    hw.gpu_name           = "NVIDIA GeForce RTX 5060";
    hw.gpu_compute_major  = 12;
    hw.gpu_compute_minor  = 0;

    // =========================================================================
    // Test 1: Llama-3.2-3B Q4_K_M (Full GPU)
    // =========================================================================
    ModelSpec llama_3b;
    llama_3b.architecture     = "llama";
    llama_3b.name             = "Llama-3.2-3B-Instruct-Q4_K_M";
    llama_3b.quant_type       = "Q4_K_M";
    llama_3b.param_count      = 3212749824ULL;   // 3.2B
    llama_3b.layers           = 28;
    llama_3b.embedding_dim    = 3072;
    llama_3b.attention_heads  = 24;
    llama_3b.kv_heads         = 8;
    llama_3b.head_dim         = 128;              // 3072 / 24
    llama_3b.ffn_dim          = 8192;
    llama_3b.context_length   = 131072;
    llama_3b.bits_per_weight  = 4.5;              // Q4_K_M

    StrategyConfig strat_full_gpu;
    strat_full_gpu.placement      = PlacementStrategy::FULL_GPU;
    strat_full_gpu.gpu_layers     = 28;            // All layers on GPU
    strat_full_gpu.context_length = 4096;          // Test with 4K context
    strat_full_gpu.batch_size     = 1;
    strat_full_gpu.kv_quant_bits  = 16;

    test_prediction(hw, llama_3b, strat_full_gpu, "Llama-3.2-3B Q4_K_M - Full GPU (4K ctx)");

    // =========================================================================
    // Test 2: Same model with 32K context
    // =========================================================================
    StrategyConfig strat_128k;
    strat_128k.placement      = PlacementStrategy::FULL_GPU;
    strat_128k.gpu_layers     = 28;
    strat_128k.context_length = 32768;             // 32K context
    strat_128k.batch_size     = 1;
    strat_128k.kv_quant_bits  = 16;

    test_prediction(hw, llama_3b, strat_128k, "Llama-3.2-3B Q4_K_M - Full GPU (32K ctx)");

    // =========================================================================
    // Test 3: CPU Only (for comparison)
    // =========================================================================
    StrategyConfig strat_cpu;
    strat_cpu.placement      = PlacementStrategy::CPU_ONLY;
    strat_cpu.gpu_layers     = 0;
    strat_cpu.context_length = 4096;
    strat_cpu.batch_size     = 1;
    strat_cpu.kv_quant_bits  = 16;

    test_prediction(hw, llama_3b, strat_cpu, "Llama-3.2-3B Q4_K_M - CPU Only (4K ctx)");

    // =========================================================================
    // Test 4: Memory-only prediction (no performance)
    // =========================================================================
    printf("\n");
    print_separator();
    printf("MEMORY BREAKDOWN ANALYSIS\n");
    print_separator();

    uint64_t weight_mem = predict_weight_memory(llama_3b);
    uint64_t kv_mem_4k = predict_kv_cache_memory(llama_3b, 4096, 16);
    uint64_t kv_mem_32k = predict_kv_cache_memory(llama_3b, 32768, 16);
    uint64_t kv_mem_128k = predict_kv_cache_memory(llama_3b, 131072, 16);
    uint64_t overhead = predict_overhead_memory(llama_3b, 1);

    printf("\nLlama-3.2-3B Q4_K_M Memory Breakdown:\n");
    printf("  Weights:           %s\n", format_bytes(weight_mem).c_str());
    printf("  KV Cache (4K):     %s\n", format_bytes(kv_mem_4k).c_str());
    printf("  KV Cache (32K):    %s\n", format_bytes(kv_mem_32k).c_str());
    printf("  KV Cache (128K):   %s\n", format_bytes(kv_mem_128k).c_str());
    printf("  Overhead:          %s\n", format_bytes(overhead).c_str());
    printf("  ----------------------------------------\n");
    printf("  Total (4K ctx):    %s\n", format_bytes(weight_mem + kv_mem_4k + overhead).c_str());
    printf("  Total (32K ctx):   %s\n", format_bytes(weight_mem + kv_mem_32k + overhead).c_str());
    printf("  Total (128K ctx):  %s\n", format_bytes(weight_mem + kv_mem_128k + overhead).c_str());
    printf("\n  VRAM Available:    %s\n", format_bytes(hw.vram_total_bytes).c_str());
    printf("  RAM Available:     %s\n", format_bytes(hw.ram_free_bytes).c_str());

    // =========================================================================
    // Summary
    // =========================================================================
    printf("\n");
    print_separator();
    printf("PREDICTOR MODULE SUMMARY\n");
    print_separator();
    printf("\nImplemented functions:\n");
    printf("  - predict_weight_memory(): param_count * bits_per_weight / 8\n");
    printf("  - predict_kv_cache_memory(): 2 * layers * ctx * kv_heads * head_dim * bits / 8\n");
    printf("  - predict_overhead_memory(): 350MB base + activations\n");
    printf("  - predict_tokens_per_sec(): bandwidth / model_size * efficiency\n");
    printf("  - predict_ttft_ms(): prompt_tokens / prompt_speed * 1000\n");
    printf("  - predict(): main orchestrator\n");
    printf("\nNext step: Calibrate efficiency factor against real llama.cpp numbers.\n");

    return 0;
}
