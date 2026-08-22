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
    llama_3b.bits_per_weight  = get_bits_per_weight("Q4_K_M");  // 4.85 bpw

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
    uint64_t kv_mem_4k = predict_kv_cache_memory(llama_3b, 4096, 16, 1);
    uint64_t kv_mem_32k = predict_kv_cache_memory(llama_3b, 32768, 16, 1);
    uint64_t kv_mem_128k = predict_kv_cache_memory(llama_3b, 131072, 16, 1);
    uint64_t overhead_gpu = predict_overhead_memory(llama_3b, 1, true);
    uint64_t overhead_cpu = predict_overhead_memory(llama_3b, 1, false);

    printf("\nLlama-3.2-3B Q4_K_M Memory Breakdown:\n");
    printf("  Weights:           %s\n", format_bytes(weight_mem).c_str());
    printf("  KV Cache (4K):     %s\n", format_bytes(kv_mem_4k).c_str());
    printf("  KV Cache (32K):    %s\n", format_bytes(kv_mem_32k).c_str());
    printf("  KV Cache (128K):   %s\n", format_bytes(kv_mem_128k).c_str());
    printf("  Overhead (GPU):    %s\n", format_bytes(overhead_gpu).c_str());
    printf("  Overhead (CPU):    %s\n", format_bytes(overhead_cpu).c_str());
    printf("  ----------------------------------------\n");
    printf("  Total (4K ctx):    %s\n", format_bytes(weight_mem + kv_mem_4k + overhead_gpu).c_str());
    printf("  Total (32K ctx):   %s\n", format_bytes(weight_mem + kv_mem_32k + overhead_gpu).c_str());
    printf("  Total (128K ctx):  %s\n", format_bytes(weight_mem + kv_mem_128k + overhead_gpu).c_str());
    printf("\n  VRAM Available:    %s\n", format_bytes(hw.vram_total_bytes).c_str());
    printf("  RAM Available:     %s\n", format_bytes(hw.ram_free_bytes).c_str());
    
    // =========================================================================
    // Test 6: Max Safe Context
    // =========================================================================
    printf("\n");
    print_separator();
    printf("MAX SAFE CONTEXT CALCULATION\n");
    print_separator();
    
    printf("\nLlama-3.2-3B Q4_K_M:\n");
    printf("  Model max context:  %u tokens\n", llama_3b.context_length);
    
    // Run prediction to get max_safe_context
    Prediction pred_max = predict(hw, llama_3b, strat_full_gpu);
    printf("  Predicted max safe: %u tokens\n", pred_max.max_safe_context);
    
    // Calculate memory at max safe context
    uint64_t kv_at_max = predict_kv_cache_memory(llama_3b, pred_max.max_safe_context, 16, 1);
    uint64_t total_at_max = weight_mem + kv_at_max + overhead_gpu;
    printf("  Memory at max:     %s\n", format_bytes(total_at_max).c_str());
    printf("  VRAM available:    %s\n", format_bytes(hw.vram_total_bytes).c_str());
    
    if (total_at_max <= hw.vram_total_bytes) {
        printf("  Status:            FITS in VRAM\n");
    } else {
        printf("  Status:            EXCEEDS VRAM (would split to CPU)\n");
    }
    
    // =========================================================================
    // Test 7: MLA Model Test (DeepSeek placeholder)
    // =========================================================================
    printf("\n");
    print_separator();
    printf("MLA ATTENTION TEST (DeepSeek-style)\n");
    print_separator();
    
    ModelSpec deepseek_test;
    deepseek_test.architecture = "deepseek2";
    deepseek_test.name = "DeepSeek-V2-Lite (test)";
    deepseek_test.quant_type = "Q4_K_M";
    deepseek_test.param_count = 16000000000ULL;  // 16B
    deepseek_test.layers = 60;
    deepseek_test.embedding_dim = 2048;
    deepseek_test.attention_heads = 16;
    deepseek_test.kv_heads = 16;
    deepseek_test.head_dim = 128;
    deepseek_test.ffn_dim = 1408;
    deepseek_test.context_length = 32768;
    deepseek_test.bits_per_weight = get_bits_per_weight("Q4_K_M");
    deepseek_test.kv_lora_rank = 512;           // MLA compression rank
    deepseek_test.qk_rope_head_dim = 64;        // Rope head dim
    
    uint64_t kv_standard = predict_kv_cache_memory(deepseek_test, 4096, 16, 1);
    
    // Temporarily set kv_lora_rank to 0 to get standard calculation
    uint32_t saved_rank = deepseek_test.kv_lora_rank;
    deepseek_test.kv_lora_rank = 0;
    uint64_t kv_no_mla = predict_kv_cache_memory(deepseek_test, 4096, 16, 1);
    deepseek_test.kv_lora_rank = saved_rank;
    
    printf("\nDeepSeek-V2-Lite (MLA model):\n");
    printf("  Architecture:      %s\n", deepseek_test.architecture.c_str());
    printf("  KV Lora Rank:      %u\n", deepseek_test.kv_lora_rank);
    printf("  QK Rope Head Dim:  %u\n", deepseek_test.qk_rope_head_dim);
    printf("  KV cache (MLA):    %s\n", format_bytes(kv_standard).c_str());
    printf("  KV cache (std):    %s (if MLA not detected)\n", format_bytes(kv_no_mla).c_str());
    printf("  MLA savings:       %.1fx smaller\n", 
           (kv_no_mla > 0) ? static_cast<double>(kv_no_mla) / kv_standard : 0.0);

    // =========================================================================
    // Summary
    // =========================================================================
    printf("\n");
    print_separator();
    printf("PREDICTOR MODULE SUMMARY\n");
    print_separator();
    // =========================================================================
    // Test 5: BPW Validation
    // =========================================================================
    printf("\n");
    print_separator();
    printf("BITS-PER-WEIGHT VALIDATION\n");
    print_separator();

    // Llama-3.2-3B Q4_K_M file size: 1,888,356,224 bytes (from Step 0)
    uint64_t file_size_llama = 1888356224ULL;
    double measured_bpw = validate_bpw_from_file(file_size_llama, llama_3b.param_count);
    double lookup_bpw = get_bits_per_weight("Q4_K_M");
    
    printf("\nLlama-3.2-3B Q4_K_M:\n");
    printf("  File size:       %llu bytes\n", file_size_llama);
    printf("  Param count:     %llu\n", llama_3b.param_count);
    printf("  Measured bpw:    %.2f (from file_size * 8 / params)\n", measured_bpw);
    printf("  Lookup bpw:      %.2f (from table)\n", lookup_bpw);
    printf("  Error:           %.1f%%\n", std::abs(measured_bpw - lookup_bpw) / lookup_bpw * 100.0);
    
    if (std::abs(measured_bpw - lookup_bpw) / lookup_bpw < 0.1) {
        printf("  Status:          PASS (<10%% error)\n");
    } else {
        printf("  Status:          NEEDS REVIEW\n");
    }

    // Print all lookup table values
    printf("\n--- Complete BPW Lookup Table ---\n");
    printf("Quant Type    bpw     Quant Type    bpw     Quant Type    bpw\n");
    printf("------------  ------  ------------  ------  ------------  ------\n");
    const char* quants[] = {
        "F32", "F16", "Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S",
        "Q4_K_M", "Q4_K_S", "Q4_0", "Q4_1", "Q3_K_L", "Q3_K_M",
        "Q3_K_S", "Q2_K", "IQ4_NL", "IQ3_XXS", "IQ2_XS", "IQ2_XXS"
    };
    for (int i = 0; i < 18; i += 3) {
        printf("%-12s  %-6.2f  %-12s  %-6.2f  %-12s  %-6.2f\n",
               quants[i], get_bits_per_weight(quants[i]),
               (i+1 < 18) ? quants[i+1] : "", (i+1 < 18) ? get_bits_per_weight(quants[i+1]) : 0.0,
               (i+2 < 18) ? quants[i+2] : "", (i+2 < 18) ? get_bits_per_weight(quants[i+2]) : 0.0);
    }

    // =========================================================================
    // Test 8: Speed Prediction Analysis
    // =========================================================================
    printf("\n");
    print_separator();
    printf("SPEED PREDICTION ANALYSIS (Phase D)\n");
    print_separator();
    
    printf("\nLlama-3.2-3B Q4_K_M Speed by Placement:\n");
    printf("  %-20s  %-15s  %-15s\n", "Strategy", "Predicted", "Theoretical Max");
    printf("  %-20s  %-15s  %-15s\n", "--------------------", "---------------", "---------------");
    
    // Full GPU speed
    double speed_full = predict_decode_speed(hw, llama_3b, 28, 4096, 16);
    double theoretical_full = (hw.gpu_bandwidth_gbs * 1e9) / predict_bytes_per_token(llama_3b);
    printf("  %-20s  %-15s  %-15s\n", "Full GPU", format_speed(speed_full).c_str(), format_speed(theoretical_full).c_str());
    
    // Split: 20/28 layers on GPU
    double speed_split_20 = predict_decode_speed(hw, llama_3b, 20, 4096, 16);
    printf("  %-20s  %-15s\n", "GPU:20 / CPU:8", format_speed(speed_split_20).c_str());
    
    // Split: 14/28 layers on GPU (50%)
    double speed_split_14 = predict_decode_speed(hw, llama_3b, 14, 4096, 16);
    printf("  %-20s  %-15s\n", "GPU:14 / CPU:14", format_speed(speed_split_14).c_str());
    
    // Split: 8/28 layers on GPU
    double speed_split_8 = predict_decode_speed(hw, llama_3b, 8, 4096, 16);
    printf("  %-20s  %-15s\n", "GPU:8 / CPU:20", format_speed(speed_split_8).c_str());
    
    // CPU only speed
    double speed_cpu = predict_decode_speed(hw, llama_3b, 0, 4096, 16);
    
    // =========================================================================
    // Test 10: Confidence Level Analysis (Phase F)
    // =========================================================================
    printf("\n");
    print_separator();
    printf("CONFIDENCE LEVEL ANALYSIS (Phase F)\n");
    print_separator();
    
    // Test with GGUF header (high quality metadata)
    ModelSpec gguf_model = llama_3b;
    gguf_model.source = MetadataSource::GGUF_HEADER;
    gguf_model.model_type = ModelType::DENSE;
    
    printf("\nTest 1: GGUF header, dense model, short context\n");
    ConfidenceResult conf1 = calculate_confidence(gguf_model, hw, 4096, 0);
    printf("  Confidence: %s\n", confidence_to_string(conf1.level));
    printf("  Reason:     %s\n", conf1.reason.c_str());
    
    printf("\nTest 2: GGUF header, dense model, long context (64K)\n");
    ConfidenceResult conf2 = calculate_confidence(gguf_model, hw, 65536, 0);
    printf("  Confidence: %s\n", confidence_to_string(conf2.level));
    printf("  Reason:     %s\n", conf2.reason.c_str());
    
    printf("\nTest 3: Config.json fallback\n");
    ModelSpec config_model = llama_3b;
    config_model.source = MetadataSource::CONFIG_JSON;
    ConfidenceResult conf3 = calculate_confidence(config_model, hw, 4096, 0);
    printf("  Confidence: %s\n", confidence_to_string(conf3.level));
    printf("  Reason:     %s\n", conf3.reason.c_str());
    
    printf("\nTest 4: Unknown quantization\n");
    ModelSpec unknown_quant = llama_3b;
    unknown_quant.bits_per_weight = 0;  // Unknown
    ConfidenceResult conf4 = calculate_confidence(unknown_quant, hw, 4096, 0);
    printf("  Confidence: %s\n", confidence_to_string(conf4.level));
    printf("  Reason:     %s\n", conf4.reason.c_str());
    
    printf("\nTest 5: With calibration records (5+)\n");
    ConfidenceResult conf5 = calculate_confidence(gguf_model, hw, 4096, 10);
    printf("  Confidence: %s\n", confidence_to_string(conf5.level));
    printf("  Reason:     %s\n", conf5.reason.c_str());
    
    printf("\nNote: Step 3 has no calibration records (Step 7), so HIGH is rare.");
    printf("\n      This is expected - mechanism will activate in Step 7.\n");
    double theoretical_cpu = (25.0 * 1e9) / predict_bytes_per_token(llama_3b);
    printf("  %-20s  %-15s  %-15s\n", "CPU Only", format_speed(speed_cpu).c_str(), format_speed(theoretical_cpu).c_str());
    
    printf("\nKey Insight: GPU/CPU split is SEQUENTIAL, not parallel.");
    printf("\n  Even 71%% on GPU (20/28) is much slower than 100%% GPU.");
    printf("\n  The CPU layers become a bottleneck.\n");
    
    // Show bytes per token
    printf("\nBytes per token: %.2f GB\n", predict_bytes_per_token(llama_3b) / 1e9);
    printf("KV bytes per token: %.2f KB\n", predict_kv_bytes_per_token(llama_3b, 16) / 1024.0);
    
    // =========================================================================
    // Test 9: TTFT Analysis (Phase E)
    // =========================================================================
    printf("\n");
    print_separator();
    printf("TTFT ANALYSIS (Phase E - Compute-bound)\n");
    print_separator();
    
    printf("\nLlama-3.2-3B Q4_K_M TTFT by Prompt Length (Full GPU):\n");
    printf("  %-15s  %-12s  %-12s  %-12s\n", "Prompt Tokens", "TTFT (ms)", "Lower Bound", "Upper Bound");
    printf("  %-15s  %-12s  %-12s  %-12s\n", "---------------", "------------", "------------", "------------");
    
    uint32_t prompt_lengths[] = {32, 128, 512, 1024, 2048, 4096};
    for (uint32_t pt : prompt_lengths) {
        double ttft = predict_ttft_ms(hw, llama_3b, pt, 28);
        double lower, upper;
        predict_ttft_bounds(hw, llama_3b, pt, 28, lower, upper);
        printf("  %-15u  %-12.1f  %-12.1f  %-12.1f\n", pt, ttft, lower, upper);
    }
    
    printf("\nKey Formula:\n");
    printf("  flops_per_token = 2 × params = %.1f TFLOPS\n", 
           2.0 * llama_3b.param_count / 1e12);
    printf("  GPU efficiency: 30%% (small batch, small model)\n");
    printf("  CPU TFLOPS estimate: 0.8 TFLOPS\n");
    printf("  TTFT confidence: ±40%% (wider than tokens/sec)\n");

    printf("\nImplemented modules:\n");
    printf("  memory_predictor.h/cpp  - Weight, KV cache, overhead\n");
    printf("  speed_predictor.h/cpp   - Decode speed, prompt eval, TTFT\n");
    printf("  context_analyzer.h/cpp  - Max safe context\n");
    printf("  predictor_validation.h/cpp - BPW validation\n");
    printf("  predictor.h/cpp         - Main orchestrator\n");
    printf("\nPhase D complete! Decode speed formulas implemented.\n");

    return 0;
}
