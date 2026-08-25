// =============================================================================
// Step 11 Phase I — Testing and Validation
// =============================================================================
// Validates platform auto-detection, cross-platform regression, and
// platform-specific features.
//
// Run: step11_test.exe
// =============================================================================

#include "platform/platform_factory.h"
#include "platform/platform_types.h"
#include "platform/hardware_profiler_interface.h"
#include "platform/execution_backend_interface.h"
#include "../predictor/predictor.h"
#include "../predictor/confidence_calculator.h"
#include "../predictor/speed_predictor.h"
#include "../predictor/memory_predictor.h"
#include "../hotcold/hotcold_predictor.h"
#include "../hotcold/hotcold_types.h"
#include "moe_placer.h"
#include <cstdio>
#include <cmath>
#include <string>
#include <memory>

// =============================================================================
// Test Helpers
// =============================================================================

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST_ASSERT(cond, name) do { \
    if (cond) { \
        printf("  ✅ PASS: %s\n", name); \
        tests_passed++; \
    } else { \
        printf("  ❌ FAIL: %s\n", name); \
        tests_failed++; \
    } \
} while(0)

#define TEST_ASSERT_NEAR(actual, expected, tolerance_pct, name) do { \
    double _diff = std::abs((actual) - (expected)); \
    double _tol = std::abs(expected) * (tolerance_pct / 100.0); \
    if (_diff <= _tol + 1e-9) { \
        printf("  ✅ PASS: %s (%.2f vs %.2f, %.1f%%)\n", name, actual, expected, \
               (expected > 0) ? (100.0 * _diff / expected) : 0.0); \
        tests_passed++; \
    } else { \
        printf("  ❌ FAIL: %s (%.2f vs %.2f, %.1f%% > %.1f%%)\n", name, actual, expected, \
               (expected > 0) ? (100.0 * _diff / expected) : 0.0, tolerance_pct); \
        tests_failed++; \
    } \
} while(0)

// =============================================================================
// I4. Cross-Platform Regression Tests
// =============================================================================

static void test_cross_platform_regression() {
    printf("\n=== I4: Cross-Platform Regression ===\n");
    
    // Create a mock hardware spec (simulating NVIDIA RTX 5060)
    HardwareSpec hw_nvidia;
    hw_nvidia.platform = Platform::NVIDIA_WINDOWS;
    hw_nvidia.backend = ComputeBackend::CUDA;
    hw_nvidia.memory_arch = MemoryArchitecture::DISCRETE;
    hw_nvidia.is_unified_memory = false;
    hw_nvidia.gpu_name = "NVIDIA GeForce RTX 5060";
    hw_nvidia.vram_total_bytes = 8ULL * 1024 * 1024 * 1024;  // 8 GB
    hw_nvidia.vram_free_bytes = 7ULL * 1024 * 1024 * 1024;   // 7 GB free
    hw_nvidia.ram_total_bytes = 32ULL * 1024 * 1024 * 1024;  // 32 GB
    hw_nvidia.ram_free_bytes = 24ULL * 1024 * 1024 * 1024;   // 24 GB free
    hw_nvidia.gpu_bandwidth_gbs = 448.0;
    hw_nvidia.gpu_tflops_fp16 = 25.0;
    hw_nvidia.ram_bandwidth_gbs = 40.0;
    hw_nvidia.nvme_sequential_mbs = 5000.0;
    hw_nvidia.nvme_random_4k_mbs = 800.0;
    hw_nvidia.gpu_compute_major = 12;
    hw_nvidia.gpu_compute_minor = 0;
    hw_nvidia.compute_capability = "sm_120";
    hw_nvidia.hardware_fingerprint = "RTX 5060|32GB";
    
    // Create a model spec (Llama 3.2 3B Q4_K_M)
    ModelSpec model;
    model.name = "Llama-3.2-3B-Instruct-Q4_K_M";
    model.architecture = "llama";
    model.quant_type = "Q4_K_M";
    model.source = MetadataSource::GGUF_HEADER;
    model.model_type = ModelType::DENSE;
    model.param_count = 3212749824ULL;
    model.layers = 28;
    model.embedding_dim = 3072;
    model.attention_heads = 24;
    model.kv_heads = 8;
    model.head_dim = 128;
    model.ffn_dim = 8192;
    model.context_length = 131072;
    model.bits_per_weight = 4.85;
    
    // Strategy: Full GPU, 4K context
    StrategyConfig strat_full_gpu;
    strat_full_gpu.placement = PlacementStrategy::FULL_GPU;
    strat_full_gpu.gpu_layers = 28;
    strat_full_gpu.context_length = 4096;
    strat_full_gpu.batch_size = 1;
    strat_full_gpu.kv_quant_bits = 16;
    
    // Get NVIDIA prediction
    Prediction pred_nvidia = predict(hw_nvidia, model, strat_full_gpu);
    
    // Test 1: NVIDIA predictions are reasonable
    TEST_ASSERT(pred_nvidia.tokens_per_sec > 0, "NVIDIA: tokens/sec > 0");
    TEST_ASSERT(pred_nvidia.tokens_per_sec < 500, "NVIDIA: tokens/sec < 500 (sanity)");
    TEST_ASSERT(pred_nvidia.memory_vram_bytes > 0, "NVIDIA: VRAM usage > 0");
    TEST_ASSERT(pred_nvidia.memory_vram_bytes < 8ULL * 1024 * 1024 * 1024, "NVIDIA: VRAM < 8GB");
    TEST_ASSERT(pred_nvidia.viable, "NVIDIA: strategy is viable");
    
    // Test 2: AMD mock (simulating RX 7800 XT)
    HardwareSpec hw_amd = hw_nvidia;
    hw_amd.platform = Platform::AMD_LINUX;
    hw_amd.backend = ComputeBackend::HIP;
    hw_amd.gpu_name = "AMD Radeon RX 7800 XT";
    hw_amd.gpu_bandwidth_gbs = 624.0;
    hw_amd.gpu_tflops_fp16 = 37.0;
    hw_amd.compute_capability = "gfx1100";
    hw_amd.hardware_fingerprint = "RX 7800 XT|32GB";
    
    Prediction pred_amd = predict(hw_amd, model, strat_full_gpu);
    
    // Test 3: AMD predictions should be different from NVIDIA (different BW/TFLOPS)
    TEST_ASSERT(pred_amd.tokens_per_sec > 0, "AMD: tokens/sec > 0");
    TEST_ASSERT(pred_amd.tokens_per_sec != pred_nvidia.tokens_per_sec,
                "AMD: different speed than NVIDIA (different BW)");
    
    // Test 4: Apple Silicon mock (simulating M2 Pro)
    HardwareSpec hw_apple = hw_nvidia;
    hw_apple.platform = Platform::APPLE_MACOS;
    hw_apple.backend = ComputeBackend::METAL;
    hw_apple.memory_arch = MemoryArchitecture::UNIFIED;
    hw_apple.is_unified_memory = true;
    hw_apple.gpu_name = "Apple M2 Pro";
    hw_apple.vram_total_bytes = 32ULL * 1024 * 1024 * 1024;  // 32 GB unified
    hw_apple.vram_free_bytes = 28ULL * 1024 * 1024 * 1024;   // 28 GB free
    hw_apple.ram_total_bytes = 32ULL * 1024 * 1024 * 1024;   // Same as VRAM
    hw_apple.ram_free_bytes = 28ULL * 1024 * 1024 * 1024;    // Same as VRAM free
    hw_apple.gpu_bandwidth_gbs = 200.0;
    hw_apple.gpu_tflops_fp16 = 13.6;
    hw_apple.ram_bandwidth_gbs = 200.0;  // Same on unified memory
    hw_apple.compute_capability = "apple_m2";
    hw_apple.hardware_fingerprint = "Apple M2 Pro|32GB";
    
    Prediction pred_apple = predict(hw_apple, model, strat_full_gpu);
    
    // Test 5: Apple Silicon predictions should be reasonable
    TEST_ASSERT(pred_apple.tokens_per_sec > 0, "Apple: tokens/sec > 0");
    TEST_ASSERT(pred_apple.memory_vram_bytes > 0, "Apple: VRAM usage > 0");
    
    // Test 6: Apple split strategy (Metal/CPU) should be close to Full Metal
    // On Apple Silicon, split doesn't have PCIe penalty
    StrategyConfig strat_split;
    strat_split.placement = PlacementStrategy::GPU_CPU_SPLIT;
    strat_split.gpu_layers = 20;  // 20/28 layers on Metal
    strat_split.context_length = 4096;
    strat_split.batch_size = 1;
    strat_split.kv_quant_bits = 16;
    
    Prediction pred_apple_split = predict(hw_apple, model, strat_split);
    
    // Split should be slower than full, but not dramatically (no PCIe penalty)
    if (pred_apple_split.tokens_per_sec > 0 && pred_apple.tokens_per_sec > 0) {
        double split_ratio = pred_apple_split.tokens_per_sec / pred_apple.tokens_per_sec;
        // On Apple Silicon, split should be > 50% of full speed (no PCIe penalty)
        TEST_ASSERT(split_ratio > 0.5, "Apple: Metal/CPU split > 50% of Full Metal speed");
    }
    
    // Test 7: CPU-only should be slower than GPU
    StrategyConfig strat_cpu;
    strat_cpu.placement = PlacementStrategy::CPU_ONLY;
    strat_cpu.gpu_layers = 0;
    strat_cpu.context_length = 4096;
    strat_cpu.batch_size = 1;
    strat_cpu.kv_quant_bits = 16;
    
    Prediction pred_cpu = predict(hw_nvidia, model, strat_cpu);
    TEST_ASSERT(pred_cpu.tokens_per_sec > 0, "CPU-only: tokens/sec > 0");
    TEST_ASSERT(pred_cpu.tokens_per_sec < pred_nvidia.tokens_per_sec,
                "CPU-only: slower than Full GPU");
    
    // Test 8: Memory prediction accuracy
    // Weight memory should be ~1.81 GB for 3B Q4_K_M
    uint64_t weight_bytes = predict_weight_memory(model);
    double weight_gb = weight_bytes / 1e9;
    TEST_ASSERT_NEAR(weight_gb, 1.81, 10.0, "Weight memory: ~1.81 GB");
    
    // Test 9: KV cache prediction accuracy
    // KV cache at 4K context should be ~448 MB for FP16
    uint64_t kv_bytes = predict_kv_cache_memory(model, 4096, 16);
    double kv_gb = kv_bytes / 1e9;
    TEST_ASSERT_NEAR(kv_gb, 0.448, 10.0, "KV cache: ~448 MB at 4K FP16");
    
    // Test 10: Confidence levels
    ConfidenceResult conf_nvidia = calculate_confidence(model, hw_nvidia, 4096, 5);
    TEST_ASSERT(conf_nvidia.level == PredictionConfidence::HIGH,
                "NVIDIA: HIGH confidence with 5 calibration records");
    
    ConfidenceResult conf_apple = calculate_confidence(model, hw_apple, 4096, 5);
    TEST_ASSERT(conf_apple.level == PredictionConfidence::HIGH,
                "Apple: HIGH confidence with 5 calibration records");
    
    ConfidenceResult conf_no_cal = calculate_confidence(model, hw_nvidia, 4096, 0);
    TEST_ASSERT(conf_no_cal.level == PredictionConfidence::MEDIUM,
                "NVIDIA: MEDIUM confidence without calibration");
}

// =============================================================================
// Platform Detection Tests
// =============================================================================

static void test_platform_detection() {
    printf("\n=== Platform Detection ===\n");
    
    // Test 1: Auto-detection finds a platform
    auto profiler = create_platform_profiler_auto();
    TEST_ASSERT(profiler != nullptr, "Auto-detect: profiler created");
    
    if (profiler) {
        TEST_ASSERT(profiler->isAvailable(), "Auto-detect: profiler available");
        printf("  Platform: %s\n", profiler->getName().c_str());
        printf("  Platform enum: %d\n", (int)profiler->getPlatform());
    }
    
    // Test 2: Platform override works
    auto nvidia_profiler = create_platform_profiler_for_platform("cuda");
    // May or may not be available depending on system
    printf("  CUDA available: %s\n", nvidia_profiler ? "yes" : "no");
    
    auto cpu_profiler = create_platform_profiler_for_platform("cpu");
    // CPU-only profiler may not be implemented yet
    printf("  CPU profiler available: %s\n", cpu_profiler ? "yes" : "no (not implemented)");
    
    // Test 3: Invalid platform returns nullptr
    auto invalid = create_platform_profiler_for_platform("invalid_platform");
    TEST_ASSERT(invalid == nullptr, "Invalid platform: returns nullptr");
    
    // Test 4: Executor auto-detection
    auto executor = create_platform_executor_auto();
    // May or may not be available
    printf("  Executor available: %s\n", executor ? "yes" : "no");
    
    // Test 5: GPU enumeration
    auto gpus = enumerate_gpus();
    printf("  GPUs found: %zu\n", gpus.size());
    for (const auto& gpu : gpus) {
        printf("    [%u] %s (%.1f GB)\n", gpu.index, gpu.name.c_str(),
               gpu.vram_bytes / 1e9);
    }
}

// =============================================================================
// Hot/Cold Strategy Tests (Apple Silicon specific)
// =============================================================================

static void test_apple_hotcold() {
    printf("\n=== Apple Silicon Hot/Cold Behavior ===\n");
    
    // Apple Silicon mock
    HardwareSpec hw_apple;
    hw_apple.platform = Platform::APPLE_MACOS;
    hw_apple.backend = ComputeBackend::METAL;
    hw_apple.memory_arch = MemoryArchitecture::UNIFIED;
    hw_apple.is_unified_memory = true;
    hw_apple.gpu_name = "Apple M2 Pro";
    hw_apple.vram_total_bytes = 32ULL * 1024 * 1024 * 1024;
    hw_apple.vram_free_bytes = 28ULL * 1024 * 1024 * 1024;
    hw_apple.ram_total_bytes = 32ULL * 1024 * 1024 * 1024;
    hw_apple.ram_free_bytes = 28ULL * 1024 * 1024 * 1024;
    hw_apple.gpu_bandwidth_gbs = 200.0;
    hw_apple.gpu_tflops_fp16 = 13.6;
    hw_apple.ram_bandwidth_gbs = 200.0;
    hw_apple.compute_capability = "apple_m2";
    hw_apple.hardware_fingerprint = "Apple M2 Pro|32GB";
    
    // Discrete GPU mock (NVIDIA)
    HardwareSpec hw_nvidia = hw_apple;
    hw_nvidia.platform = Platform::NVIDIA_WINDOWS;
    hw_nvidia.backend = ComputeBackend::CUDA;
    hw_nvidia.memory_arch = MemoryArchitecture::DISCRETE;
    hw_nvidia.is_unified_memory = false;
    hw_nvidia.gpu_name = "NVIDIA RTX 3080";
    hw_nvidia.vram_total_bytes = 10ULL * 1024 * 1024 * 1024;
    hw_nvidia.vram_free_bytes = 9ULL * 1024 * 1024 * 1024;
    hw_nvidia.ram_total_bytes = 32ULL * 1024 * 1024 * 1024;
    hw_nvidia.ram_free_bytes = 24ULL * 1024 * 1024 * 1024;
    hw_nvidia.gpu_bandwidth_gbs = 760.0;
    hw_nvidia.gpu_tflops_fp16 = 29.8;
    hw_nvidia.ram_bandwidth_gbs = 40.0;
    hw_nvidia.compute_capability = "sm_86";
    hw_nvidia.hardware_fingerprint = "RTX 3080|32GB";
    
    // Model: 13B Q4 (doesn't fit entirely on either GPU)
    ModelSpec model;
    model.name = "Llama-2-13B-Q4_K_M";
    model.architecture = "llama";
    model.quant_type = "Q4_K_M";
    model.source = MetadataSource::GGUF_HEADER;
    model.model_type = ModelType::DENSE;
    model.param_count = 13000000000ULL;
    model.layers = 40;
    model.embedding_dim = 5120;
    model.attention_heads = 40;
    model.kv_heads = 40;
    model.head_dim = 128;
    model.ffn_dim = 13824;
    model.context_length = 4096;
    model.bits_per_weight = 4.85;
    
    // Hot/Cold profile
    HotNeuronProfile profile;
    profile.model_name = model.name;
    profile.activation = ActivationType::SILU;
    profile.num_layers = model.layers;
    profile.ffn_dim = model.ffn_dim;
    profile.hidden_dim = model.embedding_dim;
    profile.hot_ratio = 0.15;
    
    // Compute hot/cold strategy for both platforms
    HotColdStrategy hc_apple = compute_hotcold_strategy(
        hw_apple, model, profile, 4096, 16, 0.15);
    HotColdStrategy hc_nvidia = compute_hotcold_strategy(
        hw_nvidia, model, profile, 4096, 16, 0.15);
    
    // Test 1: Hot/cold viability
    printf("  Apple hot/cold viable: %s\n", hc_apple.viable ? "yes" : "no");
    printf("  NVIDIA hot/cold viable: %s\n", hc_nvidia.viable ? "yes" : "no");
    
    if (hc_apple.viable && hc_nvidia.viable) {
        // Test 2: Speed comparison
        HotColdSpeedResult speed_apple = predict_hotcold_speed(hw_apple, model, hc_apple, 16);
        HotColdSpeedResult speed_nvidia = predict_hotcold_speed(hw_nvidia, model, hc_nvidia, 16);
        
        printf("  Apple hot/cold expected: %.1f tok/s\n", speed_apple.tok_s_expected);
        printf("  NVIDIA hot/cold expected: %.1f tok/s\n", speed_nvidia.tok_s_expected);
        
        // Test 3: Apple should have lower PCIe transfer time (0 on unified memory)
        TEST_ASSERT(speed_apple.pcie_transfer_ms == 0.0,
                    "Apple: no PCIe transfer for hot/cold");
        TEST_ASSERT(speed_nvidia.pcie_transfer_ms > 0.0,
                    "NVIDIA: PCIe transfer for hot/cold");
        
        // Test 4: On Apple, hot/cold benefit is less (no PCIe to avoid)
        // The speed improvement over naive split should be smaller
        // This is expected behavior
        printf("  Apple PCIe time: %.2f ms\n", speed_apple.pcie_transfer_ms);
        printf("  NVIDIA PCIe time: %.2f ms\n", speed_nvidia.pcie_transfer_ms);
    }
}

// =============================================================================
// Main
// =============================================================================

int main() {
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║     Step 11 Phase I — Validation Tests                      ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n");
    
    // Run all test suites
    test_platform_detection();
    test_cross_platform_regression();
    test_apple_hotcold();
    
    // Summary
    printf("\n╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║                     TEST SUMMARY                           ║\n");
    printf("╠═══════════════════════════════════════════════════════════════╣\n");
    printf("║  ✅ Passed:  %-3d                                           ║\n", tests_passed);
    printf("║  ❌ Failed:  %-3d                                           ║\n", tests_failed);
    printf("║  Total:      %-3d                                           ║\n", tests_passed + tests_failed);
    printf("╚═══════════════════════════════════════════════════════════════╝\n");
    
    if (tests_failed == 0) {
        printf("\n🎉 All validation tests passed!\n");
    } else {
        printf("\n⚠️  %d test(s) failed.\n", tests_failed);
    }
    
    return tests_failed > 0 ? 1 : 0;
}
