// =============================================================================
// LLM Deployment Planner — End-to-End Test Harness (Phase G)
// =============================================================================
// Tests the CORE workflow: fetch metadata from HuggingFace → predict hardware fit
// WITHOUT downloading any models. This validates the tool's main value proposition.
//
// Usage: e2e_test.exe [--local-only] [--skip-network]
//   --local-only   Skip network tests (test only with local model)
//   --skip-network Skip network tests entirely
// =============================================================================

#include "types.h"
#include "profiler.h"
#include "fetcher.h"
#include "matrix.h"
#include "ranker.h"
#include "../predictor/predictor.h"
#include "../predictor/memory_predictor.h"
#include "../predictor/speed_predictor.h"
#include "../predictor/context_analyzer.h"
#include <cstdio>
#include <cstring>
#include <chrono>
#include <string>
#include <vector>

// =============================================================================
// Test Infrastructure
// =============================================================================

struct TestCase {
    std::string name;
    std::string url;
    std::string expected_arch;      // Expected architecture (llama, qwen2, etc.)
    uint32_t min_layers;            // Minimum expected layers
    uint32_t max_layers;            // Maximum expected layers
    bool expect_viable_full_gpu;    // Should Full GPU be viable at 4K?
    bool expect_viable_any;         // Should any strategy be viable?
    bool requires_network;          // Does this test need internet?
};

struct TestResult {
    std::string name;
    bool passed;
    std::string details;
    double elapsed_ms;
};

static int tests_passed = 0;
static int tests_failed = 0;
static int tests_skipped = 0;

void print_test_header(const char* test_name) {
    printf("\n┌─────────────────────────────────────────────────────────────┐\n");
    printf("│  TEST: %-51s │\n", test_name);
    printf("└─────────────────────────────────────────────────────────────┘\n");
}

void print_test_result(const TestResult& result) {
    if (result.passed) {
        printf("  ✅ PASS: %s (%.0f ms)\n", result.name.c_str(), result.elapsed_ms);
        tests_passed++;
    } else if (result.details.find("SKIP") != std::string::npos) {
        printf("  ⏭️  SKIP: %s — %s\n", result.name.c_str(), result.details.c_str());
        tests_skipped++;
    } else {
        printf("  ❌ FAIL: %s\n", result.name.c_str());
        printf("         %s\n", result.details.c_str());
        tests_failed++;
    }
}

// =============================================================================
// Test 1: Local Model Pipeline (Happy Path)
// =============================================================================
// Tests the full pipeline with a known local model.
// This validates that profiler + fetcher + predictor + matrix all work together.

TestResult test_local_model_pipeline() {
    auto start = std::chrono::high_resolution_clock::now();
    
    std::string model_path = "D:\\projects\\software\\local_llm\\models\\Llama-3.2-3B-Instruct-Q4_K_M.gguf";
    
    printf("  Step 1: Profiling hardware...\n");
    HardwareSpec hw = profile_hardware(model_path);
    
    // Validate hardware profile
    if (hw.ram_total_bytes == 0) {
        return {"Local pipeline", false, "Hardware profiling failed — no RAM detected", 0};
    }
    if (hw.vram_total_bytes == 0) {
        return {"Local pipeline", false, "No GPU detected — cannot test GPU strategies", 0};
    }
    
    printf("    GPU: %s (%.1f GB VRAM, %.1f free)\n", 
           hw.gpu_name.c_str(), hw.vram_total_bytes/1e9, hw.vram_free_bytes/1e9);
    printf("    RAM: %.1f GB total, %.1f free\n", 
           hw.ram_total_bytes/1e9, hw.ram_free_bytes/1e9);
    
    printf("  Step 2: Fetching model metadata (local file)...\n");
    ModelSpec model = fetch_metadata(model_path);
    
    if (model.layers == 0) {
        return {"Local pipeline", false, "Failed to parse GGUF metadata", 0};
    }
    
    printf("    Model: %s | %s | %.2fB params | %u layers | %uK ctx\n",
           model.name.c_str(), model.quant_type.c_str(),
           model.param_count/1e9, model.layers, model.context_length/1024);
    
    printf("  Step 3: Generating strategy matrix...\n");
    std::vector<StrategyResult> results = generate_matrix(hw, model);
    
    if (results.empty()) {
        return {"Local pipeline", false, "No strategies generated", 0};
    }
    
    printf("    %zu strategies generated\n", results.size());
    
    // Validate results
    int viable_count = 0;
    int full_gpu_viable = 0;
    for (const auto& r : results) {
        if (r.prediction.viable) {
            viable_count++;
            if (r.strategy.placement == PlacementStrategy::FULL_GPU) {
                full_gpu_viable++;
            }
        }
    }
    
    printf("    Viable strategies: %d / %zu\n", viable_count, results.size());
    printf("    Full GPU viable: %d\n", full_gpu_viable);
    
    // Print top strategies
    printf("  Top 3 strategies:\n");
    int shown = 0;
    for (const auto& r : results) {
        if (shown >= 3) break;
        if (!r.prediction.viable) continue;
        printf("    %s — ~%.0f tok/s, %.1f GB VRAM\n",
               r.description.c_str(), r.prediction.tokens_per_sec,
               r.prediction.memory_vram_bytes / 1e9);
        shown++;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    
    // Validate expectations
    if (full_gpu_viable == 0) {
        return {"Local pipeline", false, "Expected Full GPU to be viable for 3B model on 8GB card", elapsed};
    }
    if (viable_count < 5) {
        return {"Local pipeline", false, "Expected at least 5 viable strategies", elapsed};
    }
    
    return {"Local pipeline", true, "", elapsed};
}

// =============================================================================
// Test 2: Remote GGUF Fetch (Core Value Proposition)
// =============================================================================
// Tests fetching metadata from a HuggingFace GGUF URL.
// This is the KEY test — it proves you can evaluate a model without downloading it.

TestResult test_remote_gguf_fetch(const TestCase& test_case) {
    auto start = std::chrono::high_resolution_clock::now();
    
    printf("  URL: %s\n", test_case.url.c_str());
    
    printf("  Step 1: Profiling local hardware...\n");
    HardwareSpec hw = profile_hardware("");  // No disk benchmark needed
    
    printf("    GPU: %s (%.1f GB VRAM)\n", hw.gpu_name.c_str(), hw.vram_total_bytes/1e9);
    
    printf("  Step 2: Fetching metadata from HuggingFace (64KB range request)...\n");
    ModelSpec model = fetch_metadata(test_case.url);
    
    auto fetch_end = std::chrono::high_resolution_clock::now();
    double fetch_ms = std::chrono::duration<double, std::milli>(fetch_end - start).count();
    
    if (model.layers == 0) {
        const std::string& err = get_fetch_error();
        int http_status = get_fetch_http_status();
        char buf[256];
        snprintf(buf, sizeof(buf), "Failed to fetch metadata (HTTP %d): %s", 
                 http_status, err.c_str());
        return {"Remote fetch: " + test_case.name, false, buf, fetch_ms};
    }
    
    printf("    Fetched in %.0f ms\n", fetch_ms);
    printf("    Model: %s | %s | %.2fB params | %u layers | %uK ctx\n",
           model.name.c_str(), model.quant_type.c_str(),
           model.param_count/1e9, model.layers, model.context_length/1024);
    
    // Validate architecture
    if (!test_case.expected_arch.empty() && model.architecture != test_case.expected_arch) {
        char buf[128];
        snprintf(buf, sizeof(buf), "Expected arch '%s', got '%s'", 
                 test_case.expected_arch.c_str(), model.architecture.c_str());
        return {"Remote fetch: " + test_case.name, false, buf, fetch_ms};
    }
    
    // Validate layer count
    if (model.layers < test_case.min_layers || model.layers > test_case.max_layers) {
        char buf[128];
        snprintf(buf, sizeof(buf), "Expected layers %u-%u, got %u",
                 test_case.min_layers, test_case.max_layers, model.layers);
        return {"Remote fetch: " + test_case.name, false, buf, fetch_ms};
    }
    
    printf("  Step 3: Generating strategies...\n");
    std::vector<StrategyResult> results = generate_matrix(hw, model);
    
    printf("    %zu strategies generated\n", results.size());
    
    // Count viable
    int viable_count = 0;
    int full_gpu_viable = 0;
    for (const auto& r : results) {
        if (r.prediction.viable) {
            viable_count++;
            if (r.strategy.placement == PlacementStrategy::FULL_GPU) {
                full_gpu_viable++;
            }
        }
    }
    
    printf("    Viable: %d / %zu\n", viable_count, results.size());
    printf("    Full GPU viable: %d\n", full_gpu_viable);
    
    // Print recommendation
    if (!results.empty()) {
        for (const auto& r : results) {
            if (r.prediction.viable) {
                printf("  Recommendation: %s — ~%.0f tok/s\n",
                       r.description.c_str(), r.prediction.tokens_per_sec);
                break;
            }
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    
    // Validate expectations
    if (test_case.expect_viable_full_gpu && full_gpu_viable == 0) {
        return {"Remote fetch: " + test_case.name, false, 
                "Expected Full GPU to be viable but none found", elapsed};
    }
    if (test_case.expect_viable_any && viable_count == 0) {
        return {"Remote fetch: " + test_case.name, false,
                "Expected at least one viable strategy but none found", elapsed};
    }
    
    return {"Remote fetch: " + test_case.name, true, "", elapsed};
}

// =============================================================================
// Test 3: CPU-Only Simulation (Zero VRAM)
// =============================================================================
// Simulates a machine with no GPU by setting VRAM to 0.
// Tests graceful degradation and CPU-only predictions.

TestResult test_cpu_only_simulation() {
    auto start = std::chrono::high_resolution_clock::now();
    
    printf("  Setting VRAM to 0 (simulating no GPU)...\n");
    
    HardwareSpec hw;
    hw.ram_total_bytes = 32ULL * 1024 * 1024 * 1024;  // 32 GB
    hw.ram_free_bytes = 20ULL * 1024 * 1024 * 1024;   // 20 GB free
    hw.ram_bandwidth_gbs = 40.0;
    hw.vram_total_bytes = 0;   // No GPU
    hw.vram_free_bytes = 0;
    hw.gpu_bandwidth_gbs = 0;
    hw.gpu_tflops_fp16 = 0;
    hw.gpu_name = "None (simulated)";
    hw.gpu_compute_major = 0;
    hw.gpu_compute_minor = 0;
    hw.nvme_sequential_mbs = 3000;
    hw.nvme_random_4k_mbs = 100;
    
    // Use a 3B model (known good metadata)
    printf("  Creating model metadata for 3B Q4_K_M...\n");
    ModelSpec model;
    model.architecture = "llama";
    model.name = "Llama 3.2 3B Instruct";
    model.param_count = 3212749824ULL;
    model.layers = 28;
    model.embedding_dim = 3072;
    model.attention_heads = 24;
    model.kv_heads = 8;
    model.head_dim = 128;
    model.ffn_dim = 8192;
    model.context_length = 131072;
    model.quant_type = "Q4_K_M";
    model.bits_per_weight = 4.85;
    model.source = MetadataSource::GGUF_HEADER;
    model.model_type = ModelType::DENSE;
    
    printf("  Generating strategies with zero VRAM...\n");
    std::vector<StrategyResult> results = generate_matrix(hw, model);
    
    // Validate: no division by zero, no crashes
    int viable_count = 0;
    int cpu_only_viable = 0;
    int gpu_viable = 0;
    
    for (const auto& r : results) {
        if (r.prediction.viable) {
            viable_count++;
            if (r.strategy.placement == PlacementStrategy::CPU_ONLY) {
                cpu_only_viable++;
            } else {
                gpu_viable++;
            }
        }
    }
    
    printf("    Total strategies: %zu\n", results.size());
    printf("    Viable: %d (CPU-only: %d, GPU: %d)\n", viable_count, cpu_only_viable, gpu_viable);
    
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    
    // Validate
    if (results.empty()) {
        return {"CPU-only simulation", false, "No strategies generated (possible div-by-zero)", elapsed};
    }
    if (gpu_viable > 0) {
        return {"CPU-only simulation", false, "GPU strategies should not be viable with zero VRAM", elapsed};
    }
    if (cpu_only_viable == 0) {
        return {"CPU-only simulation", false, "CPU-only strategies should be viable", elapsed};
    }
    
    // Print CPU-only results
    printf("  CPU-only viable strategies:\n");
    for (const auto& r : results) {
        if (r.prediction.viable && r.strategy.placement == PlacementStrategy::CPU_ONLY) {
            printf("    %s — ~%.1f tok/s, %.1f GB RAM\n",
                   r.description.c_str(), r.prediction.tokens_per_sec,
                   r.prediction.memory_ram_bytes / 1e9);
        }
    }
    
    return {"CPU-only simulation", true, "", elapsed};
}

// =============================================================================
// Test 4: Tight Fit Detection
// =============================================================================
// Tests that the tight-fit warning appears for strategies using >90% VRAM.

TestResult test_tight_fit_detection() {
    auto start = std::chrono::high_resolution_clock::now();
    
    printf("  Creating hardware with 4GB VRAM (tight for 3B model)...\n");
    
    HardwareSpec hw;
    hw.ram_total_bytes = 32ULL * 1024 * 1024 * 1024;
    hw.ram_free_bytes = 24ULL * 1024 * 1024 * 1024;
    hw.ram_bandwidth_gbs = 40.0;
    hw.vram_total_bytes = 4ULL * 1024 * 1024 * 1024;   // 4 GB VRAM (tight)
    hw.vram_free_bytes = 3500ULL * 1024 * 1024;         // 3.5 GB free
    hw.gpu_bandwidth_gbs = 300.0;
    hw.gpu_tflops_fp16 = 15.0;
    hw.gpu_name = "Simulated 4GB GPU";
    hw.gpu_compute_major = 8;
    hw.gpu_compute_minor = 6;
    hw.nvme_sequential_mbs = 3000;
    hw.nvme_random_4k_mbs = 100;
    
    ModelSpec model;
    model.architecture = "llama";
    model.name = "Llama 3.2 3B Instruct";
    model.param_count = 3212749824ULL;
    model.layers = 28;
    model.embedding_dim = 3072;
    model.attention_heads = 24;
    model.kv_heads = 8;
    model.head_dim = 128;
    model.ffn_dim = 8192;
    model.context_length = 131072;
    model.quant_type = "Q4_K_M";
    model.bits_per_weight = 4.85;
    model.source = MetadataSource::GGUF_HEADER;
    model.model_type = ModelType::DENSE;
    
    printf("  Generating strategies...\n");
    std::vector<StrategyResult> results = generate_matrix(hw, model);
    
    // Check for tight-fit detection
    // Full GPU at 4K should need ~2.5 GB VRAM on 3.5 GB free = ~71% → VIABLE
    // Full GPU at max context should need more → TIGHT or NO FIT
    
    int tight_count = 0;
    int no_fit_count = 0;
    int viable_count = 0;
    
    printf("  Strategy analysis:\n");
    for (const auto& r : results) {
        bool fits_vram = r.prediction.memory_vram_bytes <= hw.vram_free_bytes;
        bool fits_ram = r.prediction.memory_ram_bytes <= hw.ram_free_bytes;
        double vram_ratio = hw.vram_free_bytes > 0 ? 
            (double)r.prediction.memory_vram_bytes / hw.vram_free_bytes : 0;
        
        const char* status = "VIABLE";
        if (!fits_vram && !fits_ram) { status = "NO FIT"; no_fit_count++; }
        else if (vram_ratio > 0.9) { status = "TIGHT"; tight_count++; }
        else { viable_count++; }
        
        if (r.strategy.placement == PlacementStrategy::FULL_GPU) {
            printf("    Full GPU %uK: %.1f GB VRAM (%.0f%%) → %s\n",
                   r.strategy.context_length / 1024,
                   r.prediction.memory_vram_bytes / 1e9,
                   vram_ratio * 100, status);
        }
    }
    
    printf("  Summary: %d viable, %d tight, %d no fit\n", 
           viable_count, tight_count, no_fit_count);
    
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    
    // With 3.5 GB free and ~2.5 GB for full GPU 4K, it should be viable
    // With max context, it should be tight or no fit
    if (viable_count == 0) {
        return {"Tight fit detection", false, "Expected some viable strategies", elapsed};
    }
    if (tight_count == 0 && no_fit_count == 0) {
        return {"Tight fit detection", false, "Expected some tight or no-fit strategies at max context", elapsed};
    }
    
    return {"Tight fit detection", true, "", elapsed};
}

// =============================================================================
// Test 5: Invalid URL Handling
// =============================================================================
// Tests that invalid URLs are handled gracefully without crashes.

TestResult test_invalid_url_handling() {
    auto start = std::chrono::high_resolution_clock::now();
    
    struct InvalidUrlCase {
        std::string url;
        std::string description;
    };
    
    std::vector<InvalidUrlCase> cases = {
        {"https://huggingface.co/nonexistent/model/file.gguf", "Non-existent repo"},
        {"https://httpbin.org/status/404", "Non-GGUF URL (returns HTML)"},
        {"not-a-url", "Plain text string"},
    };
    
    int handled_gracefully = 0;
    
    for (const auto& tc : cases) {
        printf("  Testing: %s (%s)\n", tc.description.c_str(), tc.url.c_str());
        
        // This should NOT crash
        ModelSpec model = fetch_metadata(tc.url);
        
        if (model.layers == 0) {
            printf("    → Correctly returned empty model (error handled)\n");
            handled_gracefully++;
        } else {
            printf("    → Unexpectedly got model data\n");
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    
    if (handled_gracefully < cases.size()) {
        char buf[128];
        snprintf(buf, sizeof(buf), "Only %d/%zu invalid URLs handled gracefully",
                 handled_gracefully, cases.size());
        return {"Invalid URL handling", false, buf, elapsed};
    }
    
    return {"Invalid URL handling", true, "", elapsed};
}

// =============================================================================
// Test 6: Prediction Consistency
// =============================================================================
// Tests that the same inputs always produce the same outputs (pure function).

TestResult test_prediction_consistency() {
    auto start = std::chrono::high_resolution_clock::now();
    
    HardwareSpec hw;
    hw.ram_total_bytes = 32ULL * 1024 * 1024 * 1024;
    hw.ram_free_bytes = 20ULL * 1024 * 1024 * 1024;
    hw.ram_bandwidth_gbs = 40.0;
    hw.vram_total_bytes = 8ULL * 1024 * 1024 * 1024;
    hw.vram_free_bytes = 7ULL * 1024 * 1024 * 1024;
    hw.gpu_bandwidth_gbs = 448.0;
    hw.gpu_tflops_fp16 = 20.0;
    hw.gpu_name = "Test GPU";
    hw.gpu_compute_major = 8;
    hw.gpu_compute_minor = 6;
    hw.nvme_sequential_mbs = 3000;
    hw.nvme_random_4k_mbs = 100;
    
    ModelSpec model;
    model.architecture = "llama";
    model.name = "Test Model";
    model.param_count = 3212749824ULL;
    model.layers = 28;
    model.embedding_dim = 3072;
    model.attention_heads = 24;
    model.kv_heads = 8;
    model.head_dim = 128;
    model.ffn_dim = 8192;
    model.context_length = 131072;
    model.quant_type = "Q4_K_M";
    model.bits_per_weight = 4.85;
    model.source = MetadataSource::GGUF_HEADER;
    model.model_type = ModelType::DENSE;
    
    StrategyConfig strat;
    strat.placement = PlacementStrategy::FULL_GPU;
    strat.gpu_layers = 28;
    strat.context_length = 4096;
    strat.batch_size = 1;
    strat.kv_quant_bits = 16;
    
    printf("  Running predict() 100 times with identical inputs...\n");
    
    Prediction first = predict(hw, model, strat);
    bool all_match = true;
    
    for (int i = 0; i < 100; i++) {
        Prediction p = predict(hw, model, strat);
        if (p.memory_total_bytes != first.memory_total_bytes ||
            p.tokens_per_sec != first.tokens_per_sec ||
            p.viable != first.viable) {
            all_match = false;
            break;
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    
    printf("  First run: %.1f GB, ~%.0f tok/s, viable=%s\n",
           first.memory_total_bytes / 1e9, first.tokens_per_sec,
           first.viable ? "yes" : "no");
    printf("  All 100 runs identical: %s\n", all_match ? "yes" : "NO");
    
    if (!all_match) {
        return {"Prediction consistency", false, "Predictions vary across identical runs", elapsed};
    }
    
    return {"Prediction consistency", true, "", elapsed};
}

// =============================================================================
// Test 7: Ranker — Speed Priority
// =============================================================================
// Verify: Full GPU ranked #1, CPU-only near bottom, non-viable at very bottom
// =============================================================================

TestResult test_ranker_speed() {
    auto start = std::chrono::high_resolution_clock::now();
    
    std::string model_path = "D:\\projects\\software\\local_llm\\models\\Llama-3.2-3B-Instruct-Q4_K_M.gguf";
    HardwareSpec hw = profile_hardware(model_path);
    ModelSpec model = fetch_metadata(model_path);
    if (model.layers == 0) return {"Ranker speed", false, "Metadata fetch failed", 0};
    
    std::vector<StrategyResult> results = generate_matrix(hw, model);
    sort_by_priority(results, PriorityMode::SPEED, hw);
    
    // Find first viable
    const StrategyResult* first_viable = nullptr;
    for (const auto& r : results) {
        if (r.prediction.viable) { first_viable = &r; break; }
    }
    if (!first_viable) return {"Ranker speed", false, "No viable strategies", 0};
    
    // Verify: Full GPU should be #1 (fastest)
    bool full_gpu_first = (first_viable->strategy.placement == PlacementStrategy::FULL_GPU);
    
    // Verify: Non-viable at bottom
    bool non_viable_at_bottom = true;
    bool seen_viable = false;
    for (auto it = results.rbegin(); it != results.rend(); ++it) {
        if (it->prediction.viable) { seen_viable = true; continue; }
        if (seen_viable) { non_viable_at_bottom = false; break; }
    }
    
    // Verify: CPU-only ranked lower than Full GPU
    bool cpu_below_gpu = true;
    int gpu_rank = -1, cpu_rank = -1;
    for (size_t i = 0; i < results.size(); i++) {
        if (results[i].strategy.placement == PlacementStrategy::FULL_GPU && gpu_rank < 0)
            gpu_rank = static_cast<int>(i);
        if (results[i].strategy.placement == PlacementStrategy::CPU_ONLY && cpu_rank < 0)
            cpu_rank = static_cast<int>(i);
    }
    if (gpu_rank >= 0 && cpu_rank >= 0) cpu_below_gpu = (cpu_rank > gpu_rank);
    
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    
    if (!full_gpu_first)
        return {"Ranker speed", false, "Full GPU not ranked #1 for speed priority", elapsed};
    if (!cpu_below_gpu)
        return {"Ranker speed", false, "CPU-only ranked above Full GPU", elapsed};
    if (!non_viable_at_bottom)
        return {"Ranker speed", false, "Non-viable strategies not at bottom", elapsed};
    
    return {"Ranker speed", true, "", elapsed};
}

// =============================================================================
// Test 8: Ranker — Quality Priority
// =============================================================================
// Verify: FP16 KV ranks above Q8 KV, higher GPU layers rank higher
// =============================================================================

TestResult test_ranker_quality() {
    auto start = std::chrono::high_resolution_clock::now();
    
    std::string model_path = "D:\\projects\\software\\local_llm\\models\\Llama-3.2-3B-Instruct-Q4_K_M.gguf";
    HardwareSpec hw = profile_hardware(model_path);
    ModelSpec model = fetch_metadata(model_path);
    if (model.layers == 0) return {"Ranker quality", false, "Metadata fetch failed", 0};
    
    std::vector<StrategyResult> results = generate_matrix(hw, model);
    sort_by_priority(results, PriorityMode::QUALITY, hw);
    
    // Find first FP16 and first Q8 among viable
    int fp16_rank = -1, q8_rank = -1;
    for (size_t i = 0; i < results.size(); i++) {
        if (!results[i].prediction.viable) continue;
        if (results[i].strategy.kv_quant_bits == 16 && fp16_rank < 0)
            fp16_rank = static_cast<int>(i);
        if (results[i].strategy.kv_quant_bits == 8 && q8_rank < 0)
            q8_rank = static_cast<int>(i);
    }
    
    bool fp16_above_q8 = (fp16_rank >= 0 && q8_rank >= 0) ? (fp16_rank < q8_rank) : true;
    
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    
    if (!fp16_above_q8)
        return {"Ranker quality", false, "Q8 KV ranked above FP16 KV for quality", elapsed};
    
    return {"Ranker quality", true, "", elapsed};
}

// =============================================================================
// Test 9: Ranker — Safety Priority
// =============================================================================
// Verify: Strategies with most headroom rank highest
// =============================================================================

TestResult test_ranker_safety() {
    auto start = std::chrono::high_resolution_clock::now();
    
    std::string model_path = "D:\\projects\\software\\local_llm\\models\\Llama-3.2-3B-Instruct-Q4_K_M.gguf";
    HardwareSpec hw = profile_hardware(model_path);
    ModelSpec model = fetch_metadata(model_path);
    if (model.layers == 0) return {"Ranker safety", false, "Metadata fetch failed", 0};
    
    std::vector<StrategyResult> results = generate_matrix(hw, model);
    sort_by_priority(results, PriorityMode::SAFETY, hw);
    
    // Verify: top viable strategy has the most headroom
    // (scoring uses weighted combo, so strict global ordering isn't guaranteed,
    //  but the #1 strategy should have top-tier headroom)
    const StrategyResult* best = nullptr;
    for (const auto& r : results) {
        if (r.prediction.viable) { best = &r; break; }
    }
    if (!best) return {"Ranker safety", false, "No viable strategies", 0};
    
    double best_headroom = calculate_memory_headroom(hw, best->prediction);
    
    // Check that no viable strategy has significantly more headroom (>10% more)
    bool best_is_top = true;
    for (const auto& r : results) {
        if (!r.prediction.viable) continue;
        double h = calculate_memory_headroom(hw, r.prediction);
        if (h > best_headroom + 0.10) {
            best_is_top = false;
            break;
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    
    if (!best_is_top)
        return {"Ranker safety", false, "Strategy with significantly more headroom not ranked #1", elapsed};
    
    return {"Ranker safety", true, "", elapsed};
}

// =============================================================================
// Test 10: Ranker — Priority Changes Order
// =============================================================================
// Verify: Same model with 3 priorities produces different orderings
// =============================================================================

TestResult test_ranker_order_differs() {
    auto start = std::chrono::high_resolution_clock::now();
    
    std::string model_path = "D:\\projects\\software\\local_llm\\models\\Llama-3.2-3B-Instruct-Q4_K_M.gguf";
    HardwareSpec hw = profile_hardware(model_path);
    ModelSpec model = fetch_metadata(model_path);
    if (model.layers == 0) return {"Ranker order", false, "Metadata fetch failed", 0};
    
    // Generate same matrix, sort with 3 priorities
    auto base = generate_matrix(hw, model);
    
    auto speed_order = base;
    sort_by_priority(speed_order, PriorityMode::SPEED, hw);
    
    auto quality_order = base;
    sort_by_priority(quality_order, PriorityMode::QUALITY, hw);
    
    auto safety_order = base;
    sort_by_priority(safety_order, PriorityMode::SAFETY, hw);
    
    // Compare first viable strategy across priorities
    auto first_viable = [](const std::vector<StrategyResult>& v) -> const StrategyResult* {
        for (const auto& r : v) if (r.prediction.viable) return &r;
        return nullptr;
    };
    
    const StrategyResult* s = first_viable(speed_order);
    const StrategyResult* q = first_viable(quality_order);
    const StrategyResult* sa = first_viable(safety_order);
    
    // At least two of the three should differ
    bool speed_diff_quality = (s && q) && (s->strategy.gpu_layers != q->strategy.gpu_layers
                                         || s->strategy.kv_quant_bits != q->strategy.kv_quant_bits);
    bool speed_diff_safety = (s && sa) && (s->strategy.placement != sa->strategy.placement
                                         || s->strategy.gpu_layers != sa->strategy.gpu_layers);
    
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    
    if (!speed_diff_quality && !speed_diff_safety)
        return {"Ranker order", false, "All 3 priorities produce same #1 strategy", elapsed};
    
    return {"Ranker order", true, "", elapsed};
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char* argv[]) {
    bool skip_network = false;
    bool local_only = false;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--skip-network") == 0 || strcmp(argv[i], "--local-only") == 0) {
            skip_network = true;
            local_only = true;
        }
    }
    
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║   LLM Deployment Planner — End-to-End Test Suite (Phase G) ║\n");
    printf("║                                                             ║\n");
    printf("║   Testing: Fetch metadata → Predict hardware fit            ║\n");
    printf("║   WITHOUT downloading any models                            ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n");
    
    // =========================================================================
    // Test 1: Local Model Pipeline
    // =========================================================================
    print_test_header("1. Local Model Pipeline (Happy Path)");
    TestResult r1 = test_local_model_pipeline();
    print_test_result(r1);
    
    // =========================================================================
    // Test 2: Remote GGUF Fetch — Multiple Models
    // =========================================================================
    if (!skip_network) {
        // Test matrix: different sizes and architectures
        std::vector<TestCase> test_cases = {
            {
                "Llama 3.2 3B Q4_K_M",
                "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
                "llama", 25, 30,    // layers range
                true,   // Full GPU should fit on 8GB
                true,   // Any strategy should work
                true    // needs network
            },
            {
                "Qwen2.5 7B Q4_K_M",
                "https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
                "qwen2", 25, 30,
                false,  // 7B may not fully fit on 8GB GPU at 128K
                true,   // But some strategy should work
                true
            },
            {
                "Phi-3.5 Mini Q4_K_M",
                "https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF/resolve/main/Phi-3.5-mini-instruct-Q4_K_M.gguf",
                "phi3", 28, 36,
                true,   // 3.8B should fit on 8GB
                true,
                true
            },
        };
        
        for (const auto& tc : test_cases) {
            print_test_header(("2." + tc.name).c_str());
            TestResult r = test_remote_gguf_fetch(tc);
            print_test_result(r);
        }
    } else {
        printf("\n⏭️  Skipping network tests (--skip-network)\n");
        tests_skipped += 3;
    }
    
    // =========================================================================
    // Test 3: CPU-Only Simulation
    // =========================================================================
    print_test_header("3. CPU-Only Simulation (Zero VRAM)");
    TestResult r3 = test_cpu_only_simulation();
    print_test_result(r3);
    
    // =========================================================================
    // Test 4: Tight Fit Detection
    // =========================================================================
    print_test_header("4. Tight Fit Detection (4GB VRAM)");
    TestResult r4 = test_tight_fit_detection();
    print_test_result(r4);
    
    // =========================================================================
    // Test 5: Invalid URL Handling
    // =========================================================================
    if (!skip_network) {
        print_test_header("5. Invalid URL Handling");
        TestResult r5 = test_invalid_url_handling();
        print_test_result(r5);
    } else {
        printf("\n⏭️  Skipping invalid URL tests (--skip-network)\n");
        tests_skipped++;
    }
    
    // =========================================================================
    // Test 6: Prediction Consistency
    // =========================================================================
    print_test_header("6. Prediction Consistency (Pure Function)");
    TestResult r6 = test_prediction_consistency();
    print_test_result(r6);
    
    // =========================================================================
    // Test 7: Ranker — Speed Priority
    // =========================================================================
    print_test_header("7. Ranker — Speed Priority");
    TestResult r7 = test_ranker_speed();
    print_test_result(r7);
    
    // =========================================================================
    // Test 8: Ranker — Quality Priority
    // =========================================================================
    print_test_header("8. Ranker — Quality Priority");
    TestResult r8 = test_ranker_quality();
    print_test_result(r8);
    
    // =========================================================================
    // Test 9: Ranker — Safety Priority
    // =========================================================================
    print_test_header("9. Ranker — Safety Priority");
    TestResult r9 = test_ranker_safety();
    print_test_result(r9);
    
    // =========================================================================
    // Test 10: Ranker — Priority Changes Order
    // =========================================================================
    print_test_header("10. Ranker — Priority Changes Order");
    TestResult r10 = test_ranker_order_differs();
    print_test_result(r10);
    
    // =========================================================================
    // Summary
    // =========================================================================
    printf("\n╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║                      TEST SUMMARY                           ║\n");
    printf("╠═══════════════════════════════════════════════════════════════╣\n");
    printf("║  ✅ Passed:  %-46d║\n", tests_passed);
    printf("║  ❌ Failed:  %-46d║\n", tests_failed);
    printf("║  ⏭️  Skipped: %-45d║\n", tests_skipped);
    printf("║  Total:      %-46d║\n", tests_passed + tests_failed + tests_skipped);
    printf("╚═══════════════════════════════════════════════════════════════╝\n");
    
    if (tests_failed == 0) {
        printf("\n🎉 All tests passed! The tool can evaluate models without downloading them.\n");
    } else {
        printf("\n⚠️  %d test(s) failed. Review the failures above.\n", tests_failed);
    }
    
    return tests_failed > 0 ? 1 : 0;
}
