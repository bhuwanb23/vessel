// =============================================================================
// Recommendation Engine Test (Step 12, Phase B)
// =============================================================================

#include "recommend/recommendation_engine.h"
#include "recommend/catalog_loader.h"
#include "recommend/catalog_types.h"
#include "../predictor/predictor.h"
#include "../predictor/memory_predictor.h"
#include <cstdio>
#include <cmath>

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

// Mock hardware spec for testing
HardwareSpec create_mock_hw(uint64_t vram_gb, uint64_t ram_gb) {
    HardwareSpec hw;
    hw.platform = Platform::NVIDIA_WINDOWS;
    hw.backend = ComputeBackend::CUDA;
    hw.memory_arch = MemoryArchitecture::DISCRETE;
    hw.is_unified_memory = false;
    hw.gpu_name = "Mock GPU";
    hw.vram_total_bytes = vram_gb * 1024ULL * 1024 * 1024;
    hw.vram_free_bytes = (vram_gb - 1) * 1024ULL * 1024 * 1024;  // 1GB used by OS
    hw.ram_total_bytes = ram_gb * 1024ULL * 1024 * 1024;
    hw.ram_free_bytes = (ram_gb - 4) * 1024ULL * 1024 * 1024;  // 4GB used by OS
    hw.gpu_bandwidth_gbs = 500.0;
    hw.gpu_tflops_fp16 = 25.0;
    hw.ram_bandwidth_gbs = 40.0;
    hw.nvme_sequential_mbs = 5000.0;
    hw.nvme_random_4k_mbs = 800.0;
    hw.gpu_compute_major = 8;
    hw.gpu_compute_minor = 6;
    hw.compute_capability = "sm_86";
    hw.hardware_fingerprint = "Mock GPU|32GB";
    return hw;
}

int main() {
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║     Step 12 Phase B — Recommendation Engine Tests           ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n\n");
    
    // Load catalog
    ModelCatalog catalog = load_builtin_catalog();
    TEST_ASSERT(!catalog.models.empty(), "Catalog loaded");
    
    // =========================================================================
    // Test 1: Basic recommendation on 10GB VRAM / 32GB RAM
    // =========================================================================
    printf("=== Test 1: Basic Recommendation (10GB VRAM / 32GB RAM) ===\n");
    {
        HardwareSpec hw = create_mock_hw(10, 32);
        RecommendationRequest req;
        req.priority = "balanced";
        req.use_case = "all";
        req.top_n = 8;
        
        auto recs = generate_recommendations(hw, catalog, req);
        
        TEST_ASSERT(!recs.empty(), "Got recommendations");
        TEST_ASSERT(recs.size() <= 8, "At most 8 recommendations");
        
        printf("  Recommendations: %zu\n", recs.size());
        for (size_t i = 0; i < recs.size() && i < 5; i++) {
            printf("    %zu. %s %s — %.1f tok/s, %s\n",
                   i + 1, recs[i].model.name.c_str(),
                   recs[i].variant.quant.c_str(),
                   recs[i].predicted_tok_s,
                   recs[i].label.c_str());
        }
    }
    
    // =========================================================================
    // Test 2: Speed priority
    // =========================================================================
    printf("\n=== Test 2: Speed Priority ===\n");
    {
        HardwareSpec hw = create_mock_hw(10, 32);
        RecommendationRequest req;
        req.priority = "speed";
        req.use_case = "all";
        req.top_n = 5;
        
        auto recs = generate_recommendations(hw, catalog, req);
        
        TEST_ASSERT(!recs.empty(), "Got speed recommendations");
        
        // Check that smaller models rank higher for speed
        if (recs.size() >= 2) {
            bool speed_order = recs[0].predicted_tok_s >= recs[1].predicted_tok_s;
            TEST_ASSERT(speed_order, "Faster models rank higher");
        }
        
        printf("  Top 3 by speed:\n");
        for (size_t i = 0; i < recs.size() && i < 3; i++) {
            printf("    %zu. %s — %.1f tok/s\n",
                   i + 1, recs[i].model.name.c_str(), recs[i].predicted_tok_s);
        }
    }
    
    // =========================================================================
    // Test 3: Quality priority
    // =========================================================================
    printf("\n=== Test 3: Quality Priority ===\n");
    {
        HardwareSpec hw = create_mock_hw(10, 32);
        RecommendationRequest req;
        req.priority = "quality";
        req.use_case = "all";
        req.top_n = 5;
        
        auto recs = generate_recommendations(hw, catalog, req);
        
        TEST_ASSERT(!recs.empty(), "Got quality recommendations");
        
        // Check that higher quality models rank higher
        if (recs.size() >= 2) {
            bool quality_order = recs[0].model.quality_score >= recs[1].model.quality_score;
            TEST_ASSERT(quality_order, "Higher quality models rank higher");
        }
        
        printf("  Top 3 by quality:\n");
        for (size_t i = 0; i < recs.size() && i < 3; i++) {
            printf("    %zu. %s — Quality: %.1f\n",
                   i + 1, recs[i].model.name.c_str(), recs[i].model.quality_score);
        }
    }
    
    // =========================================================================
    // Test 4: Use case filter
    // =========================================================================
    printf("\n=== Test 4: Use Case Filter (coding) ===\n");
    {
        HardwareSpec hw = create_mock_hw(24, 64);
        RecommendationRequest req;
        req.priority = "balanced";
        req.use_case = "coding";
        req.top_n = 10;
        
        auto recs = generate_recommendations(hw, catalog, req);
        
        TEST_ASSERT(!recs.empty(), "Got coding recommendations");
        
        // Check all recommendations are tagged for coding
        bool all_coding = true;
        for (const auto& rec : recs) {
            bool is_coding = false;
            for (const auto& uc : rec.model.use_cases) {
                if (uc == "coding") is_coding = true;
            }
            if (!is_coding) all_coding = false;
        }
        TEST_ASSERT(all_coding, "All recommendations are tagged 'coding'");
        
        printf("  Coding models: %zu\n", recs.size());
        for (size_t i = 0; i < recs.size() && i < 3; i++) {
            printf("    %zu. %s\n", i + 1, recs[i].model.name.c_str());
        }
    }
    
    // =========================================================================
    // Test 5: Download size limit
    // =========================================================================
    printf("\n=== Test 5: Download Size Limit (3GB) ===\n");
    {
        HardwareSpec hw = create_mock_hw(10, 32);
        RecommendationRequest req;
        req.priority = "balanced";
        req.use_case = "all";
        req.max_download_gb = 3.0;
        req.top_n = 10;
        
        auto recs = generate_recommendations(hw, catalog, req);
        
        TEST_ASSERT(!recs.empty(), "Got recommendations under 3GB");
        
        // Check all recommendations are under 3GB
        bool all_under_limit = true;
        for (const auto& rec : recs) {
            if (rec.variant.file_size_gb > 3.0) all_under_limit = false;
        }
        TEST_ASSERT(all_under_limit, "All recommendations under 3GB");
        
        printf("  Small models: %zu\n", recs.size());
        for (size_t i = 0; i < recs.size() && i < 3; i++) {
            printf("    %zu. %s — %.1f GB\n",
                   i + 1, recs[i].model.name.c_str(), recs[i].variant.file_size_gb);
        }
    }
    
    // =========================================================================
    // Test 6: Constrained hardware (2GB VRAM, 8GB RAM)
    // =========================================================================
    printf("\n=== Test 6: Constrained Hardware (2GB VRAM / 8GB RAM) ===\n");
    {
        HardwareSpec hw = create_mock_hw(2, 8);
        RecommendationRequest req;
        req.priority = "balanced";
        req.use_case = "all";
        req.top_n = 10;
        
        auto recs = generate_recommendations(hw, catalog, req);
        
        // Should get some recommendations (small models)
        printf("  Recommendations: %zu\n", recs.size());
        
        // Check that only small models appear
        bool all_small = true;
        for (const auto& rec : recs) {
            if (rec.variant.file_size_gb > 2.0) all_small = false;
        }
        if (!recs.empty()) {
            TEST_ASSERT(all_small, "Only small models on constrained hardware");
        }
        
        for (size_t i = 0; i < recs.size() && i < 3; i++) {
            printf("    %zu. %s — %.1f GB\n",
                   i + 1, recs[i].model.name.c_str(), recs[i].variant.file_size_gb);
        }
    }
    
    // =========================================================================
    // Test 7: Powerful hardware (24GB VRAM, 64GB RAM)
    // =========================================================================
    printf("\n=== Test 7: Powerful Hardware (24GB VRAM / 64GB RAM) ===\n");
    {
        HardwareSpec hw = create_mock_hw(24, 64);
        RecommendationRequest req;
        req.priority = "balanced";
        req.use_case = "all";
        req.top_n = 10;
        
        auto recs = generate_recommendations(hw, catalog, req);
        
        TEST_ASSERT(recs.size() >= 5, "Got many recommendations on powerful hardware");
        
        // Check that larger models appear
        bool has_large = false;
        for (const auto& rec : recs) {
            if (rec.variant.file_size_gb > 8.0) has_large = true;
        }
        TEST_ASSERT(has_large, "Large models appear on powerful hardware");
        
        printf("  Recommendations: %zu\n", recs.size());
        for (size_t i = 0; i < recs.size() && i < 5; i++) {
            printf("    %zu. %s — %.1f GB, %.1f tok/s\n",
                   i + 1, recs[i].model.name.c_str(),
                   recs[i].variant.file_size_gb,
                   recs[i].predicted_tok_s);
        }
    }
    
    // =========================================================================
    // Test 8: Catalog-to-metadata conversion
    // =========================================================================
    printf("\n=== Test 8: Catalog-to-Metadata Conversion ===\n");
    {
        // Find a model in the catalog
        const auto& model = catalog.models[0];
        const auto& variant = model.variants[0];
        
        ModelSpec meta = catalog_to_model_spec(model, variant);
        
        TEST_ASSERT(meta.layers == model.dimensions.layers, "Layers match");
        TEST_ASSERT(meta.embedding_dim == model.dimensions.embedding_dim, "Embedding dim matches");
        TEST_ASSERT(meta.param_count > 0, "Param count > 0");
        TEST_ASSERT(meta.bits_per_weight > 0, "BPW > 0");
    }
    
    // =========================================================================
    // Test 9: Scoring function
    // =========================================================================
    printf("\n=== Test 9: Scoring Function ===\n");
    {
        Prediction pred;
        pred.tokens_per_sec = 50.0;
        pred.ttft_ms = 100.0;
        
        double speed_score = score_for_priority(pred, 7.0, "speed");
        double quality_score = score_for_priority(pred, 7.0, "quality");
        double balanced_score = score_for_priority(pred, 7.0, "balanced");
        
        TEST_ASSERT(speed_score > 0, "Speed score > 0");
        TEST_ASSERT(quality_score > 0, "Quality score > 0");
        TEST_ASSERT(balanced_score > 0, "Balanced score > 0");
        
        // Speed score should be higher for fast models
        Prediction fast_pred = pred;
        fast_pred.tokens_per_sec = 100.0;
        double fast_speed_score = score_for_priority(fast_pred, 7.0, "speed");
        TEST_ASSERT(fast_speed_score > speed_score, "Faster model gets higher speed score");
        
        printf("  Speed score: %.2f\n", speed_score);
        printf("  Quality score: %.2f\n", quality_score);
        printf("  Balanced score: %.2f\n", balanced_score);
    }
    
    // =========================================================================
    // Test 10: Label assignment
    // =========================================================================
    printf("\n=== Test 10: Label Assignment ===\n");
    {
        HardwareSpec hw = create_mock_hw(10, 32);
        RecommendationRequest req;
        req.priority = "balanced";
        req.use_case = "all";
        req.top_n = 5;
        
        auto recs = generate_recommendations(hw, catalog, req);
        
        if (!recs.empty()) {
            TEST_ASSERT(!recs[0].label.empty(), "First recommendation has label");
            TEST_ASSERT(recs[0].label.find("Best Overall") != std::string::npos,
                       "First recommendation is 'Best Overall'");
        }
        
        // Check that at least one recommendation has a label
        bool any_labeled = false;
        for (const auto& rec : recs) {
            if (!rec.label.empty()) any_labeled = true;
        }
        TEST_ASSERT(any_labeled, "At least one recommendation has a label");
    }
    
    // =========================================================================
    // Test 11: Mutual exclusion (recommend vs model)
    // =========================================================================
    printf("\n=== Test 11: Top-N Limit ===\n");
    {
        HardwareSpec hw = create_mock_hw(24, 64);
        RecommendationRequest req;
        req.priority = "balanced";
        req.use_case = "all";
        req.top_n = 3;
        
        auto recs = generate_recommendations(hw, catalog, req);
        
        TEST_ASSERT(recs.size() <= 3, "Top-N limit respected");
    }
    
    // =========================================================================
    // Summary
    // =========================================================================
    printf("\n╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║                     TEST SUMMARY                           ║\n");
    printf("╠═══════════════════════════════════════════════════════════════╣\n");
    printf("║  ✅ Passed:  %-3d                                           ║\n", tests_passed);
    printf("║  ❌ Failed:  %-3d                                           ║\n", tests_failed);
    printf("║  Total:      %-3d                                           ║\n", tests_passed + tests_failed);
    printf("╚═══════════════════════════════════════════════════════════════╝\n");
    
    if (tests_failed == 0) {
        printf("\n🎉 All recommendation engine tests passed!\n");
    } else {
        printf("\n⚠️  %d test(s) failed.\n", tests_failed);
    }
    
    return tests_failed > 0 ? 1 : 0;
}
