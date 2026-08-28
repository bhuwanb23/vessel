// =============================================================================
// Model Catalog Test (Step 12, Phase A)
// =============================================================================
// Tests catalog loading, parsing, and validation
// =============================================================================

#include "recommend/catalog_types.h"
#include "recommend/catalog_loader.h"
#include <cstdio>
#include <cstring>
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

int main() {
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║     Step 12 Phase A — Model Catalog Tests                   ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n\n");
    
    // =========================================================================
    // Test 1: Load built-in catalog
    // =========================================================================
    printf("=== Test 1: Load Built-in Catalog ===\n");
    ModelCatalog catalog = load_builtin_catalog();
    
    TEST_ASSERT(!catalog.models.empty(), "Built-in catalog loaded");
    printf("  Catalog version: %s\n", catalog.version.c_str());
    printf("  Total models: %zu\n", catalog.models.size());
    
    // =========================================================================
    // Test 2: Validate catalog structure
    // =========================================================================
    printf("\n=== Test 2: Validate Catalog Structure ===\n");
    bool valid = validate_catalog(catalog);
    TEST_ASSERT(valid, "Catalog structure is valid");
    
    // =========================================================================
    // Test 3: Check model families
    // =========================================================================
    printf("\n=== Test 3: Check Model Families ===\n");
    int families = 0;
    for (const auto& model : catalog.models) {
        if (model.family == "llama") families++;
        if (model.family == "qwen") families++;
        if (model.family == "mistral") families++;
        if (model.family == "phi") families++;
        if (model.family == "gemma") families++;
        if (model.family == "deepseek") families++;
        if (model.family == "mixtral") families++;
    }
    TEST_ASSERT(families >= 5, "At least 5 different families");
    
    // =========================================================================
    // Test 4: Check variant counts
    // =========================================================================
    printf("\n=== Test 4: Check Variant Counts ===\n");
    int total_variants = 0;
    int moe_models = 0;
    for (const auto& model : catalog.models) {
        total_variants += model.variants.size();
        if (model.is_moe) moe_models++;
    }
    TEST_ASSERT(total_variants >= 15, "At least 15 variants");
    TEST_ASSERT(total_variants <= 30, "At most 30 variants");
    TEST_ASSERT(moe_models >= 1, "At least 1 MoE model");
    
    printf("  Total variants: %d\n", total_variants);
    printf("  MoE models: %d\n", moe_models);
    
    // =========================================================================
    // Test 5: Check quality scores
    // =========================================================================
    printf("\n=== Test 5: Check Quality Scores ===\n");
    double min_quality = 10.0, max_quality = 0.0;
    for (const auto& model : catalog.models) {
        if (model.quality_score < min_quality) min_quality = model.quality_score;
        if (model.quality_score > max_quality) max_quality = model.quality_score;
    }
    TEST_ASSERT(min_quality >= 0.0, "Quality scores >= 0");
    TEST_ASSERT(max_quality <= 10.0, "Quality scores <= 10");
    TEST_ASSERT(min_quality < max_quality, "Quality scores vary");
    
    printf("  Quality range: %.1f — %.1f\n", min_quality, max_quality);
    
    // =========================================================================
    // Test 6: Check file sizes
    // =========================================================================
    printf("\n=== Test 6: Check File Sizes ===\n");
    double min_size = 100.0, max_size = 0.0;
    for (const auto& model : catalog.models) {
        for (const auto& v : model.variants) {
            if (v.file_size_gb < min_size) min_size = v.file_size_gb;
            if (v.file_size_gb > max_size) max_size = v.file_size_gb;
        }
    }
    TEST_ASSERT(min_size > 0.0, "File sizes > 0");
    TEST_ASSERT(max_size > min_size, "File sizes vary");
    
    printf("  Size range: %.1f GB — %.1f GB\n", min_size, max_size);
    
    // =========================================================================
    // Test 7: Check dimensions
    // =========================================================================
    printf("\n=== Test 7: Check Model Dimensions ===\n");
    bool all_have_dims = true;
    for (const auto& model : catalog.models) {
        if (model.dimensions.layers == 0 || model.dimensions.embedding_dim == 0) {
            all_have_dims = false;
            printf("  Missing dimensions for: %s\n", model.id.c_str());
        }
    }
    TEST_ASSERT(all_have_dims, "All models have dimensions");
    
    // =========================================================================
    // Test 8: Check download URLs
    // =========================================================================
    printf("\n=== Test 8: Check Download URLs ===\n");
    bool all_have_urls = true;
    int https_count = 0;
    for (const auto& model : catalog.models) {
        for (const auto& v : model.variants) {
            if (v.hf_url.empty()) {
                all_have_urls = false;
                printf("  Missing URL for: %s/%s\n", model.id.c_str(), v.quant.c_str());
            }
            if (v.hf_url.find("https://") == 0) https_count++;
        }
    }
    TEST_ASSERT(all_have_urls, "All variants have download URLs");
    TEST_ASSERT(https_count == total_variants, "All URLs use HTTPS");
    
    // =========================================================================
    // Test 9: Check star rating function
    // =========================================================================
    printf("\n=== Test 9: Check Star Rating Function ===\n");
    TEST_ASSERT(strcmp(quality_to_stars(9.0), "★★★★★") == 0, "Score 9.0 = 5 stars");
    TEST_ASSERT(strcmp(quality_to_stars(7.0), "★★★★☆") == 0, "Score 7.0 = 4 stars");
    TEST_ASSERT(strcmp(quality_to_stars(5.0), "★★★☆☆") == 0, "Score 5.0 = 3 stars");
    TEST_ASSERT(strcmp(quality_to_stars(3.0), "★★☆☆☆") == 0, "Score 3.0 = 2 stars");
    TEST_ASSERT(strcmp(quality_to_stars(1.0), "★☆☆☆☆") == 0, "Score 1.0 = 1 star");
    
    // =========================================================================
    // Test 10: Print catalog stats
    // =========================================================================
    printf("\n=== Test 10: Catalog Statistics ===\n");
    print_catalog_stats(catalog);
    
    // =========================================================================
    // Test 11: Load from file (if exists)
    // =========================================================================
    printf("\n=== Test 11: Load from File ===\n");
    ModelCatalog file_catalog = load_catalog_from_file("data/models_catalog.json");
    if (!file_catalog.models.empty()) {
        TEST_ASSERT(file_catalog.models.size() == catalog.models.size(),
                    "File catalog matches built-in catalog");
    } else {
        printf("  ⏭️  Skipped: File catalog not found at data/models_catalog.json\n");
    }
    
    // =========================================================================
    // Test 12: Size range covers all use cases
    // =========================================================================
    printf("\n=== Test 12: Size Range Coverage ===\n");
    bool has_small = false, has_medium = false, has_large = false;
    for (const auto& model : catalog.models) {
        double min_variant_size = 100.0;
        for (const auto& v : model.variants) {
            if (v.file_size_gb < min_variant_size) min_variant_size = v.file_size_gb;
        }
        if (min_variant_size < 2.0) has_small = true;   // <2GB = fits anywhere
        if (min_variant_size >= 2.0 && min_variant_size < 8.0) has_medium = true;  // 2-8GB = needs decent GPU
        if (min_variant_size >= 8.0) has_large = true;  // >8GB = needs serious hardware
    }
    TEST_ASSERT(has_small, "Has small models (<2GB)");
    TEST_ASSERT(has_medium, "Has medium models (2-8GB)");
    TEST_ASSERT(has_large, "Has large models (>8GB)");
    
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
        printf("\n🎉 All catalog tests passed!\n");
    } else {
        printf("\n⚠️  %d test(s) failed.\n", tests_failed);
    }
    
    return tests_failed > 0 ? 1 : 0;
}
