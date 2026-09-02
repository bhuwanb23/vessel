#include "ranker.h"
#include <algorithm>
#include <cmath>
#include <limits>

// =============================================================================
// Priority Parsing
// =============================================================================

bool is_valid_priority(const std::string& str) {
    return str == "speed" || str == "quality" || str == "safety";
}

PriorityMode parse_priority(const std::string& str) {
    if (str == "speed")   return PriorityMode::SPEED;
    if (str == "quality") return PriorityMode::QUALITY;
    if (str == "safety")  return PriorityMode::SAFETY;
    return PriorityMode::SPEED;  // should never reach here if validated
}

const char* get_priority_name(PriorityMode mode) {
    switch (mode) {
        case PriorityMode::SPEED:   return "speed";
        case PriorityMode::QUALITY: return "quality";
        case PriorityMode::SAFETY:  return "safety";
        default: return "unknown";
    }
}

// =============================================================================
// Memory Headroom Calculation
// =============================================================================
// min(vram_free - vram_used, ram_free - ram_used) / max(vram_free, ram_free)
// =============================================================================

double calculate_memory_headroom(const HardwareSpec& hw, const Prediction& pred) {
    int64_t vram_remaining = static_cast<int64_t>(hw.vram_free_bytes)
                           - static_cast<int64_t>(pred.memory_vram_bytes);
    int64_t ram_remaining  = static_cast<int64_t>(hw.ram_free_bytes)
                           - static_cast<int64_t>(pred.memory_ram_bytes);

    int64_t min_remaining = std::min(vram_remaining, ram_remaining);

    uint64_t max_available = std::max(hw.vram_free_bytes, hw.ram_free_bytes);
    if (max_available == 0) return 0.0;

    double headroom = static_cast<double>(min_remaining)
                    / static_cast<double>(max_available);

    return std::max(0.0, std::min(1.0, headroom));
}

// =============================================================================
// Metric Extraction Helpers
// =============================================================================
// Each priority has a primary, secondary, and tertiary metric.
// These functions extract the raw value for a given strategy.
// =============================================================================

// --- SPEED priority metrics ---
static double speed_primary(const StrategyResult& r) {
    return r.prediction.tokens_per_sec;       // higher is better
}
static double speed_secondary(const StrategyResult& r) {
    return 1.0 / (r.prediction.ttft_ms + 1.0); // lower TTFT → higher score
}
static double speed_tertiary(const StrategyResult& r) {
    return static_cast<double>(r.prediction.confidence);
}

// --- QUALITY priority metrics ---
// Primary: quality proxy = bpw + kv_bonus
//   (Higher bpw = less quantization loss. FP16 KV > Q8)
static double quality_primary(const StrategyResult& r) {
    double kv_bonus = (r.strategy.kv_quant_bits == 16) ? 1.0 :
                      (r.strategy.kv_quant_bits == 8)  ? 0.5 : 0.0;
    // We don't have bpw directly in StrategyResult, but we know all strategies
    // use the same model, so bpw is constant. Use GPU layers as proxy instead.
    return static_cast<double>(r.strategy.gpu_layers) + kv_bonus;
}
static double quality_secondary(const StrategyResult& r) {
    return static_cast<double>(r.strategy.gpu_layers);  // more GPU = fewer artifacts
}
static double quality_tertiary(const StrategyResult& r) {
    return static_cast<double>(r.strategy.context_length);
}

// --- SAFETY priority metrics ---
static double safety_primary_val(const StrategyResult& r, const HardwareSpec& hw) {
    return calculate_memory_headroom(hw, r.prediction);  // higher = safer
}
static double safety_secondary(const StrategyResult& r) {
    return static_cast<double>(r.prediction.confidence);
}
static double safety_tertiary(const StrategyResult& r) {
    return r.prediction.tokens_per_sec;  // among equally safe, prefer faster
}

// =============================================================================
// Unified Metric Extraction
// =============================================================================
// Single dispatch point — eliminates duplicated switch blocks in
// calculate_score and sort_by_priority.

struct PriorityMetrics {
    double primary, secondary, tertiary;
};

static PriorityMetrics get_priority_metrics(const StrategyResult& r,
                                            const HardwareSpec& hw,
                                            PriorityMode priority) {
    switch (priority) {
        case PriorityMode::SPEED:
            return { speed_primary(r), speed_secondary(r), speed_tertiary(r) };
        case PriorityMode::QUALITY:
            return { quality_primary(r), quality_secondary(r), quality_tertiary(r) };
        case PriorityMode::SAFETY:
            return { safety_primary_val(r, hw), safety_secondary(r), safety_tertiary(r) };
        default:
            return { speed_primary(r), speed_secondary(r), speed_tertiary(r) };
    }
}

// =============================================================================
// Confidence Modifier
// =============================================================================
// HIGH → 1.0, MEDIUM → 0.9, LOW → 0.7
// Slightly penalizes low-confidence predictions.
// =============================================================================

static double confidence_multiplier(PredictionConfidence conf) {
    switch (conf) {
        case PredictionConfidence::HIGH:   return 1.0;
        case PredictionConfidence::MEDIUM: return 0.9;
        case PredictionConfidence::LOW:    return 0.7;
        default: return 0.8;
    }
}

// =============================================================================
// Normalize to [0, 1]
// =============================================================================
// If all values are the same (denominator = 0), return 0.5 for all.
// =============================================================================

static double normalize(double value, double min_val, double max_val) {
    if (max_val - min_val < 1e-9) return 0.5;  // all same → middle
    return (value - min_val) / (max_val - min_val);
}

// =============================================================================
// Scoring Function (Phase B)
// =============================================================================
// score = (primary × 0.7 + secondary × 0.2 + tertiary × 0.1) × confidence_mult
//
// Non-viable strategies get score = -1.0 (always at bottom).
// =============================================================================

double calculate_score(const StrategyResult& result, const HardwareSpec& hw,
                       PriorityMode priority,
                       double min_primary, double max_primary,
                       double min_secondary, double max_secondary,
                       double min_tertiary, double max_tertiary) {
    // Non-viable → always bottom
    if (!result.prediction.viable) return -1.0;

    PriorityMetrics m = get_priority_metrics(result, hw, priority);

    // Normalize each metric to [0, 1]
    double np = normalize(m.primary, min_primary, max_primary);
    double ns = normalize(m.secondary, min_secondary, max_secondary);
    double nt = normalize(m.tertiary, min_tertiary, max_tertiary);

    // Weighted combination: 70% primary, 20% secondary, 10% tertiary
    double raw_score = np * 0.7 + ns * 0.2 + nt * 0.1;

    // Apply confidence modifier
    double conf_mult = confidence_multiplier(result.prediction.confidence);

    return raw_score * conf_mult;
}

// =============================================================================
// Sort by Priority — Scoring Function Approach
// =============================================================================
// Two-pass algorithm:
//   Pass 1: Find min/max for each metric across all viable strategies
//   Pass 2: Calculate scores and sort by score descending
// =============================================================================

void sort_by_priority(std::vector<StrategyResult>& results, PriorityMode priority,
                      const HardwareSpec& hw) {
    if (results.empty()) return;

    // =========================================================================
    // Pass 1: Find min/max for normalization (viable strategies only)
    // =========================================================================
    double min_p = std::numeric_limits<double>::max();
    double max_p = std::numeric_limits<double>::lowest();
    double min_s = std::numeric_limits<double>::max();
    double max_s = std::numeric_limits<double>::lowest();
    double min_t = std::numeric_limits<double>::max();
    double max_t = std::numeric_limits<double>::lowest();

    for (const auto& r : results) {
        if (!r.prediction.viable) continue;

        PriorityMetrics m = get_priority_metrics(r, hw, priority);
        min_p = std::min(min_p, m.primary);   max_p = std::max(max_p, m.primary);
        min_s = std::min(min_s, m.secondary);  max_s = std::max(max_s, m.secondary);
        min_t = std::min(min_t, m.tertiary);   max_t = std::max(max_t, m.tertiary);
    }

    // =========================================================================
    // Pass 2: Stable sort by score descending
    // =========================================================================
    // Using std::stable_sort so that strategies with identical scores
    // preserve their original order from the matrix generator.
    // This makes the output deterministic and reproducible.
    //
    // Sort order:
    //   1. Viable descending (viable before non-viable)
    //   2. Score descending (higher is better)
    //   3. Confidence descending (tiebreaker)
    //   4. Original order preserved for identical scores (stable sort)
    // =========================================================================
    std::stable_sort(results.begin(), results.end(),
        [priority, &hw, min_p, max_p, min_s, max_s, min_t, max_t]
        (const StrategyResult& a, const StrategyResult& b) {
            // Level 1: viable before non-viable
            if (a.prediction.viable != b.prediction.viable)
                return a.prediction.viable > b.prediction.viable;

            // Level 2: higher score is better
            double score_a = calculate_score(a, hw, priority,
                                             min_p, max_p, min_s, max_s, min_t, max_t);
            double score_b = calculate_score(b, hw, priority,
                                             min_p, max_p, min_s, max_s, min_t, max_t);
            if (score_a != score_b)
                return score_a > score_b;

            // Level 3: higher confidence is better (tiebreaker)
            return static_cast<int>(a.prediction.confidence)
                 > static_cast<int>(b.prediction.confidence);
        });
}
