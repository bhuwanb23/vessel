#include "ranker.h"
#include <algorithm>
#include <cmath>

// =============================================================================
// Priority Parsing
// =============================================================================

PriorityMode parse_priority(const std::string& str) {
    if (str == "speed") return PriorityMode::SPEED;
    if (str == "quality") return PriorityMode::QUALITY;
    if (str == "safety") return PriorityMode::SAFETY;
    return PriorityMode::SPEED;  // default
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
// From spec: min(vram_free - vram_used, ram_free - ram_used) / max(vram_free, ram_free)
// This is the percentage of the most constrained resource that remains free.
// Higher = safer (more margin for background processes, thermal throttling, etc.)
// =============================================================================

double calculate_memory_headroom(const HardwareSpec& hw, const Prediction& pred) {
    // Calculate free memory after loading model
    int64_t vram_remaining = static_cast<int64_t>(hw.vram_free_bytes)
                           - static_cast<int64_t>(pred.memory_vram_bytes);
    int64_t ram_remaining  = static_cast<int64_t>(hw.ram_free_bytes)
                           - static_cast<int64_t>(pred.memory_ram_bytes);

    // Use the most constrained resource
    int64_t min_remaining = std::min(vram_remaining, ram_remaining);

    // Normalize by the largest available resource
    uint64_t max_available = std::max(hw.vram_free_bytes, hw.ram_free_bytes);
    if (max_available == 0) return 0.0;

    double headroom = static_cast<double>(min_remaining)
                    / static_cast<double>(max_available);

    // Clamp to [0, 1] — negative means doesn't fit
    return std::max(0.0, std::min(1.0, headroom));
}

// =============================================================================
// Sort by Priority — Three Axes
// =============================================================================
// Non-viable strategies always go to the bottom, regardless of priority.
// Within viable strategies, the sort order depends on the user's priority.
// =============================================================================

void sort_by_priority(std::vector<StrategyResult>& results, PriorityMode priority,
                      const HardwareSpec& hw) {
    std::sort(results.begin(), results.end(),
        [priority, &hw](const StrategyResult& a, const StrategyResult& b) {
            // Non-viable always go to bottom
            if (a.prediction.viable != b.prediction.viable) {
                return a.prediction.viable > b.prediction.viable;
            }

            switch (priority) {

            // =================================================================
            // SPEED — Interactive chat, real-time generation, coding assistants
            // Primary:   tokens/sec descending (higher is better)
            // Secondary: TTFT ascending      (lower is better, breaks ties)
            // Tertiary:  confidence descending (prefer high-confidence when
            //            speeds are close)
            // =================================================================
            case PriorityMode::SPEED:
                if (a.prediction.tokens_per_sec != b.prediction.tokens_per_sec)
                    return a.prediction.tokens_per_sec > b.prediction.tokens_per_sec;
                if (a.prediction.ttft_ms != b.prediction.ttft_ms)
                    return a.prediction.ttft_ms < b.prediction.ttft_ms;
                return static_cast<int>(a.prediction.confidence)
                     > static_cast<int>(b.prediction.confidence);

            // =================================================================
            // QUALITY — Creative writing, research, accuracy-critical tasks
            // Primary:   KV cache precision descending (FP16 > Q8)
            // Secondary: GPU layers descending (more GPU = fewer CPU artifacts)
            // Tertiary:  context length descending (larger context = more flexibility)
            // Quaternary: speed (tiebreak)
            // =================================================================
            case PriorityMode::QUALITY:
                if (a.strategy.kv_quant_bits != b.strategy.kv_quant_bits)
                    return a.strategy.kv_quant_bits > b.strategy.kv_quant_bits;
                if (a.strategy.gpu_layers != b.strategy.gpu_layers)
                    return a.strategy.gpu_layers > b.strategy.gpu_layers;
                if (a.strategy.context_length != b.strategy.context_length)
                    return a.strategy.context_length > b.strategy.context_length;
                return a.prediction.tokens_per_sec > b.prediction.tokens_per_sec;

            // =================================================================
            // SAFETY — Production deployments, long-running sessions, laptops
            // Primary:   memory headroom descending (more free memory = safer)
            // Secondary: confidence descending (prefer strategies you're sure about)
            // Tertiary:  tokens/sec descending (among equally safe, prefer faster)
            // =================================================================
            case PriorityMode::SAFETY: {
                double headroom_a = calculate_memory_headroom(hw, a.prediction);
                double headroom_b = calculate_memory_headroom(hw, b.prediction);
                if (headroom_a != headroom_b)
                    return headroom_a > headroom_b;
                if (a.prediction.confidence != b.prediction.confidence)
                    return static_cast<int>(a.prediction.confidence)
                         > static_cast<int>(b.prediction.confidence);
                return a.prediction.tokens_per_sec > b.prediction.tokens_per_sec;
            }

            default:
                return a.prediction.tokens_per_sec > b.prediction.tokens_per_sec;
            }
        });
}
