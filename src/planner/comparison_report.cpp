#include "comparison_report.h"
#include <cstdio>
#include <cmath>
#include <string>

// =============================================================================
// Delta Classification
// =============================================================================

const char* get_delta_level_label(DeltaLevel level) {
    switch (level) {
        case DeltaLevel::CLOSE: return "Close";
        case DeltaLevel::OFF:   return "Off";
        case DeltaLevel::WRONG: return "Wrong";
        default: return "Unknown";
    }
}

DeltaLevel classify_delta(double delta_pct) {
    double abs_delta = fabs(delta_pct);
    if (abs_delta < 10.0)  return DeltaLevel::CLOSE;
    if (abs_delta < 25.0)  return DeltaLevel::OFF;
    return DeltaLevel::WRONG;
}

// =============================================================================
// Status Icon for Delta Level
// =============================================================================

static const char* get_delta_icon(DeltaLevel level) {
    switch (level) {
        case DeltaLevel::CLOSE: return "\xe2\x9c\x85";  // checkmark
        case DeltaLevel::OFF:   return "\xe2\x9a\xa0";  // warning sign
        case DeltaLevel::WRONG: return "\xe2\x9d\x8c";  // cross
        default: return "  ";
    }
}

// =============================================================================
// Calculate Delta Percentage
// =============================================================================

static double calc_delta_pct(double predicted, double actual) {
    if (actual == 0.0) return 0.0;
    return ((predicted - actual) / actual) * 100.0;
}

static double calc_delta_pct(uint64_t predicted, uint64_t actual) {
    if (actual == 0) return 0.0;
    return ((double)((int64_t)predicted - (int64_t)actual) / (double)actual) * 100.0;
}

// =============================================================================
// Strategy Description
// =============================================================================

static std::string strategy_desc(const StrategyConfig& s) {
    const char* placement;
    switch (s.placement) {
        case PlacementStrategy::FULL_GPU:       placement = "Full GPU"; break;
        case PlacementStrategy::GPU_CPU_SPLIT:  placement = "GPU+CPU Split"; break;
        case PlacementStrategy::CPU_ONLY:       placement = "CPU Only"; break;
        case PlacementStrategy::HOT_COLD_SPLIT: placement = "Hot/Cold Split"; break;
        case PlacementStrategy::LAYER_STREAM:   placement = "Layer-Stream"; break;
        default:                               placement = "Unknown"; break;
    }
    char buf[128];
    snprintf(buf, sizeof(buf), "%s, %u/%u layers, %uK context, %s KV",
             placement, s.gpu_layers, 28u,
             s.context_length / 1024,
             s.kv_quant_bits == 16 ? "FP16" : "Q8");
    return std::string(buf);
}

// =============================================================================
// Print Comparison Report
// =============================================================================

void print_comparison_report(const Prediction& predicted,
                             const ExecutionResult& actual,
                             const StrategyConfig& strategy) {
    printf("\n=== Execution Complete \xe2\x80\x94 Predicted vs Actual ===\n");
    printf("Strategy: %s\n\n", strategy_desc(strategy).c_str());

    // Header
    printf("%-20s %-12s %-12s %-10s %-10s\n",
           "Metric", "Predicted", "Actual", "Delta", "Status");
    printf("\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80"
           "\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80"
           "\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80"
           "\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80"
           "\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80"
           "\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80"
           "\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80"
           "\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80"
           "\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80"
           "\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80"
           "\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80"
           "\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80"
           "\n");

    // --- Tokens/sec ---
    {
        char pred_buf[32], act_buf[32], delta_buf[32];
        double delta = calc_delta_pct(predicted.tokens_per_sec,
                                      actual.decode_tokens_per_sec);
        DeltaLevel level = classify_delta(delta);

        if (predicted.tokens_per_sec > 0)
            snprintf(pred_buf, sizeof(pred_buf), "~%.0f", predicted.tokens_per_sec);
        else
            snprintf(pred_buf, sizeof(pred_buf), "-");

        if (actual.decode_tokens_per_sec > 0)
            snprintf(act_buf, sizeof(act_buf), "%.0f", actual.decode_tokens_per_sec);
        else
            snprintf(act_buf, sizeof(act_buf), "-");

        if (actual.decode_tokens_per_sec > 0)
            snprintf(delta_buf, sizeof(delta_buf), "%+.1f%%", delta);
        else
            snprintf(delta_buf, sizeof(delta_buf), "-");

        printf("%-20s %-12s %-12s %-10s %s %s\n",
               "Tokens/sec", pred_buf, act_buf, delta_buf,
               get_delta_icon(level), get_delta_level_label(level));
    }

    // --- TTFT ---
    {
        char pred_buf[32], act_buf[32], delta_buf[32];
        double delta = calc_delta_pct(predicted.ttft_ms, actual.prompt_eval_ms);
        DeltaLevel level = classify_delta(delta);

        if (predicted.ttft_ms > 0)
            snprintf(pred_buf, sizeof(pred_buf), "~%.0fms", predicted.ttft_ms);
        else
            snprintf(pred_buf, sizeof(pred_buf), "-");

        if (actual.prompt_eval_ms > 0)
            snprintf(act_buf, sizeof(act_buf), "%.0fms", actual.prompt_eval_ms);
        else
            snprintf(act_buf, sizeof(act_buf), "-");

        if (actual.prompt_eval_ms > 0)
            snprintf(delta_buf, sizeof(delta_buf), "%+.1f%%", delta);
        else
            snprintf(delta_buf, sizeof(delta_buf), "-");

        printf("%-20s %-12s %-12s %-10s %s %s\n",
               "TTFT", pred_buf, act_buf, delta_buf,
               get_delta_icon(level), get_delta_level_label(level));
    }

    // --- Peak VRAM ---
    {
        char pred_buf[32], act_buf[32], delta_buf[32];

        if (predicted.memory_vram_bytes > 0)
            snprintf(pred_buf, sizeof(pred_buf), "%.1f GB",
                     predicted.memory_vram_bytes / 1e9);
        else
            snprintf(pred_buf, sizeof(pred_buf), "-");

        if (actual.peak_vram_used_bytes > 0)
            snprintf(act_buf, sizeof(act_buf), "%.1f GB",
                     actual.peak_vram_used_bytes / 1e9);
        else
            snprintf(act_buf, sizeof(act_buf), "-");

        if (actual.peak_vram_used_bytes > 0 && predicted.memory_vram_bytes > 0) {
            double delta = calc_delta_pct(predicted.memory_vram_bytes,
                                          actual.peak_vram_used_bytes);
            DeltaLevel level = classify_delta(delta);
            snprintf(delta_buf, sizeof(delta_buf), "%+.1f%%", delta);
            printf("%-20s %-12s %-12s %-10s %s %s\n",
                   "Peak VRAM", pred_buf, act_buf, delta_buf,
                   get_delta_icon(level), get_delta_level_label(level));
        } else {
            printf("%-20s %-12s %-12s %-10s   %s\n",
                   "Peak VRAM", pred_buf, act_buf, "-", "-");
        }
    }

    // --- Peak RAM ---
    {
        char pred_buf[32], act_buf[32], delta_buf[32];

        if (predicted.memory_ram_bytes > 0)
            snprintf(pred_buf, sizeof(pred_buf), "%.1f GB",
                     predicted.memory_ram_bytes / 1e9);
        else
            snprintf(pred_buf, sizeof(pred_buf), "-");

        if (actual.peak_ram_used_bytes > 0)
            snprintf(act_buf, sizeof(act_buf), "%.1f GB",
                     actual.peak_ram_used_bytes / 1e9);
        else
            snprintf(act_buf, sizeof(act_buf), "-");

        if (actual.peak_ram_used_bytes > 0 && predicted.memory_ram_bytes > 0) {
            double delta = calc_delta_pct(predicted.memory_ram_bytes,
                                          actual.peak_ram_used_bytes);
            DeltaLevel level = classify_delta(delta);
            snprintf(delta_buf, sizeof(delta_buf), "%+.1f%%", delta);
            printf("%-20s %-12s %-12s %-10s %s %s\n",
                   "Peak RAM", pred_buf, act_buf, delta_buf,
                   get_delta_icon(level), get_delta_level_label(level));
        } else {
            printf("%-20s %-12s %-12s %-10s   %s\n",
                   "Peak RAM", pred_buf, act_buf, "-", "-");
        }
    }

    // --- Thermal Throttle ---
    {
        const char* pred_str = "No";
        const char* act_str = actual.throttled ? "Yes" : "No";
        bool match = (predicted.warnings.find("thermal") != std::string::npos)
                     == actual.throttled;

        printf("%-20s %-12s %-12s %-10s %s %s\n",
               "Thermal Throttle", pred_str, act_str,
               match ? "-" : "+/-",
               match ? "\xe2\x9c\x85" : "\xe2\x9a\xa0",
               match ? "Match" : "Diff");
    }

    // --- Summary ---
    printf("\nGenerated %d tokens in %.2f seconds.\n",
           actual.tokens_generated,
           (actual.prompt_eval_ms + actual.decode_ms) / 1000.0);
    printf("First token in %.0fms.\n", actual.prompt_eval_ms);

    // --- Calibration note ---
    // Find the worst delta to suggest calibration
    double worst_delta = 0.0;
    const char* worst_metric = nullptr;

    if (predicted.memory_ram_bytes > 0 && actual.peak_ram_used_bytes > 0) {
        double d = fabs(calc_delta_pct(predicted.memory_ram_bytes,
                                       actual.peak_ram_used_bytes));
        if (d > worst_delta) { worst_delta = d; worst_metric = "RAM prediction"; }
    }
    if (predicted.memory_vram_bytes > 0 && actual.peak_vram_used_bytes > 0) {
        double d = fabs(calc_delta_pct(predicted.memory_vram_bytes,
                                       actual.peak_vram_used_bytes));
        if (d > worst_delta) { worst_delta = d; worst_metric = "VRAM prediction"; }
    }
    if (predicted.tokens_per_sec > 0 && actual.decode_tokens_per_sec > 0) {
        double d = fabs(calc_delta_pct(predicted.tokens_per_sec,
                                       actual.decode_tokens_per_sec));
        if (d > worst_delta) { worst_delta = d; worst_metric = "Speed prediction"; }
    }
    if (predicted.ttft_ms > 0 && actual.prompt_eval_ms > 0) {
        double d = fabs(calc_delta_pct(predicted.ttft_ms, actual.prompt_eval_ms));
        if (d > worst_delta) { worst_delta = d; worst_metric = "TTFT prediction"; }
    }

    if (worst_delta > 10.0 && worst_metric) {
        printf("\n\xF0\x9F\x92\xA1 The %s was off by %.0f%%. This will be recalibrated\n"
               "   for future predictions on this hardware.\n",
               worst_metric, worst_delta);
    }
}
