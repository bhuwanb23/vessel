#include "output.h"
#include "profiler.h"
#include "fetcher.h"
#include <cstdio>
#include <cstring>
#include <algorithm>

// =============================================================================
// Context Mode
// =============================================================================

ContextMode parse_context(const std::string& str) {
    if (str == "4k" || str == "4K") return ContextMode::FOUR_K;
    if (str == "max") return ContextMode::MAX;
    if (str == "both") return ContextMode::BOTH;
    return ContextMode::BOTH;
}

const char* get_context_name(ContextMode mode) {
    switch (mode) {
        case ContextMode::FOUR_K: return "4K only";
        case ContextMode::MAX:    return "max-safe only";
        case ContextMode::BOTH:   return "both";
        default: return "unknown";
    }
}

// =============================================================================
// Status Determination
// =============================================================================

StrategyStatus determine_status(const HardwareSpec& hw, const Prediction& pred,
                                const StrategyConfig& strat) {
    // Low confidence → flagged
    if (pred.confidence == PredictionConfidence::LOW)
        return StrategyStatus::LOW_CONF;

    bool fits_vram = true;
    bool fits_ram  = true;
    double vram_ratio = 0.0;
    double ram_ratio  = 0.0;

    switch (strat.placement) {
        case PlacementStrategy::FULL_GPU:
            fits_vram = (pred.memory_vram_bytes <= hw.vram_free_bytes);
            if (hw.vram_free_bytes > 0)
                vram_ratio = (double)pred.memory_vram_bytes / (double)hw.vram_free_bytes;
            break;

        case PlacementStrategy::GPU_CPU_SPLIT:
            fits_vram = (pred.memory_vram_bytes <= hw.vram_free_bytes);
            if (hw.vram_free_bytes > 0 && pred.memory_vram_bytes > 0)
                vram_ratio = (double)pred.memory_vram_bytes / (double)hw.vram_free_bytes;
            fits_ram = (pred.memory_ram_bytes <= hw.ram_free_bytes);
            if (hw.ram_free_bytes > 0 && pred.memory_ram_bytes > 0)
                ram_ratio = (double)pred.memory_ram_bytes / (double)hw.ram_free_bytes;
            break;

        case PlacementStrategy::CPU_ONLY:
            fits_ram = (pred.memory_ram_bytes <= hw.ram_free_bytes);
            if (hw.ram_free_bytes > 0)
                ram_ratio = (double)pred.memory_ram_bytes / (double)hw.ram_free_bytes;
            break;
    }

    if (!fits_vram || !fits_ram)
        return StrategyStatus::NO_FIT;

    if (vram_ratio > 0.9 || ram_ratio > 0.9)
        return StrategyStatus::TIGHT;

    return StrategyStatus::VIABLE;
}

const char* get_status_icon(StrategyStatus s) {
    switch (s) {
        case StrategyStatus::VIABLE:   return "✅";
        case StrategyStatus::TIGHT:    return "⚠️";
        case StrategyStatus::NO_FIT:   return "❌";
        case StrategyStatus::LOW_CONF: return "❓";
        default: return "  ";
    }
}

const char* get_status_text(StrategyStatus s) {
    switch (s) {
        case StrategyStatus::VIABLE:   return "VIABLE";
        case StrategyStatus::TIGHT:    return "TIGHT";
        case StrategyStatus::NO_FIT:   return "NO FIT";
        case StrategyStatus::LOW_CONF: return "LOW CONF";
        default: return "UNKNOWN";
    }
}

// =============================================================================
// Number Formatting Helpers
// =============================================================================

static void fmt_memory(char* buf, size_t n, uint64_t bytes) {
    if (bytes == 0)           snprintf(buf, n, "-");
    else if (bytes >= 1e9)    snprintf(buf, n, "%.1f GB", bytes / 1e9);
    else                      snprintf(buf, n, "%.0f MB", bytes / 1e6);
}

static void fmt_speed(char* buf, size_t n, double tps) {
    if (tps <= 0)        snprintf(buf, n, "-");
    else if (tps >= 100) snprintf(buf, n, "~%.0f", tps);
    else                 snprintf(buf, n, "~%.1f", tps);
}

static void fmt_ttft(char* buf, size_t n, double ms) {
    if (ms <= 0)         snprintf(buf, n, "-");
    else if (ms < 1000)  snprintf(buf, n, "~%.0fms", ms);
    else                 snprintf(buf, n, "~%.1fs", ms / 1000);
}

// =============================================================================
// Filter by Context
// =============================================================================

std::vector<StrategyResult> filter_by_context(const std::vector<StrategyResult>& results,
                                              ContextMode mode, uint32_t model_max_ctx) {
    if (mode == ContextMode::BOTH) return results;

    uint32_t target = (mode == ContextMode::FOUR_K) ? 4096 : model_max_ctx;
    std::vector<StrategyResult> out;
    for (const auto& r : results)
        if (r.strategy.context_length == target)
            out.push_back(r);
    return out;
}

// =============================================================================
// Print Brief Summaries
// =============================================================================

void print_hardware_brief(const HardwareSpec& hw) {
    printf("Hardware: %s (%.1f GB VRAM, %.1f free) | %.0f GB RAM (%.0f free)"
           " | NVMe %.1f/%.2f GB/s\n",
           hw.gpu_name.c_str(),
           hw.vram_total_bytes / 1e9, hw.vram_free_bytes / 1e9,
           hw.ram_total_bytes / 1e9,  hw.ram_free_bytes / 1e9,
           hw.nvme_sequential_mbs / 1000.0, hw.nvme_random_4k_mbs / 1000.0);
}

void print_model_brief(const ModelSpec& model) {
    printf("Model:    %s %s | %.2fB params | %u layers | %uK max context\n",
           model.name.c_str(), model.quant_type.c_str(),
           model.param_count / 1e9, model.layers, model.context_length / 1024);
}

// =============================================================================
// Print Full Reports (--verbose)
// =============================================================================

void print_hardware_full(const HardwareSpec& hw) {
    printf("\n--- Hardware Profile (Full) ---\n");
    printf("GPU:              %s\n", hw.gpu_name.c_str());
    printf("VRAM:             %.2f GB total, %.2f GB free\n",
           hw.vram_total_bytes / 1e9, hw.vram_free_bytes / 1e9);
    printf("GPU Bandwidth:    %.1f GB/s\n", hw.gpu_bandwidth_gbs);
    printf("GPU TFLOPS:       %.1f TFLOPS (FP16)\n", hw.gpu_tflops_fp16);
    printf("RAM:              %.2f GB total, %.2f GB free\n",
           hw.ram_total_bytes / 1e9, hw.ram_free_bytes / 1e9);
    if (hw.nvme_sequential_mbs > 0)
        printf("NVMe:             %.0f MB/s seq, %.0f MB/s random 4K\n",
               hw.nvme_sequential_mbs, hw.nvme_random_4k_mbs);
    printf("Compute:          sm_%d%d\n", hw.gpu_compute_major, hw.gpu_compute_minor);
}

void print_model_full(const ModelSpec& model) {
    printf("\n--- Model Metadata (Full) ---\n");
    printf("Name:             %s\n", model.name.c_str());
    printf("Architecture:     %s\n", model.architecture.c_str());
    printf("Parameters:       %.2fB\n", model.param_count / 1e9);
    printf("Layers:           %u\n", model.layers);
    printf("Embedding:        %u\n", model.embedding_dim);
    printf("Attention Heads:  %u (KV: %u)\n", model.attention_heads, model.kv_heads);
    printf("Context:          %u tokens\n", model.context_length);
    printf("Quantization:     %s (%.2f bpw)\n", model.quant_type.c_str(),
           model.bits_per_weight);
    printf("Source:           %s\n",
           model.source == MetadataSource::GGUF_HEADER ? "GGUF Header" : "config.json");
}

// =============================================================================
// Prediction Table (Phase E + Phase A ranker-aware recommendations)
// =============================================================================

void print_prediction_table(const std::vector<StrategyResult>& results,
                            const HardwareSpec& hw, PriorityMode priority) {
    printf("\n=== LLM Deployment Planner — Strategy Comparison ===\n\n");
    printf("Ranked by: %s (use --priority to change)\n\n", get_priority_name(priority));

    // Table header
    printf(" #  %-12s %-10s %-8s %-8s %-9s %-9s %-8s %-8s %-10s\n",
           "Placement", "GPU Layers", "Context", "KV Cache",
           "VRAM", "RAM", "tok/s", "TTFT", "Status");
    printf("─── ──────────── ────────── ──────── ──────── "
           "───────── ───────── ──────── ──────── ──────────\n");

    int row = 0;
    for (const auto& r : results) {
        row++;
        StrategyStatus st = determine_status(hw, r.prediction, r.strategy);
        const auto& s = r.strategy;
        const auto& p = r.prediction;

        char placement[16], gpu_layers[16], context[16], kv_cache[16];
        char vram[16], ram[16], speed[16], ttft[16];

        snprintf(placement, sizeof(placement), "%s",
                 s.placement == PlacementStrategy::FULL_GPU    ? "Full GPU" :
                 s.placement == PlacementStrategy::GPU_CPU_SPLIT ? "Split" : "CPU Only");
        snprintf(gpu_layers, sizeof(gpu_layers), "%u/%u", s.gpu_layers, 28);
        snprintf(context, sizeof(context), "%uK", s.context_length / 1024);
        snprintf(kv_cache, sizeof(kv_cache), "%s", s.kv_quant_bits == 16 ? "FP16" : "Q8");
        fmt_memory(vram, sizeof(vram), p.memory_vram_bytes);
        fmt_memory(ram,  sizeof(ram),  p.memory_ram_bytes);

        if (st == StrategyStatus::NO_FIT) {
            snprintf(speed, sizeof(speed), "-");
            snprintf(ttft,  sizeof(ttft),  "-");
        } else {
            fmt_speed(speed, sizeof(speed), p.tokens_per_sec);
            fmt_ttft(ttft,   sizeof(ttft),  p.ttft_ms);
        }

        printf("%2d  %-12s %-10s %-8s %-8s %-9s %-9s %-8s %-8s %s %s\n",
               row, placement, gpu_layers, context, kv_cache,
               vram, ram, speed, ttft, get_status_icon(st), get_status_text(st));
    }

    // =========================================================================
    // Priority-Aware Recommendation
    // =========================================================================
    printf("\n--- Recommendation ---\n\n");

    // Find best viable strategy for current priority
    const StrategyResult* best = nullptr;
    for (const auto& r : results) {
        StrategyStatus st = determine_status(hw, r.prediction, r.strategy);
        if (st != StrategyStatus::VIABLE && st != StrategyStatus::TIGHT) continue;
        if (!best) { best = &r; break; }  // First viable after sorting = best for this priority
    }

    // Find best for speed (always show)
    const StrategyResult* best_speed = nullptr;
    for (const auto& r : results) {
        StrategyStatus st = determine_status(hw, r.prediction, r.strategy);
        if (st != StrategyStatus::VIABLE && st != StrategyStatus::TIGHT) continue;
        if (!best_speed || r.prediction.tokens_per_sec > best_speed->prediction.tokens_per_sec)
            best_speed = &r;
    }

    // Find best for long context
    const StrategyResult* best_ctx = nullptr;
    for (const auto& r : results) {
        StrategyStatus st = determine_status(hw, r.prediction, r.strategy);
        if (st != StrategyStatus::VIABLE && st != StrategyStatus::TIGHT) continue;
        if (!best_ctx || r.strategy.context_length > best_ctx->strategy.context_length)
            best_ctx = &r;
    }

    // Find best for memory safety (most headroom)
    const StrategyResult* best_safe = nullptr;
    for (const auto& r : results) {
        StrategyStatus st = determine_status(hw, r.prediction, r.strategy);
        if (st != StrategyStatus::VIABLE && st != StrategyStatus::TIGHT) continue;
        double h = calculate_memory_headroom(hw, r.prediction);
        double bh = best_safe ? calculate_memory_headroom(hw, best_safe->prediction) : -1.0;
        if (!best_safe || h > bh) best_safe = &r;
    }

    // Helper: find row number
    auto find_row = [&](const StrategyResult* target) -> int {
        for (size_t i = 0; i < results.size(); i++)
            if (&results[i] == target) return static_cast<int>(i) + 1;
        return 0;
    };

    auto placement_str = [](const StrategyConfig& s) -> const char* {
        return s.placement == PlacementStrategy::FULL_GPU    ? "Full GPU" :
               s.placement == PlacementStrategy::GPU_CPU_SPLIT ? "GPU+CPU" : "CPU";
    };

    // Priority-specific primary recommendation
    switch (priority) {
        case PriorityMode::SPEED:
            if (best_speed)
                printf("🏆 Best for speed: Strategy #%d (%s, %uK, %s) — ~%.0f tok/s\n",
                       find_row(best_speed), placement_str(best_speed->strategy),
                       best_speed->strategy.context_length / 1024,
                       best_speed->strategy.kv_quant_bits == 16 ? "FP16" : "Q8",
                       best_speed->prediction.tokens_per_sec);
            break;

        case PriorityMode::QUALITY:
            if (best)
                printf("🏆 Best for quality: Strategy #%d (%s, %uK, %s)\n"
                       "   Higher KV precision and more GPU layers = fewer quantization artifacts.\n",
                       find_row(best), placement_str(best->strategy),
                       best->strategy.context_length / 1024,
                       best->strategy.kv_quant_bits == 16 ? "FP16" : "Q8");
            break;

        case PriorityMode::SAFETY:
            if (best_safe) {
                double headroom = calculate_memory_headroom(hw, best_safe->prediction);
                printf("🏆 Best for safety: Strategy #%d (%s, %uK, %s) — %.0f%% headroom\n",
                       find_row(best_safe), placement_str(best_safe->strategy),
                       best_safe->strategy.context_length / 1024,
                       best_safe->strategy.kv_quant_bits == 16 ? "FP16" : "Q8",
                       headroom * 100.0);
            }
            break;
    }

    // Always show speed option if different from primary
    if (best_speed && best_speed != best)
        printf("💡 For fastest generation: Strategy #%d — ~%.0f tok/s\n",
               find_row(best_speed), best_speed->prediction.tokens_per_sec);

    // Always show long-context option if different
    if (best_ctx && best_ctx != best && best_ctx->strategy.context_length > 4096)
        printf("💡 For long documents: Strategy #%d uses %s with %uK context\n",
               find_row(best_ctx), placement_str(best_ctx->strategy),
               best_ctx->strategy.context_length / 1024);

    // Q8 KV tip
    printf("💡 Q8 KV cache saves ~0.3GB VRAM at negligible quality cost.\n");
}

// =============================================================================
// Warnings
// =============================================================================

void print_warnings(const HardwareSpec& hw) {
    const ProfileErrors& pe = get_profile_errors();

    if (pe.low_vram) {
        fprintf(stderr, "\n⚠️  Warning: Available VRAM is very low (%.1f GB).\n",
                pe.vram_free_gb);
        fprintf(stderr, "   Consider closing other GPU applications.\n");
    }
    if (pe.disk_slow) {
        fprintf(stderr, "\n⚠️  Warning: Storage read speed is unusually low"
                " (%.0f MB/s sequential).\n", pe.disk_seq_mbs);
        fprintf(stderr, "   Ensure the model is on a local NVMe SSD.\n");
    }
}

void print_post_table_warnings(const std::vector<StrategyResult>& results,
                               const HardwareSpec& hw) {
    int viable = 0, cpu_only = 0, gpu = 0;
    for (const auto& r : results) {
        StrategyStatus st = determine_status(hw, r.prediction, r.strategy);
        if (st == StrategyStatus::VIABLE || st == StrategyStatus::TIGHT) {
            viable++;
            if (r.strategy.placement == PlacementStrategy::CPU_ONLY) cpu_only++;
            else gpu++;
        }
    }

    if (viable == 0) {
        fprintf(stderr, "\n❌ This model does not fit on your hardware in any configuration.\n");
        fprintf(stderr, "   Consider: a smaller quantization, a smaller model,\n");
        fprintf(stderr, "   or closing other applications to free memory.\n");
    } else if (gpu == 0 && cpu_only > 0) {
        fprintf(stderr, "\n💡 GPU offload is not possible for this model on your hardware.\n");
        fprintf(stderr, "   For faster inference, consider a smaller model or more VRAM.\n");
    }
}

// =============================================================================
// Usage
// =============================================================================

void print_usage() {
    printf("Usage: llm-planner --model <url_or_path> [options]\n\n");
    printf("Required:\n");
    printf("  --model <url_or_path>               Hugging Face GGUF URL or local file\n\n");
    printf("Options:\n");
    printf("  --priority <speed|quality|safety>   Rank by (default: speed)\n");
    printf("  --context <4k|max|both>             Contexts to evaluate (default: both)\n");
    printf("  --verbose                           Full hardware & model reports\n");
    printf("  --help                              Show this help\n\n");
    printf("Examples:\n");
    printf("  llm-planner --model ./models/Llama-3.2-3B-Instruct-Q4_K_M.gguf\n");
    printf("  llm-planner --model <url> --priority safety\n");
    printf("  llm-planner --model <url> --context 4k --verbose\n");
}
