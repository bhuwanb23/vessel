#include "types.h"
#include "profiler.h"
#include "fetcher.h"
#include "matrix.h"
#include "../predictor/predictor.h"
#include <cstdio>
#include <cstring>
#include <ctime>
#include <chrono>
#include <string>
#include <algorithm>

// =============================================================================
// LLM Deployment Planner — Step 4: Output Formatting (Phase E)
// =============================================================================

// Priority modes
enum class PriorityMode { SPEED, QUALITY, SAFETY };
enum class ContextMode { FOUR_K, MAX, BOTH };

// Parse functions
PriorityMode parse_priority(const std::string& str) {
    if (str == "speed") return PriorityMode::SPEED;
    if (str == "quality") return PriorityMode::QUALITY;
    if (str == "safety") return PriorityMode::SAFETY;
    return PriorityMode::SPEED;
}

ContextMode parse_context(const std::string& str) {
    if (str == "4k" || str == "4K") return ContextMode::FOUR_K;
    if (str == "max") return ContextMode::MAX;
    if (str == "both") return ContextMode::BOTH;
    return ContextMode::BOTH;
}

const char* get_priority_name(PriorityMode mode) {
    switch (mode) {
        case PriorityMode::SPEED: return "speed";
        case PriorityMode::QUALITY: return "quality";
        case PriorityMode::SAFETY: return "safety";
        default: return "unknown";
    }
}

const char* get_context_name(ContextMode mode) {
    switch (mode) {
        case ContextMode::FOUR_K: return "4K only";
        case ContextMode::MAX: return "max-safe only";
        case ContextMode::BOTH: return "both";
        default: return "unknown";
    }
}

void print_usage() {
    printf("Usage: llm-planner --model <url_or_path> [options]\n\n");
    printf("Required:\n");
    printf("  --model <url_or_path>               Hugging Face GGUF URL or local file path\n\n");
    printf("Options:\n");
    printf("  --priority <speed|quality|safety>   How to rank strategies (default: speed)\n");
    printf("  --context <4k|max|both>             Context lengths to evaluate (default: both)\n");
    printf("  --verbose                           Print full hardware and model reports\n");
    printf("  --help                              Show this help\n\n");
    printf("Examples:\n");
    printf("  llm-planner --model ./models/Llama-3.2-3B-Instruct-Q4_K_M.gguf\n");
    printf("  llm-planner --model <url> --priority safety\n");
    printf("  llm-planner --model <url> --context 4k --verbose\n");
}

// Timer utility
class Timer {
    std::chrono::high_resolution_clock::time_point start;
public:
    Timer() : start(std::chrono::high_resolution_clock::now()) {}
    double elapsed_ms() {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count();
    }
};

// =============================================================================
// Status Determination
// =============================================================================

enum class StrategyStatus {
    VIABLE,     // Fits in memory with margin
    TIGHT,      // Fits but uses >90% of available memory
    NO_FIT,     // Exceeds available memory
    LOW_CONF    // Viable but low confidence prediction
};

StrategyStatus determine_status(const HardwareSpec& hw, const Prediction& pred, const StrategyConfig& strat) {
    // Check confidence first
    if (pred.confidence == PredictionConfidence::LOW) {
        return StrategyStatus::LOW_CONF;
    }
    
    // Determine actual memory requirements per placement
    // NOTE: We check against REAL hardware, not just pred.viable (which can be wrong)
    bool fits_vram = true;
    bool fits_ram = true;
    double vram_usage_ratio = 0.0;
    double ram_usage_ratio = 0.0;
    
    switch (strat.placement) {
        case PlacementStrategy::FULL_GPU:
            // All memory on GPU
            fits_vram = (pred.memory_vram_bytes <= hw.vram_free_bytes);
            if (hw.vram_free_bytes > 0) {
                vram_usage_ratio = (double)pred.memory_vram_bytes / (double)hw.vram_free_bytes;
            }
            break;
            
        case PlacementStrategy::GPU_CPU_SPLIT:
            // VRAM portion
            fits_vram = (pred.memory_vram_bytes <= hw.vram_free_bytes);
            if (hw.vram_free_bytes > 0 && pred.memory_vram_bytes > 0) {
                vram_usage_ratio = (double)pred.memory_vram_bytes / (double)hw.vram_free_bytes;
            }
            // RAM portion
            fits_ram = (pred.memory_ram_bytes <= hw.ram_free_bytes);
            if (hw.ram_free_bytes > 0 && pred.memory_ram_bytes > 0) {
                ram_usage_ratio = (double)pred.memory_ram_bytes / (double)hw.ram_free_bytes;
            }
            break;
            
        case PlacementStrategy::CPU_ONLY:
            // All memory on RAM
            fits_ram = (pred.memory_ram_bytes <= hw.ram_free_bytes);
            if (hw.ram_free_bytes > 0) {
                ram_usage_ratio = (double)pred.memory_ram_bytes / (double)hw.ram_free_bytes;
            }
            break;
    }
    
    // Does it fit at all?
    if (!fits_vram || !fits_ram) {
        return StrategyStatus::NO_FIT;
    }
    
    // Check tightness: >90% of available memory
    if (vram_usage_ratio > 0.9 || ram_usage_ratio > 0.9) {
        return StrategyStatus::TIGHT;
    }
    
    return StrategyStatus::VIABLE;
}

const char* get_status_icon(StrategyStatus status) {
    switch (status) {
        case StrategyStatus::VIABLE: return "✅";
        case StrategyStatus::TIGHT: return "⚠️";
        case StrategyStatus::NO_FIT: return "❌";
        case StrategyStatus::LOW_CONF: return "❓";
        default: return "  ";
    }
}

const char* get_status_text(StrategyStatus status) {
    switch (status) {
        case StrategyStatus::VIABLE: return "VIABLE";
        case StrategyStatus::TIGHT: return "TIGHT";
        case StrategyStatus::NO_FIT: return "NO FIT";
        case StrategyStatus::LOW_CONF: return "LOW CONF";
        default: return "UNKNOWN";
    }
}

// =============================================================================
// Formatting Functions
// =============================================================================

// Format memory in GB with 1 decimal
void format_memory_gb(char* buf, size_t size, uint64_t bytes) {
    if (bytes == 0) {
        snprintf(buf, size, "-");
    } else if (bytes >= 1e9) {
        snprintf(buf, size, "%.1f GB", bytes / 1e9);
    } else {
        snprintf(buf, size, "%.0f MB", bytes / 1e6);
    }
}

// Format speed with ~ prefix
void format_speed(char* buf, size_t size, double tps) {
    if (tps <= 0) {
        snprintf(buf, size, "-");
    } else if (tps >= 100) {
        snprintf(buf, size, "~%.0f", tps);
    } else {
        snprintf(buf, size, "~%.1f", tps);
    }
}

// Format TTFT
void format_ttft(char* buf, size_t size, double ms) {
    if (ms <= 0) {
        snprintf(buf, size, "-");
    } else if (ms < 1000) {
        snprintf(buf, size, "~%.0fms", ms);
    } else {
        snprintf(buf, size, "~%.1fs", ms / 1000);
    }
}

// Format placement name
void format_placement(char* buf, size_t size, const StrategyConfig& strat, uint32_t total_layers) {
    switch (strat.placement) {
        case PlacementStrategy::FULL_GPU:
            snprintf(buf, size, "Full GPU");
            break;
        case PlacementStrategy::GPU_CPU_SPLIT:
            snprintf(buf, size, "Split");
            break;
        case PlacementStrategy::CPU_ONLY:
            snprintf(buf, size, "CPU Only");
            break;
        default:
            snprintf(buf, size, "Unknown");
    }
}

// Format GPU layers
void format_gpu_layers(char* buf, size_t size, uint32_t gpu_layers, uint32_t total_layers) {
    snprintf(buf, size, "%u/%u", gpu_layers, total_layers);
}

// Format context
void format_context(char* buf, size_t size, uint32_t ctx) {
    if (ctx >= 1024) {
        snprintf(buf, size, "%uK", ctx / 1024);
    } else {
        snprintf(buf, size, "%u", ctx);
    }
}

// Format KV cache
void format_kv_cache(char* buf, size_t size, uint32_t kv_bits) {
    snprintf(buf, size, "%s", kv_bits == 16 ? "FP16" : "Q8");
}

// =============================================================================
// Print Functions
// =============================================================================

void print_hardware_brief(const HardwareSpec& hw) {
    printf("Hardware: %s (%.1f GB VRAM, %.1f free) | %.0f GB RAM (%.0f free) | NVMe %.1f/%.2f GB/s\n",
           hw.gpu_name.c_str(),
           hw.vram_total_bytes / 1e9,
           hw.vram_free_bytes / 1e9,
           hw.ram_total_bytes / 1e9,
           hw.ram_free_bytes / 1e9,
           hw.nvme_sequential_mbs / 1000.0,
           hw.nvme_random_4k_mbs / 1000.0);
}

void print_hardware_full(const HardwareSpec& hw) {
    printf("\n--- Hardware Profile (Full) ---\n");
    printf("GPU:              %s\n", hw.gpu_name.c_str());
    printf("VRAM:             %.2f GB total, %.2f GB free\n", 
           hw.vram_total_bytes / 1e9, hw.vram_free_bytes / 1e9);
    printf("GPU Bandwidth:    %.1f GB/s\n", hw.gpu_bandwidth_gbs);
    printf("GPU TFLOPS:       %.1f TFLOPS (FP16)\n", hw.gpu_tflops_fp16);
    printf("RAM:              %.2f GB total, %.2f GB free\n",
           hw.ram_total_bytes / 1e9, hw.ram_free_bytes / 1e9);
    if (hw.nvme_sequential_mbs > 0) {
        printf("NVMe:             %.0f MB/s seq, %.0f MB/s random 4K\n",
               hw.nvme_sequential_mbs, hw.nvme_random_4k_mbs);
    }
    printf("Compute:          sm_%d%d\n", hw.gpu_compute_major, hw.gpu_compute_minor);
}

void print_model_brief(const ModelSpec& model) {
    printf("Model:    %s %s | %.2fB params | %u layers | %uK max context\n",
           model.name.c_str(),
           model.quant_type.c_str(),
           model.param_count / 1e9,
           model.layers,
           model.context_length / 1024);
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
    printf("Quantization:     %s (%.2f bpw)\n", model.quant_type.c_str(), model.bits_per_weight);
    printf("Source:           %s\n", model.source == MetadataSource::GGUF_HEADER ? "GGUF Header" : "config.json");
}

// =============================================================================
// Main Prediction Table (Phase E)
// =============================================================================

void print_prediction_table(const std::vector<StrategyResult>& results, const HardwareSpec& hw, PriorityMode priority) {
    printf("\n=== LLM Deployment Planner — Strategy Comparison ===\n\n");
    
    printf("Ranked by: %s (use --priority to change)\n\n", get_priority_name(priority));
    
    // Table header
    printf(" #  %-12s %-10s %-8s %-8s %-9s %-9s %-8s %-8s %-10s\n",
           "Placement", "GPU Layers", "Context", "KV Cache", "VRAM", "RAM", "tok/s", "TTFT", "Status");
    printf("─── ──────────── ────────── ──────── ──────── ───────── ───────── ──────── ──────── ──────────\n");
    
    int row_num = 0;
    const StrategyResult* first_viable = nullptr;
    
    for (const auto& result : results) {
        row_num++;
        const auto& pred = result.prediction;
        const auto& strat = result.strategy;
        
        // Determine status
        StrategyStatus status = determine_status(hw, pred, strat);
        
        // Track first viable strategy for recommendation
        if (!first_viable && status == StrategyStatus::VIABLE) {
            first_viable = &result;
        }
        
        // Format all columns
        char placement[16], gpu_layers[16], context[16], kv_cache[16];
        char vram[16], ram[16], speed[16], ttft[16], status_icon[8], status_text[16];
        
        format_placement(placement, sizeof(placement), strat, 28);
        format_gpu_layers(gpu_layers, sizeof(gpu_layers), strat.gpu_layers, 28);
        format_context(context, sizeof(context), strat.context_length);
        format_kv_cache(kv_cache, sizeof(kv_cache), strat.kv_quant_bits);
        format_memory_gb(vram, sizeof(vram), pred.memory_vram_bytes);
        format_memory_gb(ram, sizeof(ram), pred.memory_ram_bytes);
        
        if (status == StrategyStatus::NO_FIT) {
            snprintf(speed, sizeof(speed), "-");
            snprintf(ttft, sizeof(ttft), "-");
        } else {
            format_speed(speed, sizeof(speed), pred.tokens_per_sec);
            format_ttft(ttft, sizeof(ttft), pred.ttft_ms);
        }
        
        snprintf(status_icon, sizeof(status_icon), "%s", get_status_icon(status));
        snprintf(status_text, sizeof(status_text), "%s", get_status_text(status));
        
        // Print row
        printf("%2d  %-12s %-10s %-8s %-8s %-9s %-9s %-8s %-8s %s %s\n",
               row_num,
               placement,
               gpu_layers,
               context,
               kv_cache,
               vram,
               ram,
               speed,
               ttft,
               status_icon,
               status_text);
    }
    
    // =========================================================================
    // Recommendation
    // =========================================================================
    printf("\n--- Recommendation ---\n\n");
    
    // Find best for speed
    const StrategyResult* best_speed = nullptr;
    const StrategyResult* best_context = nullptr;
    const StrategyResult* best_memory = nullptr;
    
    for (const auto& r : results) {
        StrategyStatus st = determine_status(hw, r.prediction, r.strategy);
        if (st != StrategyStatus::VIABLE && st != StrategyStatus::TIGHT) continue;
        
        if (!best_speed || r.prediction.tokens_per_sec > best_speed->prediction.tokens_per_sec) {
            best_speed = &r;
        }
        
        if (!best_context || r.strategy.context_length > best_context->strategy.context_length) {
            best_context = &r;
        }
        
        // Best memory = least total memory
        uint64_t total_mem = r.prediction.memory_vram_bytes + r.prediction.memory_ram_bytes;
        uint64_t best_mem_total = best_memory ? (best_memory->prediction.memory_vram_bytes + best_memory->prediction.memory_ram_bytes) : UINT64_MAX;
        if (!best_memory || total_mem < best_mem_total) {
            best_memory = &r;
        }
    }
    
    if (best_speed) {
        // Find row number for best_speed
        int best_row = 0;
        for (size_t i = 0; i < results.size(); i++) {
            if (&results[i] == best_speed) {
                best_row = static_cast<int>(i) + 1;
                break;
            }
        }
        
        printf("💡 For fastest generation: Strategy #%d (%s, %uK, %s) — ~%.0f tok/s\n",
               best_row,
               best_speed->strategy.placement == PlacementStrategy::FULL_GPU ? "Full GPU" :
               best_speed->strategy.placement == PlacementStrategy::GPU_CPU_SPLIT ? "Split" : "CPU Only",
               best_speed->strategy.context_length / 1024,
               best_speed->strategy.kv_quant_bits == 16 ? "FP16" : "Q8",
               best_speed->prediction.tokens_per_sec);
    }
    
    if (best_context && best_context != best_speed) {
        int ctx_row = 0;
        for (size_t i = 0; i < results.size(); i++) {
            if (&results[i] == best_context) {
                ctx_row = static_cast<int>(i) + 1;
                break;
            }
        }
        
        printf("💡 For long documents: Strategy #%d keeps everything on %s with %uK context\n",
               ctx_row,
               best_context->strategy.placement == PlacementStrategy::FULL_GPU ? "GPU" : "CPU",
               best_context->strategy.context_length / 1024);
    }
    
    // Q8 vs FP16 tip
    printf("💡 Q8 KV cache saves ~0.3GB VRAM at negligible quality cost for this model size.\n");
}

// =============================================================================
// Sort and Filter
// =============================================================================

void sort_by_priority(std::vector<StrategyResult>& results, PriorityMode priority) {
    std::sort(results.begin(), results.end(),
        [priority](const StrategyResult& a, const StrategyResult& b) {
            if (a.prediction.viable != b.prediction.viable) {
                return a.prediction.viable > b.prediction.viable;
            }
            switch (priority) {
                case PriorityMode::SPEED:
                    return a.prediction.tokens_per_sec > b.prediction.tokens_per_sec;
                case PriorityMode::QUALITY:
                    if (a.strategy.context_length != b.strategy.context_length) {
                        return a.strategy.context_length > b.strategy.context_length;
                    }
                    return a.prediction.tokens_per_sec > b.prediction.tokens_per_sec;
                case PriorityMode::SAFETY:
                    if (a.prediction.confidence != b.prediction.confidence) {
                        return static_cast<int>(a.prediction.confidence) > static_cast<int>(b.prediction.confidence);
                    }
                    return a.prediction.tokens_per_sec > b.prediction.tokens_per_sec;
                default:
                    return a.prediction.tokens_per_sec > b.prediction.tokens_per_sec;
            }
        });
}

std::vector<StrategyResult> filter_by_context(const std::vector<StrategyResult>& results, ContextMode mode, uint32_t model_max_ctx) {
    if (mode == ContextMode::BOTH) return results;
    std::vector<StrategyResult> filtered;
    uint32_t target_ctx = (mode == ContextMode::FOUR_K) ? 4096 : model_max_ctx;
    for (const auto& r : results) {
        if (r.strategy.context_length == target_ctx) {
            filtered.push_back(r);
        }
    }
    return filtered;
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char* argv[]) {
    // Parse arguments
    std::string model_path;
    PriorityMode priority = PriorityMode::SPEED;
    ContextMode context_mode = ContextMode::BOTH;
    bool verbose = false;
    bool model_specified = false;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage();
            return 0;
        } else if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
            model_specified = true;
        } else if (arg == "--priority" && i + 1 < argc) {
            priority = parse_priority(argv[++i]);
        } else if (arg == "--context" && i + 1 < argc) {
            context_mode = parse_context(argv[++i]);
        } else if (arg == "--verbose") {
            verbose = true;
        } else if (arg[0] != '-' && !model_specified) {
            model_path = arg;
            model_specified = true;
        }
    }
    
    if (!model_specified) {
        fprintf(stderr, "Error: --model <url_or_path> is required\n\n");
        print_usage();
        return 1;
    }
    
    // Step 1: Profile Hardware
    Timer timer;
    printf("\n");
    HardwareSpec hw = profile_hardware(model_path);
    if (verbose) print_hardware_full(hw);
    else print_hardware_brief(hw);
    
    // Step 2: Fetch Model Metadata
    ModelSpec model = fetch_metadata(model_path);
    if (model.layers == 0) {
        fprintf(stderr, "Error: Failed to fetch model metadata from: %s\n", model_path.c_str());
        return 1;
    }
    if (verbose) print_model_full(model);
    else print_model_brief(model);
    
    // Step 3: Generate Matrix
    std::vector<StrategyResult> results = generate_matrix(hw, model);
    results = filter_by_context(results, context_mode, model.context_length);
    
    // Step 4: Rank
    sort_by_priority(results, priority);
    
    // Step 5: Print Table
    print_prediction_table(results, hw, priority);
    
    printf("\n=================================================\n");
    
    return 0;
}