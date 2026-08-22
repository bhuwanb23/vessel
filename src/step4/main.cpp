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
// LLM Deployment Planner — Step 4: Pipeline Orchestrator (Phase C)
// =============================================================================
// End-to-end flow:
//   1. Profile hardware
//   2. Fetch model metadata
//   3. Generate method matrix
//   4. Predict each strategy
//   5. Rank and display
// =============================================================================

// Priority modes
enum class PriorityMode {
    SPEED,      // Rank by tokens/sec (fastest first)
    QUALITY,    // Rank by context length (longest first)
    SAFETY,     // Rank by confidence (highest first)
    BALANCED    // Default: speed with viability check
};

// Parse priority string
PriorityMode parse_priority(const std::string& str) {
    if (str == "speed") return PriorityMode::SPEED;
    if (str == "quality") return PriorityMode::QUALITY;
    if (str == "safety") return PriorityMode::SAFETY;
    return PriorityMode::BALANCED;
}

// Get priority name
const char* get_priority_name(PriorityMode mode) {
    switch (mode) {
        case PriorityMode::SPEED: return "Speed";
        case PriorityMode::QUALITY: return "Quality";
        case PriorityMode::SAFETY: return "Safety";
        case PriorityMode::BALANCED: return "Balanced";
        default: return "Unknown";
    }
}

void print_usage() {
    printf("Usage: llm-planner <model_url_or_path> [options]\n\n");
    printf("Options:\n");
    printf("  --priority <speed|quality|safety|balanced>  Optimize for (default: balanced)\n");
    printf("  --context <length>                          Override context length\n");
    printf("  --gpu-layers <count>                        Force specific GPU layer count\n");
    printf("  --verbose                                   Print full hardware/model reports\n");
    printf("  --help                                      Show this help\n\n");
    printf("Examples:\n");
    printf("  llm-planner ./models/model.gguf\n");
    printf("  llm-planner ./models/model.gguf --priority speed\n");
    printf("  llm-planner ./models/model.gguf --priority quality --verbose\n");
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
    void reset() { start = std::chrono::high_resolution_clock::now(); }
};

// Print brief hardware summary (1 line)
void print_hardware_brief(const HardwareSpec& hw) {
    printf("Hardware: %s (%.1f GB VRAM, %.1f free) | %.0f GB RAM (%.0f free) | NVMe %.1f GB/s\n",
           hw.gpu_name.c_str(),
           hw.vram_total_bytes / 1e9,
           hw.vram_free_bytes / 1e9,
           hw.ram_total_bytes / 1e9,
           hw.ram_free_bytes / 1e9,
           hw.nvme_sequential_mbs / 1000.0);
}

// Print full hardware report (for --verbose)
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

// Print brief model summary (1 line)
void print_model_brief(const ModelSpec& model) {
    printf("Model: %s %s | %.2fB params | %u layers | %uK context\n",
           model.name.c_str(),
           model.quant_type.c_str(),
           model.param_count / 1e9,
           model.layers,
           model.context_length / 1024);
}

// Print full model report (for --verbose)
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

// Print prediction table
void print_prediction_table(const std::vector<StrategyResult>& results) {
    printf("\n--- Deployment Strategies ---\n\n");
    
    // Header
    printf("%-28s %-10s %-10s %-10s %-10s %-8s\n",
           "Strategy", "VRAM", "RAM", "Speed", "TTFT", "Viable?");
    printf("%-28s %-10s %-10s %-10s %-10s %-8s\n",
           "----------------------------", "----------", "----------", "----------", "----------", "--------");
    
    for (const auto& result : results) {
        const auto& pred = result.prediction;
        
        // Format memory
        char vram_str[32] = "-";
        char ram_str[32] = "-";
        
        if (pred.memory_vram_bytes > 0) {
            if (pred.memory_vram_bytes >= 1e9) {
                sprintf(vram_str, "%.1f GB", pred.memory_vram_bytes / 1e9);
            } else {
                sprintf(vram_str, "%.0f MB", pred.memory_vram_bytes / 1e6);
            }
        }
        
        if (pred.memory_ram_bytes > 0) {
            if (pred.memory_ram_bytes >= 1e9) {
                sprintf(ram_str, "%.1f GB", pred.memory_ram_bytes / 1e9);
            } else {
                sprintf(ram_str, "%.0f MB", pred.memory_ram_bytes / 1e6);
            }
        }
        
        // Format speed
        char speed_str[32] = "-";
        if (pred.tokens_per_sec > 0) {
            sprintf(speed_str, "%.1f t/s", pred.tokens_per_sec);
        }
        
        // Format TTFT
        char ttft_str[32] = "-";
        if (pred.ttft_ms > 0) {
            if (pred.ttft_ms < 1000) {
                sprintf(ttft_str, "%.0f ms", pred.ttft_ms);
            } else {
                sprintf(ttft_str, "%.1f s", pred.ttft_ms / 1000);
            }
        }
        
        printf("%-28s %-10s %-10s %-10s %-10s %-8s\n",
               result.description.c_str(),
               vram_str,
               ram_str,
               speed_str,
               ttft_str,
               pred.viable ? "YES" : "NO");
    }
}

// Sort by priority
void sort_by_priority(std::vector<StrategyResult>& results, PriorityMode priority) {
    std::sort(results.begin(), results.end(),
        [priority](const StrategyResult& a, const StrategyResult& b) {
            // Viable always comes first
            if (a.prediction.viable != b.prediction.viable) {
                return a.prediction.viable > b.prediction.viable;
            }
            
            switch (priority) {
                case PriorityMode::SPEED:
                    return a.prediction.tokens_per_sec > b.prediction.tokens_per_sec;
                    
                case PriorityMode::QUALITY:
                    // Prefer longer context, then faster speed
                    if (a.strategy.context_length != b.strategy.context_length) {
                        return a.strategy.context_length > b.strategy.context_length;
                    }
                    return a.prediction.tokens_per_sec > b.prediction.tokens_per_sec;
                    
                case PriorityMode::SAFETY:
                    // Prefer higher confidence, then faster speed
                    if (a.prediction.confidence != b.prediction.confidence) {
                        return static_cast<int>(a.prediction.confidence) > static_cast<int>(b.prediction.confidence);
                    }
                    return a.prediction.tokens_per_sec > b.prediction.tokens_per_sec;
                    
                case PriorityMode::BALANCED:
                default:
                    // Speed first, but penalize high memory usage
                    return a.prediction.tokens_per_sec > b.prediction.tokens_per_sec;
            }
        });
}

int main(int argc, char* argv[]) {
    printf("\n");
    printf("=================================================\n");
    printf("LLM Deployment Planner\n");
    printf("=================================================\n");
    
    // Parse arguments
    if (argc < 2) {
        print_usage();
        return 1;
    }
    
    std::string model_path = argv[1];
    
    // Check for --help
    if (model_path == "--help" || model_path == "-h") {
        print_usage();
        return 0;
    }
    
    // Parse options
    PriorityMode priority = PriorityMode::BALANCED;
    uint32_t context_override = 0;
    uint32_t gpu_layers_override = 0;
    bool verbose = false;
    
    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--help" || arg == "-h") {
            print_usage();
            return 0;
        } else if (arg == "--verbose") {
            verbose = true;
        } else if (arg == "--priority" && i + 1 < argc) {
            priority = parse_priority(argv[++i]);
        } else if (arg == "--context" && i + 1 < argc) {
            context_override = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--gpu-layers" && i + 1 < argc) {
            gpu_layers_override = static_cast<uint32_t>(std::stoul(argv[++i]));
        }
    }
    
    // =========================================================================
    // Step 1: Profile Hardware (~2-3 seconds for disk benchmark)
    // =========================================================================
    Timer timer;
    printf("\n[1/5] Profiling hardware...\n");
    
    HardwareSpec hw = profile_hardware(model_path);
    
    if (verbose) {
        print_hardware_full(hw);
    } else {
        print_hardware_brief(hw);
    }
    printf("  (%.1f seconds)\n", timer.elapsed_ms() / 1000.0);
    
    // =========================================================================
    // Step 2: Fetch Model Metadata (1-5 seconds network)
    // =========================================================================
    timer.reset();
    printf("\n[2/5] Fetching model metadata...\n");
    
    ModelSpec model = fetch_metadata(model_path);
    
    if (model.layers == 0) {
        printf("Error: Failed to fetch model metadata from: %s\n", model_path.c_str());
        return 1;
    }
    
    if (verbose) {
        print_model_full(model);
    } else {
        print_model_brief(model);
    }
    printf("  (%.1f seconds)\n", timer.elapsed_ms() / 1000.0);
    
    // Apply overrides
    if (context_override > 0) {
        model.context_length = context_override;
    }
    
    // =========================================================================
    // Step 3: Generate Method Matrix (<1 ms)
    // =========================================================================
    timer.reset();
    printf("\n[3/5] Generating deployment strategies...\n");
    
    std::vector<StrategyResult> results;
    
    if (gpu_layers_override > 0) {
        // Generate strategies for specific GPU layer count
        StrategyConfig strat;
        strat.gpu_layers = gpu_layers_override;
        strat.batch_size = 1;
        strat.kv_quant_bits = 16;
        
        uint32_t contexts[] = {4096, model.context_length};
        
        for (uint32_t ctx : contexts) {
            strat.context_length = ctx;
            strat.placement = (gpu_layers_override >= model.layers) ? 
                              PlacementStrategy::FULL_GPU : PlacementStrategy::GPU_CPU_SPLIT;
            
            Prediction pred = predict(hw, model, strat);
            
            StrategyResult result;
            result.strategy = strat;
            result.prediction = pred;
            result.description = format_strategy_description(strat, model.layers);
            
            results.push_back(result);
        }
    } else {
        results = generate_matrix(hw, model);
    }
    
    printf("  %zu strategies generated (%.2f ms)\n", results.size(), timer.elapsed_ms());
    
    // =========================================================================
    // Step 4: Rank by Priority (<1 ms)
    // =========================================================================
    printf("\n[4/5] Ranking strategies by priority: %s\n", get_priority_name(priority));
    sort_by_priority(results, priority);
    
    // =========================================================================
    // Step 5: Display Results
    // =========================================================================
    printf("\n[5/5] Results:\n");
    print_prediction_table(results);
    
    // =========================================================================
    // Recommendation
    // =========================================================================
    printf("\n--- Recommendation ---\n");
    
    // Find best viable strategy
    const StrategyResult* best = nullptr;
    for (const auto& r : results) {
        if (r.prediction.viable) {
            if (!best || r.prediction.tokens_per_sec > best->prediction.tokens_per_sec) {
                best = &r;
            }
        }
    }
    
    if (best) {
        printf("\nBest: %s\n", best->description.c_str());
        printf("  Speed:     %.1f tokens/sec\n", best->prediction.tokens_per_sec);
        printf("  Memory:    %.1f GB VRAM + %.1f GB RAM\n",
               best->prediction.memory_vram_bytes / 1e9,
               best->prediction.memory_ram_bytes / 1e9);
        printf("  TTFT:      %.1f ms\n", best->prediction.ttft_ms);
        printf("  Context:   %u tokens\n", best->strategy.context_length);
        printf("  KV Cache:  %s\n", best->strategy.kv_quant_bits == 16 ? "FP16" : "Q8");
    } else {
        printf("\nNo viable strategies found for this hardware/model combination.\n");
    }
    
    printf("\n=================================================\n");
    
    return 0;
}