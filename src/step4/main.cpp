#include "types.h"
#include "profiler.h"
#include "fetcher.h"
#include "matrix.h"
#include "../predictor/predictor.h"
#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>

// =============================================================================
// LLM Deployment Planner — Step 4: Wire It Together
// =============================================================================
// Single binary that:
// 1. Profiles hardware
// 2. Fetches model metadata
// 3. Generates all viable strategies
// 4. Predicts performance for each
// 5. Prints comparison table
// =============================================================================

void print_usage() {
    printf("Usage: llm-planner <model_url_or_path> [options]\n\n");
    printf("Options:\n");
    printf("  --priority <speed|memory|balanced>  Optimize for (default: balanced)\n");
    printf("  --context <length>                  Override context length\n");
    printf("  --gpu-layers <count>                Force specific GPU layer count\n");
    printf("  --help                              Show this help\n\n");
    printf("Examples:\n");
    printf("  llm-planner https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf\n");
    printf("  llm-planner ./models/model.gguf --priority speed\n");
    printf("  llm-planner ./models/model.gguf --context 4096 --gpu-layers 28\n");
}

std::string get_timestamp() {
    std::time_t now = std::time(nullptr);
    char buf[64];
    struct tm tm_buf;
    localtime_s(&tm_buf, &now);
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tm_buf);
    return std::string(buf);
}

void print_hardware_summary(const HardwareSpec& hw) {
    printf("\n--- Hardware Profile ---\n");
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
}

void print_model_summary(const ModelSpec& model) {
    printf("\n--- Model Metadata ---\n");
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

void print_method_matrix(const std::vector<StrategyResult>& results) {
    printf("\n--- Deployment Strategies ---\n\n");
    
    // Header
    printf("%-20s %-12s %-12s %-12s %-10s %-8s\n",
           "Strategy", "VRAM", "RAM", "Speed", "TTFT", "Viable?");
    printf("%-20s %-12s %-12s %-12s %-10s %-8s\n",
           "--------------------", "------------", "------------", "------------", "----------", "--------");
    
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
        
        printf("%-20s %-12s %-12s %-12s %-10s %-8s\n",
               result.description.c_str(),
               vram_str,
               ram_str,
               speed_str,
               ttft_str,
               pred.viable ? "YES" : "NO");
    }
}

int main(int argc, char* argv[]) {
    printf("\n");
    printf("=================================================\n");
    printf("LLM Deployment Planner — Step 4\n");
    printf("=================================================\n");
    printf("Timestamp: %s\n", get_timestamp().c_str());
    
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
    
    // Simple argument parsing
    std::string priority = "balanced";
    uint32_t context_override = 0;
    uint32_t gpu_layers_override = 0;
    
    for (int i = 2; i < argc; i += 2) {
        if (i + 1 >= argc) break;
        
        std::string arg = argv[i];
        std::string value = argv[i + 1];
        
        if (arg == "--priority") {
            priority = value;
        } else if (arg == "--context") {
            context_override = static_cast<uint32_t>(std::stoul(value));
        } else if (arg == "--gpu-layers") {
            gpu_layers_override = static_cast<uint32_t>(std::stoul(value));
        }
    }
    
    // =========================================================================
    // Step 1: Profile Hardware
    // =========================================================================
    printf("\n[1/4] Profiling hardware...\n");
    HardwareSpec hw = profile_hardware(model_path);
    print_hardware_summary(hw);
    
    // =========================================================================
    // Step 2: Fetch Model Metadata
    // =========================================================================
    printf("\n[2/4] Fetching model metadata...\n");
    ModelSpec model = fetch_metadata(model_path);
    
    if (model.layers == 0) {
        printf("Error: Failed to fetch model metadata from: %s\n", model_path.c_str());
        return 1;
    }
    
    print_model_summary(model);
    
    // Apply overrides
    if (context_override > 0) {
        model.context_length = context_override;
    }
    
    // =========================================================================
    // Step 3: Generate Method Matrix
    // =========================================================================
    printf("\n[3/4] Generating deployment strategies...\n");
    
    // If GPU layers override, generate limited strategies
    std::vector<StrategyResult> results;
    
    if (gpu_layers_override > 0) {
        // Generate strategies for specific GPU layer count
        StrategyConfig strat;
        strat.gpu_layers = gpu_layers_override;
        strat.batch_size = 1;
        strat.kv_quant_bits = 16;
        
        auto ctx_lengths = get_context_lengths(model.context_length);
        
        for (uint32_t ctx : ctx_lengths) {
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
        // Generate full matrix
        results = generate_matrix(hw, model);
    }
    
    // =========================================================================
    // Step 4: Print Results
    // =========================================================================
    printf("\n[4/4] Predictions complete!\n");
    print_method_matrix(results);
    
    // =========================================================================
    // Summary
    // =========================================================================
    printf("\n--- Summary ---\n");
    
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
        printf("\nRecommended: %s\n", best->description.c_str());
        printf("  Speed:     %.1f tokens/sec\n", best->prediction.tokens_per_sec);
        printf("  Memory:    %.1f GB VRAM + %.1f GB RAM\n",
               best->prediction.memory_vram_bytes / 1e9,
               best->prediction.memory_ram_bytes / 1e9);
        printf("  TTFT:      %.1f ms\n", best->prediction.ttft_ms);
        printf("  Viable:    %s\n", best->prediction.viable ? "YES" : "NO");
    } else {
        printf("\nNo viable strategies found for this hardware/model combination.\n");
    }
    
    printf("\n=================================================\n");
    
    return 0;
}
