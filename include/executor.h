#pragma once

#include "types.h"
#include <string>
#include <functional>
#include <vector>

// =============================================================================
// Step 6 — Executor: llama.cpp Integration
// =============================================================================
// Loads the model, configures it per StrategyConfig, runs inference,
// samples hardware state live, and reports predicted-vs-actual.
//
// Lifecycle:
//   1. executor_init()               — once at start (backends + NVML)
//   2. execute()                     — load model, run inference, cleanup
//   3. executor_shutdown()           — once at end (backends + NVML)
// =============================================================================

// Execution result — actual performance measured during inference
struct ExecutionResult {
    // Timing
    double prompt_eval_ms = 0.0;      // TTFT: time to process prompt
    double prompt_eval_tokens_per_sec = 0.0;
    double decode_ms = 0.0;           // Total generation time
    double decode_tokens_per_sec = 0.0;  // Actual tok/s
    int tokens_generated = 0;

    // Memory (sampled at peak during inference)
    uint64_t peak_vram_used_bytes = 0;
    uint64_t peak_ram_used_bytes = 0;

    // Hardware state during run
    double peak_gpu_temp_c = 0.0;     // Peak GPU temperature (Celsius)
    bool throttled = false;           // Did GPU thermal throttle?

    // Generated text
    std::string generated_text;

    // Status
    bool success = false;
    std::string error_message;
};

// Progress callback for inference loop
using InferProgressCallback = std::function<void(int tokens_generated, double tokens_per_sec)>;

// Initialize llama.cpp backend + NVML (call once at program start)
bool executor_init();

// Shutdown llama.cpp backend + NVML (call once at program end)
void executor_shutdown();

// Execute inference with given model and strategy
// This is the main entry point — loads model, runs inference, returns results
ExecutionResult execute(const std::string& model_path,
                        const StrategyConfig& strategy,
                        const std::string& prompt,
                        int max_tokens = 100,
                        InferProgressCallback progress = nullptr);

// Get CPU thread count (for n_threads parameter)
int get_cpu_thread_count();
