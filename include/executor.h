#pragma once

#include "types.h"
#include <string>
#include <functional>

// =============================================================================
// Step 6 — Executor: llama.cpp Integration
// =============================================================================
// Loads the model, configures it per StrategyConfig, runs inference,
// samples hardware state live, and reports predicted-vs-actual.
//
// Lifecycle:
//   1. llama_backend_init()         — once at start
//   2. llama_model_load_from_file() — load weights from GGUF
//   3. llama_init_from_model()      — create inference context
//   4. [inference loop]             — prompt + token generation
//   5. llama_free()                 — free context
//   6. llama_model_free()           — free model
//   7. llama_backend_free()         — once at shutdown
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

    // Generated text
    std::string generated_text;

    // Status
    bool success = false;
    std::string error_message;
};

// Progress callback for inference loop
using InferProgressCallback = std::function<void(int tokens_generated, double tokens_per_sec)>;

// Initialize llama.cpp backend (call once at program start)
bool executor_init();

// Shutdown llama.cpp backend (call once at program end)
void executor_shutdown();

// Execute inference with given model and strategy
// This is the main entry point — loads model, runs inference, returns results
ExecutionResult execute(const std::string& model_path,
                        const StrategyConfig& strategy,
                        const std::string& prompt,
                        int max_tokens = 50,
                        InferProgressCallback progress = nullptr);

// Get CPU thread count (for n_threads parameter)
int get_cpu_thread_count();
