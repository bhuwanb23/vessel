#pragma once

#include "types.h"
#include "stream_sampler.h"
#include <string>
#include <functional>
#include <vector>
#include <atomic>

// Forward declare llama.cpp types
struct llama_model;
struct llama_context;

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

// Execute with MoE model metadata (enables tensor override generation)
ExecutionResult execute_moe(const std::string& model_path,
                            const StrategyConfig& strategy,
                            const ModelSpec& model_metadata,
                            const HardwareSpec& hw,
                            const std::string& prompt,
                            int max_tokens = 100,
                            InferProgressCallback progress = nullptr);

// Get CPU thread count (for n_threads parameter)
int get_cpu_thread_count();

// =============================================================================
// Step 14 — Streaming Inference (for API Server)
// =============================================================================
// Uses pre-loaded model+context (owned by ModelManager).
// Calls on_token for each generated token for SSE streaming.
// Clears KV cache before each request.
// =============================================================================

struct StreamingConfig {
    SamplingConfig sampling;
    int max_tokens = 512;
    bool echo_prompt = false;  // Include prompt in output
};

struct StreamCallbacks {
    // Called for each generated token (text)
    std::function<void(const std::string& token_text)> on_token;
    // Called when generation finishes
    std::function<void(const std::string& finish_reason, int prompt_tokens, int completion_tokens)> on_done;
    // Called on error
    std::function<void(const std::string& error)> on_error;
};

// Streaming execute — calls on_token for each generated token.
// Uses pre-loaded model+context (model_manager owns lifecycle).
// Returns true on success, false on error (calls on_error).
bool execute_streaming(
    struct llama_model* model,
    struct llama_context* ctx,
    const std::string& formatted_prompt,
    const StreamingConfig& config,
    StreamCallbacks callbacks
);

// =============================================================================
// Graceful Abort — Ctrl+C handling
// =============================================================================
// Set by signal handler, checked by decode loop every iteration.
// Never call exit() — clean up llama.cpp resources first.
extern std::atomic<bool> abort_requested;

// Register Ctrl+C handler (call once at program start)
void register_abort_handler();

// Check if abort was requested (used in decode loop)
bool is_abort_requested();
