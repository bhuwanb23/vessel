// =============================================================================
// Executor Smoke Test — Step 6 Phase A + Phase D
// =============================================================================
// Tests: model loading, context creation, tokenization, inference,
//        live hardware sampling, and cleanup.
// =============================================================================

#include "executor.h"
#include "types.h"
#include <cstdio>
#include <string>

int main(int argc, char* argv[]) {
    printf("=== Executor Smoke Test (Step 6) ===\n\n");

    // Default model path
    std::string model_path = "D:\\projects\\software\\local_llm\\models\\Llama-3.2-3B-Instruct-Q4_K_M.gguf";

    if (argc > 1) {
        model_path = argv[1];
    }

    printf("Model: %s\n", model_path.c_str());
    printf("CPU threads: %d\n\n", get_cpu_thread_count());

    // D1: Initialize backend + NVML
    printf("Initializing backend + NVML...\n");
    if (!executor_init()) {
        fprintf(stderr, "Failed to initialize\n");
        return 1;
    }
    printf("  Backend initialized.\n\n");

    // Strategy: Full GPU, 4K context, Q8 KV
    StrategyConfig strategy;
    strategy.placement = PlacementStrategy::FULL_GPU;
    strategy.gpu_layers = 28;
    strategy.context_length = 4096;
    strategy.batch_size = 1;
    strategy.kv_quant_bits = 8;

    printf("Strategy: Full GPU, %u layers, %uK context, Q%d KV\n",
           strategy.gpu_layers, strategy.context_length / 1024,
           strategy.kv_quant_bits);
    printf("Prompt: \"The capital of France is\"\n");
    printf("Max tokens: 50\n\n");

    // Execute
    printf("--- Running Inference ---\n");
    ExecutionResult result = execute(
        model_path,
        strategy,
        "The capital of France is",
        50,
        nullptr
    );

    // Print results
    printf("\n--- Results ---\n");
    if (result.success) {
        printf("Status:            SUCCESS\n");
        printf("Prompt eval:       %.1f ms (%.1f tokens/sec)\n",
               result.prompt_eval_ms, result.prompt_eval_tokens_per_sec);
        printf("Decode:            %.1f ms (%.1f tokens/sec)\n",
               result.decode_ms, result.decode_tokens_per_sec);
        printf("Tokens generated:  %d\n", result.tokens_generated);
        printf("Peak VRAM used:    %.2f GB\n", result.peak_vram_used_bytes / 1e9);
        printf("Peak RAM used:     %.2f GB\n", result.peak_ram_used_bytes / 1e9);
        printf("Peak GPU temp:     %.0f C\n", result.peak_gpu_temp_c);
        printf("Throttled:         %s\n", result.throttled ? "YES" : "No");
        printf("Generated text:    \"%s\"\n", result.generated_text.c_str());
    } else {
        printf("Status:            FAILED\n");
        printf("Error:             %s\n", result.error_message.c_str());
    }

    // Shutdown
    executor_shutdown();
    printf("\nBackend shutdown.\n");

    return result.success ? 0 : 1;
}
