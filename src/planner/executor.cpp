#include "executor.h"
#include <llama.h>
#include <ggml-backend.h>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <thread>
#include <vector>

// =============================================================================
// Backend Lifecycle
// =============================================================================

bool executor_init() {
    llama_backend_init();
    ggml_backend_load_all();  // Load CPU + CUDA backends
    return true;
}

void executor_shutdown() {
    llama_backend_free();
}

// =============================================================================
// CPU Thread Count
// =============================================================================

int get_cpu_thread_count() {
    unsigned int n = std::thread::hardware_concurrency();
    // Use physical cores (not hyperthreads) for decode
    // Rough heuristic: divide by 2 for hyperthreading
    if (n > 4) n = n / 2;
    return static_cast<int>(n);
}

// =============================================================================
// KV Cache Type Mapping
// =============================================================================

static enum ggml_type kv_bits_to_ggml_type(uint32_t kv_bits) {
    switch (kv_bits) {
        case 16: return GGML_TYPE_F16;
        case 8:  return GGML_TYPE_Q8_0;
        case 4:  return GGML_TYPE_Q4_0;
        default: return GGML_TYPE_F16;
    }
}

// =============================================================================
// Model Parameters
// =============================================================================

static llama_model_params make_model_params(const StrategyConfig& strategy) {
    llama_model_params params = llama_model_default_params();

    // GPU layers: the most critical field
    // -1 = all layers on GPU, 0 = CPU only
    if (strategy.placement == PlacementStrategy::FULL_GPU) {
        params.n_gpu_layers = -1;  // all layers
    } else if (strategy.placement == PlacementStrategy::GPU_CPU_SPLIT) {
        params.n_gpu_layers = static_cast<int32_t>(strategy.gpu_layers);
    } else {
        params.n_gpu_layers = 0;  // CPU only
    }

    params.main_gpu = 0;  // single GPU
    params.vocab_only = false;

    // Use mmap for CPU-heavy strategies (saves RAM)
    // Don't use mmap for full GPU (GPU needs contiguous memory)
    // Note: llama_model_params doesn't have use_mmap directly,
    // it's controlled by load_mode

    return params;
}

// =============================================================================
// Context Parameters
// =============================================================================

static llama_context_params make_context_params(const StrategyConfig& strategy) {
    llama_context_params params = llama_context_default_params();

    params.n_ctx = strategy.context_length;
    params.n_batch = 512;        // prompt processing batch size
    params.n_ubatch = 512;       // physical batch size
    params.n_threads = get_cpu_thread_count();
    params.n_threads_batch = get_cpu_thread_count();

    // KV cache precision
    params.type_k = kv_bits_to_ggml_type(strategy.kv_quant_bits);
    params.type_v = kv_bits_to_ggml_type(strategy.kv_quant_bits);

    // GPU offload KQV
    params.offload_kqv = (strategy.gpu_layers > 0);

    // Flash attention (saves VRAM on long contexts)
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;

    return params;
}

// =============================================================================
// Batch Helpers
// =============================================================================

static llama_batch make_batch(int32_t n_tokens) {
    llama_batch batch = {};
    batch.n_tokens = n_tokens;
    batch.token    = new llama_token[n_tokens];
    batch.pos      = new llama_pos[n_tokens];
    batch.n_seq_id = new int32_t[n_tokens];
    batch.seq_id   = new llama_seq_id*[n_tokens];
    batch.logits   = new int8_t[n_tokens];

    for (int32_t i = 0; i < n_tokens; i++) {
        batch.n_seq_id[i] = 1;
        batch.seq_id[i]   = new llama_seq_id[1];
        batch.seq_id[i][0] = 0;
    }

    return batch;
}

static void free_batch(llama_batch& batch) {
    delete[] batch.token;
    delete[] batch.pos;
    delete[] batch.n_seq_id;
    for (int32_t i = 0; i < batch.n_tokens; i++) {
        delete[] batch.seq_id[i];
    }
    delete[] batch.seq_id;
    delete[] batch.logits;
    batch = {};
}

// =============================================================================
// Main Execute Function
// =============================================================================

ExecutionResult execute(const std::string& model_path,
                        const StrategyConfig& strategy,
                        const std::string& prompt,
                        int max_tokens,
                        InferProgressCallback progress) {
    ExecutionResult result;

    auto t_start = std::chrono::high_resolution_clock::now();

    // =========================================================================
    // 1. Load Model
    // =========================================================================
    llama_model_params model_params = make_model_params(strategy);

    struct llama_model* model = llama_model_load_from_file(
        model_path.c_str(), model_params);

    if (!model) {
        result.error_message = "Failed to load model from: " + model_path;
        return result;
    }

    // =========================================================================
    // 2. Create Context
    // =========================================================================
    llama_context_params ctx_params = make_context_params(strategy);

    struct llama_context* ctx = llama_init_from_model(model, ctx_params);

    if (!ctx) {
        result.error_message = "Failed to create inference context";
        llama_model_free(model);
        return result;
    }

    // =========================================================================
    // 3. Tokenize Prompt
    // =========================================================================
    const llama_vocab* vocab = llama_model_get_vocab(model);

    // Allocate a generous buffer for tokens
    const int32_t MAX_TOKENS = 4096;
    std::vector<llama_token> tokens(MAX_TOKENS);

    // Tokenize with add_special=false, parse_special=true
    int32_t n_tokens = llama_tokenize(
        vocab, prompt.c_str(), static_cast<int32_t>(prompt.size()),
        tokens.data(), MAX_TOKENS, false, true);

    if (n_tokens < 0) {
        // Negative return = error, but also indicates needed size
        result.error_message = "Failed to tokenize prompt (error code: " + std::to_string(n_tokens) + ")";
        llama_free(ctx);
        llama_model_free(model);
        return result;
    }

    if (n_tokens == 0) {
        result.error_message = "Prompt tokenized to 0 tokens";
        llama_free(ctx);
        llama_model_free(model);
        return result;
    }

    tokens.resize(n_tokens);

    printf("  Prompt: %d tokens\n", n_tokens);

    // =========================================================================
    // 4. Prefill (Prompt Processing)
    // =========================================================================
    auto t_prefill_start = std::chrono::high_resolution_clock::now();

    llama_batch batch = make_batch(n_tokens);

    for (int32_t i = 0; i < n_tokens; i++) {
        batch.token[i]   = tokens[i];
        batch.pos[i]     = i;
        batch.logits[i]  = (i == n_tokens - 1) ? 1 : 0;  // only last token needs logits
    }

    if (llama_decode(ctx, batch) != 0) {
        result.error_message = "Failed to decode prompt";
        free_batch(batch);
        llama_free(ctx);
        llama_model_free(model);
        return result;
    }

    auto t_prefill_end = std::chrono::high_resolution_clock::now();
    result.prompt_eval_ms = std::chrono::duration<double, std::milli>(
        t_prefill_end - t_prefill_start).count();
    result.prompt_eval_tokens_per_sec = (result.prompt_eval_ms > 0) ?
        (n_tokens * 1000.0 / result.prompt_eval_ms) : 0;

    printf("  Prefill: %.1f ms (%.1f tokens/sec)\n",
           result.prompt_eval_ms, result.prompt_eval_tokens_per_sec);

    free_batch(batch);

    // =========================================================================
    // 5. Decode (Token Generation)
    // =========================================================================
    auto t_decode_start = std::chrono::high_resolution_clock::now();

    std::vector<llama_token> generated;
    llama_token new_token;

    // Get the logits from the last position
    float* logits = llama_get_logits_ith(ctx, -1);

    // Simple sampling: argmax (greedy) for MVP
    // TODO: Use llama_sampler_chain for proper sampling in later phases
    auto sample_greedy = [](const float* logits, int n_vocab) -> llama_token {
        llama_token best = 0;
        float best_score = logits[0];
        for (int i = 1; i < n_vocab; i++) {
            if (logits[i] > best_score) {
                best_score = logits[i];
                best = i;
            }
        }
        return best;
    };

    int n_vocab = llama_vocab_n_tokens(vocab);

    for (int i = 0; i < max_tokens; i++) {
        // Sample next token
        new_token = sample_greedy(logits, n_vocab);

        // Check for stop token
        if (new_token == llama_vocab_eos(vocab)) {
            break;
        }

        generated.push_back(new_token);

        // Report progress
        if (progress) {
            auto t_now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double, std::milli>(
                t_now - t_decode_start).count();
            double tps = (elapsed > 0) ? ((i + 1) * 1000.0 / elapsed) : 0;
            progress(i + 1, tps);
        }

        // Feed token back for next decode
        llama_batch decode_batch = make_batch(1);
        decode_batch.token[0]   = new_token;
        decode_batch.pos[0]     = n_tokens + i;
        decode_batch.logits[0]  = 1;

        if (llama_decode(ctx, decode_batch) != 0) {
            result.error_message = "Failed to decode token at step " + std::to_string(i);
            free_batch(decode_batch);
            break;
        }

        free_batch(decode_batch);

        // Get logits for next iteration
        logits = llama_get_logits_ith(ctx, -1);
    }

    auto t_decode_end = std::chrono::high_resolution_clock::now();
    result.decode_ms = std::chrono::duration<double, std::milli>(
        t_decode_end - t_decode_start).count();
    result.tokens_generated = static_cast<int>(generated.size());
    result.decode_tokens_per_sec = (result.decode_ms > 0) ?
        (result.tokens_generated * 1000.0 / result.decode_ms) : 0;

    // Convert tokens to text
    for (llama_token tok : generated) {
        char buf[256];
        int n = llama_token_to_piece(vocab, tok, buf, sizeof(buf), 0, true);
        if (n > 0) {
            result.generated_text.append(buf, n);
        }
    }

    printf("  Generated: %d tokens in %.1f ms (%.1f tokens/sec)\n",
           result.tokens_generated, result.decode_ms, result.decode_tokens_per_sec);

    // =========================================================================
    // 6. Cleanup
    // =========================================================================
    llama_free(ctx);
    llama_model_free(model);

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(
        t_end - t_start).count();
    printf("  Total: %.1f ms\n", total_ms);

    result.success = true;
    return result;
}
