#include "executor.h"
#include "hardware_sampler.h"
#include "moe_placer.h"
#include "moe_tensor_overrides.h"
#include "../hotcold/hotcold_executor.h"
#include "../hotcold/mask_file.h"
#include <llama.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>
#ifdef GGML_USE_CUDA
#include <ggml-cuda.h>
#endif
#include <cstdio>
#include <cstring>
#include <chrono>
#include <thread>
#include <vector>

// =============================================================================
// D1: Backend Lifecycle
// =============================================================================

bool executor_init() {
    // Initialize llama.cpp backends (CPU + CUDA)
    llama_backend_init();
    ggml_backend_load_all();

    // If no backends loaded (static build), register them manually
    if (ggml_backend_reg_count() == 0) {
        fprintf(stderr, "[executor] No backends loaded dynamically, registering statically-linked backends...\n");
        ggml_backend_register(ggml_backend_cpu_reg());
#ifdef GGML_USE_CUDA
        ggml_backend_register(ggml_backend_cuda_reg());
#endif
        fprintf(stderr, "[executor] Registered %zu backend(s)\n", ggml_backend_reg_count());
    }

    // NVML is initialized on first HardwareSampler construction
    return ggml_backend_reg_count() > 0;
}

void executor_shutdown() {
    llama_backend_free();
    // NVML shutdown happens in HardwareSampler destructor
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
// D2: Model Parameters
// =============================================================================

static llama_model_params make_model_params(const StrategyConfig& strategy,
                                             const MoETensorOverrides* moe_overrides = nullptr,
                                             const HotColdExecConfig* hotcold_config = nullptr) {
    llama_model_params params = llama_model_default_params();

    // GPU layers: the most critical field
    if (strategy.placement == PlacementStrategy::FULL_GPU) {
        params.n_gpu_layers = -1;  // all layers on GPU
    } else if (strategy.placement == PlacementStrategy::GPU_CPU_SPLIT) {
        params.n_gpu_layers = static_cast<int32_t>(strategy.gpu_layers);
    } else {
        params.n_gpu_layers = 0;  // CPU only
    }

    params.main_gpu = 0;  // single GPU
    params.vocab_only = false;

    // Load mode: mmap for CPU-only (saves RAM), auto for GPU strategies
    if (strategy.placement == PlacementStrategy::CPU_ONLY) {
        params.load_mode = LLAMA_LOAD_MODE_MMAP;
    } else {
        params.load_mode = LLAMA_LOAD_MODE_AUTO;
    }

    // Apply MoE tensor overrides if available
    if (moe_overrides && moe_overrides->needed) {
        static float moe_split[2] = {0.0f, 0.0f};
        moe_split[0] = moe_overrides->tensor_split[0];
        moe_split[1] = moe_overrides->tensor_split[1];
        params.tensor_split = moe_split;
        
        printf("  MoE tensor split: %.1f%% GPU, %.1f%% CPU\n",
               moe_overrides->tensor_split[0] * 100.0f,
               moe_overrides->tensor_split[1] * 100.0f);
    }
    // Apply hot/cold tensor split if available (and no MoE override)
    else if (hotcold_config && hotcold_config->enabled) {
        static float hc_split[2] = {0.0f, 0.0f};
        hc_split[0] = hotcold_config->tensor_split_gpu;
        hc_split[1] = hotcold_config->tensor_split_cpu;
        params.tensor_split = hc_split;
        
        printf("  Hot/Cold tensor split: %.1f%% GPU, %.1f%% CPU\n",
               hotcold_config->tensor_split_gpu * 100.0f,
               hotcold_config->tensor_split_cpu * 100.0f);
    }

    return params;
}

// =============================================================================
// D3: Context Parameters
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
// Main Execute Function (D1-D10)
// =============================================================================

ExecutionResult execute(const std::string& model_path,
                        const StrategyConfig& strategy,
                        const std::string& prompt,
                        int max_tokens,
                        InferProgressCallback progress) {
    ExecutionResult result;

    auto t_start = std::chrono::high_resolution_clock::now();

    // =========================================================================
    // D1.5: Check for Hot/Cold Mask File
    // =========================================================================
    HotColdExecConfig hotcold_config = create_hotcold_config(model_path);
    
    // =========================================================================
    // D2: Load Model
    // =========================================================================
    // For hot/cold, we need to compute tensor_split BEFORE loading.
    // Use a temporary full load to get dimensions, then reload with tensor_split.
    if (hotcold_config.enabled) {
        // First load: get dimensions only
        llama_model_params dim_params = llama_model_default_params();
        dim_params.n_gpu_layers = 0;
        struct llama_model* dim_model = llama_model_load_from_file(model_path.c_str(), dim_params);
        if (dim_model) {
            int32_t actual_n_embd = llama_model_n_embd(dim_model);
            int32_t actual_n_layer = llama_model_n_layer(dim_model);
            
            if (actual_n_embd > 0 && actual_n_layer > 0) {
                uint32_t actual_ffn = hotcold_config.profile.ffn_dim;
                if (actual_ffn == 0) actual_ffn = actual_n_embd * 4;
                
                double bytes_per_param = 4.85 / 8.0;
                
                calculate_tensor_split(
                    hotcold_config.profile,
                    static_cast<uint32_t>(actual_n_embd),
                    actual_ffn,
                    static_cast<uint32_t>(actual_n_layer),
                    bytes_per_param,
                    hotcold_config.tensor_split_gpu,
                    hotcold_config.tensor_split_cpu);
            }
            llama_model_free(dim_model);
        }
        print_hotcold_exec_info(hotcold_config);
    }
    
    llama_model_params model_params = make_model_params(strategy, nullptr,
                                                         hotcold_config.enabled ? &hotcold_config : nullptr);

    struct llama_model* model = llama_model_load_from_file(
        model_path.c_str(), model_params);

    if (!model) {
        result.error_message = "Failed to load model from: " + model_path + "\n"
            "  Possible causes:\n"
            "  1. File does not exist or is corrupted\n"
            "  2. Not enough VRAM/RAM for requested n_gpu_layers ("
            + std::to_string(strategy.gpu_layers) + " layers)\n"
            "  3. GGUF version mismatch with this llama.cpp build\n"
            "  Try: reduce --model gpu_layers, close other GPU apps, "
            "or use a smaller model.";
        return result;
    }

    // =========================================================================
    // D3: Create Context
    // =========================================================================
    llama_context_params ctx_params = make_context_params(strategy);

    struct llama_context* ctx = llama_init_from_model(model, ctx_params);

    if (!ctx) {
        result.error_message = "Failed to create inference context\n"
            "  Possible causes:\n"
            "  1. Not enough VRAM/RAM for context length "
            + std::to_string(strategy.context_length) + " with "
            + std::to_string(strategy.kv_quant_bits) + "-bit KV cache\n"
            "  2. GPU driver issue (try restarting the driver)\n"
            "  Try: reduce --context or increase kv_quant_bits.";
        llama_model_free(model);
        return result;
    }

    // =========================================================================
    // D4: Tokenize Prompt
    // =========================================================================
    const llama_vocab* vocab = llama_model_get_vocab(model);

    // Allocate buffer with headroom
    std::vector<llama_token> tokens(prompt.size() + 32);

    // Tokenize: add_special=true (adds BOS), parse_special=true
    int32_t n_tokens = llama_tokenize(
        vocab, prompt.c_str(), static_cast<int32_t>(prompt.size()),
        tokens.data(), static_cast<int32_t>(tokens.size()),
        true,   // add_special — adds BOS token for Llama models
        true);  // parse_special — handles <|start_header_id|> etc.

    if (n_tokens < 0) {
        result.error_message = "Failed to tokenize prompt (error: "
            + std::to_string(n_tokens) + ")";
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
    // D5: Start Live Hardware Sampler
    // =========================================================================
    HardwareSampler sampler;
    sampler.start(0);  // pass max_clock_mhz from Step 1 if available

    // =========================================================================
    // D6: Prefill (Prompt Processing)
    // =========================================================================
    auto t_prefill_start = std::chrono::high_resolution_clock::now();

    // Use llama_batch_get_one — the official API for single-sequence batches
    llama_batch batch = llama_batch_get_one(tokens.data(), n_tokens);

    if (llama_decode(ctx, batch) != 0) {
        result.error_message = "Failed to decode prompt (" + std::to_string(n_tokens)
            + " tokens). Error: CUDA OOM or compute buffer too small."
            " Try reducing context length or using fewer GPU layers.";
        sampler.stop();
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

    // =========================================================================
    // D7: Decode Loop (Token Generation)
    // =========================================================================
    auto t_decode_start = std::chrono::high_resolution_clock::now();

    // Create a greedy sampler chain
    llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    llama_sampler* smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    std::vector<llama_token> generated;
    llama_token new_token;

    for (int i = 0; i < max_tokens; i++) {
        // Check for Ctrl+C abort (Phase H: graceful abort)
        if (is_abort_requested()) {
            fprintf(stderr, "\n  [!] Generation aborted by user (Ctrl+C)\n");
            break;
        }

        // Sample next token using the sampler chain
        new_token = llama_sampler_sample(smpl, ctx, -1);

        // Check for end-of-generation
        if (llama_vocab_is_eog(vocab, new_token)) {
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

        // Feed the token back for next iteration
        llama_batch decode_batch = llama_batch_get_one(&new_token, 1);

        if (llama_decode(ctx, decode_batch) != 0) {
            result.error_message = "Failed to decode token at step "
                + std::to_string(i) + ". Possible causes:\n"
                "  1. CUDA error (driver crash or GPU fell off bus)\n"
                "  2. Memory corruption during generation\n"
                "  3. Hardware fault (check nvidia-smi for XID errors)";
            break;
        }
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
    // D8: Stop Hardware Sampler & Collect Results
    // =========================================================================
    sampler.stop();

    HardwareMetrics hw_metrics = sampler.get_metrics();
    result.peak_vram_used_bytes = hw_metrics.peak_vram_bytes;
    result.peak_ram_used_bytes = hw_metrics.peak_ram_bytes;
    result.peak_gpu_temp_c = hw_metrics.max_temp_celsius;
    result.throttled = hw_metrics.throttled;

    // Phase H: Report hardware warnings
    if (hw_metrics.gpu_bus_off) {
        fprintf(stderr, "  [!] WARNING: GPU may have fallen off the bus during generation."
                " Performance data is unreliable.\n");
    }
    if (hw_metrics.os_swapping) {
        fprintf(stderr, "  [!] WARNING: OS swapping detected (RAM oversubscribed)."
                " Generation speed may be severely degraded.\n");
    }

    // =========================================================================
    // D9: Cleanup
    // =========================================================================
    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(
        t_end - t_start).count();
    printf("  Total: %.1f ms\n", total_ms);

    // =========================================================================
    // D10: Return ExecutionResult
    // =========================================================================
    result.success = true;
    return result;
}

// =============================================================================
// MoE-Aware Execute (Step 9, Phase C)
// =============================================================================
// Same as execute() but generates and applies MoE tensor overrides.
// This enables expert-level GPU/CPU splitting for MoE models.
// =============================================================================

ExecutionResult execute_moe(const std::string& model_path,
                            const StrategyConfig& strategy,
                            const ModelSpec& model_metadata,
                            const HardwareSpec& hw,
                            const std::string& prompt,
                            int max_tokens,
                            InferProgressCallback progress) {
    ExecutionResult result;
    
    // Generate MoE tensor overrides
    MoETensorOverrides moe_overrides = generateMoETensorOverrides(model_metadata, strategy, hw);
    
    if (moe_overrides.needed) {
        printf("  MoE Placement: %s\n", moe_overrides.summary.c_str());
        printf("  GPU weights: %.2f GB, CPU weights: %.2f GB\n",
               moe_overrides.gpu_weight_bytes / 1e9,
               moe_overrides.cpu_weight_bytes / 1e9);
    }
    
    auto t_start = std::chrono::high_resolution_clock::now();
    
    // =========================================================================
    // D2: Load Model (with MoE overrides)
    // =========================================================================
    llama_model_params model_params = make_model_params(strategy, &moe_overrides);
    
    struct llama_model* model = llama_model_load_from_file(
        model_path.c_str(), model_params);
    
    if (!model) {
        result.error_message = "Failed to load MoE model from: " + model_path + "\n"
            "  Possible causes:\n"
            "  1. File does not exist or is corrupted\n"
            "  2. Not enough VRAM/RAM for requested tensor split\n"
            "  3. GGUF version mismatch with this llama.cpp build\n"
            "  Try: reduce GPU expert count, close other GPU apps, "
            "or use a smaller model.";
        return result;
    }
    
    // =========================================================================
    // D3: Create Context
    // =========================================================================
    llama_context_params ctx_params = make_context_params(strategy);
    
    struct llama_context* ctx = llama_init_from_model(model, ctx_params);
    
    if (!ctx) {
        result.error_message = "Failed to create MoE inference context\n"
            "  Possible causes:\n"
            "  1. Not enough VRAM/RAM for context length "
            + std::to_string(strategy.context_length) + "\n"
            "  2. GPU driver issue (try restarting the driver)\n"
            "  Try: reduce --context or use fewer GPU experts.";
        llama_model_free(model);
        return result;
    }
    
    // =========================================================================
    // D4: Tokenize Prompt
    // =========================================================================
    const llama_vocab* vocab = llama_model_get_vocab(model);
    
    std::vector<llama_token> tokens(prompt.size() + 32);
    
    int32_t n_tokens = llama_tokenize(
        vocab, prompt.c_str(), static_cast<int32_t>(prompt.size()),
        tokens.data(), static_cast<int32_t>(tokens.size()),
        true,   // add_special
        true);  // parse_special
    
    if (n_tokens <= 0) {
        result.error_message = "Failed to tokenize prompt for MoE model";
        llama_free(ctx);
        llama_model_free(model);
        return result;
    }
    
    tokens.resize(n_tokens);
    printf("  Prompt: %d tokens\n", n_tokens);
    
    // =========================================================================
    // D5: Start Live Hardware Sampler
    // =========================================================================
    HardwareSampler sampler;
    sampler.start(0);
    
    // =========================================================================
    // D6: Prefill (Prompt Processing)
    // =========================================================================
    auto t_prefill_start = std::chrono::high_resolution_clock::now();
    
    llama_batch batch = llama_batch_get_one(tokens.data(), n_tokens);
    
    if (llama_decode(ctx, batch) != 0) {
        result.error_message = "Failed to decode MoE prompt. Error: CUDA OOM or compute buffer too small.";
        sampler.stop();
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
    
    // =========================================================================
    // D7: Decode Loop (Token Generation)
    // =========================================================================
    auto t_decode_start = std::chrono::high_resolution_clock::now();
    
    llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    llama_sampler* smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    
    std::vector<llama_token> generated;
    llama_token new_token;
    
    for (int i = 0; i < max_tokens; i++) {
        if (is_abort_requested()) {
            fprintf(stderr, "\n  [!] MoE generation aborted by user (Ctrl+C)\n");
            break;
        }
        
        new_token = llama_sampler_sample(smpl, ctx, -1);
        
        if (llama_vocab_is_eog(vocab, new_token)) {
            break;
        }
        
        generated.push_back(new_token);
        
        if (progress) {
            auto t_now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double, std::milli>(
                t_now - t_decode_start).count();
            double tps = (elapsed > 0) ? ((i + 1) * 1000.0 / elapsed) : 0;
            progress(i + 1, tps);
        }
        
        llama_batch decode_batch = llama_batch_get_one(&new_token, 1);
        
        if (llama_decode(ctx, decode_batch) != 0) {
            result.error_message = "Failed to decode MoE token at step " + std::to_string(i);
            break;
        }
    }
    
    auto t_decode_end = std::chrono::high_resolution_clock::now();
    result.decode_ms = std::chrono::duration<double, std::milli>(
        t_decode_end - t_decode_start).count();
    result.tokens_generated = static_cast<int>(generated.size());
    result.decode_tokens_per_sec = (result.decode_ms > 0) ?
        (result.tokens_generated * 1000.0 / result.decode_ms) : 0;
    
    for (llama_token tok : generated) {
        char buf[256];
        int n = llama_token_to_piece(vocab, tok, buf, sizeof(buf), 0, true);
        if (n > 0) result.generated_text.append(buf, n);
    }
    
    printf("  Generated: %d tokens in %.1f ms (%.1f tokens/sec)\n",
           result.tokens_generated, result.decode_ms, result.decode_tokens_per_sec);
    
    // =========================================================================
    // D8: Stop Hardware Sampler & Collect Results
    // =========================================================================
    sampler.stop();
    
    HardwareMetrics hw_metrics = sampler.get_metrics();
    result.peak_vram_used_bytes = hw_metrics.peak_vram_bytes;
    result.peak_ram_used_bytes = hw_metrics.peak_ram_bytes;
    result.peak_gpu_temp_c = hw_metrics.max_temp_celsius;
    result.throttled = hw_metrics.throttled;
    
    // =========================================================================
    // D9: Cleanup
    // =========================================================================
    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    
    auto t_end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(
        t_end - t_start).count();
    printf("  Total: %.1f ms\n", total_ms);
    
    result.success = true;
    return result;
}

// =============================================================================
// Step 14: Streaming Inference (for API Server)
// =============================================================================
// Uses pre-loaded model+context. Clears KV cache, tokenizes,
// prefills, then decodes one token at a time with callback.
// =============================================================================

bool execute_streaming(
    struct llama_model* model,
    struct llama_context* ctx,
    const std::string& formatted_prompt,
    const StreamingConfig& config,
    StreamCallbacks callbacks) {

    if (!model || !ctx) {
        if (callbacks.on_error) callbacks.on_error("Model or context is null");
        return false;
    }

    if (formatted_prompt.empty()) {
        if (callbacks.on_error) callbacks.on_error("Prompt is empty");
        return false;
    }

    // Clear KV cache for this request
    llama_memory_clear(llama_get_memory(ctx), true);

    // Tokenize prompt
    const llama_vocab* vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tokens(formatted_prompt.size() + 32);

    int32_t n_tokens = llama_tokenize(
        vocab, formatted_prompt.c_str(), static_cast<int32_t>(formatted_prompt.size()),
        tokens.data(), static_cast<int32_t>(tokens.size()),
        true,   // add_special
        true);  // parse_special

    if (n_tokens < 0) {
        if (callbacks.on_error) callbacks.on_error("Failed to tokenize prompt (error: " + std::to_string(n_tokens) + ")");
        return false;
    }

    if (n_tokens == 0) {
        if (callbacks.on_error) callbacks.on_error("Prompt tokenized to 0 tokens");
        return false;
    }

    tokens.resize(n_tokens);

    // Prefill
    llama_batch batch = llama_batch_get_one(tokens.data(), n_tokens);
    if (llama_decode(ctx, batch) != 0) {
        if (callbacks.on_error) callbacks.on_error("Failed to decode prompt (" + std::to_string(n_tokens) + " tokens). CUDA OOM or compute buffer too small.");
        return false;
    }

    // Build sampler chain
    llama_sampler* smpl = build_sampler_chain(config.sampling);

    // Decode loop
    std::vector<llama_token> generated;
    llama_token new_token;
    int max_tokens = config.max_tokens;
    std::string generated_text;
    bool stopped = false;

    for (int i = 0; i < max_tokens; i++) {
        // Check for abort
        if (is_abort_requested()) break;

        // Sample next token
        new_token = llama_sampler_sample(smpl, ctx, -1);

        // Check for end of generation
        if (llama_vocab_is_eog(vocab, new_token)) {
            stopped = true;
            break;
        }

        generated.push_back(new_token);

        // Convert token to text
        char buf[256];
        int n = llama_token_to_piece(vocab, new_token, buf, sizeof(buf), 0, true);
        std::string token_text;
        if (n > 0) {
            token_text.assign(buf, n);
            generated_text += token_text;
        }

        // Call streaming callback
        if (callbacks.on_token && !token_text.empty()) {
            callbacks.on_token(token_text);
        }

        // Check stop sequences
        if (!config.sampling.stop_sequences.empty()) {
            for (const auto& stop : config.sampling.stop_sequences) {
                if (generated_text.size() >= stop.size()) {
                    if (generated_text.compare(
                            generated_text.size() - stop.size(),
                            stop.size(), stop) == 0) {
                        stopped = true;
                        break;
                    }
                }
            }
            if (stopped) break;
        }

        // Feed token back for next iteration
        llama_batch decode_batch = llama_batch_get_one(&new_token, 1);
        if (llama_decode(ctx, decode_batch) != 0) {
            if (callbacks.on_error) callbacks.on_error("Failed to decode token at step " + std::to_string(i));
            llama_sampler_free(smpl);
            return false;
        }
    }

    llama_sampler_free(smpl);

    // Determine finish reason
    std::string finish_reason = stopped ? "stop" : "length";

    // Report completion
    if (callbacks.on_done) {
        callbacks.on_done(finish_reason, n_tokens, static_cast<int>(generated.size()));
    }

    return true;
}
