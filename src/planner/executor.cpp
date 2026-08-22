#include "executor.h"
#include <llama.h>
#include <ggml-backend.h>
#include <nvml.h>
#include <windows.h>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <thread>
#include <vector>
#include <atomic>
#include <mutex>

// =============================================================================
// NVML State (for live hardware sampling)
// =============================================================================

static bool nvml_initialized = false;
static nvmlDevice_t nvml_device = nullptr;

// =============================================================================
// D1: Backend Lifecycle
// =============================================================================

bool executor_init() {
    // Initialize llama.cpp backends (CPU + CUDA)
    llama_backend_init();
    ggml_backend_load_all();

    // Initialize NVML for live hardware sampling
    nvmlReturn_t ret = nvmlInit();
    if (ret == NVML_SUCCESS) {
        nvml_initialized = true;
        unsigned int device_count = 0;
        nvmlDeviceGetCount(&device_count);
        if (device_count > 0) {
            nvmlDeviceGetHandleByIndex(0, &nvml_device);
        }
    }

    return true;
}

void executor_shutdown() {
    if (nvml_initialized) {
        nvmlShutdown();
        nvml_initialized = false;
    }
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
// Live Hardware Sampler (D5)
// =============================================================================

struct HardwareSample {
    uint64_t vram_used_bytes = 0;
    uint64_t ram_used_bytes = 0;
    double gpu_temp_c = 0.0;
    double gpu_power_w = 0.0;
};

class HardwareSampler {
    std::thread thread_;
    std::atomic<bool> running_{false};
    std::vector<HardwareSample> samples_;
    std::mutex mutex_;

public:
    void start() {
        running_ = true;
        samples_.clear();
        thread_ = std::thread([this]() { poll_loop(); });
    }

    void stop() {
        running_ = false;
        if (thread_.joinable()) thread_.join();
    }

    HardwareSample get_peak() const {
        HardwareSample peak;
        for (const auto& s : samples_) {
            if (s.vram_used_bytes > peak.vram_used_bytes)
                peak.vram_used_bytes = s.vram_used_bytes;
            if (s.ram_used_bytes > peak.ram_used_bytes)
                peak.ram_used_bytes = s.ram_used_bytes;
            if (s.gpu_temp_c > peak.gpu_temp_c)
                peak.gpu_temp_c = s.gpu_temp_c;
            if (s.gpu_power_w > peak.gpu_power_w)
                peak.gpu_power_w = s.gpu_power_w;
        }
        return peak;
    }

    bool was_throttled() const {
        // Check if temperature exceeded 83°C (typical thermal throttle point)
        for (const auto& s : samples_) {
            if (s.gpu_temp_c > 83.0) return true;
        }
        return false;
    }

private:
    void poll_loop() {
        while (running_) {
            HardwareSample sample;

            // Sample VRAM via NVML
            if (nvml_initialized && nvml_device) {
                nvmlMemory_t mem;
                if (nvmlDeviceGetMemoryInfo(nvml_device, &mem) == NVML_SUCCESS) {
                    sample.vram_used_bytes = mem.used;
                }

                // Sample temperature
                unsigned int temp = 0;
                if (nvmlDeviceGetTemperature(nvml_device, NVML_TEMPERATURE_GPU, &temp) == NVML_SUCCESS) {
                    sample.gpu_temp_c = static_cast<double>(temp);
                }

                // Sample power
                unsigned int power = 0;
                if (nvmlDeviceGetPowerUsage(nvml_device, &power) == NVML_SUCCESS) {
                    sample.gpu_power_w = power / 1000.0;  // milliwatts to watts
                }
            }

            // Sample RAM (approximate via GlobalMemoryStatusEx)
            MEMORYSTATUSEX mem_status;
            mem_status.dwLength = sizeof(MEMORYSTATUSEX);
            if (GlobalMemoryStatusEx(&mem_status)) {
                sample.ram_used_bytes = mem_status.ullTotalPhys - mem_status.ullAvailPhys;
            }

            {
                std::lock_guard<std::mutex> lock(mutex_);
                samples_.push_back(sample);
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    }
};

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

static llama_model_params make_model_params(const StrategyConfig& strategy) {
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
    if (batch.n_tokens == 0) return;
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
// Main Execute Function (D1-D5)
// =============================================================================

ExecutionResult execute(const std::string& model_path,
                        const StrategyConfig& strategy,
                        const std::string& prompt,
                        int max_tokens,
                        InferProgressCallback progress) {
    ExecutionResult result;

    auto t_start = std::chrono::high_resolution_clock::now();

    // =========================================================================
    // D2: Load Model
    // =========================================================================
    llama_model_params model_params = make_model_params(strategy);

    struct llama_model* model = llama_model_load_from_file(
        model_path.c_str(), model_params);

    if (!model) {
        result.error_message = "Failed to load model from: " + model_path
            + " (check file exists and VRAM/RAM is sufficient)";
        return result;
    }

    // =========================================================================
    // D3: Create Context
    // =========================================================================
    llama_context_params ctx_params = make_context_params(strategy);

    struct llama_context* ctx = llama_init_from_model(model, ctx_params);

    if (!ctx) {
        result.error_message = "Failed to create inference context"
            " (VRAM/RAM may be insufficient for requested context length)";
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
    sampler.start();

    // =========================================================================
    // Prefill (Prompt Processing)
    // =========================================================================
    auto t_prefill_start = std::chrono::high_resolution_clock::now();

    llama_batch batch = make_batch(n_tokens);

    for (int32_t i = 0; i < n_tokens; i++) {
        batch.token[i]   = tokens[i];
        batch.pos[i]     = i;
        batch.logits[i]  = (i == n_tokens - 1) ? 1 : 0;  // only last needs logits
    }

    if (llama_decode(ctx, batch) != 0) {
        result.error_message = "Failed to decode prompt (out of memory?)";
        free_batch(batch);
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

    free_batch(batch);

    // =========================================================================
    // Decode (Token Generation)
    // =========================================================================
    auto t_decode_start = std::chrono::high_resolution_clock::now();

    std::vector<llama_token> generated;
    llama_token new_token;

    // Get logits from the last position
    float* logits = llama_get_logits_ith(ctx, -1);

    // Greedy sampling (argmax) for MVP
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
            result.error_message = "Failed to decode token at step "
                + std::to_string(i) + " (out of memory?)";
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
    // D5: Stop Hardware Sampler & Collect Results
    // =========================================================================
    sampler.stop();

    HardwareSample peak = sampler.get_peak();
    result.peak_vram_used_bytes = peak.vram_used_bytes;
    result.peak_ram_used_bytes = peak.ram_used_bytes;
    result.peak_gpu_temp_c = peak.gpu_temp_c;
    result.throttled = sampler.was_throttled();

    // =========================================================================
    // Cleanup
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
