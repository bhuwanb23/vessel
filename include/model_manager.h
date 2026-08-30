#pragma once

#include "types.h"
#include "executor.h"
#include <string>
#include <vector>
#include <mutex>
#include <atomic>
#include <memory>

// Forward declare llama.cpp types
struct llama_model;
struct llama_context;

// =============================================================================
// Step 14 — Model Manager
// =============================================================================
// Scans directories for .gguf files, auto-selects the best strategy per model,
// manages model loading/unloading lifecycle, and serializes inference access.
//
// This is the "killer advantage" over Ollama: every model gets hardware-aware
// auto-tuned deployment strategy instead of a fixed config.
// =============================================================================

// One loaded model session
struct ModelSession {
    // Identity
    std::string model_id;          // "llama-3.2-3b-instruct-q4-k-m"
    std::string file_path;         // "/path/to/model.gguf"
    std::string display_name;      // "Llama 3.2 3B Instruct Q4_K_M"
    int64_t created_at = 0;        // Unix timestamp

    // Metadata and strategy (populated at scan time)
    ModelSpec metadata;
    StrategyConfig best_strategy;
    Prediction prediction;

    // llama.cpp objects (populated on demand)
    struct llama_model* model = nullptr;
    struct llama_context* ctx = nullptr;
    std::mutex infer_mutex;        // Serialize inference per model
    bool loaded = false;

    // Usage stats (atomic for thread safety)
    std::atomic<uint64_t> prompt_tokens{0};
    std::atomic<uint64_t> completion_tokens{0};
    std::atomic<uint32_t> request_count{0};
};

// =============================================================================
// Model Manager
// =============================================================================

class ModelManager {
public:
    ModelManager() = default;
    ~ModelManager();

    // Non-copyable
    ModelManager(const ModelManager&) = delete;
    ModelManager& operator=(const ModelManager&) = delete;

    // Initialize: scan directories, profile hardware, auto-select strategies
    bool init(const std::vector<std::string>& model_dirs, int context_length = 4096);

    // List all discovered models (for /v1/models)
    std::vector<const ModelSession*> list_models() const;

    // Find model by ID (for /v1/chat/completions model field)
    ModelSession* find_model(const std::string& model_id);

    // Load model on demand (first request triggers load)
    // Thread-safe: blocks if another thread is loading the same model
    bool ensure_loaded(ModelSession& session);

    // Unload model (free memory)
    void unload(const std::string& model_id);

    // Get hardware profile
    const HardwareSpec& get_hardware() const { return hw_; }

    // Get number of loaded models
    int loaded_count() const;

private:
    HardwareSpec hw_;
    std::vector<ModelSession> models_;
    bool initialized_ = false;

    // Generate model ID from file path
    std::string model_id_from_path(const std::string& path);

    // Generate display name from path
    std::string display_name_from_path(const std::string& path);

    // Auto-select best strategy for a model
    StrategyConfig auto_select_strategy(const ModelSpec& model);

    // Load a GGUF model and create context
    bool load_model(ModelSession& session);
};
