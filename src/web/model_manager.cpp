#include "model_manager.h"
#include "profiler.h"
#include "fetcher.h"
#include "matrix.h"
#include "ranker.h"
#include "calibration_log.h"
#include "calibration_aggregator.h"

#include <llama.h>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <algorithm>
#include <chrono>
#include <fstream>

namespace fs = std::filesystem;

// =============================================================================
// Model Manager — Destructor
// =============================================================================

ModelManager::~ModelManager() {
    for (auto& session : models_) {
        if (session.ctx) {
            llama_free(session.ctx);
            session.ctx = nullptr;
        }
        if (session.model) {
            llama_model_free(session.model);
            session.model = nullptr;
        }
    }
}

// =============================================================================
// Model ID Generation
// =============================================================================

std::string ModelManager::model_id_from_path(const std::string& path) {
    // Extract filename without extension
    fs::path p(path);
    std::string stem = p.stem().string();

    // Convert to lowercase, replace spaces/underscores with hyphens
    std::string id;
    for (char c : stem) {
        if (c == ' ' || c == '_') {
            if (!id.empty() && id.back() != '-') {
                id += '-';
            }
        } else {
            id += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
    }
    // Remove trailing hyphens
    while (!id.empty() && id.back() == '-') id.pop_back();
    return id;
}

std::string ModelManager::display_name_from_path(const std::string& path) {
    fs::path p(path);
    std::string stem = p.stem().string();
    // Replace underscores with spaces for readability
    for (char& c : stem) {
        if (c == '_') c = ' ';
    }
    return stem;
}

// =============================================================================
// Auto-Strategy Selection
// =============================================================================

StrategyConfig ModelManager::auto_select_strategy(const ModelSpec& model) {
    // Use existing matrix + ranker to find the best strategy
    CalibrationAggregator cal_agg(hw_.hardware_fingerprint);
    CalibrationData cal = cal_agg.get_calibration_data();

    std::vector<StrategyResult> results = generate_matrix(hw_, model, cal);
    sort_by_priority(results, PriorityMode::SPEED, hw_);

    if (!results.empty()) {
        return results[0].strategy;
    }

    // Fallback: Full GPU with 4K context
    StrategyConfig fallback;
    fallback.placement = PlacementStrategy::FULL_GPU;
    fallback.gpu_layers = model.layers;
    fallback.context_length = 4096;
    fallback.kv_quant_bits = 16;
    return fallback;
}

// =============================================================================
// Initialization — Scan Directories
// =============================================================================

bool ModelManager::init(const std::vector<std::string>& model_dirs, int context_length) {
    if (initialized_) return true;

    // Profile hardware once
    std::string fp;
    hw_ = profile_hardware(fp);

    printf("[ModelManager] Hardware: %s\n", hw_.gpu_name.empty() ? "CPU-only" : hw_.gpu_name.c_str());
    printf("[ModelManager] RAM: %.1f GB free / %.1f GB total\n",
           hw_.ram_free_bytes / 1e9, hw_.ram_total_bytes / 1e9);
    if (hw_.vram_total_bytes > 0) {
        printf("[ModelManager] VRAM: %.1f GB free / %.1f GB total\n",
               hw_.vram_free_bytes / 1e9, hw_.vram_total_bytes / 1e9);
    }

    // Scan each directory for .gguf files
    for (const auto& dir : model_dirs) {
        if (!fs::exists(dir) || !fs::is_directory(dir)) {
            printf("[ModelManager] Skipping directory (not found): %s\n", dir.c_str());
            continue;
        }

        printf("[ModelManager] Scanning: %s\n", dir.c_str());

        for (const auto& entry : fs::directory_iterator(dir)) {
            if (!entry.is_regular_file()) continue;

            std::string ext = entry.path().extension().string();
            // Case-insensitive extension check
            for (auto& c : ext) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            if (ext != ".gguf") continue;

            std::string path = entry.path().string();
            std::string id = model_id_from_path(path);

            // Check for duplicate
            bool duplicate = false;
            for (const auto& existing : models_) {
                if (existing.model_id == id) {
                    duplicate = true;
                    break;
                }
            }
            if (duplicate) continue;

            // Create session
            ModelSession session;
            session.model_id = id;
            session.file_path = path;
            session.display_name = display_name_from_path(path);
            session.created_at = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();

            // Fetch metadata from local GGUF file
            printf("[ModelManager]   Loading metadata: %s\n", entry.path().filename().string().c_str());
            session.metadata = fetch_gguf_metadata(path);

            if (session.metadata.layers == 0) {
                printf("[ModelManager]   WARNING: Could not read metadata from %s, skipping\n",
                       entry.path().filename().string().c_str());
                continue;
            }

            // Auto-select best strategy
            session.best_strategy = auto_select_strategy(session.metadata);

            // Quick prediction for display
            CalibrationAggregator cal_agg(hw_.hardware_fingerprint);
            CalibrationData cal = cal_agg.get_calibration_data();
            session.prediction = predict(hw_, session.metadata, session.best_strategy, cal);

            printf("[ModelManager]   Found: %s (%.1fB params, %s, %s)\n",
                   session.display_name.c_str(),
                   session.metadata.param_count / 1e9,
                   session.metadata.quant_type.c_str(),
                   session.prediction.viable ? "VIABLE" : "NOT VIABLE");

            models_.push_back(std::move(session));
        }
    }

    printf("[ModelManager] Found %zu model(s)\n", models_.size());
    initialized_ = true;
    return true;
}

// =============================================================================
// List Models
// =============================================================================

std::vector<const ModelSession*> ModelManager::list_models() const {
    std::vector<const ModelSession*> result;
    result.reserve(models_.size());
    for (const auto& s : models_) {
        result.push_back(&s);
    }
    return result;
}

// =============================================================================
// Find Model by ID
// =============================================================================

ModelSession* ModelManager::find_model(const std::string& model_id) {
    // Exact match first
    for (auto& s : models_) {
        if (s.model_id == model_id) return &s;
    }

    // Case-insensitive fallback
    std::string lower_id = model_id;
    for (auto& c : lower_id) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    for (auto& s : models_) {
        if (s.model_id == lower_id) return &s;
    }

    // Partial match (model_id is a substring of a session ID)
    for (auto& s : models_) {
        if (s.model_id.find(lower_id) != std::string::npos) return &s;
    }

    return nullptr;
}

// =============================================================================
// Ensure Model Loaded (On-Demand)
// =============================================================================

bool ModelManager::ensure_loaded(ModelSession& session) {
    if (session.loaded && session.model && session.ctx) {
        return true;
    }
    return load_model(session);
}

// =============================================================================
// Load Model
// =============================================================================

bool ModelManager::load_model(ModelSession& session) {
    printf("[ModelManager] Loading model: %s\n", session.display_name.c_str());

    auto t0 = std::chrono::high_resolution_clock::now();

    // Build model params from auto-selected strategy
    llama_model_params model_params = llama_model_default_params();
    if (session.best_strategy.placement == PlacementStrategy::FULL_GPU) {
        model_params.n_gpu_layers = -1;
    } else if (session.best_strategy.placement == PlacementStrategy::GPU_CPU_SPLIT) {
        model_params.n_gpu_layers = static_cast<int32_t>(session.best_strategy.gpu_layers);
    } else {
        model_params.n_gpu_layers = 0;
    }
    model_params.load_mode = (session.best_strategy.placement == PlacementStrategy::CPU_ONLY)
        ? LLAMA_LOAD_MODE_MMAP : LLAMA_LOAD_MODE_AUTO;

    // Load model
    session.model = llama_model_load_from_file(session.file_path.c_str(), model_params);
    if (!session.model) {
        fprintf(stderr, "[ModelManager] Failed to load model: %s\n", session.file_path.c_str());
        return false;
    }

    // Build context params
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = session.best_strategy.context_length > 0
        ? session.best_strategy.context_length : 4096;
    ctx_params.n_batch = 512;
    ctx_params.n_ubatch = 512;
    ctx_params.n_threads = get_cpu_thread_count();
    ctx_params.n_threads_batch = get_cpu_thread_count();
    ctx_params.type_k = (session.best_strategy.kv_quant_bits == 8) ? GGML_TYPE_Q8_0 : GGML_TYPE_F16;
    ctx_params.type_v = ctx_params.type_k;
    ctx_params.offload_kqv = (session.best_strategy.gpu_layers > 0);
    ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;

    // Create context
    session.ctx = llama_init_from_model(session.model, ctx_params);
    if (!session.ctx) {
        fprintf(stderr, "[ModelManager] Failed to create context for: %s\n", session.display_name.c_str());
        llama_model_free(session.model);
        session.model = nullptr;
        return false;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double load_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    session.loaded = true;
    printf("[ModelManager] Loaded %s in %.0f ms (ctx=%d, gpu_layers=%d)\n",
           session.display_name.c_str(), load_ms,
           ctx_params.n_ctx, model_params.n_gpu_layers);

    return true;
}

// =============================================================================
// Unload Model
// =============================================================================

void ModelManager::unload(const std::string& model_id) {
    for (auto& s : models_) {
        if (s.model_id == model_id) {
            std::lock_guard<std::mutex> lock(s.infer_mutex);
            if (s.ctx) {
                llama_free(s.ctx);
                s.ctx = nullptr;
            }
            if (s.model) {
                llama_model_free(s.model);
                s.model = nullptr;
            }
            s.loaded = false;
            printf("[ModelManager] Unloaded: %s\n", s.display_name.c_str());
            return;
        }
    }
}

// =============================================================================
// Loaded Count
// =============================================================================

int ModelManager::loaded_count() const {
    int count = 0;
    for (const auto& s : models_) {
        if (s.loaded) count++;
    }
    return count;
}
