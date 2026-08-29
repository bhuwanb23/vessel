// =============================================================================
// Vessel Web Dashboard Server — Phase E: Task Management + SSE Streaming
// =============================================================================

#include "web_server.h"
#include "types.h"
#include "profiler.h"
#include "../predictor/predictor.h"
#include "matrix.h"
#include "ranker.h"
#include "output.h"
#include "fetcher.h"
#include "download_manager.h"
#include "calibration_log.h"
#include "calibration_aggregator.h"
#include "recommend/catalog_loader.h"
#include "recommend/recommendation_engine.h"
#include "platform/platform_factory.h"
#include "executor.h"

#include <cstdio>
#include <string>
#include <mutex>
#include <atomic>
#include <thread>
#include <chrono>
#include <queue>
#include <condition_variable>
#include <filesystem>
#include <unordered_map>
#include <sstream>
#include <memory>

namespace fs = std::filesystem;

static int g_server_port = 8080;

// =============================================================================
// Security: Path Validation (F5)
// =============================================================================

static bool is_safe_model_path(const std::string& path, std::string& error) {
    if (path.empty()) {
        error = "Path is empty";
        return false;
    }

    // Must be an absolute path
    fs::path p(path);
    if (!p.is_absolute()) {
        error = "Path must be absolute (e.g. C:\\models\\model.gguf)";
        return false;
    }

    // Must not contain .. components (path traversal)
    try {
        fs::path canonical = fs::weakly_canonical(p);
        std::string canonical_str = canonical.string();
        // Check for .. after canonicalization (shouldn't happen, but defensive)
        if (canonical_str.find("..") != std::string::npos) {
            error = "Path contains path traversal (..)";
            return false;
        }
    } catch (...) {
        error = "Path could not be resolved";
        return false;
    }

    // Must end in .gguf
    std::string ext = p.extension().string();
    // Handle case-insensitive extension
    for (auto& c : ext) c = (char)tolower((unsigned char)c);
    if (ext != ".gguf") {
        error = "File must be a .gguf model file";
        return false;
    }

    // Must exist and be readable
    std::error_code ec;
    if (!fs::exists(p, ec) || ec) {
        error = "File does not exist: " + path;
        return false;
    }
    if (!fs::is_regular_file(p, ec) || ec) {
        error = "Path is not a regular file: " + path;
        return false;
    }

    return true;
}

// Timer utility
class Timer {
    std::chrono::high_resolution_clock::time_point t0;
public:
    Timer() : t0(std::chrono::high_resolution_clock::now()) {}
    double elapsed_ms() {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(now - t0).count();
    }
};

// SSL disabled
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
#undef CPPHTTPLIB_OPENSSL_SUPPORT
#endif
#include "httplib.h"

#include <nlohmann/json.hpp>

#ifdef _WIN32
#include <windows.h>
#include <shellapi.h>
#endif

using json = nlohmann::json;

// =============================================================================
// Embedded web assets (hex-encoded byte arrays from CMake)
// =============================================================================
#include "embedded_web.h"

// =============================================================================
// JSON Envelope
// =============================================================================
static json makeSuccess(const json& data) {
    return { {"success", true}, {"data", data}, {"error", nullptr} };
}

static json makeError(const std::string& code, const std::string& msg, const std::string& details = "") {
    return { {"success", false}, {"data", nullptr},
             {"error", { {"code", code}, {"message", msg}, {"details", details} }} };
}

static std::string get_cors_origin() {
    return "http://localhost:" + std::to_string(g_server_port);
}

static void sendJson(httplib::Response& res, int status, const json& envelope) {
    res.status = status;
    res.set_header("Access-Control-Allow-Origin", get_cors_origin());
    res.set_header("Content-Type", "application/json");
    res.set_content(envelope.dump(), "application/json");
}

static void sendSSEHeader(httplib::Response& res) {
    res.set_header("Access-Control-Allow-Origin", get_cors_origin());
    res.set_header("Content-Type", "text/event-stream");
    res.set_header("Cache-Control", "no-cache");
    res.set_header("Connection", "keep-alive");
}

// =============================================================================
// Task Management System (Phase E)
// =============================================================================

struct Task {
    std::string id;
    std::string type;               // "download" or "execute"
    std::atomic<bool> running{false};
    std::atomic<bool> aborted{false};

    // Event queue: SSE events waiting to be sent to client
    std::mutex queue_mutex;
    std::queue<std::string> event_queue;
    std::condition_variable queue_cv;

    // Final result
    json result;

    void pushEvent(const std::string& event, const json& data) {
        std::string sse = "event: " + event + "\ndata: " + data.dump() + "\n\n";
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            event_queue.push(sse);
        }
        queue_cv.notify_all();
    }

    void pushEvent(const std::string& raw_sse) {
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            event_queue.push(raw_sse);
        }
        queue_cv.notify_all();
    }

    bool popEvent(std::string& out) {
        std::lock_guard<std::mutex> lock(queue_mutex);
        if (event_queue.empty()) return false;
        out = event_queue.front();
        event_queue.pop();
        return true;
    }

    // Wait for an event or timeout (returns false on timeout)
    bool waitForEvent(std::string& out, int timeout_ms = 1000) {
        std::unique_lock<std::mutex> lock(queue_mutex);
        if (!event_queue.empty()) {
            out = event_queue.front();
            event_queue.pop();
            return true;
        }
        lock.unlock();

        std::unique_lock<std::mutex> cv_lock(queue_mutex);
        queue_cv.wait_for(cv_lock, std::chrono::milliseconds(timeout_ms),
            [this] { return !event_queue.empty() || !running; });

        if (!event_queue.empty()) {
            out = event_queue.front();
            event_queue.pop();
            return true;
        }
        return false;
    }
};

static std::mutex g_tasks_mutex;
static std::unordered_map<std::string, std::shared_ptr<Task>> g_tasks;
static int g_task_counter = 0;

static std::string generateTaskId(const std::string& prefix) {
    std::lock_guard<std::mutex> lock(g_tasks_mutex);
    char buf[32];
    snprintf(buf, sizeof(buf), "%s_%04x_%d", prefix.c_str(),
        (unsigned)(std::chrono::steady_clock::now().time_since_epoch().count() & 0xFFFF),
        g_task_counter++);
    return std::string(buf);
}

static std::shared_ptr<Task> getTask(const std::string& id) {
    std::lock_guard<std::mutex> lock(g_tasks_mutex);
    auto it = g_tasks.find(id);
    return (it != g_tasks.end()) ? it->second : nullptr;
}

// Cross-platform browser opener
static void openBrowser(const std::string& url) {
#ifdef _WIN32
    ShellExecuteA(NULL, "open", url.c_str(), NULL, NULL, SW_SHOWNORMAL);
#elif __APPLE__
    std::system(("open " + url).c_str());
#else
    std::system(("xdg-open " + url + " &").c_str());
#endif
}

// =============================================================================
// JSON Helpers
// =============================================================================
static json hardwareToJson(const HardwareSpec& hw) {
    json j;
    j["platform"] = hw.gpu_name.empty() ? "cpu" : (hw.is_nvidia() ? "nvidia" : hw.is_amd() ? "amd" : hw.is_apple() ? "apple" : "unknown");
    j["gpu_name"] = hw.gpu_name;
    j["vram_total_bytes"] = hw.vram_total_bytes;
    j["vram_free_bytes"] = hw.vram_free_bytes;
    j["vram_used_pct"] = hw.vram_total_bytes > 0 ? 100.0 * (hw.vram_total_bytes - hw.vram_free_bytes) / hw.vram_total_bytes : 0.0;
    j["ram_total_bytes"] = hw.ram_total_bytes;
    j["ram_free_bytes"] = hw.ram_free_bytes;
    j["ram_used_pct"] = hw.ram_total_bytes > 0 ? 100.0 * (hw.ram_total_bytes - hw.ram_free_bytes) / hw.ram_total_bytes : 0.0;
    j["gpu_bandwidth_gbs"] = hw.gpu_bandwidth_gbs;
    j["gpu_tflops_fp16"] = hw.gpu_tflops_fp16;
    j["ram_bandwidth_gbs"] = hw.ram_bandwidth_gbs;
    j["nvme_sequential_mbs"] = hw.nvme_sequential_mbs;
    j["nvme_random_4k_mbs"] = hw.nvme_random_4k_mbs;
    j["gpu_temp_celsius"] = hw.gpu_temp_celsius;
    j["gpu_clock_mhz"] = hw.gpu_clock_mhz;
    j["hardware_fingerprint"] = hw.hardware_fingerprint;
    j["is_unified_memory"] = hw.is_unified_memory;
    return j;
}

static json hardwareToLiveJson(const HardwareSpec& hw) {
    return {
        {"vram_free_bytes", hw.vram_free_bytes},
        {"vram_used_bytes", hw.vram_total_bytes - hw.vram_free_bytes},
        {"ram_free_bytes", hw.ram_free_bytes},
        {"gpu_temp_celsius", hw.gpu_temp_celsius},
        {"gpu_clock_mhz", hw.gpu_clock_mhz},
        {"gpu_utilization", hw.gpu_utilization}
    };
}

static json predictionToJson(const Prediction& p, const StrategyConfig& s, int rank = 0) {
    json j;
    j["rank"] = rank;
    j["vram_bytes"] = p.memory_vram_bytes;
    j["ram_bytes"] = p.memory_ram_bytes;
    j["tokens_per_sec"] = p.tokens_per_sec;
    j["ttft_ms"] = p.ttft_ms;
    j["viable"] = p.viable;
    switch (p.confidence) {
        case PredictionConfidence::HIGH:   j["confidence"] = "HIGH"; break;
        case PredictionConfidence::MEDIUM: j["confidence"] = "MEDIUM"; break;
        default:                           j["confidence"] = "LOW"; break;
    }
    switch (s.placement) {
        case PlacementStrategy::FULL_GPU:       j["placement"] = "FULL_GPU"; break;
        case PlacementStrategy::GPU_CPU_SPLIT:  j["placement"] = "GPU_CPU_SPLIT"; break;
        case PlacementStrategy::CPU_ONLY:       j["placement"] = "CPU_ONLY"; break;
        case PlacementStrategy::HOT_COLD_SPLIT: j["placement"] = "HOT_COLD_SPLIT"; break;
        case PlacementStrategy::LAYER_STREAM:   j["placement"] = "LAYER_STREAM"; break;
        default:                               j["placement"] = "UNKNOWN"; break;
    }
    j["gpu_layers"] = s.gpu_layers;
    j["context_length"] = s.context_length;
    j["kv_quant_bits"] = s.kv_quant_bits;
    return j;
}

static std::string starsString(double score) {
    int full = (int)(score / 2.0 + 0.5);
    if (full > 5) full = 5; if (full < 0) full = 0;
    std::string s;
    for (int i = 0; i < full; i++) s += "\xe2\x98\x85";
    for (int i = full; i < 5; i++) s += "\xe2\x98\x86";
    return s;
}

// =============================================================================
// API Route Handlers
// =============================================================================

static void handleHealth(const httplib::Request&, httplib::Response& res) {
    sendJson(res, 200, makeSuccess({ {"status", "ok"}, {"version", "0.1.0"} }));
}

static void handleHardware(const httplib::Request&, httplib::Response& res) {
    std::string fp;
    HardwareSpec hw = profile_hardware(fp);
    json data = hardwareToJson(hw);
    sendJson(res, 200, makeSuccess(data));
}

// SSE keepalive comment (prevents proxy/firewall from closing idle connections)
static const char SSE_KEEPALIVE[] = ":keepalive\n\n";

static void handleHardwareLive(const httplib::Request&, httplib::Response& res) {
    sendSSEHeader(res);
    res.set_chunked_content_provider("text/event-stream",
        [](size_t, httplib::DataSink& sink) -> bool {
            static std::string fp;
            static auto last_send = std::chrono::steady_clock::now();
            static auto last_keepalive = std::chrono::steady_clock::now();
            auto now = std::chrono::steady_clock::now();

            // G: Send keepalive every 15 seconds
            if (std::chrono::duration_cast<std::chrono::seconds>(now - last_keepalive).count() >= 15) {
                last_keepalive = now;
                if (!sink.write(SSE_KEEPALIVE, strlen(SSE_KEEPALIVE))) return false;
            }

            if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_send).count() < 500) {
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
                return true;
            }
            HardwareSpec hw = profile_hardware(fp);
            std::string sse = "event: hardware\ndata: " + hardwareToLiveJson(hw).dump() + "\n\n";
            last_send = now;
            return sink.write(sse.c_str(), sse.size());
        });
}

static void handlePredict(const httplib::Request& req, httplib::Response& res) {
    json body;
    try { body = json::parse(req.body); } catch (...) {
        sendJson(res, 400, makeError("INVALID_JSON", "Request body is not valid JSON"));
        return;
    }
    std::string model_url = body.value("model_url", "");
    if (model_url.empty()) {
        sendJson(res, 400, makeError("MISSING_PARAM", "model_url is required"));
        return;
    }
    std::string priority = body.value("priority", "speed");
    std::string fp;
    HardwareSpec hw = profile_hardware(fp);
    Timer t_meta;
    ModelSpec model = fetch_metadata(model_url);
    double meta_ms = t_meta.elapsed_ms();
    if (model.layers == 0) {
        sendJson(res, 404, makeError("MODEL_NOT_FOUND", "Failed to fetch model metadata", get_fetch_error()));
        return;
    }
    CalibrationAggregator cal_agg(hw.hardware_fingerprint);
    CalibrationData cal = cal_agg.get_calibration_data();
    Timer t_matrix;
    std::vector<StrategyResult> results = generate_matrix(hw, model, cal);
    double matrix_ms = t_matrix.elapsed_ms();
    PriorityMode pm = (priority == "quality") ? PriorityMode::QUALITY :
                      (priority == "safety") ? PriorityMode::SAFETY : PriorityMode::SPEED;
    sort_by_priority(results, pm, hw);

    json strategies = json::array();
    for (size_t i = 0; i < results.size(); i++) {
        json sj = predictionToJson(results[i].prediction, results[i].strategy, (int)i + 1);
        StrategyStatus st = determine_status(hw, results[i].prediction, results[i].strategy);
        switch (st) {
            case StrategyStatus::VIABLE:   sj["status"] = "VIABLE"; break;
            case StrategyStatus::TIGHT:    sj["status"] = "TIGHT"; break;
            case StrategyStatus::NO_FIT:   sj["status"] = "NO_FIT"; break;
            case StrategyStatus::LOW_CONF: sj["status"] = "LOW_CONF"; break;
            default:                       sj["status"] = "UNKNOWN"; break;
        }
        sj["warnings"] = results[i].prediction.warnings;
        strategies.push_back(sj);
    }
    json data;
    data["model"]["name"] = model.name;
    data["model"]["params"] = model.param_count;
    data["model"]["layers"] = model.layers;
    data["model"]["architecture"] = model.architecture;
    data["model"]["is_moe"] = model.is_moe;
    data["model"]["quant"] = model.quant_type;
    data["model"]["context"] = model.context_length;
    data["hardware"] = hardwareToJson(hw);
    data["strategies"] = strategies;
    data["time_ms"] = (int)(meta_ms + matrix_ms);
    sendJson(res, 200, makeSuccess(data));
}

static void handleRecommend(const httplib::Request& req, httplib::Response& res) {
    std::string priority = req.get_param_value("priority").empty() ? "balanced" : req.get_param_value("priority");
    std::string use_case = req.get_param_value("use_case").empty() ? "all" : req.get_param_value("use_case");
    int top_n = 8;
    try { top_n = std::stoi(req.get_param_value("top")); } catch (...) {}
    std::string fp;
    HardwareSpec hw = profile_hardware(fp);
    ModelCatalog catalog = load_builtin_catalog();
    RecommendationRequest req2;
    req2.priority = priority;
    req2.use_case = use_case;
    req2.top_n = top_n;
    auto recs = generate_recommendations(hw, catalog, req2);
    json recs_json = json::array();
    for (size_t i = 0; i < recs.size(); i++) {
        const auto& r = recs[i];
        json rj;
        rj["model"] = r.model.name;
        rj["quant"] = r.variant.quant;
        rj["strategy"] = r.best_strategy_desc;
        rj["vram_bytes"] = (uint64_t)(r.predicted_vram_gb * 1e9);
        rj["tokens_per_sec"] = r.predicted_tok_s;
        rj["quality_score"] = r.model.quality_score;
        rj["quality_stars"] = starsString(r.model.quality_score);
        rj["download_gb"] = r.variant.file_size_gb;
        rj["hf_url"] = r.variant.hf_url;
        rj["label"] = r.label;
        recs_json.push_back(rj);
    }
    json data;
    data["recommendations"] = recs_json;
    data["count"] = recs_json.size();
    sendJson(res, 200, makeSuccess(data));
}

static void handleCatalog(const httplib::Request&, httplib::Response& res) {
    ModelCatalog catalog = load_builtin_catalog();
    json models = json::array();
    for (const auto& m : catalog.models) {
        models.push_back({ {"id", m.id}, {"name", m.name}, {"family", m.family},
            {"params_billions", m.params_billions}, {"architecture", m.architecture},
            {"quality_score", m.quality_score}, {"is_moe", m.is_moe},
            {"max_context", m.max_context}, {"variant_count", (int)m.variants.size()} });
    }
    sendJson(res, 200, makeSuccess({ {"models", models}, {"count", models.size()} }));
}

static void handleLocalModels(const httplib::Request&, httplib::Response& res) {
    json models = json::array();
    const char* dirs[] = { "models", "../models", "./models", "C:/dev/models", "C:/models" };
    for (const char* dir : dirs) {
        if (!fs::exists(dir)) continue;
        for (const auto& entry : fs::directory_iterator(dir)) {
            if (!entry.is_regular_file()) continue;
            std::string ext = entry.path().extension().string();
            for (auto& c : ext) c = (char)tolower(c);
            if (ext != ".gguf") continue;
            models.push_back({ {"path", entry.path().string()},
                {"filename", entry.path().filename().string()},
                {"size_bytes", entry.file_size()} });
        }
    }
    sendJson(res, 200, makeSuccess({ {"models", models}, {"count", models.size()} }));
}

// =============================================================================
// Download Task (Phase E)
// =============================================================================

static void startDownloadThread(std::shared_ptr<Task> task, std::string url, std::string dir) {
    std::thread([task, url, dir]() {
        task->running = true;
        task->pushEvent("starting", { {"message", "Starting download..."}, {"url", url} });

        std::atomic<bool> abort_flag{false};
        task->aborted = false;

        // Get file size
        uint64_t file_size = get_file_size_via_head(url);
        if (file_size == 0) {
            task->pushEvent("error", { {"message", "Could not determine file size"} });
            task->running = false;
            return;
        }

        task->pushEvent("progress", {
            {"bytes_downloaded", 0}, {"bytes_total", file_size},
            {"speed_mbs", 0}, {"eta_seconds", 0}
        });

        // Monitor progress in a separate thread
        std::thread monitor([task, dir, url, file_size, &abort_flag]() {
            std::string partial = get_partial_path(url, dir);
            auto last_time = std::chrono::steady_clock::now();
            uint64_t last_bytes = get_partial_file_size(partial);

            while (!abort_flag && task->running) {
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
                uint64_t current = get_partial_file_size(partial);
                auto now = std::chrono::steady_clock::now();
                double elapsed = std::chrono::duration<double>(now - last_time).count();

                if (elapsed >= 0.5 && current > last_bytes) {
                    double speed = (current - last_bytes) / elapsed / 1e6;
                    uint64_t remaining = (current < file_size) ? file_size - current : 0;
                    int eta = (speed > 0) ? (int)(remaining / (speed * 1e6)) : 0;
                    last_bytes = current;
                    last_time = now;

                    task->pushEvent("progress", {
                        {"bytes_downloaded", current}, {"bytes_total", file_size},
                        {"speed_mbs", speed}, {"eta_seconds", eta}
                    });
                }
            }
        });
        monitor.detach();

        // Run the actual download
        DownloadResult result = download_model_file(url, dir, file_size, abort_flag, false);
        abort_flag = true;  // Stop monitor

        if (task->aborted) {
            task->pushEvent("aborted", { {"message", "Download cancelled"} });
        } else if (result.success) {
            task->pushEvent("verifying", { {"message", "Download complete, verifying..."} });
            task->pushEvent("complete", {
                {"local_path", result.final_path},
                {"verified", result.verified},
                {"bytes_downloaded", result.bytes_downloaded}
            });
        } else {
            task->pushEvent("error", { {"message", result.error_message} });
        }

        task->running = false;
    }).detach();
}

static void handleDownloadStart(const httplib::Request& req, httplib::Response& res) {
    json body;
    try { body = json::parse(req.body); } catch (...) {
        sendJson(res, 400, makeError("INVALID_JSON", "Request body is not valid JSON"));
        return;
    }
    std::string model_url = body.value("model_url", "");
    if (model_url.empty()) {
        sendJson(res, 400, makeError("MISSING_PARAM", "model_url is required"));
        return;
    }
    std::string target_dir = body.value("target_dir", get_default_download_dir());

    auto task = std::make_shared<Task>();
    task->id = generateTaskId("dl");
    task->type = "download";

    {
        std::lock_guard<std::mutex> lock(g_tasks_mutex);
        g_tasks[task->id] = task;
    }

    startDownloadThread(task, model_url, target_dir);
    sendJson(res, 200, makeSuccess({
        {"task_id", task->id},
        {"stream_url", "/api/download/" + task->id + "/progress"}
    }));
}

static void handleDownloadProgress(const httplib::Request& req, httplib::Response& res) {
    std::string task_id;
    auto it = req.path_params.find("id");
    if (it != req.path_params.end()) task_id = it->second;
    auto task = getTask(task_id);
    if (!task) {
        sendJson(res, 404, makeError("TASK_NOT_FOUND", "No task with this ID"));
        return;
    }

    sendSSEHeader(res);
    res.set_chunked_content_provider("text/event-stream",
        [task](size_t, httplib::DataSink& sink) -> bool {
            auto last_keepalive = std::chrono::steady_clock::now();
            while (task->running || !task->event_queue.empty()) {
                // G: Send keepalive every 15s during long downloads
                auto now = std::chrono::steady_clock::now();
                if (std::chrono::duration_cast<std::chrono::seconds>(now - last_keepalive).count() >= 15) {
                    last_keepalive = now;
                    if (!sink.write(SSE_KEEPALIVE, strlen(SSE_KEEPALIVE))) return false;
                }
                std::string event;
                if (task->waitForEvent(event, 1000)) {
                    if (!sink.write(event.c_str(), event.size())) return false;
                }
                if (!task->running && task->event_queue.empty()) break;
            }
            return true;
        });
}

// =============================================================================
// Execute Task (Phase E)
// =============================================================================

static void handleExecuteStart(const httplib::Request& req, httplib::Response& res) {
    json body;
    try { body = json::parse(req.body); } catch (...) {
        sendJson(res, 400, makeError("INVALID_JSON", "Request body is not valid JSON"));
        return;
    }
    std::string model_path = body.value("model_path", "");
    if (model_path.empty()) {
        sendJson(res, 400, makeError("MISSING_PARAM", "model_path is required"));
        return;
    }
    // F5: Path traversal validation
    {
        std::string path_error;
        if (!is_safe_model_path(model_path, path_error)) {
            sendJson(res, 400, makeError("INVALID_PATH", path_error));
            return;
        }
    }

    auto task = std::make_shared<Task>();
    task->id = generateTaskId("ex");
    task->type = "execute";
    {
        std::lock_guard<std::mutex> lock(g_tasks_mutex);
        g_tasks[task->id] = task;
    }

    // Parse optional strategy/prompt from body
    std::string prompt = body.value("prompt", "Hello, how are you?");
    int max_tokens = body.value("max_tokens", 200);

    // Start execution in background thread using actual llama.cpp executor
    std::thread([task, model_path, prompt, max_tokens]() {
        task->running = true;
        task->pushEvent("loading", { {"message", "Loading model..."} });

        // Build a default strategy config (Full GPU)
        StrategyConfig strategy;
        strategy.placement = PlacementStrategy::FULL_GPU;
        strategy.gpu_layers = -1;  // All layers on GPU
        strategy.context_length = 4096;
        strategy.kv_quant_bits = 16;

        // Run actual inference via llama.cpp
        ExecutionResult result = execute(
            model_path, strategy, prompt, max_tokens,
            // Progress callback: fires on each generated token
            [task](int tokens_generated, double tok_per_sec) {
                task->pushEvent("token", {
                    {"tokens_generated", tokens_generated},
                    {"current_tok_per_sec", tok_per_sec}
                });
            }
        );

        if (result.success) {
            task->pushEvent("complete", {
                {"tokens_generated", result.tokens_generated},
                {"actual_tokens_per_sec", result.decode_tokens_per_sec},
                {"actual_ttft_ms", result.prompt_eval_ms},
                {"peak_vram_bytes", result.peak_vram_used_bytes},
                {"throttled", result.throttled},
                {"generated_text", result.generated_text}
            });
        } else {
            task->pushEvent("error", {
                {"message", result.error_message.empty() ? "Execution failed" : result.error_message}
            });
        }

        task->running = false;
    }).detach();

    sendJson(res, 200, makeSuccess({
        {"task_id", task->id},
        {"stream_url", "/api/execute/" + task->id + "/stream"}
    }));
}

static void handleExecuteStream(const httplib::Request& req, httplib::Response& res) {
    std::string task_id;
    auto it = req.path_params.find("id");
    if (it != req.path_params.end()) task_id = it->second;
    auto task = getTask(task_id);
    if (!task) {
        sendJson(res, 404, makeError("TASK_NOT_FOUND", "No task with this ID"));
        return;
    }    sendSSEHeader(res);
    res.set_chunked_content_provider("text/event-stream",
        [task](size_t, httplib::DataSink& sink) -> bool {
            auto last_keepalive = std::chrono::steady_clock::now();
            while (task->running || !task->event_queue.empty()) {
                auto now = std::chrono::steady_clock::now();
                if (std::chrono::duration_cast<std::chrono::seconds>(now - last_keepalive).count() >= 15) {
                    last_keepalive = now;
                    if (!sink.write(SSE_KEEPALIVE, strlen(SSE_KEEPALIVE))) return false;
                }
                std::string event;
                if (task->waitForEvent(event, 1000)) {
                    if (!sink.write(event.c_str(), event.size())) return false;
                }
                if (!task->running && task->event_queue.empty()) break;
            }
            return true;
        });
}




static void handleExecuteAbort(const httplib::Request& req, httplib::Response& res) {
    std::string task_id;
    auto it = req.path_params.find("id");
    if (it != req.path_params.end()) task_id = it->second;
    auto task = getTask(task_id);
    if (!task) {
        sendJson(res, 404, makeError("TASK_NOT_FOUND", "No task with this ID"));
        return;
    }
    task->aborted = true;
    task->running = false;
    sendJson(res, 200, makeSuccess({ {"message", "Task aborted"}, {"task_id", task_id} }));
}

// =============================================================================
// Calibration
// =============================================================================

static void handleCalibration(const httplib::Request&, httplib::Response& res) {
    std::string fp;
    HardwareSpec hw = profile_hardware(fp);
    std::string log_path = get_log_path();
    auto records = read_all_records(log_path);
    int match_count = 0;
    for (const auto& r : records) if (r.hardware_fingerprint == hw.hardware_fingerprint) match_count++;

    json data;
    data["records"] = (int)records.size();
    data["matching_records"] = match_count;
    data["fingerprint"] = hw.hardware_fingerprint;
    data["log_path"] = log_path;
    if (match_count > 0) {
        CalibrationAggregator agg(hw.hardware_fingerprint);
        CalibrationData cal = agg.get_calibration_data();
        data["gpu_overhead_mb"] = cal.adjusted_gpu_overhead_bytes / (1024 * 1024);
        data["gpu_decode_efficiency"] = cal.adjusted_gpu_decode_efficiency > 0 ? cal.adjusted_gpu_decode_efficiency : 0.27;
        data["cpu_decode_efficiency"] = cal.adjusted_cpu_decode_efficiency > 0 ? cal.adjusted_cpu_decode_efficiency : 0.80;
        data["gpu_prefill_efficiency"] = cal.adjusted_gpu_prefill_efficiency > 0 ? cal.adjusted_gpu_prefill_efficiency : 0.23;
    }
    sendJson(res, 200, makeSuccess(data));
}

static void handleCalibrationHistory(const httplib::Request&, httplib::Response& res) {
    std::string log_path = get_log_path();
    auto records = read_all_records(log_path);
    json entries = json::array();
    for (const auto& r : records) {
        entries.push_back({
            {"hardware_fingerprint", r.hardware_fingerprint}, {"model_id", r.model_id},
            {"placement", r.placement}, {"gpu_layers", r.gpu_layers}, {"context", r.context},
            {"predicted_tps", r.predicted_tokens_per_sec}, {"actual_tps", r.actual_tokens_per_sec},
            {"predicted_vram", r.predicted_vram_bytes}, {"actual_vram", r.actual_peak_vram_bytes},
            {"tokens_generated", r.actual_tokens_generated}, {"timestamp", r.timestamp}
        });
    }
    sendJson(res, 200, makeSuccess({ {"entries", entries}, {"count", (int)entries.size()} }));
}

static void handleCalibrationReset(const httplib::Request&, httplib::Response& res) {
    std::string log_path = get_log_path();
    auto records = read_all_records(log_path);
    int count = (int)records.size();
    if (remove(log_path.c_str()) != 0 && count > 0) {
        sendJson(res, 500, makeError("DELETE_FAILED", "Could not delete calibration log"));
        return;
    }
    sendJson(res, 200, makeSuccess({ {"deleted", count}, {"message", "Calibration log reset"} }));
}

// =============================================================================
// Static + API Route Setup
// =============================================================================

static void setupRoutes(httplib::Server& server) {
    // Static
    server.Get("/", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(WEB_INDEX_HTML, WEB_INDEX_HTML_MIME);
    });
    server.Get("/app.js", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(WEB_APP_JS, WEB_APP_JS_MIME);
    });
    server.Get("/styles.css", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(WEB_STYLES_CSS, WEB_STYLES_CSS_MIME);
    });
    server.Get("/chart.min.js", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(WEB_CHART_JS, WEB_CHART_JS_MIME);
    });

    // API
    server.Get("/api/health", handleHealth);
    server.Get("/api/hardware", handleHardware);
    server.Get("/api/hardware/live", handleHardwareLive);
    server.Post("/api/predict", [](const httplib::Request& req, httplib::Response& res) {
        handlePredict(req, res);
    });
    server.Get("/api/recommend", handleRecommend);
    server.Get("/api/catalog", handleCatalog);
    server.Get("/api/models/local", handleLocalModels);
    server.Post("/api/download", handleDownloadStart);
    server.Get("/api/download/:id/progress", handleDownloadProgress);
    server.Post("/api/execute", [](const httplib::Request& req, httplib::Response& res) {
        handleExecuteStart(req, res);
    });
    server.Get("/api/execute/:id/stream", handleExecuteStream);
    server.Post("/api/execute/:id/abort", handleExecuteAbort);
    server.Get("/api/calibration", handleCalibration);
    server.Get("/api/calibration/history", handleCalibrationHistory);
    server.Delete("/api/calibration", handleCalibrationReset);

    // CORS (F3) - restrict to localhost only
    server.Options(".*", [&server](const httplib::Request&, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "http://localhost:" + std::to_string(g_server_port));
        res.set_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type");
        res.status = 204;
    });
}

// =============================================================================
// Public API
// =============================================================================

bool start_web_server(int port, const std::string&, bool open_browser_flag, const std::string& bind_address) {
    // F1: Localhost-only binding by default
    if (bind_address == "0.0.0.0") {
        printf("\n  WARNING: Binding to 0.0.0.0 - the dashboard is accessible from other machines on the network.\n");
        printf("  For production use, consider adding authentication.\n");
        printf("  Default (safer): bind to 127.0.0.1 (localhost only)\n\n");
    }

    // G1: Auto-port selection - try requested port, then scan 8080-8089
    // We probe each port with a quick connect to detect if it's in use,
    // because httplib's SO_REUSEPORT allows multiple binds to the same port.
    auto is_port_available = [](const std::string& host, int p) -> bool {
#ifdef _WIN32
        SOCKET sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (sock == INVALID_SOCKET) return true;
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons((u_short)p);
        inet_pton(AF_INET, host.c_str(), &addr.sin_addr);
        bool available = (connect(sock, (sockaddr*)&addr, sizeof(addr)) == SOCKET_ERROR);
        closesocket(sock);
        return available;
#else
        int sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (sock < 0) return true;
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(p);
        inet_pton(AF_INET, host.c_str(), &addr.sin_addr);
        bool available = (connect(sock, (sockaddr*)&addr, sizeof(addr)) != 0);
        close(sock);
        return available;
#endif
    };

    int actual_port = port;
    const int port_max = 8090;
    while (actual_port < port_max && !is_port_available(bind_address, actual_port)) {
        actual_port++;
    }
    if (actual_port >= port_max) {
        fprintf(stderr, "\n  ERROR: No available port in range %d-%d.\n", port, port_max - 1);
        fprintf(stderr, "  Close other services using these ports and try again.\n\n");
        return false;
    }

    httplib::Server server;
    setupRoutes(server);
    server.set_error_handler([](const httplib::Request& req, httplib::Response& res) {
        if (!res.body.empty()) return;
        fprintf(stderr, "[vessel-web] Error %d for %s %s\n", res.status, req.method.c_str(), req.path.c_str());
        if (res.status == 0) res.status = 500;
        sendJson(res, res.status, makeError("NOT_FOUND", "Endpoint not found", req.method + " " + req.path));
    });

    g_server_port = actual_port;

    if (actual_port != port) {
        printf("\n  Port %d is in use. Using port %d instead.\n", port, actual_port);
    }
    printf("\n  Vessel Dashboard\n  http://%s:%d\n  Press Ctrl+C to stop\n\n",
           bind_address.c_str(), actual_port);
    if (open_browser_flag) openBrowser("http://localhost:" + std::to_string(actual_port));
    return server.listen(bind_address, actual_port);
}
