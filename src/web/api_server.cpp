// =============================================================================
// Step 14 — OpenAI-Compatible API Server
// =============================================================================
// Implements the core OpenAI API endpoints for local LLM inference.
// Uses cpp-httplib (same library as Step 13 web dashboard).
// =============================================================================

#include "api_server.h"
#include "model_manager.h"
#include "executor.h"
#include "types.h"
#include "profiler.h"

#include <cstdio>
#include <string>
#include <mutex>
#include <atomic>
#include <thread>
#include <chrono>
#include <sstream>
#include <memory>
#include <algorithm>

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
// Global State
// =============================================================================

static std::unique_ptr<ModelManager> g_model_manager;
static std::atomic<uint64_t> g_request_counter{0};
static std::atomic<uint64_t> g_total_prompt_tokens{0};
static std::atomic<uint64_t> g_total_completion_tokens{0};

// =============================================================================
// JSON Helpers
// =============================================================================

static json make_error_json(const std::string& message, const std::string& type = "invalid_request_error") {
    return {
        {"error", {
            {"message", message},
            {"type", type},
            {"param", nullptr},
            {"code", nullptr}
        }}
    };
}

static std::string generate_id() {
    uint64_t counter = g_request_counter.fetch_add(1);
    char buf[64];
    snprintf(buf, sizeof(buf), "chatcmpl-%llx-%llx",
             (unsigned long long)std::chrono::duration_cast<std::chrono::seconds>(
                 std::chrono::system_clock::now().time_since_epoch()).count(),
             (unsigned long long)counter);
    return std::string(buf);
}

static int64_t unix_timestamp() {
    return std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

// =============================================================================
// Chat Template Application
// =============================================================================
// Uses llama.cpp's built-in chat template support to properly format
// messages according to the model's native template.

static std::string apply_chat_template(
    struct llama_model* model,
    const std::vector<std::pair<std::string, std::string>>& messages
) {
    const char* tmpl = llama_model_chat_template(model, nullptr);
    if (!tmpl) {
        // Fallback: simple concatenation
        std::string result;
        for (const auto& [role, content] : messages) {
            result += "<|" + role + "|>\n" + content + "\n";
        }
        result += "<|assistant|>\n";
        return result;
    }

    // Convert to llama_chat_message format
    std::vector<llama_chat_message> chat_msgs;
    std::vector<std::string> contents; // Keep strings alive
    for (const auto& [role, content] : messages) {
        contents.push_back(content);
    }

    for (size_t i = 0; i < messages.size(); i++) {
        llama_chat_message msg;
        msg.role = messages[i].first.c_str();
        msg.content = contents[i].c_str();
        chat_msgs.push_back(msg);
    }

    // Estimate buffer size
    size_t total_chars = 0;
    for (const auto& [role, content] : messages) {
        total_chars += role.size() + content.size() + 10;
    }
    size_t buf_size = total_chars * 2 + 256;

    std::string buf(buf_size, '\0');
    int32_t result = llama_chat_apply_template(
        tmpl, chat_msgs.data(), chat_msgs.size(),
        true,  // add_ass — end with assistant header
        &buf[0], static_cast<int32_t>(buf_size)
    );

    if (result < 0) {
        // Template failed, use fallback
        std::string fallback;
        for (const auto& [role, content] : messages) {
            fallback += "<|" + role + "|>\n" + content + "\n";
        }
        fallback += "<|assistant|>\n";
        return fallback;
    }

    buf.resize(result);
    return buf;
}

// =============================================================================
// Route: GET /v1/health
// =============================================================================

static void handle_health(const httplib::Request&, httplib::Response& res) {
    json j = {
        {"status", "ok"},
        {"version", "0.2.0"},
        {"models_loaded", g_model_manager ? g_model_manager->loaded_count() : 0},
        {"models_total", g_model_manager ? (int)g_model_manager->list_models().size() : 0},
        {"requests_served", g_request_counter.load()},
        {"total_prompt_tokens", g_total_prompt_tokens.load()},
        {"total_completion_tokens", g_total_completion_tokens.load()}
    };
    res.set_header("Content-Type", "application/json");
    res.set_content(j.dump(), "application/json");
}

// =============================================================================
// Route: GET /v1/hardware
// =============================================================================

static void handle_hardware(const httplib::Request&, httplib::Response& res) {
    if (!g_model_manager) {
        res.status = 503;
        res.set_content(make_error_json("Server not initialized").dump(), "application/json");
        return;
    }

    const HardwareSpec& hw = g_model_manager->get_hardware();
    json j = {
        {"platform", hw.gpu_name.empty() ? "cpu" : (hw.is_nvidia() ? "nvidia" : hw.is_amd() ? "amd" : hw.is_apple() ? "apple" : "unknown")},
        {"gpu_name", hw.gpu_name},
        {"vram_total_bytes", hw.vram_total_bytes},
        {"vram_free_bytes", hw.vram_free_bytes},
        {"ram_total_bytes", hw.ram_total_bytes},
        {"ram_free_bytes", hw.ram_free_bytes},
        {"gpu_bandwidth_gbs", hw.gpu_bandwidth_gbs},
        {"gpu_tflops_fp16", hw.gpu_tflops_fp16},
        {"ram_bandwidth_gbs", hw.ram_bandwidth_gbs},
        {"is_unified_memory", hw.is_unified_memory}
    };
    res.set_header("Content-Type", "application/json");
    res.set_content(j.dump(), "application/json");
}

// =============================================================================
// Route: GET /v1/models
// =============================================================================

static void handle_list_models(const httplib::Request&, httplib::Response& res) {
    if (!g_model_manager) {
        res.status = 503;
        res.set_content(make_error_json("Server not initialized").dump(), "application/json");
        return;
    }

    auto models = g_model_manager->list_models();
    json data = json::array();

    for (const auto* session : models) {
        json model_obj = {
            {"id", session->model_id},
            {"object", "model"},
            {"created", session->created_at},
            {"owned_by", "local"},
            {"permission", json::array()},
            {"root", session->model_id},
            {"parent", nullptr}
        };

        // Vessel-specific extensions
        model_obj["vessel"] = {
            {"display_name", session->display_name},
            {"file_path", session->file_path},
            {"architecture", session->metadata.architecture},
            {"params_billions", session->metadata.param_count / 1e9},
            {"quant", session->metadata.quant_type},
            {"layers", session->metadata.layers},
            {"is_moe", session->metadata.is_moe},
            {"loaded", session->loaded},
            {"strategy", format_strategy_description(session->best_strategy, session->metadata.layers)},
            {"predicted_tps", session->prediction.tokens_per_sec},
            {"predicted_vram_gb", session->prediction.memory_vram_bytes / 1e9},
            {"viable", session->prediction.viable},
            {"prompt_tokens", session->prompt_tokens.load()},
            {"completion_tokens", session->completion_tokens.load()},
            {"request_count", session->request_count.load()}
        };

        data.push_back(model_obj);
    }

    json j = {{"object", "list"}, {"data", data}};
    res.set_header("Content-Type", "application/json");
    res.set_content(j.dump(), "application/json");
}

// =============================================================================
// Route: GET /v1/models/:id
// =============================================================================

static void handle_get_model(const httplib::Request& req, httplib::Response& res) {
    if (!g_model_manager) {
        res.status = 503;
        res.set_content(make_error_json("Server not initialized").dump(), "application/json");
        return;
    }

    std::string model_id;
    auto it = req.path_params.find("id");
    if (it != req.path_params.end()) model_id = it->second;

    ModelSession* session = g_model_manager->find_model(model_id);
    if (!session) {
        res.status = 404;
        res.set_content(make_error_json("Model not found: " + model_id, "model_not_found").dump(), "application/json");
        return;
    }

    json j = {
        {"id", session->model_id},
        {"object", "model"},
        {"created", session->created_at},
        {"owned_by", "local"},
        {"vessel", {
            {"display_name", session->display_name},
            {"file_path", session->file_path},
            {"architecture", session->metadata.architecture},
            {"params_billions", session->metadata.param_count / 1e9},
            {"quant", session->metadata.quant_type},
            {"layers", session->metadata.layers},
            {"loaded", session->loaded}
        }}
    };
    res.set_header("Content-Type", "application/json");
    res.set_content(j.dump(), "application/json");
}

// =============================================================================
// Streaming SSE Helper
// =============================================================================

static void send_sse_chunk(httplib::DataSink& sink, const json& chunk) {
    std::string data = "data: " + chunk.dump() + "\n\n";
    sink.write(data.c_str(), data.size());
}

static void send_sse_done(httplib::DataSink& sink) {
    const char* done = "data: [DONE]\n\n";
    sink.write(done, strlen(done));
}

// =============================================================================
// Core Chat/Completion Handler
// =============================================================================

static void handle_completion(
    const httplib::Request& req,
    httplib::Response& res,
    bool is_chat  // true = /v1/chat/completions, false = /v1/completions
) {
    // Parse request body
    json body;
    try {
        body = json::parse(req.body);
    } catch (...) {
        res.status = 400;
        res.set_content(make_error_json("Request body is not valid JSON").dump(), "application/json");
        return;
    }

    // Extract model
    std::string model_id = body.value("model", "");
    if (model_id.empty()) {
        res.status = 400;
        res.set_content(make_error_json("'model' field is required").dump(), "application/json");
        return;
    }

    // Find model
    ModelSession* session = g_model_manager->find_model(model_id);
    if (!session) {
        res.status = 404;
        res.set_content(make_error_json("Model not found: " + model_id, "model_not_found").dump(), "application/json");
        return;
    }

    // Ensure model is loaded
    if (!g_model_manager->ensure_loaded(*session)) {
        res.status = 500;
        res.set_content(make_error_json("Failed to load model: " + session->display_name, "server_error").dump(), "application/json");
        return;
    }

    // Parse sampling parameters
    SamplingConfig sampling;
    sampling.temperature = body.value("temperature", 1.0f);
    sampling.top_p = body.value("top_p", 1.0f);
    sampling.top_k = body.value("top_k", -1);
    sampling.seed = body.value("seed", -1);
    sampling.frequency_penalty = body.value("frequency_penalty", 0);
    sampling.presence_penalty = body.value("presence_penalty", 0);

    // Min-p (OpenAI doesn't have this, but it's common in local LLM APIs)
    if (body.contains("min_p")) {
        sampling.min_p = body["min_p"].get<float>();
    }

    // Stop sequences
    if (body.contains("stop")) {
        if (body["stop"].is_string()) {
            sampling.stop_sequences.push_back(body["stop"].get<std::string>());
        } else if (body["stop"].is_array()) {
            for (const auto& s : body["stop"]) {
                if (s.is_string()) sampling.stop_sequences.push_back(s.get<std::string>());
            }
        }
    }

    int max_tokens = body.value("max_tokens", 512);
    bool stream = body.value("stream", false);

    // Build formatted prompt
    std::string formatted_prompt;
    if (is_chat) {
        // Parse messages array
        std::vector<std::pair<std::string, std::string>> messages;
        if (body.contains("messages") && body["messages"].is_array()) {
            for (const auto& msg : body["messages"]) {
                std::string role = msg.value("role", "user");
                std::string content = msg.value("content", "");
                // Handle content that's an array (multi-part)
                if (msg.contains("content") && msg["content"].is_array()) {
                    content = "";
                    for (const auto& part : msg["content"]) {
                        if (part.is_object() && part.contains("text")) {
                            content += part["text"].get<std::string>();
                        } else if (part.is_string()) {
                            content += part.get<std::string>();
                        }
                    }
                }
                messages.push_back({role, content});
            }
        }

        if (messages.empty()) {
            res.status = 400;
            res.set_content(make_error_json("'messages' array is required and must not be empty").dump(), "application/json");
            return;
        }

        formatted_prompt = apply_chat_template(session->model, messages);
    } else {
        // Raw completion
        std::string prompt = body.value("prompt", "");
        if (prompt.empty()) {
            res.status = 400;
            res.set_content(make_error_json("'prompt' field is required for completions").dump(), "application/json");
            return;
        }
        formatted_prompt = prompt;
    }

    // Build streaming config
    StreamingConfig stream_config;
    stream_config.sampling = sampling;
    stream_config.max_tokens = max_tokens;

    // Generate response ID
    std::string response_id = generate_id();
    int64_t created = unix_timestamp();
    std::string model_name = session->model_id;

    // Lock the model for this request
    std::lock_guard<std::mutex> lock(session->infer_mutex);

    if (stream) {
        // =====================================================================
        // Streaming Response
        // =====================================================================
        res.set_header("Content-Type", "text/event-stream");
        res.set_header("Cache-Control", "no-cache");
        res.set_header("Connection", "keep-alive");
        res.set_chunked_content_provider("text/event-stream",
            [session, response_id, created, model_name, formatted_prompt, stream_config, is_chat]
            (size_t, httplib::DataSink& sink) -> bool {

                // Send initial role chunk
                {
                    json initial = {
                        {"id", response_id},
                        {"object", is_chat ? "chat.completion.chunk" : "text_completion"},
                        {"created", created},
                        {"model", model_name},
                        {"choices", {{
                            {"index", 0},
                            {"delta", is_chat ? json{{"role", "assistant"}} : json{{"text", ""}}},
                            {"finish_reason", nullptr}
                        }}}
                    };
                    send_sse_chunk(sink, initial);
                }

                std::string full_text;
                std::string finish_reason;
                int prompt_tokens = 0;
                int completion_tokens = 0;

                // Set up callbacks
                StreamCallbacks callbacks;

                callbacks.on_token = [&sink, &full_text, &response_id, &created, &model_name, is_chat]
                    (const std::string& token_text) {
                    full_text += token_text;

                    json chunk = {
                        {"id", response_id},
                        {"object", is_chat ? "chat.completion.chunk" : "text_completion"},
                        {"created", created},
                        {"model", model_name},
                        {"choices", {{
                            {"index", 0},
                            {"delta", is_chat
                                ? json{{"content", token_text}}
                                : json{{"text", token_text}}},
                            {"finish_reason", nullptr}
                        }}}
                    };
                    send_sse_chunk(sink, chunk);
                };

                callbacks.on_done = [&sink, &finish_reason, &prompt_tokens, &completion_tokens,
                                     &response_id, &created, &model_name, is_chat, session]
                    (const std::string& reason, int p_tokens, int c_tokens) {
                    finish_reason = reason;
                    prompt_tokens = p_tokens;
                    completion_tokens = c_tokens;

                    // Send final chunk with finish_reason
                    json final_chunk = {
                        {"id", response_id},
                        {"object", is_chat ? "chat.completion.chunk" : "text_completion"},
                        {"created", created},
                        {"model", model_name},
                        {"choices", {{
                            {"index", 0},
                            {"delta", json::object()},
                            {"finish_reason", finish_reason}
                        }}},
                        {"usage", {
                            {"prompt_tokens", prompt_tokens},
                            {"completion_tokens", completion_tokens},
                            {"total_tokens", prompt_tokens + completion_tokens}
                        }}
                    };
                    send_sse_chunk(sink, final_chunk);

                    // Send [DONE]
                    send_sse_done(sink);

                    // Update stats
                    session->prompt_tokens += prompt_tokens;
                    session->completion_tokens += completion_tokens;
                    session->request_count++;
                    g_total_prompt_tokens += prompt_tokens;
                    g_total_completion_tokens += completion_tokens;
                };

                callbacks.on_error = [&sink](const std::string& error) {
                    json err = make_error_json(error, "server_error");
                    std::string data = "data: " + err.dump() + "\n\n";
                    sink.write(data.c_str(), data.size());
                    send_sse_done(sink);
                };

                // Run streaming inference
                execute_streaming(session->model, session->ctx, formatted_prompt, stream_config, callbacks);

                return true;
            });

    } else {
        // =====================================================================
        // Non-Streaming Response
        // =====================================================================
        std::string generated_text;
        std::string finish_reason;
        int prompt_tokens = 0;
        int completion_tokens = 0;

        StreamCallbacks callbacks;

        callbacks.on_token = [&generated_text](const std::string& token_text) {
            generated_text += token_text;
        };

        callbacks.on_done = [&finish_reason, &prompt_tokens, &completion_tokens]
            (const std::string& reason, int p_tokens, int c_tokens) {
            finish_reason = reason;
            prompt_tokens = p_tokens;
            completion_tokens = c_tokens;
        };

        callbacks.on_error = [&res](const std::string& error) {
            res.status = 500;
            res.set_content(make_error_json(error, "server_error").dump(), "application/json");
        };

        bool success = execute_streaming(session->model, session->ctx, formatted_prompt, stream_config, callbacks);

        if (!success) {
            return; // Error already set in callbacks
        }

        // Build OpenAI-compatible response
        json j = {
            {"id", response_id},
            {"object", is_chat ? "chat.completion" : "text_completion"},
            {"created", created},
            {"model", model_name},
            {"choices", {{
                {"index", 0},
                is_chat
                    ? json{{"message", {{"role", "assistant"}, {"content", generated_text}}}}
                    : json{{"text", generated_text}},
                {"logprobs", nullptr},
                {"finish_reason", finish_reason}
            }}},
            {"usage", {
                {"prompt_tokens", prompt_tokens},
                {"completion_tokens", completion_tokens},
                {"total_tokens", prompt_tokens + completion_tokens}
            }}
        };

        // Update stats
        session->prompt_tokens += prompt_tokens;
        session->completion_tokens += completion_tokens;
        session->request_count++;
        g_total_prompt_tokens += prompt_tokens;
        g_total_completion_tokens += completion_tokens;

        res.set_header("Content-Type", "application/json");
        res.set_content(j.dump(), "application/json");
    }
}

// =============================================================================
// Route Handlers
// =============================================================================

static void handle_chat_completions(const httplib::Request& req, httplib::Response& res) {
    handle_completion(req, res, true);
}

static void handle_completions(const httplib::Request& req, httplib::Response& res) {
    handle_completion(req, res, false);
}

// =============================================================================
// Route Setup
// =============================================================================

static void setup_routes(httplib::Server& server) {
    // OpenAI-compatible endpoints
    server.Get("/v1/health", handle_health);
    server.Get("/v1/hardware", handle_hardware);
    server.Get("/v1/models", handle_list_models);
    server.Get(R"(/v1/models/([^/]+))", handle_get_model);
    server.Post("/v1/chat/completions", handle_chat_completions);
    server.Post("/v1/completions", handle_completions);

    // Also accept /chat/completions and /completions (without /v1/ prefix)
    // for clients that set a custom base_url
    server.Post("/chat/completions", handle_chat_completions);
    server.Post("/completions", handle_completions);

    // Root redirect to health
    server.Get("/", [](const httplib::Request&, httplib::Response& res) {
        json j = {
            {"name", "Vessel API Server"},
            {"version", "0.2.0"},
            {"description", "OpenAI-compatible API for local GGUF models with auto-optimized performance"},
            {"endpoints", {
                {"GET /v1/models", "List available models"},
                {"GET /v1/models/:id", "Get model details"},
                {"POST /v1/chat/completions", "Chat completions (OpenAI-compatible)"},
                {"POST /v1/completions", "Text completions (OpenAI-compatible)"},
                {"GET /v1/hardware", "Hardware profile"},
                {"GET /v1/health", "Health check"}
            }}
        };
        res.set_header("Content-Type", "application/json");
        res.set_content(j.dump(2), "application/json");
    });

    // CORS (permissive for local development)
    server.Options(".*", [](const httplib::Request&, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");
        res.status = 204;
    });

    // Error handler
    server.set_error_handler([](const httplib::Request& req, httplib::Response& res) {
        if (res.status == 0) res.status = 404;
        if (res.body.empty()) {
            res.set_header("Content-Type", "application/json");
            res.set_content(make_error_json("Not found: " + req.method + " " + req.path).dump(), "application/json");
        }
    });
}

// =============================================================================
// Public API
// =============================================================================

bool start_api_server(
    int port,
    const std::string& host,
    const std::vector<std::string>& model_dirs
) {
    printf("\n");
    printf("  ╔══════════════════════════════════════════╗\n");
    printf("  ║       Vessel API Server (Step 14)        ║\n");
    printf("  ║   OpenAI-Compatible Local LLM API        ║\n");
    printf("  ╚══════════════════════════════════════════╝\n");
    printf("\n");

    // Initialize model manager
    g_model_manager = std::make_unique<ModelManager>();
    if (!g_model_manager->init(model_dirs)) {
        fprintf(stderr, "  ERROR: Failed to initialize model manager\n");
        return false;
    }

    // Initialize llama.cpp backend
    if (!executor_init()) {
        fprintf(stderr, "  ERROR: Failed to initialize llama.cpp backend\n");
        return false;
    }

    // Setup and start server
    httplib::Server server;
    setup_routes(server);

    printf("  API Server: http://%s:%d\n", host.c_str(), port);
    printf("  OpenAI Base URL: http://%s:%d/v1\n", host.c_str(), port);
    printf("\n");
    printf("  Example usage:\n");
    printf("    curl http://localhost:%d/v1/models\n", port);
    printf("    curl http://localhost:%d/v1/chat/completions \\\\\n", port);
    printf("      -H 'Content-Type: application/json' \\\\\n");
    printf("      -d '{\"model\":\"<model-id>\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello!\"}]}'\n");
    printf("\n");
    printf("  Or configure your client:\n");
    printf("    Base URL: http://localhost:%d/v1\n", port);
    printf("    API Key:  (any string)\n");
    printf("    Model:    <model-id> (see /v1/models)\n");
    printf("\n");
    printf("  Press Ctrl+C to stop.\n\n");

    // Handle port conflicts
    auto is_port_available = [](const std::string& h, int p) -> bool {
#ifdef _WIN32
        SOCKET sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (sock == INVALID_SOCKET) return true;
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons((u_short)p);
        inet_pton(AF_INET, h.c_str(), &addr.sin_addr);
        bool available = (connect(sock, (sockaddr*)&addr, sizeof(addr)) == SOCKET_ERROR);
        closesocket(sock);
        return available;
#else
        int sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (sock < 0) return true;
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(p);
        inet_pton(AF_INET, h.c_str(), &addr.sin_addr);
        bool available = (connect(sock, (sockaddr*)&addr, sizeof(addr)) != 0);
        close(sock);
        return available;
#endif
    };

    int actual_port = port;
    while (actual_port < port + 100 && !is_port_available(host, actual_port)) {
        actual_port++;
    }
    if (actual_port != port) {
        printf("  Port %d in use. Using port %d instead.\n", port, actual_port);
        printf("  Updated Base URL: http://%s:%d/v1\n", host.c_str(), actual_port);
        printf("\n");
    }

    bool result = server.listen(host, actual_port);

    // Cleanup
    executor_shutdown();
    g_model_manager.reset();

    return result;
}
