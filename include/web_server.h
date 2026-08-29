#pragma once

#include "types.h"
#include <string>
#include <functional>

// Callback types for decoupling web server from specific modules
using HardwareProfileCallback = std::function<HardwareSpec()>;
using PredictCallback = std::function<std::string(const std::string& model_url)>;
using RecommendCallback = std::function<std::string(const std::string& priority, const std::string& use_case, int top_n)>;
using CalibrationCallback = std::function<std::string()>;

// Start the web dashboard server.
// bind_address: "127.0.0.1" (default, localhost only) or "0.0.0.0" (all interfaces)
// Blocks until the server is stopped (Ctrl+C).
// Returns true if the server started successfully.
bool start_web_server(
    int port,
    const std::string& model_for_fingerprint,
    bool open_browser = true,
    const std::string& bind_address = "127.0.0.1"
);
