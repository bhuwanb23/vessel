#include "calibration_log.h"
#include "executor.h"

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <shlobj.h>
#endif
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <sstream>
#include <algorithm>

// =============================================================================
// JSON Escaping
// =============================================================================

static std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '\"':  out += "\\\""; break;
            case '\\':  out += "\\\\"; break;
            case '\n':  out += "\\n";  break;
            case '\r':  out += "\\r";  break;
            case '\t':  out += "\\t";  break;
            default:    out += c;      break;
        }
    }
    return out;
}

// =============================================================================
// Timestamp — ISO 8601 UTC
// =============================================================================

static std::string get_timestamp_utc() {
#if defined(_WIN32)
    SYSTEMTIME st_utc;
    GetSystemTime(&st_utc);
    char buf[64];
    snprintf(buf, sizeof(buf), "%04d-%02d-%02dT%02d:%02d:%02dZ",
             st_utc.wYear, st_utc.wMonth, st_utc.wDay,
             st_utc.wHour, st_utc.wMinute, st_utc.wSecond);
    return std::string(buf);
#else
    // POSIX: use gmtime
    auto now = std::time(nullptr);
    struct tm utc;
    gmtime_r(&now, &utc);
    char buf[64];
    snprintf(buf, sizeof(buf), "%04d-%02d-%02dT%02d:%02d:%02dZ",
             utc.tm_year + 1900, utc.tm_mon + 1, utc.tm_mday,
             utc.tm_hour, utc.tm_min, utc.tm_sec);
    return std::string(buf);
#endif
}

// =============================================================================
// Placement Enum to String
// =============================================================================

static const char* placement_to_string(PlacementStrategy p) {
    switch (p) {
        case PlacementStrategy::FULL_GPU:       return "FULL_GPU";
        case PlacementStrategy::GPU_CPU_SPLIT:  return "GPU_CPU_SPLIT";
        case PlacementStrategy::CPU_ONLY:       return "CPU_ONLY";
        case PlacementStrategy::HOT_COLD_SPLIT: return "HOT_COLD_SPLIT";
        case PlacementStrategy::LAYER_STREAM:   return "LAYER_STREAM";
        default: return "UNKNOWN";
    }
}

// =============================================================================
// JSON Serialization
// =============================================================================

std::string record_to_json(const CalibrationRecord& r) {
    std::ostringstream o;
    o << "{";

    // Identity
    o << "\"hardware_fingerprint\":\"" << json_escape(r.hardware_fingerprint) << "\",";
    o << "\"model_id\":\"" << json_escape(r.model_id) << "\",";

    // Strategy
    o << "\"strategy\":{";
    o << "\"backend\":\"" << json_escape(r.backend) << "\",";
    o << "\"quant\":\"" << json_escape(r.quant) << "\",";
    o << "\"placement\":\"" << json_escape(r.placement) << "\",";
    o << "\"gpu_layers\":" << r.gpu_layers << ",";
    o << "\"context\":" << r.context << ",";
    o << "\"kv_quant_bits\":" << r.kv_quant_bits;
    // MoE-specific strategy fields (Step 9, Phase E)
    if (r.gpu_experts_per_layer > 0 || r.total_experts_per_layer > 0) {
        o << ",\"gpu_experts_per_layer\":" << r.gpu_experts_per_layer;
        o << ",\"total_experts_per_layer\":" << r.total_experts_per_layer;
    }
    // Hot/Cold-specific strategy fields (Step 10, Phase H)
    if (r.hot_neuron_pct > 0) {
        o << ",\"hot_neuron_pct\":" << r.hot_neuron_pct;
        o << ",\"profiled\":" << (r.profiled ? "true" : "false");
        o << ",\"cold_activation_rate\":" << r.cold_activation_rate;
    }
    o << "},";

    // Predicted
    o << "\"predicted\":{";
    o << "\"tokens_per_sec\":" << r.predicted_tokens_per_sec << ",";
    o << "\"ttft_ms\":" << r.predicted_ttft_ms << ",";
    o << "\"vram_bytes\":" << r.predicted_vram_bytes << ",";
    o << "\"ram_bytes\":" << r.predicted_ram_bytes << ",";
    o << "\"confidence\":\"" << json_escape(r.predicted_confidence) << "\"";
    // MoE-specific predicted ranges (Step 9, Phase D)
    if (r.predicted_tokens_per_sec_min > 0 || r.predicted_tokens_per_sec_max > 0) {
        o << ",\"tokens_per_sec_min\":" << r.predicted_tokens_per_sec_min;
        o << ",\"tokens_per_sec_max\":" << r.predicted_tokens_per_sec_max;
        o << ",\"tokens_per_sec_expected\":" << r.predicted_tokens_per_sec_expected;
    }
    // Hot/Cold-specific predicted ranges (Step 10, Phase H)
    if (r.predicted_tok_s_range_min > 0 || r.predicted_tok_s_range_max > 0) {
        o << ",\"tok_s_range_min\":" << r.predicted_tok_s_range_min;
        o << ",\"tok_s_range_max\":" << r.predicted_tok_s_range_max;
    }
    o << "},";

    // Actual
    o << "\"actual\":{";
    o << "\"tokens_per_sec\":" << r.actual_tokens_per_sec << ",";
    o << "\"ttft_ms\":" << r.actual_ttft_ms << ",";
    o << "\"peak_vram_bytes\":" << r.actual_peak_vram_bytes << ",";
    o << "\"peak_ram_bytes\":" << r.actual_peak_ram_bytes << ",";
    o << "\"throttled\":" << (r.actual_throttled ? "true" : "false") << ",";
    o << "\"tokens_generated\":" << r.actual_tokens_generated << ",";
    o << "\"duration_sec\":" << r.actual_duration_sec;
    // MoE-specific actual telemetry (Step 9, Phase E)
    if (r.actual_tokens_per_sec_min > 0 || r.actual_tokens_per_sec_max > 0) {
        o << ",\"tokens_per_sec_min\":" << r.actual_tokens_per_sec_min;
        o << ",\"tokens_per_sec_max\":" << r.actual_tokens_per_sec_max;
    }
    if (r.actual_pcie_throughput_mbs > 0) {
        o << ",\"pcie_throughput_mbs\":" << r.actual_pcie_throughput_mbs;
    }
    if (r.actual_token_time_variance > 0) {
        o << ",\"token_time_variance\":" << r.actual_token_time_variance;
    }
    // Hot/Cold-specific actual telemetry (Step 10, Phase H)
    if (r.actual_cold_compute_pct > 0) {
        o << ",\"cold_compute_pct\":" << r.actual_cold_compute_pct;
    }
    o << "},";

    // Metadata
    o << "\"timestamp\":\"" << json_escape(r.timestamp) << "\",";
    o << "\"tool_version\":\"" << json_escape(r.tool_version) << "\"";

    o << "}";
    return o.str();
}

// =============================================================================
// JSON Deserialization — Minimal parser for the fixed schema
// =============================================================================
// We only parse our own schema. This is not a general JSON parser.

// Find the value for a given key in a JSON object string.
// Returns the value string (without quotes for strings, raw for numbers/bools).
// Returns empty string if key not found.
static std::string find_json_value(const std::string& json, const std::string& key) {
    std::string search_key = "\"" + key + "\"";
    size_t pos = json.find(search_key);
    if (pos == std::string::npos) return "";

    // Skip past the key and the colon
    pos += search_key.size();
    pos = json.find(':', pos);
    if (pos == std::string::npos) return "";
    pos++;  // skip ':'

    // Skip whitespace
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t'))
        pos++;

    if (pos >= json.size()) return "";

    // String value
    if (json[pos] == '"') {
        pos++;  // skip opening quote
        std::string value;
        while (pos < json.size() && json[pos] != '"') {
            if (json[pos] == '\\' && pos + 1 < json.size()) {
                pos++;  // skip escape
                switch (json[pos]) {
                    case '"':  value += '"';  break;
                    case '\\': value += '\\'; break;
                    case 'n':  value += '\n'; break;
                    case 'r':  value += '\r'; break;
                    case 't':  value += '\t'; break;
                    default:   value += json[pos]; break;
                }
            } else {
                value += json[pos];
            }
            pos++;
        }
        return value;
    }

    // Number or boolean value
    std::string value;
    while (pos < json.size() && json[pos] != ',' && json[pos] != '}'
           && json[pos] != ' ' && json[pos] != '\t') {
        value += json[pos];
        pos++;
    }
    return value;
}

// Find a nested JSON object by key, return it as a string.
// e.g., find_json_object(json, "strategy") returns the {...} content.
static std::string find_json_object(const std::string& json, const std::string& key) {
    std::string search_key = "\"" + key + "\"";
    size_t pos = json.find(search_key);
    if (pos == std::string::npos) return "";

    pos = json.find(':', pos);
    if (pos == std::string::npos) return "";
    pos++;

    while (pos < json.size() && json[pos] == ' ') pos++;
    if (pos >= json.size() || json[pos] != '{') return "";

    // Find matching closing brace
    int depth = 1;
    size_t start = pos;
    pos++;
    while (pos < json.size() && depth > 0) {
        if (json[pos] == '{') depth++;
        else if (json[pos] == '}') depth--;
        pos++;
    }

    return json.substr(start, pos - start);
}

bool json_to_record(const std::string& json, CalibrationRecord& record) {
    // Validate it starts with {
    if (json.empty() || json[0] != '{') return false;

    // Top-level fields
    record.hardware_fingerprint = find_json_value(json, "hardware_fingerprint");
    record.model_id = find_json_value(json, "model_id");
    record.timestamp = find_json_value(json, "timestamp");
    record.tool_version = find_json_value(json, "tool_version");

    // Strategy sub-object
    std::string strategy_json = find_json_object(json, "strategy");
    if (!strategy_json.empty()) {
        record.backend = find_json_value(strategy_json, "backend");
        record.quant = find_json_value(strategy_json, "quant");
        record.placement = find_json_value(strategy_json, "placement");
        std::string gl = find_json_value(strategy_json, "gpu_layers");
        if (!gl.empty()) record.gpu_layers = (uint32_t)std::stoul(gl);
        std::string ctx = find_json_value(strategy_json, "context");
        if (!ctx.empty()) record.context = (uint32_t)std::stoul(ctx);
        std::string kv = find_json_value(strategy_json, "kv_quant_bits");
        if (!kv.empty()) record.kv_quant_bits = (uint32_t)std::stoul(kv);
        // MoE-specific strategy fields (Step 9, Phase E)
        std::string ge = find_json_value(strategy_json, "gpu_experts_per_layer");
        if (!ge.empty()) record.gpu_experts_per_layer = (uint32_t)std::stoul(ge);
        std::string te = find_json_value(strategy_json, "total_experts_per_layer");
        if (!te.empty()) record.total_experts_per_layer = (uint32_t)std::stoul(te);
        // Hot/Cold-specific strategy fields (Step 10, Phase H)
        std::string hnp = find_json_value(strategy_json, "hot_neuron_pct");
        if (!hnp.empty()) record.hot_neuron_pct = std::stod(hnp);
        std::string prof = find_json_value(strategy_json, "profiled");
        if (!prof.empty()) record.profiled = (prof == "true");
        std::string car = find_json_value(strategy_json, "cold_activation_rate");
        if (!car.empty()) record.cold_activation_rate = std::stod(car);
    }

    // Predicted sub-object
    std::string predicted_json = find_json_object(json, "predicted");
    if (!predicted_json.empty()) {
        std::string v;
        v = find_json_value(predicted_json, "tokens_per_sec");
        if (!v.empty()) record.predicted_tokens_per_sec = std::stod(v);
        v = find_json_value(predicted_json, "ttft_ms");
        if (!v.empty()) record.predicted_ttft_ms = std::stod(v);
        v = find_json_value(predicted_json, "vram_bytes");
        if (!v.empty()) record.predicted_vram_bytes = std::stoull(v);
        v = find_json_value(predicted_json, "ram_bytes");
        if (!v.empty()) record.predicted_ram_bytes = std::stoull(v);
        record.predicted_confidence = find_json_value(predicted_json, "confidence");
        // MoE-specific predicted ranges (Step 9, Phase D)
        v = find_json_value(predicted_json, "tokens_per_sec_min");
        if (!v.empty()) record.predicted_tokens_per_sec_min = std::stod(v);
        v = find_json_value(predicted_json, "tokens_per_sec_max");
        if (!v.empty()) record.predicted_tokens_per_sec_max = std::stod(v);
        v = find_json_value(predicted_json, "tokens_per_sec_expected");
        if (!v.empty()) record.predicted_tokens_per_sec_expected = std::stod(v);
        // Hot/Cold-specific predicted ranges (Step 10, Phase H)
        v = find_json_value(predicted_json, "tok_s_range_min");
        if (!v.empty()) record.predicted_tok_s_range_min = std::stod(v);
        v = find_json_value(predicted_json, "tok_s_range_max");
        if (!v.empty()) record.predicted_tok_s_range_max = std::stod(v);
    }

    // Actual sub-object
    std::string actual_json = find_json_object(json, "actual");
    if (!actual_json.empty()) {
        std::string v;
        v = find_json_value(actual_json, "tokens_per_sec");
        if (!v.empty()) record.actual_tokens_per_sec = std::stod(v);
        v = find_json_value(actual_json, "ttft_ms");
        if (!v.empty()) record.actual_ttft_ms = std::stod(v);
        v = find_json_value(actual_json, "peak_vram_bytes");
        if (!v.empty()) record.actual_peak_vram_bytes = std::stoull(v);
        v = find_json_value(actual_json, "peak_ram_bytes");
        if (!v.empty()) record.actual_peak_ram_bytes = std::stoull(v);
        v = find_json_value(actual_json, "throttled");
        if (!v.empty()) record.actual_throttled = (v == "true");
        v = find_json_value(actual_json, "tokens_generated");
        if (!v.empty()) record.actual_tokens_generated = std::stoi(v);
        v = find_json_value(actual_json, "duration_sec");
        if (!v.empty()) record.actual_duration_sec = std::stod(v);
        // MoE-specific actual telemetry (Step 9, Phase E)
        v = find_json_value(actual_json, "tokens_per_sec_min");
        if (!v.empty()) record.actual_tokens_per_sec_min = std::stod(v);
        v = find_json_value(actual_json, "tokens_per_sec_max");
        if (!v.empty()) record.actual_tokens_per_sec_max = std::stod(v);
        v = find_json_value(actual_json, "pcie_throughput_mbs");
        if (!v.empty()) record.actual_pcie_throughput_mbs = std::stoull(v);
        v = find_json_value(actual_json, "token_time_variance");
        if (!v.empty()) record.actual_token_time_variance = std::stod(v);
        // Hot/Cold-specific actual telemetry (Step 10, Phase H)
        v = find_json_value(actual_json, "cold_compute_pct");
        if (!v.empty()) record.actual_cold_compute_pct = std::stod(v);
    }

    return true;
}

// =============================================================================
// Record Creation
// =============================================================================

CalibrationRecord make_record(
    const HardwareSpec& hw,
    const ModelSpec& model,
    const StrategyConfig& strategy,
    const Prediction& prediction,
    const ExecutionResult& result,
    const std::string& model_id)
{
    CalibrationRecord r;

    // Identity
    r.hardware_fingerprint = hw.hardware_fingerprint;
    r.model_id = model_id;

    // Strategy
    r.backend = "llama.cpp";
    r.quant = model.quant_type;
    r.placement = placement_to_string(strategy.placement);
    r.gpu_layers = strategy.gpu_layers;
    r.context = strategy.context_length;
    r.kv_quant_bits = strategy.kv_quant_bits;
    // MoE-specific strategy fields (Step 9, Phase E)
    if (prediction.is_moe_range) {
        // For MoE, compute experts from the placement plan
        // The strategy.gpu_layers is set to model.layers for MoE Expert Offload
        // but we need to store actual expert counts
        r.gpu_experts_per_layer = (model.expert_count > 0) ? 
            (uint32_t)(prediction.gpu_hit_probability * model.expert_count + 0.5) : 0;
        r.total_experts_per_layer = model.expert_count;
    }
    
    // Hot/Cold-specific strategy fields (Step 10, Phase H)
    if (strategy.placement == PlacementStrategy::HOT_COLD_SPLIT) {
        // Hot neuron percentage (from the strategy description or default 15%)
        r.hot_neuron_pct = 0.15;  // Default; actual value from profiling
        r.profiled = false;       // Will be set to true if mask file exists
        r.cold_activation_rate = 0.12;  // Default estimate
    }

    // Predicted
    r.predicted_tokens_per_sec = prediction.tokens_per_sec;
    r.predicted_ttft_ms = prediction.ttft_ms;
    r.predicted_vram_bytes = prediction.memory_vram_bytes;
    r.predicted_ram_bytes = prediction.memory_ram_bytes;
    switch (prediction.confidence) {
        case PredictionConfidence::HIGH:   r.predicted_confidence = "HIGH";   break;
        case PredictionConfidence::MEDIUM: r.predicted_confidence = "MEDIUM"; break;
        case PredictionConfidence::LOW:    r.predicted_confidence = prediction.is_moe_range ? "LOW_MOE" : "LOW";
        default: r.predicted_confidence = prediction.is_moe_range ? "LOW_MOE" : "LOW"; break;
    }
    // MoE-specific predicted ranges (Step 9, Phase D)
    if (prediction.is_moe_range) {
        r.predicted_tokens_per_sec_min = prediction.tok_s_worst;
        r.predicted_tokens_per_sec_max = prediction.tok_s_best;
        r.predicted_tokens_per_sec_expected = prediction.tok_s_expected;
    }
    
    // Hot/Cold-specific predicted ranges (Step 10, Phase H)
    if (strategy.placement == PlacementStrategy::HOT_COLD_SPLIT) {
        r.predicted_tok_s_range_min = prediction.tok_s_worst;
        r.predicted_tok_s_range_max = prediction.tok_s_best;
    }

    // Actual
    r.actual_tokens_per_sec = result.decode_tokens_per_sec;
    r.actual_ttft_ms = result.prompt_eval_ms;
    r.actual_peak_vram_bytes = result.peak_vram_used_bytes;
    r.actual_peak_ram_bytes = result.peak_ram_used_bytes;
    r.actual_throttled = result.throttled;
    r.actual_tokens_generated = result.tokens_generated;
    r.actual_duration_sec = result.decode_ms / 1000.0;

    // Metadata
    r.timestamp = get_timestamp_utc();
    r.tool_version = CALIBRATION_TOOL_VERSION;

    return r;
}

// =============================================================================
// File I/O
// =============================================================================

bool append_record(const CalibrationRecord& record, const std::string& log_path) {
    std::ofstream file(log_path, std::ios::app);
    if (!file.is_open()) {
        fprintf(stderr, "Error: Cannot open calibration log file: %s\n", log_path.c_str());
        return false;
    }

    std::string json = record_to_json(record);
    file << json << "\n";
    file.close();

    return true;
}

std::vector<CalibrationRecord> read_all_records(const std::string& log_path) {
    std::vector<CalibrationRecord> records;

    std::ifstream file(log_path);
    if (!file.is_open()) {
        return records;  // File doesn't exist yet — empty is fine
    }

    std::string line;
    int line_num = 0;
    while (std::getline(file, line)) {
        line_num++;

        // Skip empty lines
        if (line.empty()) continue;

        // Skip whitespace-only lines
        bool all_space = true;
        for (char c : line) {
            if (c != ' ' && c != '\t' && c != '\r' && c != '\n') {
                all_space = false;
                break;
            }
        }
        if (all_space) continue;

        CalibrationRecord record;
        if (json_to_record(line, record)) {
            records.push_back(record);
        } else {
            fprintf(stderr, "Warning: Failed to parse calibration record on line %d, skipping.\n",
                    line_num);
        }
    }

    return records;
}

int count_records_for_hardware(const std::string& fingerprint,
                               const std::string& log_path) {
    auto records = read_all_records(log_path);
    int count = 0;
    for (const auto& r : records) {
        if (r.hardware_fingerprint == fingerprint) {
            count++;
        }
    }
    return count;
}

// =============================================================================
// Platform-Specific Log Path
// =============================================================================

std::string get_log_path() {
#if defined(_WIN32)
    // Windows: %APPDATA%\vessel\calibration.jsonl
    char appdata[MAX_PATH] = {};
    HRESULT hr = SHGetFolderPathA(NULL, CSIDL_APPDATA, NULL, 0, appdata);
    if (SUCCEEDED(hr) && appdata[0] != '\0') {
        std::string dir = std::string(appdata) + "\\vessel";
        CreateDirectoryA(dir.c_str(), NULL);
        return dir + "\\calibration.jsonl";
    }
    // Fallback: next to executable
    char exe_path[MAX_PATH] = {};
    GetModuleFileNameA(NULL, exe_path, MAX_PATH);
    std::string path(exe_path);
    auto pos = path.find_last_of('\\');
    if (pos != std::string::npos) path = path.substr(0, pos + 1) + "calibration.jsonl";
    else path = "calibration.jsonl";
    return path;
#elif defined(__APPLE__)
    // macOS: ~/Library/Application Support/vessel/calibration.jsonl
    const char* home = getenv("HOME");
    if (home) {
        std::string dir = std::string(home) + "/Library/Application Support/vessel";
        std::string cmd = "mkdir -p \"" + dir + "\"";
        system(cmd.c_str());
        return dir + "/calibration.jsonl";
    }
    return "calibration.jsonl";
#else
    // Linux: ~/.config/vessel/calibration.jsonl
    const char* home = getenv("HOME");
    if (home) {
        std::string dir = std::string(home) + "/.config/vessel";
        std::string cmd = "mkdir -p \"" + dir + "\"";
        system(cmd.c_str());
        return dir + "/calibration.jsonl";
    }
    return "calibration.jsonl";
#endif
}

// =============================================================================
// High-Level Write Entry (Phase C)
// =============================================================================

bool write_calibration_entry(
    const HardwareSpec& hw,
    const ModelSpec& model,
    const StrategyConfig& strategy,
    const Prediction& prediction,
    const ExecutionResult& result,
    const std::string& model_id)
{
    // Validation: skip if run was not meaningful
    if (result.tokens_generated < 10) {
        // Don't warn — this is expected for short benchmark runs
        return false;
    }

    // Get the platform-specific log path
    std::string log_path = get_log_path();

    // Create the record
    CalibrationRecord record = make_record(hw, model, strategy, prediction, result, model_id);

    // Try to append
    if (!append_record(record, log_path)) {
        fprintf(stderr, "\nWarning: Could not write calibration log to: %s\n",
                log_path.c_str());
        fprintf(stderr, "   Execution results are still valid.\n");
        return false;
    }

    // Success — print path in verbose mode
    printf("Calibration record saved to: %s\n", log_path.c_str());
    return true;
}
