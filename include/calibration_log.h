#pragma once

#include "types.h"
#include "executor.h"
#include <string>
#include <vector>
#include <cstdint>

// =============================================================================
// Calibration Log — Step 7
// =============================================================================
// Records predicted vs actual metrics after each execution run.
// Stored as JSONL (one JSON object per line) in a local file.
// Keyed by hardware_fingerprint + model_id for aggregation.
//
// Log file location: next to the executable, "calibration_log.jsonl"
// =============================================================================

// Tool version — increment when formulas/constants change
static const char* CALIBRATION_TOOL_VERSION = "0.1.0";

// Default log file path
static const char* CALIBRATION_LOG_FILE = "calibration_log.jsonl";

// =============================================================================
// Calibration Record — matches the locked JSON schema from §7
// =============================================================================

struct CalibrationRecord {
    // Identity
    std::string hardware_fingerprint;   // From Phase A
    std::string model_id;               // HuggingFace repo/filename (not local path)

    // Strategy
    std::string backend;                // "llama.cpp" for MVP
    std::string quant;                  // "Q4_K_M", "Q8_0", etc.
    std::string placement;              // "FULL_GPU", "GPU_CPU_SPLIT", "CPU_ONLY"
    uint32_t gpu_layers = 0;
    uint32_t context = 0;
    uint32_t kv_quant_bits = 16;

    // Predicted (from Step 3 predictor)
    double predicted_tokens_per_sec = 0.0;
    double predicted_ttft_ms = 0.0;
    uint64_t predicted_vram_bytes = 0;
    uint64_t predicted_ram_bytes = 0;
    std::string predicted_confidence;   // "HIGH", "MEDIUM", "LOW"

    // Actual (from Step 6 executor)
    double actual_tokens_per_sec = 0.0;
    double actual_ttft_ms = 0.0;
    uint64_t actual_peak_vram_bytes = 0;
    uint64_t actual_peak_ram_bytes = 0;
    bool actual_throttled = false;
    int actual_tokens_generated = 0;
    double actual_duration_sec = 0.0;

    // Metadata
    std::string timestamp;              // ISO 8601 UTC
    std::string tool_version;           // CALIBRATION_TOOL_VERSION
};

// =============================================================================
// Record Lifecycle
// =============================================================================

// Create a CalibrationRecord from Prediction + ExecutionResult + context info
CalibrationRecord make_record(
    const HardwareSpec& hw,
    const ModelSpec& model,
    const StrategyConfig& strategy,
    const Prediction& prediction,
    const ExecutionResult& result,
    const std::string& model_id  // HuggingFace-style identifier
);

// =============================================================================
// File I/O — JSONL format (one JSON object per line)
// =============================================================================

// Append a record to the calibration log file.
// Creates the file if it doesn't exist.
// Returns true on success.
bool append_record(const CalibrationRecord& record,
                   const std::string& log_path = CALIBRATION_LOG_FILE);

// Read all records from the log file.
// Returns empty vector if file doesn't exist or is empty.
std::vector<CalibrationRecord> read_all_records(
    const std::string& log_path = CALIBRATION_LOG_FILE);

// Count records matching a specific hardware fingerprint.
int count_records_for_hardware(const std::string& fingerprint,
                               const std::string& log_path = CALIBRATION_LOG_FILE);

// =============================================================================
// JSON Serialization (internal, exposed for testing)
// =============================================================================

// Serialize a single record to a JSON string (one line).
std::string record_to_json(const CalibrationRecord& record);

// Deserialize a single JSON string to a record.
// Returns true on success, false on parse error.
bool json_to_record(const std::string& json, CalibrationRecord& record);
