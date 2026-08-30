#pragma once

#include "types.h"
#include <string>
#include <vector>
#include <cstdint>

// Forward declaration — ExecutionResult is defined in executor.h.
// We only need it by reference here, avoiding a transitive #include <llama.h>
// which would break targets that don't link llama.cpp (e.g. predictor_test).
struct ExecutionResult;

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

// Default log file path (relative, for testing)
static const char* CALIBRATION_LOG_FILE = "calibration_log.jsonl";

// Get the platform-specific log file path.
// Windows: %APPDATA%\llm-planner\calibration.jsonl
// Fallback: ./calibration_log.jsonl
std::string get_log_path();

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
    std::string placement;              // "FULL_GPU", "GPU_CPU_SPLIT", "CPU_ONLY", "MoE-Expert-Offload", etc.
    uint32_t gpu_layers = 0;
    uint32_t context = 0;
    uint32_t kv_quant_bits = 16;
    
    // MoE-specific strategy fields (Step 9, Phase E)
    uint32_t gpu_experts_per_layer = 0;     // Number of routed experts on GPU per layer
    uint32_t total_experts_per_layer = 0;   // Total routed experts per layer (N)
    
    // Hot/Cold-specific strategy fields (Step 10, Phase H)
    double hot_neuron_pct = 0.0;            // Fraction of neurons on GPU (0.15 = 15%)
    bool profiled = false;                  // Was a real profiling mask used?
    double cold_activation_rate = 0.0;      // Observed sparsity during decode

    // Predicted (from Step 3 predictor)
    double predicted_tokens_per_sec = 0.0;
    double predicted_ttft_ms = 0.0;
    uint64_t predicted_vram_bytes = 0;
    uint64_t predicted_ram_bytes = 0;
    std::string predicted_confidence;   // "HIGH", "MEDIUM", "LOW", "LOW_MOE"
    
    // MoE-specific predicted ranges (Step 9, Phase D)
    double predicted_tokens_per_sec_min = 0.0;   // Worst case (0%% GPU hit)
    double predicted_tokens_per_sec_max = 0.0;   // Best case (100%% GPU hit)
    double predicted_tokens_per_sec_expected = 0.0; // Expected (uniform routing)
    
    // Hot/Cold-specific predicted ranges (Step 10, Phase H)
    double predicted_tok_s_range_min = 0.0;   // Worst case (all cold neurons activate)
    double predicted_tok_s_range_max = 0.0;   // Best case (few cold neurons activate)

    // Actual (from Step 6 executor)
    double actual_tokens_per_sec = 0.0;
    double actual_ttft_ms = 0.0;
    uint64_t actual_peak_vram_bytes = 0;
    uint64_t actual_peak_ram_bytes = 0;
    bool actual_throttled = false;
    int actual_tokens_generated = 0;
    double actual_duration_sec = 0.0;
    
    // MoE-specific actual telemetry (Step 9, Phase E)
    double actual_tokens_per_sec_min = 0.0;   // Min measured tok/s (variance)
    double actual_tokens_per_sec_max = 0.0;   // Max measured tok/s (variance)
    uint64_t actual_pcie_throughput_mbs = 0;   // Average PCIe RX throughput
    double actual_token_time_variance = 0.0;   // Variance in token generation time
    
    // Hot/Cold-specific actual telemetry (Step 10, Phase H)
    double actual_cold_compute_pct = 0.0;     // Fraction of time spent on CPU cold path

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

// High-level convenience: create record from run results and append to log.
// Handles directory creation, error messages, and validation.
// Returns true on success, false on write error (warning printed).
// Does NOT fail the run — the execution results are still valid.
bool write_calibration_entry(
    const HardwareSpec& hw,
    const ModelSpec& model,
    const StrategyConfig& strategy,
    const Prediction& prediction,
    const ExecutionResult& result,
    const std::string& model_id
);

// =============================================================================
// JSON Serialization (internal, exposed for testing)
// =============================================================================

// Serialize a single record to a JSON string (one line).
std::string record_to_json(const CalibrationRecord& record);

// Deserialize a single JSON string to a record.
// Returns true on success, false on parse error.
bool json_to_record(const std::string& json, CalibrationRecord& record);
