#pragma once

#include "types.h"
#include "calibration_log.h"
#include <string>
#include <vector>

// =============================================================================
// Calibration Data — Adjusted Constants from Log Aggregation
// =============================================================================
// Computed at the start of each run by reading the calibration log.
// These replace the hardcoded defaults in the predictor formulas.
//
// For MVP: Computed in memory, not persisted to a config file.
// The JSONL log is the single source of truth.
// =============================================================================

struct CalibrationData {
    // ---- Memory overhead (bytes) ----
    // Replaces the 512MB/128MB defaults in predict_overhead_memory()
    uint64_t adjusted_gpu_overhead_bytes = 0;  // 0 = use default (512MB)
    uint64_t adjusted_cpu_overhead_bytes = 0;  // 0 = use default (128MB)

    // ---- Decode efficiency (0.0 to 1.0) ----
    // Replaces the 0.27/0.80 defaults in predict_decode_speed()
    double adjusted_gpu_decode_efficiency = 0.0;  // 0 = use default (0.27)
    double adjusted_cpu_decode_efficiency = 0.0;  // 0 = use default (0.80)

    // ---- Prefill efficiency (0.0 to 1.0) ----
    // Replaces the 0.23 default in predict_prompt_eval_speed() and predict_ttft_ms()
    double adjusted_gpu_prefill_efficiency = 0.0;  // 0 = use default (0.23)

    // ---- Metadata ----
    int matching_record_count = 0;     // How many records matched this hardware
    int total_record_count = 0;        // Total records in log
    bool has_calibration_data = false; // false = no matching records, use all defaults

    // ---- Per-metric record counts (for debugging) ----
    int overhead_gpu_records = 0;
    int overhead_cpu_records = 0;
    int decode_gpu_records = 0;
    int decode_cpu_records = 0;
    int prefill_records = 0;
};

// =============================================================================
// Calibration Aggregator — Reads and aggregates calibration log
// =============================================================================

class CalibrationAggregator {
public:
    // Create an aggregator for a specific hardware fingerprint.
    // Reads the log file and filters by fingerprint.
    // log_path_override: if empty, uses get_log_path() (platform-specific)
    CalibrationAggregator(const std::string& hardware_fingerprint,
                          const std::string& log_path_override = "");

    // Get the aggregated calibration data.
    // Returns a CalibrationData struct with adjusted constants.
    // If no matching records exist, all fields are 0 (use defaults).
    CalibrationData get_calibration_data() const;

    // Get the number of matching records for this hardware.
    int get_matching_record_count() const;

    // Get the number of total records in the log.
    int get_total_record_count() const;

private:
    // All records matching this hardware fingerprint
    std::vector<CalibrationRecord> matching_records_;

    // Total records in the log
    int total_records_ = 0;

    // Compute adjusted constants from matching records
    CalibrationData compute_adjusted_constants() const;

    // Confidence-weighted blend of adjusted value with default
    // Uses the sample-size thresholds from Phase D Step D5
    double blend_with_default(double adjusted, double default_val,
                              int sample_count) const;

    // Filter records by placement and quality criteria
    std::vector<const CalibrationRecord*> filter_for_overhead(
        const std::string& placement, bool gpu) const;
    std::vector<const CalibrationRecord*> filter_for_decode_speed(
        const std::string& placement) const;
    std::vector<const CalibrationRecord*> filter_for_prefill() const;
    
    // Hot/Cold and Layer-Streaming filters (Step 10, Phase H)
    std::vector<const CalibrationRecord*> filter_for_hotcold() const;
    std::vector<const CalibrationRecord*> filter_for_layer_stream() const;
};

// =============================================================================
// Derived Constants from Hot/Cold Records (Step 10, Phase H)
// =============================================================================

struct HotColdDerivedConstants {
    double cold_activation_rate = 0.12;      // How many cold neurons actually fire per token
    double gpu_cpu_sync_overhead_ms = 0.0;   // Gap between max(t_hot, t_cold) and actual per-layer time
    int sample_count = 0;
};

// Compute derived constants from hot/cold calibration records
// Returns defaults if no records available
HotColdDerivedConstants compute_hotcold_derived_constants();

// =============================================================================
// Derived Constants from Layer-Streaming Records (Step 10, Phase H)
// =============================================================================

struct LayerStreamDerivedConstants {
    double layer_stream_actual_io_speed_mbs = 0.0;  // Real-world sequential read speed during streaming
    int sample_count = 0;
};

// Compute derived constants from layer-streaming calibration records
// Returns defaults if no records available
LayerStreamDerivedConstants compute_layer_stream_derived_constants();
