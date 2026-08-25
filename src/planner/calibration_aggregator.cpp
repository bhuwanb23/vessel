#include "calibration_aggregator.h"
#include <cmath>
#include <algorithm>

// =============================================================================
// Default Constants (matching predictor defaults)
// =============================================================================

static constexpr uint64_t DEFAULT_GPU_OVERHEAD_BYTES = 512ULL * 1024 * 1024;  // 512 MB
static constexpr uint64_t DEFAULT_CPU_OVERHEAD_BYTES = 128ULL * 1024 * 1024;  // 128 MB
static constexpr double   DEFAULT_GPU_DECODE_EFFICIENCY = 0.27;
static constexpr double   DEFAULT_CPU_DECODE_EFFICIENCY = 0.80;
static constexpr double   DEFAULT_GPU_PREFILL_EFFICIENCY = 0.23;

// =============================================================================
// Constructor — Load and Filter
// =============================================================================

CalibrationAggregator::CalibrationAggregator(
    const std::string& hardware_fingerprint,
    const std::string& log_path_override)
{
    std::string log_path = log_path_override.empty() ? get_log_path() : log_path_override;
    std::vector<CalibrationRecord> all_records = read_all_records(log_path);
    total_records_ = static_cast<int>(all_records.size());

    for (const auto& record : all_records) {
        if (record.hardware_fingerprint == hardware_fingerprint
            && record.tool_version == CALIBRATION_TOOL_VERSION) {
            matching_records_.push_back(record);
        }
    }
}

// =============================================================================
// Filter Functions (Step D3)
// =============================================================================

std::vector<const CalibrationRecord*> CalibrationAggregator::filter_for_overhead(
    const std::string& placement, bool gpu) const
{
    std::vector<const CalibrationRecord*> result;
    for (const auto& r : matching_records_) {
        bool placement_match = (r.placement == placement);
        if (!placement_match && placement == "GPU_CPU_SPLIT" && r.placement == "HOT_COLD_SPLIT") {
            placement_match = true;
        }
        if (!placement_match) continue;
        if (r.actual_throttled) continue;
        if (r.actual_tokens_generated < 10) continue;
        if (gpu && r.actual_peak_vram_bytes == 0) continue;
        if (!gpu && r.actual_peak_ram_bytes == 0) continue;
        result.push_back(&r);
    }
    return result;
}

std::vector<const CalibrationRecord*> CalibrationAggregator::filter_for_decode_speed(
    const std::string& placement) const
{
    std::vector<const CalibrationRecord*> result;
    for (const auto& r : matching_records_) {
        bool placement_match = (r.placement == placement);
        if (!placement_match && placement == "GPU_CPU_SPLIT" && r.placement == "HOT_COLD_SPLIT") {
            placement_match = true;
        }
        if (!placement_match) continue;
        if (r.actual_throttled) continue;
        if (r.actual_tokens_generated < 50) continue;
        if (r.actual_tokens_per_sec <= 0) continue;
        result.push_back(&r);
    }
    return result;
}

std::vector<const CalibrationRecord*> CalibrationAggregator::filter_for_prefill() const
{
    std::vector<const CalibrationRecord*> result;
    for (const auto& r : matching_records_) {
        if (r.placement != "FULL_GPU") continue;
        if (r.actual_throttled) continue;
        if (r.actual_ttft_ms <= 0) continue;
        if (r.actual_ttft_ms < 5.0) continue;
        result.push_back(&r);
    }
    return result;
}

// =============================================================================
// Hot/Cold and Layer-Streaming Filters (Step 10, Phase H)
// =============================================================================

std::vector<const CalibrationRecord*> CalibrationAggregator::filter_for_hotcold() const
{
    std::vector<const CalibrationRecord*> result;
    for (const auto& r : matching_records_) {
        if (r.placement != "HOT_COLD_SPLIT") continue;
        if (r.actual_throttled) continue;
        if (r.actual_tokens_generated < 10) continue;
        if (r.actual_tokens_per_sec <= 0) continue;
        result.push_back(&r);
    }
    return result;
}

std::vector<const CalibrationRecord*> CalibrationAggregator::filter_for_layer_stream() const
{
    std::vector<const CalibrationRecord*> result;
    for (const auto& r : matching_records_) {
        if (r.placement != "LAYER_STREAM") continue;
        if (r.actual_tokens_generated < 10) continue;
        if (r.actual_tokens_per_sec <= 0) continue;
        result.push_back(&r);
    }
    return result;
}

// =============================================================================
// Confidence-Weighted Blending (Step D5)
// =============================================================================

double CalibrationAggregator::blend_with_default(
    double adjusted, double default_val, int sample_count) const
{
    if (sample_count >= 10) {
        return adjusted;
    } else if (sample_count >= 5) {
        return 0.7 * adjusted + 0.3 * default_val;
    } else if (sample_count >= 2) {
        return 0.4 * adjusted + 0.6 * default_val;
    } else {
        return default_val;
    }
}

// =============================================================================
// Compute Adjusted Constants (Step D4)
// =============================================================================

CalibrationData CalibrationAggregator::compute_adjusted_constants() const {
    CalibrationData data;
    data.total_record_count = total_records_;
    data.matching_record_count = static_cast<int>(matching_records_.size());

    if (matching_records_.empty()) {
        return data;
    }

    // GPU Overhead Adjustment
    {
        auto records = filter_for_overhead("FULL_GPU", true);
        data.overhead_gpu_records = static_cast<int>(records.size());

        if (!records.empty()) {
            double sum_overhead = 0.0;
            for (const auto* r : records) {
                double overhead_delta = static_cast<double>(r->actual_peak_vram_bytes)
                                      - static_cast<double>(r->predicted_vram_bytes);
                sum_overhead += DEFAULT_GPU_OVERHEAD_BYTES + overhead_delta;
            }
            double avg_overhead = sum_overhead / records.size();
            avg_overhead = std::max(200.0 * 1024 * 1024, avg_overhead);
            avg_overhead = std::min(2.0 * 1024 * 1024 * 1024, avg_overhead);

            data.adjusted_gpu_overhead_bytes = static_cast<uint64_t>(
                blend_with_default(avg_overhead, DEFAULT_GPU_OVERHEAD_BYTES,
                                   static_cast<int>(records.size())));
            data.has_calibration_data = true;
        }
    }

    // CPU Overhead Adjustment
    {
        auto records = filter_for_overhead("CPU_ONLY", false);
        data.overhead_cpu_records = static_cast<int>(records.size());

        if (!records.empty()) {
            double sum_overhead = 0.0;
            for (const auto* r : records) {
                double overhead_delta = static_cast<double>(r->actual_peak_ram_bytes)
                                      - static_cast<double>(r->predicted_ram_bytes);
                sum_overhead += DEFAULT_CPU_OVERHEAD_BYTES + overhead_delta;
            }
            double avg_overhead = sum_overhead / records.size();
            avg_overhead = std::max(64.0 * 1024 * 1024, avg_overhead);
            avg_overhead = std::min(1.0 * 1024 * 1024 * 1024, avg_overhead);

            data.adjusted_cpu_overhead_bytes = static_cast<uint64_t>(
                blend_with_default(avg_overhead, DEFAULT_CPU_OVERHEAD_BYTES,
                                   static_cast<int>(records.size())));
            data.has_calibration_data = true;
        }
    }

    // GPU Decode Efficiency Adjustment
    {
        auto records = filter_for_decode_speed("FULL_GPU");
        data.decode_gpu_records = static_cast<int>(records.size());

        if (!records.empty()) {
            double sum_efficiency = 0.0;
            int count = 0;
            for (const auto* r : records) {
                if (r->predicted_tokens_per_sec > 0 && r->actual_tokens_per_sec > 0) {
                    double ratio = r->actual_tokens_per_sec / r->predicted_tokens_per_sec;
                    double implied_eff = DEFAULT_GPU_DECODE_EFFICIENCY * ratio;
                    implied_eff = std::max(0.05, implied_eff);
                    implied_eff = std::min(0.80, implied_eff);
                    sum_efficiency += implied_eff;
                    count++;
                }
            }

            if (count > 0) {
                double avg_eff = sum_efficiency / count;
                data.adjusted_gpu_decode_efficiency =
                    blend_with_default(avg_eff, DEFAULT_GPU_DECODE_EFFICIENCY, count);
                data.has_calibration_data = true;
            }
        }
    }

    // CPU Decode Efficiency Adjustment
    {
        auto records = filter_for_decode_speed("CPU_ONLY");
        data.decode_cpu_records = static_cast<int>(records.size());

        if (!records.empty()) {
            double sum_efficiency = 0.0;
            int count = 0;
            for (const auto* r : records) {
                if (r->predicted_tokens_per_sec > 0 && r->actual_tokens_per_sec > 0) {
                    double ratio = r->actual_tokens_per_sec / r->predicted_tokens_per_sec;
                    double implied_eff = DEFAULT_CPU_DECODE_EFFICIENCY * ratio;
                    implied_eff = std::max(0.10, implied_eff);
                    implied_eff = std::min(0.95, implied_eff);
                    sum_efficiency += implied_eff;
                    count++;
                }
            }

            if (count > 0) {
                double avg_eff = sum_efficiency / count;
                data.adjusted_cpu_decode_efficiency =
                    blend_with_default(avg_eff, DEFAULT_CPU_DECODE_EFFICIENCY, count);
                data.has_calibration_data = true;
            }
        }
    }

    // GPU Prefill Efficiency Adjustment
    {
        auto records = filter_for_prefill();
        data.prefill_records = static_cast<int>(records.size());

        if (!records.empty()) {
            double sum_efficiency = 0.0;
            int count = 0;
            for (const auto* r : records) {
                if (r->predicted_ttft_ms > 0 && r->actual_ttft_ms > 0) {
                    double ratio = r->predicted_ttft_ms / r->actual_ttft_ms;
                    double implied_eff = DEFAULT_GPU_PREFILL_EFFICIENCY * ratio;
                    implied_eff = std::max(0.05, implied_eff);
                    implied_eff = std::min(0.80, implied_eff);
                    sum_efficiency += implied_eff;
                    count++;
                }
            }

            if (count > 0) {
                double avg_eff = sum_efficiency / count;
                data.adjusted_gpu_prefill_efficiency =
                    blend_with_default(avg_eff, DEFAULT_GPU_PREFILL_EFFICIENCY, count);
                data.has_calibration_data = true;
            }
        }
    }

    return data;
}

// =============================================================================
// Derived Constants from Hot/Cold Records (Step 10, Phase H)
// =============================================================================

HotColdDerivedConstants compute_hotcold_derived_constants() {
    HotColdDerivedConstants result;
    // For now, return defaults
    // In production, this would read the calibration log and compute
    return result;
}

// =============================================================================
// Derived Constants from Layer-Streaming Records (Step 10, Phase H)
// =============================================================================

LayerStreamDerivedConstants compute_layer_stream_derived_constants() {
    LayerStreamDerivedConstants result;
    // For now, return defaults
    // In production, this would read the calibration log and compute
    return result;
}

// =============================================================================
// Public API
// =============================================================================

CalibrationData CalibrationAggregator::get_calibration_data() const {
    return compute_adjusted_constants();
}

int CalibrationAggregator::get_matching_record_count() const {
    return static_cast<int>(matching_records_.size());
}

int CalibrationAggregator::get_total_record_count() const {
    return total_records_;
}
