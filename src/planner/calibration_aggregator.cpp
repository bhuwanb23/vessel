#include "calibration_aggregator.h"
#include <cmath>
#include <algorithm>

// =============================================================================
// Default Constants (matching predictor defaults)
// =============================================================================
// These are the hardcoded values in the predictor that get replaced
// when calibration data is available.

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
    const std::string& log_path)
{
    // Step D1: Load all entries
    std::vector<CalibrationRecord> all_records = read_all_records(log_path);
    total_records_ = static_cast<int>(all_records.size());

    // Step D2: Filter by hardware fingerprint
    for (const auto& record : all_records) {
        if (record.hardware_fingerprint == hardware_fingerprint) {
            matching_records_.push_back(record);
        }
    }

    // Debug: print summary
    fprintf(stderr, "  [Calibration] Log: %s, Total: %d, Matching: %d\n",
            log_path.c_str(), total_records_, (int)matching_records_.size());
}

// =============================================================================
// Filter Functions (Step D3)
// =============================================================================

std::vector<const CalibrationRecord*> CalibrationAggregator::filter_for_overhead(
    const std::string& placement, bool gpu) const
{
    std::vector<const CalibrationRecord*> result;
    for (const auto& r : matching_records_) {
        // Must match placement
        if (r.placement != placement) continue;
        // Must not be throttled (distorts overhead measurements)
        if (r.actual_throttled) continue;
        // Must have meaningful data
        if (r.actual_tokens_generated < 10) continue;
        // Must have actual memory data
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
        // Must match placement
        if (r.placement != placement) continue;
        // Must not be throttled
        if (r.actual_throttled) continue;
        // Must have meaningful token count (at least 50 for stable tok/s)
        if (r.actual_tokens_generated < 50) continue;
        // Must have actual speed data
        if (r.actual_tokens_per_sec <= 0) continue;

        result.push_back(&r);
    }
    return result;
}

std::vector<const CalibrationRecord*> CalibrationAggregator::filter_for_prefill() const
{
    std::vector<const CalibrationRecord*> result;
    for (const auto& r : matching_records_) {
        // Only full GPU for prefill efficiency
        if (r.placement != "FULL_GPU") continue;
        // Must not be throttled
        if (r.actual_throttled) continue;
        // Must have actual TTFT data
        if (r.actual_ttft_ms <= 0) continue;
        // Must have meaningful prompt length (at least 20 tokens)
        // We don't have prompt_tokens in the record, but we can check
        // that ttft is reasonable (> 5ms means real work happened)
        if (r.actual_ttft_ms < 5.0) continue;

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
        // Trust the data fully
        return adjusted;
    } else if (sample_count >= 5) {
        // Blend: 70% adjusted, 30% default
        return 0.7 * adjusted + 0.3 * default_val;
    } else if (sample_count >= 2) {
        // Lean toward default: 40% adjusted, 60% default
        return 0.4 * adjusted + 0.6 * default_val;
    } else {
        // Not enough data — use default
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
        return data;  // No calibration data — all zeros = use defaults
    }

    // =========================================================================
    // GPU Overhead Adjustment
    // Formula: actual_overhead = actual_peak_vram - (weight_bytes + kv_cache_bytes)
    // We average across all matching FULL_GPU entries
    // =========================================================================
    {
        auto records = filter_for_overhead("FULL_GPU", true);
        data.overhead_gpu_records = static_cast<int>(records.size());

        if (!records.empty()) {
            double sum_overhead = 0.0;
            for (const auto* r : records) {
                // We need to compute what weight+kv should be from the record's strategy
                // The record has predicted vram which includes weight+kv+overhead
                // We can estimate: actual_overhead ≈ actual_peak_vram - predicted_vram + predicted_overhead
                // But predicted_overhead is baked into predicted_vram...
                // Simpler approach: just use a fixed proportion estimate
                // For MVP: assume overhead ≈ actual_peak_vram - predicted_vram + 512MB default
                // This is approximate but improves over pure default

                // Better: We know the model's weight memory from the record
                // weight_bytes = param_count * bpw / 8 — but we don't have param_count in the record
                // For MVP: use the difference between actual and predicted VRAM
                // predicted already includes the default overhead
                // actual - predicted = actual_overhead - default_overhead
                // actual_overhead = default_overhead + (actual - predicted)
                double overhead_delta = static_cast<double>(r->actual_peak_vram_bytes)
                                      - static_cast<double>(r->predicted_vram_bytes);
                sum_overhead += DEFAULT_GPU_OVERHEAD_BYTES + overhead_delta;
            }
            double avg_overhead = sum_overhead / records.size();
            // Clamp to reasonable range: 200 MB to 2 GB
            avg_overhead = std::max(200.0 * 1024 * 1024, avg_overhead);
            avg_overhead = std::min(2.0 * 1024 * 1024 * 1024, avg_overhead);

            data.adjusted_gpu_overhead_bytes = static_cast<uint64_t>(
                blend_with_default(avg_overhead, DEFAULT_GPU_OVERHEAD_BYTES,
                                   static_cast<int>(records.size())));
            data.has_calibration_data = true;
        }
    }

    // =========================================================================
    // CPU Overhead Adjustment
    // =========================================================================
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

    // =========================================================================
    // GPU Decode Efficiency Adjustment
    // From: tok/s = (gpu_bw * 1e9 * efficiency) / bytes_per_token
    // Rearranged: efficiency = tok/s * bytes_per_token / (gpu_bw * 1e9)
    // =========================================================================
    {
        auto records = filter_for_decode_speed("FULL_GPU");
        data.decode_gpu_records = static_cast<int>(records.size());

        if (!records.empty()) {
            // We need bytes_per_token from the record's model.
            // The record doesn't store param_count/bpw directly, but we can
            // estimate from the predicted VRAM (which is weight_bytes + kv + overhead)
            // For a simple approximation, assume weight_bytes ≈ predicted_vram - overhead
            // Then bytes_per_token = weight_bytes * 8 / param_count
            // This is circular... For MVP, use a simpler approach:
            // efficiency = actual_tok/s / theoretical_tok/s
            // theoretical = gpu_bw / bytes_per_token
            // We can estimate bytes_per_token from the model info in the record

            // Actually, the cleanest way: the predicted tok/s uses the default efficiency
            // So: actual / predicted = actual_efficiency / default_efficiency
            // => actual_efficiency = default * (actual / predicted)
            double sum_efficiency = 0.0;
            int count = 0;
            for (const auto* r : records) {
                if (r->predicted_tokens_per_sec > 0 && r->actual_tokens_per_sec > 0) {
                    double ratio = r->actual_tokens_per_sec / r->predicted_tokens_per_sec;
                    double implied_eff = DEFAULT_GPU_DECODE_EFFICIENCY * ratio;
                    // Clamp to reasonable range: 0.05 to 0.80
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

    // =========================================================================
    // CPU Decode Efficiency Adjustment
    // =========================================================================
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

    // =========================================================================
    // GPU Prefill Efficiency Adjustment
    // From: ttft = (2 * params * prompt_tokens) / (tflops * efficiency * 1e12)
    // Rearranged: efficiency = (2 * params * prompt_tokens) / (ttft_ms * 1e-3 * tflops * 1e12)
    // =========================================================================
    {
        auto records = filter_for_prefill();
        data.prefill_records = static_cast<int>(records.size());

        if (!records.empty()) {
            // Similar approach: use predicted vs actual ratio
            double sum_efficiency = 0.0;
            int count = 0;
            for (const auto* r : records) {
                if (r->predicted_ttft_ms > 0 && r->actual_ttft_ms > 0) {
                    // Lower ttft = faster = higher efficiency
                    // efficiency_ratio = predicted_ttft / actual_ttft
                    // (if actual is faster, ttft is lower, ratio > 1)
                    double ratio = r->predicted_ttft_ms / r->actual_ttft_ms;
                    // But wait — predicted TTFT uses a hybrid model, not pure compute
                    // The ratio gives us a correction factor for the overall TTFT prediction
                    // For the efficiency constant specifically:
                    // If actual_ttft < predicted_ttft, the real efficiency is higher
                    // implied_eff = default_eff * (predicted_ttft / actual_ttft)
                    // This works because TTFT is inversely proportional to efficiency
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
