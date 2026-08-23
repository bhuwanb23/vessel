#pragma once

#include <cstdint>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>

// =============================================================================
// Hardware Sampler — Live GPU/RAM monitoring during inference
// =============================================================================
// Runs on a background thread, polling NVML and RAM every 500ms.
// Used by the executor to measure peak VRAM, RAM, temperature, and
// detect thermal throttling during model execution.
// =============================================================================

// Single snapshot of hardware state
struct HardwareSample {
    uint64_t vram_used_bytes = 0;
    uint64_t ram_used_bytes = 0;
    uint32_t gpu_temp_celsius = 0;
    uint32_t gpu_clock_mhz = 0;
    uint64_t timestamp_ms = 0;
};

// Aggregated results after a sampling session
struct HardwareMetrics {
    uint64_t peak_vram_bytes = 0;
    uint64_t peak_ram_bytes = 0;
    uint32_t max_temp_celsius = 0;
    bool throttled = false;       // clock dropped below 85% of max
    bool gpu_bus_off = false;     // NVML query failed — GPU may have fallen off bus
    bool os_swapping = false;     // page file usage spiked — RAM oversubscribed
    int sample_count = 0;
};

// =============================================================================
// HardwareSampler — Background thread that polls hardware state
// =============================================================================

class HardwareSampler {
public:
    HardwareSampler();
    ~HardwareSampler();

    // Non-copyable
    HardwareSampler(const HardwareSampler&) = delete;
    HardwareSampler& operator=(const HardwareSampler&) = delete;

    // Start sampling on a background thread
    // max_clock_mhz: GPU's max clock from Step 1 (for throttle detection)
    void start(uint32_t max_clock_mhz = 0);

    // Stop sampling and wait for thread to exit
    void stop();

    // Get aggregated metrics after stop()
    HardwareMetrics get_metrics() const;

    // Check if thermal throttling was detected
    bool was_throttled() const;

private:
    void poll_loop();

    std::thread thread_;
    std::atomic<bool> running_{false};
    std::vector<HardwareSample> samples_;
    mutable std::mutex mutex_;

    uint32_t max_clock_mhz_ = 0;  // for throttle detection
    uint64_t prev_page_file_avail_ = 0;  // for swap detection
    bool nvml_failed_ = false;           // for GPU bus-off detection
};
