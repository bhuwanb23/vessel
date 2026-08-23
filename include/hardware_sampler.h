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
    uint32_t pcie_rx_mbs = 0;       // PCIe RX throughput (MB/s)
    uint32_t pcie_tx_mbs = 0;       // PCIe TX throughput (MB/s)
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
    
    // MoE-specific telemetry (Step 9, Phase E)
    uint64_t pcie_throughput_mbs = 0;     // Average PCIe RX throughput (MB/s)
    uint64_t pcie_tx_throughput_mbs = 0;  // Average PCIe TX throughput (MB/s)
    double token_time_variance_ms = 0.0;  // Variance in token generation time (ms^2)
    double avg_token_time_ms = 0.0;       // Average token generation time (ms)
    bool vram_verified = false;           // VRAM usage verified against prediction
    double vram_verification_delta = 0.0; // Actual vs predicted VRAM delta (bytes)
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
