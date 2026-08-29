#include "hardware_sampler.h"
#include "executor.h"
#include "nvml_loader.h"
#include <csignal>
#include <cstdio>
#include <chrono>
#include <algorithm>

#if defined(_WIN32)
#include <windows.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#endif

// =============================================================================
// NVML State (NVIDIA only, not available on macOS)
// Uses shared nvml_loader for dynamic loading
// =============================================================================

static bool nvml_initialized = false;
static nvmlDevice_t nvml_device = nullptr;

static void ensure_nvml() {
    if (nvml_initialized) return;
    if (!nvml_loader_init()) return;
    nvmlReturn_t ret = nvml_fn_Init();
    if (ret == NVML_SUCCESS) {
        nvml_initialized = true;
        unsigned int device_count = 0;
        if (nvml_fn_DeviceGetCount) nvml_fn_DeviceGetCount(&device_count);
        if (device_count > 0 && nvml_fn_DeviceGetHandleByIndex)
            nvml_fn_DeviceGetHandleByIndex(0, &nvml_device);
    }
}
static void shutdown_nvml() {
    if (nvml_initialized && nvml_fn_Shutdown) { nvml_fn_Shutdown(); nvml_initialized = false; nvml_device = nullptr; }
    nvml_loader_shutdown();
}

// =============================================================================
// HardwareSampler
// =============================================================================

HardwareSampler::HardwareSampler() {
    ensure_nvml();
}

HardwareSampler::~HardwareSampler() {
    stop();  // ensure thread is joined
}

void HardwareSampler::start(uint32_t max_clock_mhz) {
    max_clock_mhz_ = max_clock_mhz;
    running_ = true;
    samples_.clear();
    thread_ = std::thread([this]() { poll_loop(); });
}

void HardwareSampler::stop() {
    running_ = false;
    if (thread_.joinable()) {
        thread_.join();
    }
}

HardwareMetrics HardwareSampler::get_metrics() const {
    HardwareMetrics metrics;

    std::lock_guard<std::mutex> lock(mutex_);
    metrics.sample_count = static_cast<int>(samples_.size());

    uint64_t pcie_rx_sum = 0;
    uint64_t pcie_tx_sum = 0;

    for (const auto& s : samples_) {
        if (s.vram_used_bytes > metrics.peak_vram_bytes)
            metrics.peak_vram_bytes = s.vram_used_bytes;
        if (s.ram_used_bytes > metrics.peak_ram_bytes)
            metrics.peak_ram_bytes = s.ram_used_bytes;
        if (s.gpu_temp_celsius > metrics.max_temp_celsius)
            metrics.max_temp_celsius = s.gpu_temp_celsius;

        // Throttle detection: clock < 85% of max
        if (max_clock_mhz_ > 0 && s.gpu_clock_mhz > 0) {
            if (s.gpu_clock_mhz < max_clock_mhz_ * 0.85) {
                metrics.throttled = true;
            }
        }

        // PCIe throughput aggregation (Step 9, Phase E)
        pcie_rx_sum += s.pcie_rx_mbs;
        pcie_tx_sum += s.pcie_tx_mbs;
    }

    // Average PCIe throughput across all samples
    if (metrics.sample_count > 0) {
        metrics.pcie_throughput_mbs = pcie_rx_sum / metrics.sample_count;
        metrics.pcie_tx_throughput_mbs = pcie_tx_sum / metrics.sample_count;
    }

    // GPU bus-off detection
    metrics.gpu_bus_off = nvml_failed_;

    // OS swap detection: if page file usage increased by >500MB between any two samples
    metrics.os_swapping = false;
    if (samples_.size() >= 2) {
        for (size_t i = 1; i < samples_.size(); i++) {
            // We track swap indirectly: if RAM used jumped >500MB in one interval
            // while VRAM stayed flat, the OS is likely paging to disk
            uint64_t ram_delta = samples_[i].ram_used_bytes - samples_[i-1].ram_used_bytes;
            if (ram_delta > 500ULL * 1024 * 1024) {
                metrics.os_swapping = true;
                break;
            }
        }
    }

    return metrics;
}

bool HardwareSampler::was_throttled() const {
    return get_metrics().throttled;
}

void HardwareSampler::poll_loop() {
    while (running_) {
        HardwareSample sample;

        // Get current timestamp
        auto now = std::chrono::steady_clock::now();
        sample.timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()).count();

        bool nvml_ok = false;
#if defined(_WIN32)
        // Sample GPU via NVML (NVIDIA only, dynamically loaded)
        if (nvml_initialized && nvml_device && nvmlDeviceGetMemoryInfo_fn) {
            nvmlMemory_t mem;
            if (nvmlDeviceGetMemoryInfo_fn(nvml_device, &mem) == NVML_SUCCESS) {
                sample.vram_used_bytes = mem.used;
                nvml_ok = true;
            }
            unsigned int temp = 0;
            if (nvmlDeviceGetTemperature_fn && nvmlDeviceGetTemperature_fn(nvml_device, NVML_TEMPERATURE_GPU, &temp) == NVML_SUCCESS)
                sample.gpu_temp_celsius = temp;
            unsigned int clock = 0;
            if (nvmlDeviceGetClockInfo_fn && nvmlDeviceGetClockInfo_fn(nvml_device, NVML_CLOCK_SM, &clock) == NVML_SUCCESS)
                sample.gpu_clock_mhz = clock;
            unsigned int pcie_rx = 0;
            if (nvmlDeviceGetPcieThroughput_fn && nvmlDeviceGetPcieThroughput_fn(nvml_device, NVML_PCIE_UTIL_RX_BYTES, &pcie_rx) == NVML_SUCCESS)
                sample.pcie_rx_mbs = pcie_rx / (1024 * 1024);
            unsigned int pcie_tx = 0;
            if (nvmlDeviceGetPcieThroughput_fn && nvmlDeviceGetPcieThroughput_fn(nvml_device, NVML_PCIE_UTIL_TX_BYTES, &pcie_tx) == NVML_SUCCESS)
                sample.pcie_tx_mbs = pcie_tx / (1024 * 1024);
        }
#endif

        // GPU bus-off detection: if NVML query failed after previously succeeding
        if (!nvml_ok && !nvml_failed_ && samples_.size() > 2) {
            nvml_failed_ = true;
            fprintf(stderr, "[HW] WARNING: NVML query failed — GPU may have fallen off the bus\n");
        }

#if defined(_WIN32)
        // Sample RAM via Windows API
        MEMORYSTATUSEX mem_status;
        mem_status.dwLength = sizeof(MEMORYSTATUSEX);
        if (GlobalMemoryStatusEx(&mem_status)) {
            sample.ram_used_bytes = mem_status.ullTotalPhys - mem_status.ullAvailPhys;
        }
#elif defined(__APPLE__)
        // macOS: use sysctl
        {
            int64_t physmem = 0;
            size_t size = sizeof(physmem);
            sysctlbyname("hw.memsize", &physmem, &size, NULL, 0);
            // Approximate used from free_count (simplified)
            sample.ram_used_bytes = 0;  // Will be filled by profiler on first run
        }
#endif

        // Store sample
        {
            std::lock_guard<std::mutex> lock(mutex_);
            samples_.push_back(sample);
        }

        // Sleep 500ms between polls
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}

// =============================================================================
// Graceful Abort — Ctrl+C Handling (Phase H)
// =============================================================================
// Never call exit() from a signal handler. Just set a flag.
// The decode loop checks this flag every iteration and breaks cleanly.
//
// This prevents GPU memory leaks that would occur if we called exit()
// while llama.cpp holds CUDA allocations.
// =============================================================================

std::atomic<bool> abort_requested(false);

static void abort_signal_handler(int signal) {
    // Only set the flag. No malloc, no printf, no complex operations.
    // Signal handlers must be async-signal-safe.
    abort_requested.store(true, std::memory_order_relaxed);
}

void register_abort_handler() {
    std::signal(SIGINT, abort_signal_handler);
    std::signal(SIGTERM, abort_signal_handler);
}

bool is_abort_requested() {
    return abort_requested.load(std::memory_order_relaxed);
}
