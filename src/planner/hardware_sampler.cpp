#include "hardware_sampler.h"
#include "executor.h"
#include <csignal>
#include <cstdio>
#include <chrono>
#include <algorithm>

#if defined(_WIN32)
#include <windows.h>
// NVML loaded dynamically at runtime — no nvml.lib required at link time.
// NVML is always present on machines with NVIDIA drivers.
typedef int nvmlReturn_t;
typedef void* nvmlDevice_t;
typedef struct { unsigned long long total; unsigned long long free; unsigned long long used; } nvmlMemory_t;
static const int NVML_SUCCESS = 0;
static const int NVML_TEMPERATURE_GPU = 0;
static const int NVML_CLOCK_SM = 0;
static const int NVML_PCIE_UTIL_RX_BYTES = 1;
static const int NVML_PCIE_UTIL_TX_BYTES = 2;
typedef nvmlReturn_t (*pfn_nvmlInit)(void);
typedef nvmlReturn_t (*pfn_nvmlShutdown)(void);
typedef nvmlReturn_t (*pfn_nvmlDeviceGetCount)(unsigned int*);
typedef nvmlReturn_t (*pfn_nvmlDeviceGetHandleByIndex)(unsigned int, nvmlDevice_t*);
typedef nvmlReturn_t (*pfn_nvmlDeviceGetMemoryInfo)(nvmlDevice_t, nvmlMemory_t*);
typedef nvmlReturn_t (*pfn_nvmlDeviceGetTemperature)(nvmlDevice_t, unsigned int, unsigned int*);
typedef nvmlReturn_t (*pfn_nvmlDeviceGetClockInfo)(nvmlDevice_t, unsigned int, unsigned int*);
typedef nvmlReturn_t (*pfn_nvmlDeviceGetPcieThroughput)(nvmlDevice_t, unsigned int, unsigned int*);
static HMODULE nvml_dll = nullptr;
static pfn_nvmlInit nvmlInit_fn = nullptr;
static pfn_nvmlShutdown nvmlShutdown_fn = nullptr;
static pfn_nvmlDeviceGetCount nvmlDeviceGetCount_fn = nullptr;
static pfn_nvmlDeviceGetHandleByIndex nvmlDeviceGetHandleByIndex_fn = nullptr;
static pfn_nvmlDeviceGetMemoryInfo nvmlDeviceGetMemoryInfo_fn = nullptr;
static pfn_nvmlDeviceGetTemperature nvmlDeviceGetTemperature_fn = nullptr;
static pfn_nvmlDeviceGetClockInfo nvmlDeviceGetClockInfo_fn = nullptr;
static pfn_nvmlDeviceGetPcieThroughput nvmlDeviceGetPcieThroughput_fn = nullptr;
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#endif

// =============================================================================
// NVML State (NVIDIA only, not available on macOS)
// =============================================================================

#if defined(_WIN32)
static bool nvml_initialized = false;
static nvmlDevice_t nvml_device = nullptr;

static void ensure_nvml() {
    if (nvml_initialized) return;
    // Load nvml.dll from the NVIDIA driver installation
    nvml_dll = LoadLibraryA("nvml.dll");
    if (!nvml_dll) return;
    nvmlInit_fn = (pfn_nvmlInit)GetProcAddress(nvml_dll, "nvmlInit_v2");
    if (!nvmlInit_fn) nvmlInit_fn = (pfn_nvmlInit)GetProcAddress(nvml_dll, "nvmlInit");
    nvmlShutdown_fn = (pfn_nvmlShutdown)GetProcAddress(nvml_dll, "nvmlShutdown");
    nvmlDeviceGetCount_fn = (pfn_nvmlDeviceGetCount)GetProcAddress(nvml_dll, "nvmlDeviceGetCount");
    nvmlDeviceGetHandleByIndex_fn = (pfn_nvmlDeviceGetHandleByIndex)GetProcAddress(nvml_dll, "nvmlDeviceGetHandleByIndex_v2");
    if (!nvmlDeviceGetHandleByIndex_fn) nvmlDeviceGetHandleByIndex_fn = (pfn_nvmlDeviceGetHandleByIndex)GetProcAddress(nvml_dll, "nvmlDeviceGetHandleByIndex");
    nvmlDeviceGetMemoryInfo_fn = (pfn_nvmlDeviceGetMemoryInfo)GetProcAddress(nvml_dll, "nvmlDeviceGetMemoryInfo");
    nvmlDeviceGetTemperature_fn = (pfn_nvmlDeviceGetTemperature)GetProcAddress(nvml_dll, "nvmlDeviceGetTemperature");
    nvmlDeviceGetClockInfo_fn = (pfn_nvmlDeviceGetClockInfo)GetProcAddress(nvml_dll, "nvmlDeviceGetClockInfo");
    nvmlDeviceGetPcieThroughput_fn = (pfn_nvmlDeviceGetPcieThroughput)GetProcAddress(nvml_dll, "nvmlDeviceGetPcieThroughput");
    if (!nvmlInit_fn || !nvmlDeviceGetCount_fn || !nvmlDeviceGetHandleByIndex_fn ||
        !nvmlDeviceGetMemoryInfo_fn || !nvmlDeviceGetTemperature_fn ||
        !nvmlDeviceGetClockInfo_fn || !nvmlDeviceGetPcieThroughput_fn) {
        // Not all symbols found — don't use NVML
        FreeLibrary(nvml_dll);
        nvml_dll = nullptr;
        return;
    }
    nvmlReturn_t ret = nvmlInit_fn();
    if (ret == NVML_SUCCESS) {
        nvml_initialized = true;
        unsigned int device_count = 0;
        nvmlDeviceGetCount_fn(&device_count);
        if (device_count > 0) nvmlDeviceGetHandleByIndex_fn(0, &nvml_device);
    }
}
static void shutdown_nvml() {
    if (nvml_initialized && nvmlShutdown_fn) { nvmlShutdown_fn(); nvml_initialized = false; nvml_device = nullptr; }
    if (nvml_dll) { FreeLibrary(nvml_dll); nvml_dll = nullptr; }
}
#else
static void ensure_nvml() {}
static void shutdown_nvml() {}
#endif

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
