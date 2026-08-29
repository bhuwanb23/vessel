#include "platform/hardware_profiler_interface.h"
#include "nvml_loader.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <thread>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <unordered_map>

// =============================================================================
// NVIDIA Profiler (Windows + CUDA)
// =============================================================================
// Implements IHardwareProfiler for NVIDIA GPUs using NVML.
// =============================================================================

class NvidiaProfiler : public IHardwareProfiler {
public:
    NvidiaProfiler() : initialized_(false) {
        // Dynamically load NVML
        if (!nvml_loader_init()) {
            fprintf(stderr, "[NvidiaProfiler] NVML not available\n");
            return;
        }
        // Try to initialize NVML
        nvmlReturn_t result = nvml_fn_Init();
        if (result == NVML_SUCCESS) {
            initialized_ = true;
            // Get device handle
            result = nvml_fn_DeviceGetHandleByIndex(0, &device_);
            if (result != NVML_SUCCESS) {
                fprintf(stderr, "[NvidiaProfiler] Failed to get device handle: %s\n",
                        nvml_fn_ErrorString ? nvml_fn_ErrorString(result) : "unknown");
                initialized_ = false;
            }
        } else {
            fprintf(stderr, "[NvidiaProfiler] NVML initialization failed: %s\n",
                    nvml_fn_ErrorString ? nvml_fn_ErrorString(result) : "unknown");
        }
    }
    
    ~NvidiaProfiler() override {
        if (initialized_ && nvml_fn_Shutdown) {
            nvml_fn_Shutdown();
        }
    }
    
    // =========================================================================
    // IHardwareProfiler Interface
    // =========================================================================
    
    bool isAvailable() const override {
        return initialized_;
    }
    
    Platform getPlatform() const override {
        return Platform::NVIDIA_WINDOWS;
    }
    
    std::string getName() const override {
        return "NVIDIA NVML";
    }
    
    HardwareSpec profile(const std::string& model_path_for_disk_bench = "") override {
        HardwareSpec spec;
        
        if (!initialized_) {
            fprintf(stderr, "[NvidiaProfiler] Not initialized, returning empty spec\n");
            return spec;
        }
        
        // Platform info
        spec.platform = Platform::NVIDIA_WINDOWS;
        spec.backend = ComputeBackend::CUDA;
        spec.memory_arch = MemoryArchitecture::DISCRETE;
        spec.is_unified_memory = false;
        
        // GPU info
        char name[NVML_DEVICE_NAME_V2_BUFFER_SIZE];
        if (nvml_fn_DeviceGetName && nvml_fn_DeviceGetName(device_, name, sizeof(name)) == NVML_SUCCESS) {
            spec.gpu_name = name;
        }
        
        // Compute capability
        int major, minor;
        if (nvml_fn_DeviceGetCudaComputeCapability && nvml_fn_DeviceGetCudaComputeCapability(device_, &major, &minor) == NVML_SUCCESS) {
            spec.gpu_compute_major = major;
            spec.gpu_compute_minor = minor;
            spec.compute_capability = "sm_" + std::to_string(major) + std::to_string(minor);
        }
        
        // VRAM
        nvmlMemory_t mem;
        if (nvml_fn_DeviceGetMemoryInfo && nvml_fn_DeviceGetMemoryInfo(device_, &mem) == NVML_SUCCESS) {
            spec.vram_total_bytes = mem.total;
            spec.vram_free_bytes = mem.free;
        }
        
        // GPU temperature
        unsigned int temp;
        if (nvml_fn_DeviceGetTemperature && nvml_fn_DeviceGetTemperature(device_, NVML_TEMPERATURE_GPU, &temp) == NVML_SUCCESS) {
            spec.gpu_temp_celsius = temp;
        }
        
        // GPU clock
        unsigned int clock;
        if (nvml_fn_DeviceGetClockInfo && nvml_fn_DeviceGetClockInfo(device_, NVML_CLOCK_SM, &clock) == NVML_SUCCESS) {
            spec.gpu_clock_mhz = clock;
        }
        
        // GPU utilization
        nvmlUtilization_t util;
        if (nvml_fn_DeviceGetUtilizationRates && nvml_fn_DeviceGetUtilizationRates(device_, &util) == NVML_SUCCESS) {
            spec.gpu_utilization = util.gpu;
        }
        
        // GPU count
        unsigned int count;
        if (nvml_fn_DeviceGetCount && nvml_fn_DeviceGetCount(&count) == NVML_SUCCESS) {
            spec.gpu_count = count;
        }
        
        // RAM (Windows API)
        MEMORYSTATUSEX mem_status;
        mem_status.dwLength = sizeof(mem_status);
        if (GlobalMemoryStatusEx(&mem_status)) {
            spec.ram_total_bytes = mem_status.ullTotalPhys;
            spec.ram_free_bytes = mem_status.ullAvailPhys;
        }
        
        // Disk I/O (if model path provided)
        if (!model_path_for_disk_bench.empty()) {
            spec.nvme_sequential_mbs = measure_disk_sequential(model_path_for_disk_bench);
            spec.nvme_random_4k_mbs = measure_disk_random_4k(model_path_for_disk_bench);
        }
        
        // GPU memory bandwidth (GB/s)
        // Always prefer the curated fallback table — it has accurate spec values
        // for every GPU including GDDR7 (Blackwell) which doesn't follow the
        // standard DDR formula. The formula-derived value is only used as a last
        // resort when the GPU isn't in any table.
        double fallback_bw = lookup_value(BW_FALLBACK, spec.gpu_name, 0.0);
        if (fallback_bw > 0) {
            spec.gpu_bandwidth_gbs = fallback_bw;
        } else {
            // Unknown GPU: derive from memory clock + bus width
            unsigned int mem_clock_mhz = 0;
            if (nvml_fn_DeviceGetMaxClockInfo && nvml_fn_DeviceGetMaxClockInfo(device_, NVML_CLOCK_MEM, &mem_clock_mhz) != NVML_SUCCESS)
                mem_clock_mhz = 0;
            int bus_width = lookup_int(BUS_WIDTH_TABLE, spec.gpu_name, 0);
            if (bus_width > 0 && mem_clock_mhz > 0) {
                spec.gpu_bandwidth_gbs = (double(mem_clock_mhz) * 2.0 * bus_width) / 8.0 / 1000.0;
            }
        }
        
        // TFLOPS (FP16) from lookup table
        spec.gpu_tflops_fp16 = lookup_value(TFLOPS_TABLE, spec.gpu_name, 20.0);
        
        // RAM bandwidth estimate (DDR4/DDR5)
        spec.ram_bandwidth_gbs = 40.0;  // Conservative DDR5 estimate
        
        // Generate fingerprint
        spec.hardware_fingerprint = generate_fingerprint(spec);
        
        return spec;
    }
    
    uint64_t getFreeVRAM() const override {
        if (!initialized_) return 0;
        nvmlMemory_t mem;
        if (nvml_fn_DeviceGetMemoryInfo && nvml_fn_DeviceGetMemoryInfo(device_, &mem) == NVML_SUCCESS) {
            return mem.free;
        }
        return 0;
    }
    
    uint32_t getGPUTemp() const override {
        if (!initialized_) return 0;
        unsigned int temp;
        if (nvml_fn_DeviceGetTemperature && nvml_fn_DeviceGetTemperature(device_, NVML_TEMPERATURE_GPU, &temp) == NVML_SUCCESS) {
            return temp;
        }
        return 0;
    }
    
    uint32_t getGPUClock() const override {
        if (!initialized_) return 0;
        unsigned int clock;
        if (nvml_fn_DeviceGetClockInfo && nvml_fn_DeviceGetClockInfo(device_, NVML_CLOCK_SM, &clock) == NVML_SUCCESS) {
            return clock;
        }
        return 0;
    }
    
    uint32_t getGPUUtilization() const override {
        if (!initialized_) return 0;
        nvmlUtilization_t util;
        if (nvml_fn_DeviceGetUtilizationRates && nvml_fn_DeviceGetUtilizationRates(device_, &util) == NVML_SUCCESS) {
            return util.gpu;
        }
        return 0;
    }
    
    bool supportsUnifiedMemory() const override {
        return false;  // NVIDIA discrete GPUs don't have unified memory
    }
    
    bool supportsMultiGPU() const override {
        return true;  // NVIDIA supports multi-GPU via NVLink or PCIe
    }
    
    uint32_t getGPUCount() const override {
        if (!initialized_) return 1;
        unsigned int count;
        if (nvml_fn_DeviceGetCount && nvml_fn_DeviceGetCount(&count) == NVML_SUCCESS) {
            return count;
        }
        return 1;
    }
    
    std::string getComputeCapability() const override {
        return compute_capability_;
    }
    
private:
    bool initialized_;
    nvmlDevice_t device_;
    std::string compute_capability_;
    
    // =========================================================================
    // GPU Bandwidth & TFLOPS Lookup Tables
    // =========================================================================
    
    // Bus width (bits) by GPU model substring
    static inline const std::unordered_map<std::string, int> BUS_WIDTH_TABLE = {
        {"RTX 3060", 192}, {"RTX 3070", 256}, {"RTX 3080", 320}, {"RTX 3090", 384},
        {"RTX 4060", 128}, {"RTX 4070", 192}, {"RTX 4080", 256}, {"RTX 4090", 384},
        {"RTX 5060", 128}, {"RTX 5070", 192}, {"RTX 5080", 256}, {"RTX 5090", 512},
        {"RTX 2060", 192}, {"RTX 2070", 256}, {"RTX 2080", 256}, {"RTX 2080 Ti", 352},
        {"GTX 1660", 192},
        {"A100", 5120}, {"A40", 384}, {"A30", 384},
        {"H100", 5120}, {"H200", 5120}, {"H800", 5120},
    };
    
    // Fallback memory bandwidth (GB/s) — spec-sheet values for when NVML clock fails
    static inline const std::unordered_map<std::string, double> BW_FALLBACK = {
        {"RTX 3060", 360.0}, {"RTX 3070", 448.0}, {"RTX 3080", 760.0}, {"RTX 3090", 936.0},
        {"RTX 4060", 272.0}, {"RTX 4070", 504.0}, {"RTX 4080", 717.0}, {"RTX 4090", 1008.0},
        {"RTX 5060", 336.0}, {"RTX 5070", 504.0}, {"RTX 5080", 896.0}, {"RTX 5090", 1792.0},
        {"RTX 2060", 336.0}, {"RTX 2070", 448.0}, {"RTX 2080", 448.0}, {"RTX 2080 Ti", 616.0},
        {"A100", 2039.0}, {"A40", 696.0}, {"H100", 3350.0}, {"H200", 4800.0},
    };
    
    // TFLOPS (FP16) by GPU model substring
    static inline const std::unordered_map<std::string, double> TFLOPS_TABLE = {
        {"RTX 3060", 13.0}, {"RTX 3070", 20.0}, {"RTX 3080", 30.0}, {"RTX 3090", 36.0},
        {"RTX 4060", 15.0}, {"RTX 4070", 20.0}, {"RTX 4080", 30.0}, {"RTX 4090", 40.0},
        {"RTX 5060", 20.0}, {"RTX 5070", 25.0}, {"RTX 5080", 35.0}, {"RTX 5090", 50.0},
        {"RTX 2060", 6.5},  {"RTX 2070", 7.5},  {"RTX 2080", 10.0},
        {"A100", 78.0}, {"A40", 37.0}, {"H100", 135.0}, {"H200", 140.0},
    };
    
    static double lookup_value(const std::unordered_map<std::string, double>& table,
                               const std::string& name, double fallback) {
        // Prefer longer matches; break ties by earlier position in name
        double result = fallback;
        size_t best_len = 0;
        size_t best_pos = std::string::npos;
        for (const auto& [pattern, val] : table) {
            size_t pos = name.find(pattern);
            if (pos != std::string::npos) {
                if (pattern.size() > best_len ||
                    (pattern.size() == best_len && pos < best_pos)) {
                    result = val;
                    best_len = pattern.size();
                    best_pos = pos;
                }
            }
        }
        return result;
    }
    
    static int lookup_int(const std::unordered_map<std::string, int>& table,
                          const std::string& name, int fallback) {
        int result = fallback;
        size_t best_len = 0;
        size_t best_pos = std::string::npos;
        for (const auto& [pattern, val] : table) {
            size_t pos = name.find(pattern);
            if (pos != std::string::npos) {
                if (pattern.size() > best_len ||
                    (pattern.size() == best_len && pos < best_pos)) {
                    result = val;
                    best_len = pattern.size();
                    best_pos = pos;
                }
            }
        }
        return result;
    }
    
    // =========================================================================
    // Disk I/O Measurement
    // =========================================================================
    
    double measure_disk_sequential(const std::string& file_path) {
        // Create a test file if it doesn't exist
        std::string test_file = file_path + ".bench";
        bool created = false;
        
        if (!std::filesystem::exists(test_file)) {
            // Create 128MB test file
            std::ofstream ofs(test_file, std::ios::binary);
            if (!ofs.is_open()) return 0.0;
            
            std::vector<char> buffer(1024 * 1024, 'A');  // 1MB buffer
            for (int i = 0; i < 128; i++) {
                ofs.write(buffer.data(), buffer.size());
            }
            ofs.close();
            created = true;
        }
        
        // Measure sequential read
        auto start = std::chrono::high_resolution_clock::now();
        
        std::ifstream ifs(test_file, std::ios::binary);
        if (!ifs.is_open()) {
            if (created) std::filesystem::remove(test_file);
            return 0.0;
        }
        
        std::vector<char> buffer(1024 * 1024);  // 1MB buffer
        while (ifs.read(buffer.data(), buffer.size())) {
            // Read entire file
        }
        ifs.close();
        
        auto end = std::chrono::high_resolution_clock::now();
        double seconds = std::chrono::duration<double>(end - start).count();
        
        // Cleanup
        if (created) std::filesystem::remove(test_file);
        
        if (seconds <= 0) return 0.0;
        
        // 128 MB / seconds = MB/s
        return 128.0 / seconds;
    }
    
    double measure_disk_random_4k(const std::string& file_path) {
        // Create a test file if it doesn't exist
        std::string test_file = file_path + ".bench4k";
        bool created = false;
        
        if (!std::filesystem::exists(test_file)) {
            // Create 64MB test file
            std::ofstream ofs(test_file, std::ios::binary);
            if (!ofs.is_open()) return 0.0;
            
            std::vector<char> buffer(1024 * 1024, 'B');
            for (int i = 0; i < 64; i++) {
                ofs.write(buffer.data(), buffer.size());
            }
            ofs.close();
            created = true;
        }
        
        // Measure random 4K reads
        auto start = std::chrono::high_resolution_clock::now();
        
        HANDLE hFile = CreateFileA(test_file.c_str(), GENERIC_READ, FILE_SHARE_READ,
                                   NULL, OPEN_EXISTING, 0, NULL);
        if (hFile == INVALID_HANDLE_VALUE) {
            if (created) std::filesystem::remove(test_file);
            return 0.0;
        }
        
        OVERLAPPED overlapped = {};
        char buffer[4096];
        int num_reads = 1000;
        uint64_t file_size = std::filesystem::file_size(test_file);
        
        for (int i = 0; i < num_reads; i++) {
            uint64_t offset = (static_cast<uint64_t>(rand()) * 4096) % (file_size - 4096);
            overlapped.Offset = static_cast<DWORD>(offset & 0xFFFFFFFF);
            overlapped.OffsetHigh = static_cast<DWORD>((offset >> 32) & 0xFFFFFFFF);
            
            DWORD bytes_read;
            ReadFile(hFile, buffer, 4096, &bytes_read, &overlapped);
        }
        
        CloseHandle(hFile);
        
        auto end = std::chrono::high_resolution_clock::now();
        double seconds = std::chrono::duration<double>(end - start).count();
        
        // Cleanup
        if (created) std::filesystem::remove(test_file);
        
        if (seconds <= 0) return 0.0;
        
        // (num_reads * 4KB) / seconds = MB/s
        return (num_reads * 4.0 / 1024.0) / seconds;
    }
    
    // =========================================================================
    // Fingerprint Generation
    // =========================================================================
    
    std::string generate_fingerprint(const HardwareSpec& spec) {
        // Simple fingerprint: GPU name + RAM (rounded to GB)
        uint64_t ram_gb = (spec.ram_total_bytes + 512ULL * 1024 * 1024) / (1024ULL * 1024 * 1024);
        return spec.gpu_name + "|" + std::to_string(ram_gb) + "GB";
    }
};

// =============================================================================
// Factory Implementation (NVIDIA-specific)
// =============================================================================
// Note: create_platform_profiler(Platform) is defined in cpu_profiler.cpp
// with cases for CPU_ONLY and default. We add NVIDIA case here.
// =============================================================================

std::unique_ptr<IHardwareProfiler> create_nvidia_profiler() {
    return std::make_unique<NvidiaProfiler>();
}

