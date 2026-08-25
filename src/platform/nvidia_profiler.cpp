#include "platform/hardware_profiler_interface.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <nvml.h>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <thread>
#include <filesystem>
#include <fstream>
#include <algorithm>

// =============================================================================
// NVIDIA Profiler (Windows + CUDA)
// =============================================================================
// Implements IHardwareProfiler for NVIDIA GPUs using NVML.
// =============================================================================

class NvidiaProfiler : public IHardwareProfiler {
public:
    NvidiaProfiler() : initialized_(false) {
        // Try to initialize NVML
        nvmlReturn_t result = nvmlInit();
        if (result == NVML_SUCCESS) {
            initialized_ = true;
            // Get device handle
            result = nvmlDeviceGetHandleByIndex(0, &device_);
            if (result != NVML_SUCCESS) {
                fprintf(stderr, "[NvidiaProfiler] Failed to get device handle: %s\n",
                        nvmlErrorString(result));
                initialized_ = false;
            }
        } else {
            fprintf(stderr, "[NvidiaProfiler] NVML initialization failed: %s\n",
                    nvmlErrorString(result));
        }
    }
    
    ~NvidiaProfiler() override {
        if (initialized_) {
            nvmlShutdown();
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
        if (nvmlDeviceGetName(device_, name, sizeof(name)) == NVML_SUCCESS) {
            spec.gpu_name = name;
        }
        
        // Compute capability
        int major, minor;
        if (nvmlDeviceGetCudaComputeCapability(device_, &major, &minor) == NVML_SUCCESS) {
            spec.gpu_compute_major = major;
            spec.gpu_compute_minor = minor;
            spec.compute_capability = "sm_" + std::to_string(major) + std::to_string(minor);
        }
        
        // VRAM
        nvmlMemory_t mem;
        if (nvmlDeviceGetMemoryInfo(device_, &mem) == NVML_SUCCESS) {
            spec.vram_total_bytes = mem.total;
            spec.vram_free_bytes = mem.free;
        }
        
        // GPU temperature
        unsigned int temp;
        if (nvmlDeviceGetTemperature(device_, NVML_TEMPERATURE_GPU, &temp) == NVML_SUCCESS) {
            spec.gpu_temp_celsius = temp;
        }
        
        // GPU clock
        unsigned int clock;
        if (nvmlDeviceGetClockInfo(device_, NVML_CLOCK_SM, &clock) == NVML_SUCCESS) {
            spec.gpu_clock_mhz = clock;
        }
        
        // GPU utilization
        nvmlUtilization_t util;
        if (nvmlDeviceGetUtilizationRates(device_, &util) == NVML_SUCCESS) {
            spec.gpu_utilization = util.gpu;
        }
        
        // GPU count
        unsigned int count;
        if (nvmlDeviceGetCount(&count) == NVML_SUCCESS) {
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
        
        // Generate fingerprint
        spec.hardware_fingerprint = generate_fingerprint(spec);
        
        return spec;
    }
    
    uint64_t getFreeVRAM() const override {
        if (!initialized_) return 0;
        nvmlMemory_t mem;
        if (nvmlDeviceGetMemoryInfo(device_, &mem) == NVML_SUCCESS) {
            return mem.free;
        }
        return 0;
    }
    
    uint32_t getGPUTemp() const override {
        if (!initialized_) return 0;
        unsigned int temp;
        if (nvmlDeviceGetTemperature(device_, NVML_TEMPERATURE_GPU, &temp) == NVML_SUCCESS) {
            return temp;
        }
        return 0;
    }
    
    uint32_t getGPUClock() const override {
        if (!initialized_) return 0;
        unsigned int clock;
        if (nvmlDeviceGetClockInfo(device_, NVML_CLOCK_SM, &clock) == NVML_SUCCESS) {
            return clock;
        }
        return 0;
    }
    
    uint32_t getGPUUtilization() const override {
        if (!initialized_) return 0;
        nvmlUtilization_t util;
        if (nvmlDeviceGetUtilizationRates(device_, &util) == NVML_SUCCESS) {
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
        if (nvmlDeviceGetCount(&count) == NVML_SUCCESS) {
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
// Factory Implementation
// =============================================================================

std::unique_ptr<IHardwareProfiler> create_platform_profiler() {
    return std::make_unique<NvidiaProfiler>();
}

std::unique_ptr<IHardwareProfiler> create_platform_profiler(Platform platform) {
    if (platform == Platform::NVIDIA_WINDOWS || platform == Platform::NVIDIA_LINUX) {
        return std::make_unique<NvidiaProfiler>();
    }
    // Other platforms not yet implemented
    return nullptr;
}

// =============================================================================
// Legacy Interface (delegates to platform profiler)
// =============================================================================
// These functions are defined in the existing profiler files (ram_profiler.cpp,
// gpu_profiler.cpp, disk_profiler.cpp). We don't redefine them here.
// The nvidia_profiler.cpp only provides the IHardwareProfiler implementation.
// =============================================================================

