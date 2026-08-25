// =============================================================================
// CPU-Only Profiler (Universal Fallback)
// =============================================================================
// Implements IHardwareProfiler for systems without a dedicated GPU.
// Always available — no external dependencies.
//
// This profiler:
//   - Reports zero VRAM (no GPU)
//   - Reports system RAM (from OS)
//   - Estimates RAM bandwidth from CPU model
//   - Runs disk I/O benchmarks
//   - Generates a hardware fingerprint from CPU + RAM
// =============================================================================

#include "platform/hardware_profiler_interface.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <thread>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <psapi.h>

// =============================================================================
// CPU-Only Profiler Class
// =============================================================================

class CpuOnlyProfiler : public IHardwareProfiler {
public:
    CpuOnlyProfiler() : initialized_(true) {
        fprintf(stderr, "[CpuOnlyProfiler] Initialized (CPU-only mode)\n");
    }
    
    ~CpuOnlyProfiler() override = default;
    
    // =========================================================================
    // IHardwareProfiler Interface
    // =========================================================================
    
    bool isAvailable() const override {
        return true;  // Always available
    }
    
    Platform getPlatform() const override {
        return Platform::CPU_ONLY;
    }
    
    std::string getName() const override {
        return "CPU Only";
    }
    
    HardwareSpec profile(const std::string& model_path_for_disk_bench = "") override {
        HardwareSpec spec;
        
        // Platform info
        spec.platform = Platform::CPU_ONLY;
        spec.backend = ComputeBackend::CPU;
        spec.memory_arch = MemoryArchitecture::DISCRETE;  // No GPU
        spec.is_unified_memory = false;
        spec.gpu_name = "None (CPU-only)";
        spec.compute_capability = "cpu";
        
        // No GPU
        spec.vram_total_bytes = 0;
        spec.vram_free_bytes = 0;
        spec.gpu_bandwidth_gbs = 0;
        spec.gpu_tflops_fp16 = 0;
        spec.gpu_temp_celsius = 0;
        spec.gpu_clock_mhz = 0;
        spec.gpu_utilization = 0;
        spec.gpu_count = 0;
        spec.gpu_compute_major = 0;
        spec.gpu_compute_minor = 0;
        
        // RAM (Windows API)
        MEMORYSTATUSEX mem_status;
        mem_status.dwLength = sizeof(mem_status);
        if (GlobalMemoryStatusEx(&mem_status)) {
            spec.ram_total_bytes = mem_status.ullTotalPhys;
            spec.ram_free_bytes = mem_status.ullAvailPhys;
        }
        
        // Estimate RAM bandwidth from CPU model
        spec.ram_bandwidth_gbs = estimate_ram_bandwidth();
        
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
        return 0;  // No GPU
    }
    
    uint32_t getGPUTemp() const override {
        return 0;  // No GPU
    }
    
    uint32_t getGPUClock() const override {
        return 0;  // No GPU
    }
    
    uint32_t getGPUUtilization() const override {
        return 0;  // No GPU
    }
    
    bool supportsUnifiedMemory() const override {
        return false;  // No GPU
    }
    
    bool supportsMultiGPU() const override {
        return false;  // No GPU
    }
    
    uint32_t getGPUCount() const override {
        return 0;  // No GPU
    }
    
    std::string getComputeCapability() const override {
        return "cpu";
    }
    
private:
    bool initialized_;
    
    // =========================================================================
    // RAM Bandwidth Estimation
    // =========================================================================
    
    double estimate_ram_bandwidth() {
        // Try to detect RAM type from Windows
        // DDR4-3200: ~25 GB/s theoretical, ~20 GB/s achievable
        // DDR5-5600: ~45 GB/s theoretical, ~35 GB/s achievable
        // DDR5-6400: ~51 GB/s theoretical, ~40 GB/s achievable
        
        // For now, return a conservative estimate
        // The actual bandwidth should be measured via memcpy benchmark
        // but that requires a model path for the test file
        
        // Check if we have DDR5 (Windows 10 1903+ reports this)
        // For now, assume DDR4-3200 (most common)
        return 35.0;  // Conservative estimate for DDR4
    }
    
    // =========================================================================
    // Disk I/O Measurement
    // =========================================================================
    
    double measure_disk_sequential(const std::string& file_path) {
        std::string test_file = file_path + ".bench";
        bool created = false;
        
        if (!std::filesystem::exists(test_file)) {
            std::ofstream ofs(test_file, std::ios::binary);
            if (!ofs.is_open()) return 0.0;
            
            std::vector<char> buffer(1024 * 1024, 'A');
            for (int i = 0; i < 128; i++) {
                ofs.write(buffer.data(), buffer.size());
            }
            ofs.close();
            created = true;
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        
        std::ifstream ifs(test_file, std::ios::binary);
        if (!ifs.is_open()) {
            if (created) std::filesystem::remove(test_file);
            return 0.0;
        }
        
        std::vector<char> buffer(1024 * 1024);
        while (ifs.read(buffer.data(), buffer.size())) {}
        ifs.close();
        
        auto end = std::chrono::high_resolution_clock::now();
        double seconds = std::chrono::duration<double>(end - start).count();
        
        if (created) std::filesystem::remove(test_file);
        
        return (seconds > 0) ? 128.0 / seconds : 0.0;
    }
    
    double measure_disk_random_4k(const std::string& file_path) {
        std::string test_file = file_path + ".bench4k";
        bool created = false;
        
        if (!std::filesystem::exists(test_file)) {
            std::ofstream ofs(test_file, std::ios::binary);
            if (!ofs.is_open()) return 0.0;
            
            std::vector<char> buffer(1024 * 1024, 'B');
            for (int i = 0; i < 64; i++) {
                ofs.write(buffer.data(), buffer.size());
            }
            ofs.close();
            created = true;
        }
        
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
        
        if (created) std::filesystem::remove(test_file);
        
        return (seconds > 0) ? (num_reads * 4.0 / 1024.0) / seconds : 0.0;
    }
    
    // =========================================================================
    // Fingerprint Generation
    // =========================================================================
    
    std::string generate_fingerprint(const HardwareSpec& spec) {
        // CPU-only fingerprint: CPU model + RAM (rounded to GB)
        // Get CPU name from Windows registry
        char cpu_name[256] = "Unknown CPU";
        HKEY hKey;
        if (RegOpenKeyExA(HKEY_LOCAL_MACHINE,
                          "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
                          0, KEY_READ, &hKey) == ERROR_SUCCESS) {
            DWORD size = sizeof(cpu_name);
            RegQueryValueExA(hKey, "ProcessorNameString", NULL, NULL,
                            (LPBYTE)cpu_name, &size);
            RegCloseKey(hKey);
        }
        
        // Normalize CPU name: strip clock speed and "CPU" suffix
        std::string normalized = cpu_name;
        // Remove "(R)" and "(TM)"
        auto remove_parens = [&](const std::string& pattern) {
            size_t pos;
            while ((pos = normalized.find(pattern)) != std::string::npos) {
                normalized.erase(pos, pattern.length());
            }
        };
        remove_parens("(R)");
        remove_parens("(TM)");
        remove_parens("@");
        
        // Strip everything after " @ " (clock speed)
        size_t at_pos = normalized.find("@");
        if (at_pos != std::string::npos) {
            normalized = normalized.substr(0, at_pos);
        }
        
        // Trim whitespace
        while (!normalized.empty() && normalized.back() == ' ') normalized.pop_back();
        while (!normalized.empty() && normalized.front() == ' ') normalized.erase(0, 1);
        
        uint64_t ram_gb = (spec.ram_total_bytes + 512ULL * 1024 * 1024) / (1024ULL * 1024 * 1024);
        return normalized + "|" + std::to_string(ram_gb) + "GB";
    }
};

// =============================================================================
// Factory Implementation
// =============================================================================

std::unique_ptr<IHardwareProfiler> create_cpu_profiler() {
    return std::make_unique<CpuOnlyProfiler>();
}
