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

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <psapi.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#include <mach/mach.h>
#include <unistd.h>
#include <fcntl.h>
#else
#include <fstream>
#include <unistd.h>
#include <fcntl.h>
#endif

#include <cstdio>
#include <cstring>
#include <chrono>
#include <thread>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <vector>
#include <string>

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
        
        // RAM (platform-specific)
        spec.ram_total_bytes = get_ram_total();
        spec.ram_free_bytes = get_ram_free();
        
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
    // RAM Profiling (Cross-Platform)
    // =========================================================================
    
    uint64_t get_ram_total() {
#if defined(_WIN32)
        MEMORYSTATUSEX mem_status;
        mem_status.dwLength = sizeof(mem_status);
        if (GlobalMemoryStatusEx(&mem_status)) {
            return mem_status.ullTotalPhys;
        }
        return 0;
#elif defined(__APPLE__)
        uint64_t total_mem = 0;
        size_t size = sizeof(total_mem);
        if (sysctlbyname("hw.memsize", &total_mem, &size, NULL, 0) == 0) {
            return total_mem;
        }
        return 0;
#else
        // Linux: read /proc/meminfo
        std::ifstream meminfo("/proc/meminfo");
        std::string line;
        while (std::getline(meminfo, line)) {
            uint64_t total_kb = 0;
            if (sscanf(line.c_str(), "MemTotal: %lu kB", &total_kb) == 1) {
                return total_kb * 1024ULL;
            }
        }
        return 0;
#endif
    }
    
    uint64_t get_ram_free() {
#if defined(_WIN32)
        MEMORYSTATUSEX mem_status;
        mem_status.dwLength = sizeof(mem_status);
        if (GlobalMemoryStatusEx(&mem_status)) {
            return mem_status.ullAvailPhys;
        }
        return 0;
#elif defined(__APPLE__)
        mach_port_t host = mach_host_self();
        vm_statistics64_data_t vm_stats;
        mach_msg_type_number_t info_count = HOST_VM_INFO64_COUNT;
        if (host_statistics64(host, HOST_VM_INFO64, (host_info64_t)&vm_stats, &info_count) == KERN_SUCCESS) {
            uint64_t page_size = 0;
            size_t size = sizeof(page_size);
            sysctlbyname("hw.pagesize", &page_size, &size, NULL, 0);
            return (uint64_t)(vm_stats.free_count + vm_stats.inactive_count) * page_size;
        }
        return 0;
#else
        // Linux: read /proc/meminfo
        std::ifstream meminfo("/proc/meminfo");
        std::string line;
        while (std::getline(meminfo, line)) {
            uint64_t avail_kb = 0;
            if (sscanf(line.c_str(), "MemAvailable: %lu kB", &avail_kb) == 1) {
                return avail_kb * 1024ULL;
            }
        }
        return 0;
#endif
    }
    
    // =========================================================================
    // RAM Bandwidth Estimation
    // =========================================================================
    
    double estimate_ram_bandwidth() {
        // Conservative estimate for DDR4-3200
        // Real measurement would require a memcpy benchmark
        return 35.0;
    }
    
    // =========================================================================
    // Disk I/O Measurement (Cross-Platform)
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
        
#if defined(_WIN32)
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
#else
        // Unix: use pread for random reads
        int fd = open(test_file.c_str(), O_RDONLY);
        if (fd < 0) {
            if (created) std::filesystem::remove(test_file);
            return 0.0;
        }
        
        char buffer[4096];
        int num_reads = 1000;
        uint64_t file_size = std::filesystem::file_size(test_file);
        
        for (int i = 0; i < num_reads; i++) {
            uint64_t offset = (static_cast<uint64_t>(rand()) * 4096) % (file_size - 4096);
            pread(fd, buffer, 4096, offset);
        }
        
        close(fd);
#endif
        
        auto end = std::chrono::high_resolution_clock::now();
        double seconds = std::chrono::duration<double>(end - start).count();
        
        if (created) std::filesystem::remove(test_file);
        
        return (seconds > 0) ? (num_reads * 4.0 / 1024.0) / seconds : 0.0;
    }
    
    // =========================================================================
    // Fingerprint Generation (Cross-Platform)
    // =========================================================================
    
    std::string get_cpu_model() {
#if defined(_WIN32)
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
        return cpu_name;
#elif defined(__APPLE__)
        char buf[256] = "Unknown CPU";
        size_t size = sizeof(buf);
        sysctlbyname("machdep.cpu.brand_string", buf, &size, NULL, 0);
        return buf;
#else
        // Linux: read /proc/cpuinfo
        std::ifstream cpuinfo("/proc/cpuinfo");
        std::string line;
        while (std::getline(cpuinfo, line)) {
            if (line.find("model name") != std::string::npos) {
                size_t pos = line.find(": ");
                if (pos != std::string::npos) {
                    return line.substr(pos + 2);
                }
            }
        }
        return "Unknown CPU";
#endif
    }
    
    std::string generate_fingerprint(const HardwareSpec& spec) {
        std::string cpu = get_cpu_model();
        
        // Normalize CPU name: strip clock speed and common suffixes
        auto remove_pattern = [&](const std::string& pattern) {
            size_t pos;
            while ((pos = cpu.find(pattern)) != std::string::npos) {
                cpu.erase(pos, pattern.length());
            }
        };
        remove_pattern("(R)");
        remove_pattern("(TM)");
        remove_pattern("CPU");
        
        // Strip everything after " @ " (clock speed)
        size_t at_pos = cpu.find("@");
        if (at_pos != std::string::npos) {
            cpu = cpu.substr(0, at_pos);
        }
        
        // Trim whitespace
        while (!cpu.empty() && cpu.back() == ' ') cpu.pop_back();
        while (!cpu.empty() && cpu.front() == ' ') cpu.erase(0, 1);
        
        uint64_t ram_gb = (spec.ram_total_bytes + 512ULL * 1024 * 1024) / (1024ULL * 1024 * 1024);
        return cpu + "|" + std::to_string(ram_gb) + "GB";
    }
};

// =============================================================================
// Factory Implementation
// =============================================================================

std::unique_ptr<IHardwareProfiler> create_cpu_profiler() {
    return std::make_unique<CpuOnlyProfiler>();
}
