#include "platform/hardware_profiler_interface.h"

// =============================================================================
// Apple Silicon Metal Profiler (macOS)
// =============================================================================
// Implements IHardwareProfiler for Apple Silicon using Metal APIs.
// This profiler is only compiled on macOS.
//
// Requirements:
//   - macOS 14+ (Sonoma or later)
//   - Apple Silicon (M1/M2/M3/M4)
//   - Metal framework
//
// Build: -DGGML_METAL=ON (enables Metal backend)
// =============================================================================

#if defined(__APPLE__) && defined(__MACH__) && defined(GGML_METAL)

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <sys/sysctl.h>
#include <mach/mach.h>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <algorithm>

// =============================================================================
// Apple Silicon Bandwidth Lookup Table
// =============================================================================

struct AppleChipInfo {
    const char* name;
    double bandwidth_gbs;
    uint32_t gpu_cores;
    uint64_t max_memory_bytes;
};

static const AppleChipInfo apple_chips[] = {
    // M1 series
    {"Apple M1",         68.25,  8,  16ULL * 1024 * 1024 * 1024},
    {"Apple M1 Pro",    200.0,  16, 32ULL * 1024 * 1024 * 1024},
    {"Apple M1 Max",    400.0,  32, 64ULL * 1024 * 1024 * 1024},
    {"Apple M1 Ultra",  800.0,  64, 128ULL * 1024 * 1024 * 1024},
    
    // M2 series
    {"Apple M2",        100.0,  10, 24ULL * 1024 * 1024 * 1024},
    {"Apple M2 Pro",    200.0,  19, 32ULL * 1024 * 1024 * 1024},
    {"Apple M2 Max",    400.0,  38, 96ULL * 1024 * 1024 * 1024},
    {"Apple M2 Ultra",  800.0,  76, 192ULL * 1024 * 1024 * 1024},
    
    // M3 series
    {"Apple M3",        100.0,  10, 24ULL * 1024 * 1024 * 1024},
    {"Apple M3 Pro",    150.0,  18, 36ULL * 1024 * 1024 * 1024},
    {"Apple M3 Max",    400.0,  40, 128ULL * 1024 * 1024 * 1024},
    
    // M4 series
    {"Apple M4",        120.0,  10, 32ULL * 1024 * 1024 * 1024},
    {"Apple M4 Pro",    273.0,  20, 48ULL * 1024 * 1024 * 1024},
    {"Apple M4 Max",    546.0,  40, 128ULL * 1024 * 1024 * 1024},
};

static const int NUM_APPLE_CHIPS = sizeof(apple_chips) / sizeof(apple_chips[0]);

// =============================================================================
// Apple Silicon Profiler Class
// =============================================================================

class AppleMetalProfiler : public IHardwareProfiler {
public:
    AppleMetalProfiler() : initialized_(false), device_(nil) {
        // Initialize Metal
        device_ = MTLCreateSystemDefaultDevice();
        if (device_) {
            initialized_ = true;
            fprintf(stderr, "[AppleMetalProfiler] Initialized: %s\n",
                    [[device_ name] UTF8String]);
        } else {
            fprintf(stderr, "[AppleMetalProfiler] No Metal-compatible GPU found\n");
        }
    }
    
    ~AppleMetalProfiler() override {
        if (device_) {
            [device_ release];
            device_ = nil;
        }
    }
    
    // =========================================================================
    // IHardwareProfiler Interface
    // =========================================================================
    
    bool isAvailable() const override {
        return initialized_;
    }
    
    Platform getPlatform() const override {
        return Platform::APPLE_MACOS;
    }
    
    std::string getName() const override {
        return "Apple Metal";
    }
    
    HardwareSpec profile(const std::string& model_path_for_disk_bench = "") override {
        HardwareSpec spec;
        
        if (!initialized_) {
            fprintf(stderr, "[AppleMetalProfiler] Not initialized, returning empty spec\n");
            return spec;
        }
        
        // Platform info
        spec.platform = Platform::APPLE_MACOS;
        spec.backend = ComputeBackend::METAL;
        spec.memory_arch = MemoryArchitecture::UNIFIED;
        spec.is_unified_memory = true;
        
        // GPU name
        spec.gpu_name = [[device_ name] UTF8String];
        
        // Compute capability (Apple doesn't use gfx/sm codes)
        spec.compute_capability = "apple_" + get_chip_family();
        spec.gpu_compute_major = 0;
        spec.gpu_compute_minor = 0;
        
        // Unified memory (same as RAM on Apple Silicon)
        uint64_t total_memory = get_total_memory();
        uint64_t available_memory = get_available_memory();
        
        spec.vram_total_bytes = total_memory;  // Unified: VRAM = RAM
        spec.vram_free_bytes = available_memory;
        spec.ram_total_bytes = total_memory;
        spec.ram_free_bytes = available_memory;
        
        // Memory bandwidth (from lookup table)
        spec.gpu_bandwidth_gbs = lookup_bandwidth(spec.gpu_name);
        spec.ram_bandwidth_gbs = spec.gpu_bandwidth_gbs;  // Same on unified memory
        
        // GPU info
        spec.gpu_temp_celsius = 0;  // Apple doesn't expose GPU temperature
        spec.gpu_clock_mhz = 0;     // Apple doesn't expose GPU clock
        spec.gpu_utilization = 0;   // Would need Metal performance counters
        spec.gpu_count = 1;         // Apple Silicon has one GPU
        
        // Recommended max working set size
        uint64_t recommended_max = [device_ recommendedMaxWorkingSetSize];
        fprintf(stderr, "[AppleMetalProfiler] Recommended max working set: %llu MB\n",
                recommended_max / (1024 * 1024));
        
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
        return get_available_memory();  // Unified memory
    }
    
    uint32_t getGPUTemp() const override {
        // Apple Silicon doesn't expose GPU temperature via public APIs
        return 0;
    }
    
    uint32_t getGPUClock() const override {
        // Apple Silicon doesn't expose GPU clock via public APIs
        return 0;
    }
    
    uint32_t getGPUUtilization() const override {
        // Would need Metal performance counters
        return 0;
    }
    
    bool supportsUnifiedMemory() const override {
        return true;  // Apple Silicon always uses unified memory
    }
    
    bool supportsMultiGPU() const override {
        return false;  // Apple Silicon has one GPU
    }
    
    uint32_t getGPUCount() const override {
        return 1;
    }
    
    std::string getComputeCapability() const override {
        return compute_capability_;
    }
    
private:
    bool initialized_;
    id<MTLDevice> device_;
    std::string compute_capability_;
    
    // =========================================================================
    // Apple Silicon Memory Queries
    // =========================================================================
    
    uint64_t get_total_memory() const {
        int64_t total_memory = 0;
        size_t size = sizeof(total_memory);
        sysctlbyname("hw.memsize", &total_memory, &size, NULL, 0);
        return static_cast<uint64_t>(total_memory);
    }
    
    uint64_t get_available_memory() const {
        mach_port_t host = mach_host_self();
        vm_statistics64_data_t vm_stats;
        mach_msg_type_number_t info_count = HOST_VM_INFO64_COUNT;
        
        if (host_statistics64(host, HOST_VM_INFO64, 
                             (host_info64_t)&vm_stats, &info_count) == KERN_SUCCESS) {
            int64_t page_size = 0;
            size_t size = sizeof(page_size);
            sysctlbyname("hw.pagesize", &page_size, &size, NULL, 0);
            
            // Available = free + inactive pages
            uint64_t available = (vm_stats.free_count + vm_stats.inactive_count) * page_size;
            
            // Account for memory compression
            // If compressor is active, some "inactive" memory is actually compressed
            // Use a conservative estimate: 70% of inactive memory is truly available
            uint64_t compressed = vm_stats.compressor_page_count * page_size;
            uint64_t inactive_available = vm_stats.inactive_count * page_size * 0.7;
            
            return vm_stats.free_count * page_size + inactive_available;
        }
        
        return 0;
    }
    
    // =========================================================================
    // Chip Family Detection
    // =========================================================================
    
    std::string get_chip_family() const {
        NSString* name = [device_ name];
        
        if ([name containsString:@"M4"]) return "m4";
        if ([name containsString:@"M3"]) return "m3";
        if ([name containsString:@"M2"]) return "m2";
        if ([name containsString:@"M1"]) return "m1";
        
        return "unknown";
    }
    
    // =========================================================================
    // Bandwidth Lookup
    // =========================================================================
    
    double lookup_bandwidth(const std::string& gpu_name) const {
        for (int i = 0; i < NUM_APPLE_CHIPS; i++) {
            if (gpu_name.find(apple_chips[i].name) != std::string::npos) {
                return apple_chips[i].bandwidth_gbs;
            }
        }
        
        // Default: assume M2-level performance
        fprintf(stderr, "[AppleMetalProfiler] Unknown chip: %s, using default bandwidth\n",
                gpu_name.c_str());
        return 100.0;
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
        
        FILE* fp = fopen(test_file.c_str(), "rb");
        if (!fp) {
            if (created) std::filesystem::remove(test_file);
            return 0.0;
        }
        
        char buffer[4096];
        int num_reads = 1000;
        uint64_t file_size = std::filesystem::file_size(test_file);
        
        for (int i = 0; i < num_reads; i++) {
            uint64_t offset = (static_cast<uint64_t>(rand()) * 4096) % (file_size - 4096);
            fseek(fp, offset, SEEK_SET);
            fread(buffer, 1, 4096, fp);
        }
        
        fclose(fp);
        
        auto end = std::chrono::high_resolution_clock::now();
        double seconds = std::chrono::duration<double>(end - start).count();
        
        if (created) std::filesystem::remove(test_file);
        
        return (seconds > 0) ? (num_reads * 4.0 / 1024.0) / seconds : 0.0;
    }
    
    // =========================================================================
    // Fingerprint Generation
    // =========================================================================
    
    std::string generate_fingerprint(const HardwareSpec& spec) {
        uint64_t ram_gb = (spec.ram_total_bytes + 512ULL * 1024 * 1024) / (1024ULL * 1024 * 1024);
        return spec.gpu_name + "|" + std::to_string(ram_gb) + "GB";
    }
};

// =============================================================================
// Factory Implementation
// =============================================================================

std::unique_ptr<IHardwareProfiler> create_platform_profiler(Platform platform) {
    if (platform == Platform::APPLE_MACOS) {
        return std::make_unique<AppleMetalProfiler>();
    }
    return nullptr;
}

#endif  // __APPLE__ && __MACH__ && GGML_METAL
