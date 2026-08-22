#include "profiler.h"
#include "../profiler/ram_profiler.h"
#include "../profiler/gpu_profiler.h"
#include "../profiler/disk_profiler.h"
#include <vector>

// =============================================================================
// Hardware Profiler (Step 1)
// =============================================================================

HardwareSpec profile_hardware(const std::string& model_path_for_disk_bench) {
    HardwareSpec hw;
    
    // Profile RAM
    RamProfile ram = profile_ram();
    hw.ram_total_bytes = static_cast<uint64_t>(ram.total_ram_gb * 1024.0 * 1024.0 * 1024.0);
    hw.ram_free_bytes = static_cast<uint64_t>(ram.available_ram_gb * 1024.0 * 1024.0 * 1024.0);
    hw.ram_bandwidth_gbs = 40.0;  // Default estimate for DDR5
    
    // Profile GPU
    std::vector<GpuProfile> gpus = profile_gpus();
    if (!gpus.empty()) {
        const auto& gpu = gpus[0];
        hw.gpu_name = gpu.name;
        hw.vram_total_bytes = static_cast<uint64_t>(gpu.vram_total_gb * 1024.0 * 1024.0 * 1024.0);
        hw.vram_free_bytes = static_cast<uint64_t>(gpu.vram_free_gb * 1024.0 * 1024.0 * 1024.0);
        hw.gpu_bandwidth_gbs = gpu.memory_bandwidth_gb_per_sec;
        hw.gpu_compute_major = gpu.compute_capability_major;
        hw.gpu_compute_minor = gpu.compute_capability_minor;
        
        // Estimate TFLOPS based on GPU model
        // RTX 5060 ~20 TFLOPS, RTX 3080 ~30 TFLOPS, etc.
        if (gpu.name.find("5060") != std::string::npos) hw.gpu_tflops_fp16 = 20.0;
        else if (gpu.name.find("5070") != std::string::npos) hw.gpu_tflops_fp16 = 25.0;
        else if (gpu.name.find("5080") != std::string::npos) hw.gpu_tflops_fp16 = 35.0;
        else if (gpu.name.find("5090") != std::string::npos) hw.gpu_tflops_fp16 = 50.0;
        else if (gpu.name.find("3080") != std::string::npos) hw.gpu_tflops_fp16 = 30.0;
        else if (gpu.name.find("3090") != std::string::npos) hw.gpu_tflops_fp16 = 36.0;
        else if (gpu.name.find("4060") != std::string::npos) hw.gpu_tflops_fp16 = 15.0;
        else if (gpu.name.find("4070") != std::string::npos) hw.gpu_tflops_fp16 = 20.0;
        else if (gpu.name.find("4080") != std::string::npos) hw.gpu_tflops_fp16 = 30.0;
        else if (gpu.name.find("4090") != std::string::npos) hw.gpu_tflops_fp16 = 40.0;
        else hw.gpu_tflops_fp16 = 20.0;  // Default estimate
    }
    
    // Profile disk if model path provided
    if (!model_path_for_disk_bench.empty()) {
        DiskProfile disk = profile_disk(model_path_for_disk_bench);
        hw.nvme_sequential_mbs = disk.sequential_read_mb_per_sec;
        hw.nvme_random_4k_mbs = disk.random_read_mb_per_sec;
    }
    
    return hw;
}

uint64_t profile_ram_total() {
    RamProfile ram = profile_ram();
    return static_cast<uint64_t>(ram.total_ram_gb * 1024.0 * 1024.0 * 1024.0);
}

uint64_t profile_ram_free() {
    RamProfile ram = profile_ram();
    return static_cast<uint64_t>(ram.available_ram_gb * 1024.0 * 1024.0 * 1024.0);
}

std::string profile_gpu_name() {
    std::vector<GpuProfile> gpus = profile_gpus();
    return gpus.empty() ? "Unknown" : gpus[0].name;
}

uint64_t profile_vram_total() {
    std::vector<GpuProfile> gpus = profile_gpus();
    return gpus.empty() ? 0 : static_cast<uint64_t>(gpus[0].vram_total_gb * 1024.0 * 1024.0 * 1024.0);
}

uint64_t profile_vram_free() {
    std::vector<GpuProfile> gpus = profile_gpus();
    return gpus.empty() ? 0 : static_cast<uint64_t>(gpus[0].vram_free_gb * 1024.0 * 1024.0 * 1024.0);
}

double profile_gpu_bandwidth() {
    std::vector<GpuProfile> gpus = profile_gpus();
    return gpus.empty() ? 0.0 : gpus[0].memory_bandwidth_gb_per_sec;
}

double profile_disk_sequential(const std::string& file_path) {
    DiskProfile disk = profile_disk(file_path);
    return disk.sequential_read_mb_per_sec;
}

double profile_disk_random_4k(const std::string& file_path) {
    DiskProfile disk = profile_disk(file_path);
    return disk.random_read_mb_per_sec;
}
