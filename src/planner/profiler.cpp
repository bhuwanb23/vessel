#include "profiler.h"
#include "hardware_fingerprint.h"
#include "../profiler/ram_profiler.h"
#include "../profiler/gpu_profiler.h"
#include "../profiler/disk_profiler.h"
#include <vector>
#include <cstdio>

// =============================================================================
// Hardware Profiler (Step 1)
// =============================================================================

static ProfileErrors profile_errors;

const ProfileErrors& get_profile_errors() {
    return profile_errors;
}

static void clear_profile_errors() {
    profile_errors = ProfileErrors();
}

// GPU TFLOPS lookup by model name
double estimate_gpu_tflops(const std::string& gpu_name) {
    // RTX 50 series
    if (gpu_name.find("5060") != std::string::npos) return 20.0;
    if (gpu_name.find("5070") != std::string::npos) return 25.0;
    if (gpu_name.find("5080") != std::string::npos) return 35.0;
    if (gpu_name.find("5090") != std::string::npos) return 50.0;
    // RTX 40 series
    if (gpu_name.find("4060") != std::string::npos) return 15.0;
    if (gpu_name.find("4070") != std::string::npos) return 20.0;
    if (gpu_name.find("4080") != std::string::npos) return 30.0;
    if (gpu_name.find("4090") != std::string::npos) return 40.0;
    // RTX 30 series
    if (gpu_name.find("3060") != std::string::npos) return 13.0;
    if (gpu_name.find("3070") != std::string::npos) return 20.0;
    if (gpu_name.find("3080") != std::string::npos) return 30.0;
    if (gpu_name.find("3090") != std::string::npos) return 36.0;
    // RTX 20 series
    if (gpu_name.find("2060") != std::string::npos) return 6.5;
    if (gpu_name.find("2070") != std::string::npos) return 7.5;
    if (gpu_name.find("2080") != std::string::npos) return 10.0;
    // A-series
    if (gpu_name.find("A100") != std::string::npos) return 78.0;
    if (gpu_name.find("A40") != std::string::npos) return 37.0;
    // H-series
    if (gpu_name.find("H100") != std::string::npos) return 135.0;
    if (gpu_name.find("H200") != std::string::npos) return 140.0;
    // Unknown GPU - conservative default
    return 20.0;
}

HardwareSpec profile_hardware(const std::string& model_path_for_disk_bench) {
    clear_profile_errors();
    HardwareSpec hw;
    
    // Profile RAM
    RamProfile ram = profile_ram();
    hw.ram_total_bytes = static_cast<uint64_t>(ram.total_ram_gb * 1024.0 * 1024.0 * 1024.0);
    hw.ram_free_bytes = static_cast<uint64_t>(ram.available_ram_gb * 1024.0 * 1024.0 * 1024.0);
    hw.ram_bandwidth_gbs = 40.0;  // Default estimate for DDR5
    
    if (ram.total_ram_gb <= 0) {
        profile_errors.ram_failed = true;
        fprintf(stderr, "Warning: Could not detect system RAM. Using conservative defaults.\n");
        // Set minimal defaults so predictions don't crash
        hw.ram_total_bytes = 8ULL * 1024 * 1024 * 1024;  // 8 GB fallback
        hw.ram_free_bytes = 4ULL * 1024 * 1024 * 1024;   // 4 GB fallback
    }
    
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
        hw.gpu_tflops_fp16 = estimate_gpu_tflops(gpu.name);
        
        // Check for low VRAM warning
        profile_errors.vram_free_gb = gpu.vram_free_gb;
        if (gpu.vram_free_gb < 1.0) {
            profile_errors.low_vram = true;
        }
    } else {
        // No GPU detected
        profile_errors.gpu_failed = true;
        profile_errors.gpu_error_msg = "No NVIDIA GPU detected via NVML.";
        fprintf(stderr, "Warning: No NVIDIA GPU detected. Check NVIDIA driver installation.\n");
        fprintf(stderr, "  GPU-accelerated strategies will not be available.\n");
        // Set conservative defaults for CPU-only predictions
        hw.gpu_bandwidth_gbs = 0.0;
        hw.gpu_tflops_fp16 = 0.0;
    }
    
    // Profile disk if model path provided
    if (!model_path_for_disk_bench.empty()) {
        DiskProfile disk = profile_disk(model_path_for_disk_bench);
        hw.nvme_sequential_mbs = disk.sequential_read_mb_per_sec;
        hw.nvme_random_4k_mbs = disk.random_read_mb_per_sec;
        
        // Check for abnormally slow disk
        profile_errors.disk_seq_mbs = disk.sequential_read_mb_per_sec;
        if (disk.sequential_read_mb_per_sec > 0 && disk.sequential_read_mb_per_sec < 100.0) {
            profile_errors.disk_slow = true;
        }
    }
    
    // Generate hardware fingerprint for calibration log
    hw.hardware_fingerprint = generateHardwareFingerprint(hw, model_path_for_disk_bench);
    
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
