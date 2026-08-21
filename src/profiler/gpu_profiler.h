#pragma once

#include <cstdint>
#include <string>
#include <vector>

// Result of GPU profiling for a single device
struct GpuProfile {
    int device_index;
    std::string name;
    uint64_t vram_total_bytes;
    uint64_t vram_free_bytes;
    uint64_t vram_used_bytes;
    double vram_total_gb;
    double vram_free_gb;
    int memory_clock_mhz;
    int bus_width_bits;
    double memory_bandwidth_gb_per_sec; // derived: (clock * 2 * bus_width) / 8 / 1000
    int temperature_c;
    int compute_capability_major;
    int compute_capability_minor;
    bool bandwidth_known; // false if GPU not in lookup table
};

// Profile all NVIDIA GPUs using NVML
// Returns a vector of GpuProfile structs (one per GPU).
std::vector<GpuProfile> profile_gpus();

// Print GPU profile(s) to stdout in a readable format
void print_gpu_profiles(const std::vector<GpuProfile>& profiles);
