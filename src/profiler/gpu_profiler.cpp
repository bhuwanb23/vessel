#include "gpu_profiler.h"

#include <nvml.h>
#include <cstdio>
#include <cstring>
#include <unordered_map>

// Bus width lookup table for common consumer GPUs
// Key: GPU name substring (case-insensitive match)
// Value: memory bus width in bits
static const std::unordered_map<std::string, int> BUS_WIDTH_TABLE = {
    // RTX 30 series
    {"RTX 3060", 192},
    {"RTX 3070", 256},
    {"RTX 3080", 320},
    {"RTX 3090", 384},
    // RTX 40 series
    {"RTX 4060", 128},
    {"RTX 4070", 192},
    {"RTX 4080", 256},
    {"RTX 4090", 384},
    // RTX 50 series (Blackwell)
    {"RTX 5060", 128},
    {"RTX 5070", 192},
    {"RTX 5080", 256},
    {"RTX 5090", 512},
};

// Find bus width for a GPU name by substring matching
static int lookup_bus_width(const std::string& gpu_name) {
    for (const auto& [pattern, width] : BUS_WIDTH_TABLE) {
        if (gpu_name.find(pattern) != std::string::npos) {
            return width;
        }
    }
    return 0; // not found
}

std::vector<GpuProfile> profile_gpus() {
    std::vector<GpuProfile> profiles;

    nvmlReturn_t result = nvmlInit();
    if (result != NVML_SUCCESS) {
        fprintf(stderr, "Error: nvmlInit() failed: %s\n", nvmlErrorString(result));
        return profiles;
    }

    unsigned int device_count = 0;
    result = nvmlDeviceGetCount(&device_count);
    if (result != NVML_SUCCESS) {
        fprintf(stderr, "Error: nvmlDeviceGetCount() failed: %s\n", nvmlErrorString(result));
        nvmlShutdown();
        return profiles;
    }

    for (unsigned int i = 0; i < device_count; i++) {
        GpuProfile profile = {};
        profile.device_index = static_cast<int>(i);

        nvmlDevice_t device;
        result = nvmlDeviceGetHandleByIndex(i, &device);
        if (result != NVML_SUCCESS) {
            fprintf(stderr, "Error: nvmlDeviceGetHandleByIndex(%u) failed: %s\n", i, nvmlErrorString(result));
            continue;
        }

        // GPU name
        char name[NVML_DEVICE_NAME_V2_BUFFER_SIZE];
        result = nvmlDeviceGetName(device, name, sizeof(name));
        if (result == NVML_SUCCESS) {
            profile.name = name;
        } else {
            profile.name = "Unknown GPU";
            fprintf(stderr, "Warning: nvmlDeviceGetName() failed: %s\n", nvmlErrorString(result));
        }

        // VRAM
        nvmlMemory_t memory;
        result = nvmlDeviceGetMemoryInfo(device, &memory);
        if (result == NVML_SUCCESS) {
            profile.vram_total_bytes = memory.total;
            profile.vram_free_bytes = memory.free;
            profile.vram_used_bytes = memory.used;
            profile.vram_total_gb = static_cast<double>(memory.total) / (1024.0 * 1024.0 * 1024.0);
            profile.vram_free_gb = static_cast<double>(memory.free) / (1024.0 * 1024.0 * 1024.0);
        } else {
            fprintf(stderr, "Warning: nvmlDeviceGetMemoryInfo() failed: %s\n", nvmlErrorString(result));
        }

        // Memory clock speed
        unsigned int clock_mhz = 0;
        result = nvmlDeviceGetMaxClockInfo(device, NVML_CLOCK_MEM, &clock_mhz);
        if (result == NVML_SUCCESS) {
            profile.memory_clock_mhz = static_cast<int>(clock_mhz);
        } else {
            fprintf(stderr, "Warning: nvmlDeviceGetMaxClockInfo() failed: %s\n", nvmlErrorString(result));
        }

        // Bus width from lookup table
        profile.bus_width_bits = lookup_bus_width(profile.name);
        profile.bandwidth_known = (profile.bus_width_bits > 0);

        // Derive memory bandwidth
        if (profile.bandwidth_known) {
            // bandwidth_GB_per_sec = (clock_MHz * 2 * bus_width_bits) / 8 / 1000
            profile.memory_bandwidth_gb_per_sec =
                (static_cast<double>(profile.memory_clock_mhz) * 2.0 * profile.bus_width_bits) / 8.0 / 1000.0;
        }

        // Temperature
        unsigned int temp_c = 0;
        result = nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temp_c);
        if (result == NVML_SUCCESS) {
            profile.temperature_c = static_cast<int>(temp_c);
        } else {
            fprintf(stderr, "Warning: nvmlDeviceGetTemperature() failed: %s\n", nvmlErrorString(result));
        }

        // Compute capability
        int major = 0, minor = 0;
        result = nvmlDeviceGetCudaComputeCapability(device, &major, &minor);
        if (result == NVML_SUCCESS) {
            profile.compute_capability_major = major;
            profile.compute_capability_minor = minor;
        } else {
            fprintf(stderr, "Warning: nvmlDeviceGetCudaComputeCapability() failed: %s\n", nvmlErrorString(result));
        }

        profiles.push_back(profile);
    }

    nvmlShutdown();
    return profiles;
}

void print_gpu_profiles(const std::vector<GpuProfile>& profiles) {
    if (profiles.empty()) {
        printf("=== GPU Profile ===\n");
        printf("  No NVIDIA GPUs found.\n\n");
        return;
    }

    for (const auto& p : profiles) {
        printf("=== GPU Profile [%d] ===\n", p.device_index);
        printf("  Name:              %s\n", p.name.c_str());
        printf("  VRAM Total:        %.2f GB (%llu bytes)\n", p.vram_total_gb, p.vram_total_bytes);
        printf("  VRAM Free:         %.2f GB (%llu bytes)\n", p.vram_free_gb, p.vram_free_bytes);
        printf("  VRAM Used:         %.2f GB\n", p.vram_total_gb - p.vram_free_gb);
        printf("  Memory Clock:      %d MHz\n", p.memory_clock_mhz);
        if (p.bandwidth_known) {
            printf("  Bus Width:         %d-bit\n", p.bus_width_bits);
            printf("  Memory Bandwidth:  %.2f GB/s (derived)\n", p.memory_bandwidth_gb_per_sec);
        } else {
            printf("  Bus Width:         Unknown (GPU not in lookup table)\n");
            printf("  Memory Bandwidth:  Unknown\n");
        }
        printf("  Temperature:       %d C\n", p.temperature_c);
        printf("  Compute Capability: %d.%d (sm_%d%d)\n",
               p.compute_capability_major, p.compute_capability_minor,
               p.compute_capability_major, p.compute_capability_minor);
        printf("\n");
    }
}
