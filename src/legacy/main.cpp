#include <cstdio>
#include <ctime>
#include <string>
#include "profiler/ram_profiler.h"
#include "profiler/gpu_profiler.h"
#include "profiler/disk_profiler.h"
    
static std::string get_timestamp() {
    std::time_t now = std::time(nullptr);
    char buf[64];
    struct tm tm_buf;
    localtime_s(&tm_buf, &now);
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tm_buf);
    return std::string(buf);
}

int main() {
    // Collect all profiles
    RamProfile ram = profile_ram();
    std::vector<GpuProfile> gpus = profile_gpus();
    std::string model_path = "D:\\projects\\software\\local_llm\\models\\Llama-3.2-3B-Instruct-Q4_K_M.gguf";
    DiskProfile disk = profile_disk(model_path);

    // Print formatted report
    printf("\n");
    printf("=== LLM Deployment Planner - Hardware Profile ===\n");
    printf("Timestamp: %s\n", get_timestamp().c_str());
    printf("\n");

    // --- System Memory ---
    printf("--- System Memory ---\n");
    printf("Total RAM:        %.2f GB\n", ram.total_ram_gb);
    printf("Available RAM:    %.2f GB\n", ram.available_ram_gb);
    printf("Page File:        %.2f GB total, %.2f GB available\n", ram.total_pagefile_gb, ram.available_pagefile_gb);
    printf("\n");

    // --- GPU ---
    if (!gpus.empty()) {
        const auto& g = gpus[0];
        printf("--- GPU ---\n");
        printf("Model:            %s\n", g.name.c_str());
        printf("VRAM Total:       %.2f GB\n", g.vram_total_gb);
        printf("VRAM Free:        %.2f GB\n", g.vram_free_gb);
        printf("Memory Clock:     %d MHz\n", g.memory_clock_mhz);
        if (g.bandwidth_known) {
            printf("Bus Width:        %d-bit\n", g.bus_width_bits);
            printf("Bandwidth:        %.1f GB/s (derived)\n", g.memory_bandwidth_gb_per_sec);
        } else {
            printf("Bus Width:        Unknown\n");
            printf("Bandwidth:        Unknown (GPU not in lookup table)\n");
        }
        printf("Temperature:      %d C\n", g.temperature_c);
        printf("Compute:          sm_%d%d\n", g.compute_capability_major, g.compute_capability_minor);
        printf("\n");
    }

    // --- Storage ---
    printf("--- Storage (%s) ---\n", model_path.c_str());
    if (disk.sequential_read_mb_per_sec > 0) {
        printf("Sequential Read:  %.0f MB/s  (%.1fs benchmark, 4MB blocks)\n",
               disk.sequential_read_mb_per_sec, disk.sequential_elapsed_sec);
    } else {
        printf("Sequential Read:  FAILED\n");
    }
    if (disk.random_read_mb_per_sec > 0) {
        printf("Random 4K Read:   %.0f MB/s  (%.1fs benchmark, 4KB blocks, %.0f IOPS)\n",
               disk.random_read_mb_per_sec, disk.random_elapsed_sec, disk.random_read_iops);
    } else {
        printf("Random 4K Read:   FAILED\n");
    }
    printf("\n");

    printf("=================================================\n");

    return 0;
}
