#include <cstdio>
#include "profiler/ram_profiler.h"
#include "profiler/gpu_profiler.h"

int main() {
    printf("Local LLM Planner - Hardware Profiler\n");
    printf("======================================\n\n");

    // RAM Profiler
    RamProfile ram = profile_ram();
    print_ram_profile(ram);

    // GPU Profiler
    std::vector<GpuProfile> gpus = profile_gpus();
    print_gpu_profiles(gpus);

    printf("=== Validation ===\n");
    printf("Compare GPU name and VRAM with: nvidia-smi\n");
    printf("Compare Available RAM with: Task Manager -> Performance -> Memory\n");
    printf("Temperature should be within a few degrees of nvidia-smi.\n");

    return 0;
}
