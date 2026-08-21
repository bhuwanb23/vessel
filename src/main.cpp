#include <cstdio>
#include "profiler/ram_profiler.h"
#include "profiler/gpu_profiler.h"
#include "profiler/disk_profiler.h"

int main() {
    printf("Local LLM Planner - Hardware Profiler\n");
    printf("======================================\n\n");

    // RAM Profiler
    RamProfile ram = profile_ram();
    print_ram_profile(ram);

    // GPU Profiler
    std::vector<GpuProfile> gpus = profile_gpus();
    print_gpu_profiles(gpus);

    // Disk Profiler (using the test GGUF model as benchmark target)
    std::string model_path = "D:\\projects\\software\\local_llm\\models\\Llama-3.2-3B-Instruct-Q4_K_M.gguf";
    DiskProfile disk = profile_disk(model_path);
    print_disk_profile(disk);

    printf("=== Validation ===\n");
    printf("RAM:    Compare 'Available RAM' with Task Manager -> Memory -> 'Available'\n");
    printf("GPU:    Compare name/VRAM with: nvidia-smi\n");
    printf("Disk:   Sequential should be 50-80%% of drive spec. Random should be 10-50x lower.\n");

    return 0;
}
