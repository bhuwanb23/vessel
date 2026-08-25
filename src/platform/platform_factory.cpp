// =============================================================================
// Platform Auto-Detection Factory (Step 11, Phase F)
// =============================================================================
// Runtime detection of available platforms and creation of appropriate
// profiler and executor instances.
//
// Detection order (most common for LLM workloads first):
//   1. NVIDIA (CUDA) — most common for local LLM inference
//   2. AMD (ROCm/HIP on Linux, Vulkan/DirectML on Windows)
//   3. Apple (Metal on macOS)
//   4. CPU-only (always available as fallback)
//
// The factory tries each platform in order and returns the first one that
// reports isAvailable() == true.
// =============================================================================

#include "platform/hardware_profiler_interface.h"
#include "platform/execution_backend_interface.h"
#include <cstdio>
#include <string>
#include <vector>
#include <algorithm>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#elif defined(__linux__)
#include <dlfcn.h>
#include <cstdio>
#endif

// =============================================================================
// Platform Detection Helpers
// =============================================================================

// Check if NVIDIA GPU is available (Windows: NVML, Linux: libnvidia-ml)
static bool detect_nvidia() {
#if defined(_WIN32)
    // Try loading NVML
    HMODULE nvml = LoadLibraryA("nvml.dll");
    if (nvml) {
        FreeLibrary(nvml);
        return true;
    }
    // Also try the CUDA installation path
    nvml = LoadLibraryA("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/bin/nvml.dll");
    if (nvml) {
        FreeLibrary(nvml);
        return true;
    }
    return false;
#elif defined(__linux__)
    // Try loading libnvidia-ml.so
    void* handle = dlopen("libnvidia-ml.so.1", RTLD_LAZY);
    if (handle) {
        dlclose(handle);
        return true;
    }
    handle = dlopen("libnvidia-ml.so", RTLD_LAZY);
    if (handle) {
        dlclose(handle);
        return true;
    }
    return false;
#else
    return false;
#endif
}

// Check if AMD GPU is available (Linux: ROCm-SMI, Windows: ADL/DXGI)
static bool detect_amd() {
#if defined(__linux__)
    // Try loading ROCm-SMI
    void* handle = dlopen("librocm_smi64.so", RTLD_LAZY);
    if (handle) {
        dlclose(handle);
        return true;
    }
    handle = dlopen("librocm_smi64.so.1", RTLD_LAZY);
    if (handle) {
        dlclose(handle);
        return true;
    }
    // Fallback: check if rocm-smi CLI exists
    FILE* pipe = popen("which rocm-smi 2>/dev/null", "r");
    if (pipe) {
        char buffer[128];
        bool found = (fgets(buffer, sizeof(buffer), pipe) != nullptr);
        pclose(pipe);
        return found;
    }
    return false;
#elif defined(_WIN32)
    // Try loading AMD Display Library
    HMODULE adl = LoadLibraryA("atiadlxx.dll");
    if (adl) {
        FreeLibrary(adl);
        return true;
    }
    adl = LoadLibraryA("atiadlxy.dll");
    if (adl) {
        FreeLibrary(adl);
        return true;
    }
    // Fallback: check DXGI for AMD adapter
    // This is more complex and would require DXGI enumeration
    return false;
#else
    return false;
#endif
}

// Check if Apple Silicon Metal is available
static bool detect_apple_metal() {
#if defined(__APPLE__) && defined(__MACH__)
    // Metal availability is checked at runtime via MTLCreateSystemDefaultDevice()
    // For now, assume Apple Silicon if on macOS
    #if defined(GGML_METAL)
        return true;  // Compiled with Metal support
    #else
        return false;  // No Metal support compiled in
    #endif
#else
    return false;
#endif
}

// =============================================================================
// Platform Auto-Detection Factory
// =============================================================================

// Create a profiler for the best available platform
// Tries NVIDIA → AMD → Apple → CPU-only in order
std::unique_ptr<IHardwareProfiler> create_platform_profiler_auto() {
    // Try NVIDIA first (most common for LLM workloads)
    if (detect_nvidia()) {
        auto profiler = create_platform_profiler(Platform::NVIDIA_WINDOWS);
        if (profiler && profiler->isAvailable()) {
            fprintf(stderr, "[PlatformFactory] Detected NVIDIA GPU\n");
            return profiler;
        }
    }
    
    // Try AMD
    if (detect_amd()) {
#if defined(__linux__)
        auto profiler = create_platform_profiler(Platform::AMD_LINUX);
#else
        auto profiler = create_platform_profiler(Platform::AMD_WINDOWS);
#endif
        if (profiler && profiler->isAvailable()) {
            fprintf(stderr, "[PlatformFactory] Detected AMD GPU\n");
            return profiler;
        }
    }
    
    // Try Apple
    if (detect_apple_metal()) {
        auto profiler = create_platform_profiler(Platform::APPLE_MACOS);
        if (profiler && profiler->isAvailable()) {
            fprintf(stderr, "[PlatformFactory] Detected Apple Silicon\n");
            return profiler;
        }
    }
    
    // Fallback: CPU-only
    fprintf(stderr, "[PlatformFactory] No GPU detected, using CPU-only mode\n");
    return create_platform_profiler(Platform::CPU_ONLY);
}

// Create a profiler for a specific platform (from --platform flag)
std::unique_ptr<IHardwareProfiler> create_platform_profiler_for_platform(const std::string& platform_name) {
    Platform platform;
    
    // Normalize to lowercase
    std::string lower = platform_name;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    
    if (lower == "cuda" || lower == "nvidia") {
#if defined(_WIN32)
        platform = Platform::NVIDIA_WINDOWS;
#elif defined(__linux__)
        platform = Platform::NVIDIA_LINUX;
#else
        fprintf(stderr, "[PlatformFactory] CUDA not available on this OS\n");
        return nullptr;
#endif
    } else if (lower == "hip" || lower == "rocm" || lower == "amd") {
#if defined(__linux__)
        platform = Platform::AMD_LINUX;
#elif defined(_WIN32)
        platform = Platform::AMD_WINDOWS;
#else
        fprintf(stderr, "[PlatformFactory] ROCm not available on this OS\n");
        return nullptr;
#endif
    } else if (lower == "metal" || lower == "apple") {
#if defined(__APPLE__) && defined(__MACH__)
        platform = Platform::APPLE_MACOS;
#else
        fprintf(stderr, "[PlatformFactory] Metal not available on this OS\n");
        return nullptr;
#endif
    } else if (lower == "cpu" || lower == "cpu-only" || lower == "none") {
        platform = Platform::CPU_ONLY;
    } else {
        fprintf(stderr, "[PlatformFactory] Unknown platform: %s\n", platform_name.c_str());
        fprintf(stderr, "  Valid options: cuda, hip, metal, cpu\n");
        return nullptr;
    }
    
    auto profiler = create_platform_profiler(platform);
    if (profiler && profiler->isAvailable()) {
        fprintf(stderr, "[PlatformFactory] Using %s platform\n", platform_name.c_str());
        return profiler;
    }
    
    fprintf(stderr, "[PlatformFactory] Platform %s is not available on this system\n", 
            platform_name.c_str());
    return nullptr;
}

// Create an executor for the best available platform
std::unique_ptr<IExecutionBackend> create_platform_executor_auto() {
    // Try NVIDIA first
    if (detect_nvidia()) {
        auto executor = create_platform_backend(ComputeBackend::CUDA);
        if (executor && executor->isAvailable()) {
            return executor;
        }
    }
    
    // Try AMD (HIP on Linux, Vulkan on Windows)
    if (detect_amd()) {
#if defined(__linux__)
        auto executor = create_platform_backend(ComputeBackend::HIP);
#else
        auto executor = create_platform_backend(ComputeBackend::VULKAN);
#endif
        if (executor && executor->isAvailable()) {
            return executor;
        }
    }
    
    // Try Apple Metal
    if (detect_apple_metal()) {
        auto executor = create_platform_backend(ComputeBackend::METAL);
        if (executor && executor->isAvailable()) {
            return executor;
        }
    }
    
    // Fallback: CPU-only (no GPU acceleration)
    // Note: CPU-only backend may not be implemented yet
    auto cpu = create_platform_backend(ComputeBackend::CPU);
    if (cpu) return cpu;
    
    // If no backend available at all, return nullptr
    return nullptr;
}

// Create an executor for a specific platform
std::unique_ptr<IExecutionBackend> create_platform_executor_for_platform(const std::string& platform_name) {
    std::string lower = platform_name;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    
    ComputeBackend backend;
    
    if (lower == "cuda" || lower == "nvidia") {
        backend = ComputeBackend::CUDA;
    } else if (lower == "hip" || lower == "rocm" || lower == "amd") {
        backend = ComputeBackend::HIP;
    } else if (lower == "metal" || lower == "apple") {
        backend = ComputeBackend::METAL;
    } else if (lower == "vulkan") {
        backend = ComputeBackend::VULKAN;
    } else if (lower == "cpu" || lower == "cpu-only" || lower == "none") {
        backend = ComputeBackend::CPU;
    } else {
        return nullptr;
    }
    
    return create_platform_backend(backend);
}

// =============================================================================
// Multi-GPU Detection
// =============================================================================

// Get the number of available GPUs on this platform
uint32_t get_gpu_count() {
    // Try NVIDIA
    if (detect_nvidia()) {
        auto profiler = create_platform_profiler(Platform::NVIDIA_WINDOWS);
        if (profiler && profiler->isAvailable()) {
            return profiler->getGPUCount();
        }
    }
    
    // For other platforms, assume single GPU for MVP
    return 1;
}

// Get GPU info for display (name, VRAM, platform)
struct GpuInfo {
    std::string name;
    uint64_t vram_bytes;
    Platform platform;
    uint32_t index;
};

std::vector<GpuInfo> enumerate_gpus() {
    std::vector<GpuInfo> gpus;
    
    // For now, only enumerate NVIDIA GPUs (most common multi-GPU scenario)
    // AMD and Apple typically have single GPUs
#if defined(_WIN32) || defined(__linux__)
    if (detect_nvidia()) {
        auto profiler = create_platform_profiler(Platform::NVIDIA_WINDOWS);
        if (profiler && profiler->isAvailable()) {
            uint32_t count = profiler->getGPUCount();
            for (uint32_t i = 0; i < count; i++) {
                GpuInfo info;
                info.name = profiler->getName();
                info.vram_bytes = profiler->getFreeVRAM();  // TODO: get total VRAM
                info.platform = Platform::NVIDIA_WINDOWS;
                info.index = i;
                gpus.push_back(info);
            }
        }
    }
#endif
    
    return gpus;
}

// =============================================================================
// Platform Profiler Factory (Central Definition)
// =============================================================================
// This is the main factory function declared in hardware_profiler_interface.h.
// It delegates to platform-specific factory functions.
// =============================================================================

// Forward declarations of platform-specific factory functions
// (defined in nvidia_profiler.cpp, cpu_profiler.cpp, etc.)
extern std::unique_ptr<IHardwareProfiler> create_nvidia_profiler();
extern std::unique_ptr<IHardwareProfiler> create_cpu_profiler();

std::unique_ptr<IHardwareProfiler> create_platform_profiler(Platform platform) {
    switch (platform) {
        case Platform::NVIDIA_WINDOWS:
        case Platform::NVIDIA_LINUX: {
            auto p = create_nvidia_profiler();
            if (p && p->isAvailable()) return p;
            break;
        }
        case Platform::CPU_ONLY: {
            return create_cpu_profiler();
        }
        default:
            break;
    }
    return nullptr;
}

std::unique_ptr<IHardwareProfiler> create_platform_profiler() {
    return create_platform_profiler_auto();
}


