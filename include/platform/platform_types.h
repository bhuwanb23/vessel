#pragma once

#include <string>

// =============================================================================
// Platform Types (Step 11, Phase A)
// =============================================================================
// Defines the platform enumeration and platform-specific types used across
// the multi-platform architecture.
// =============================================================================

// =============================================================================
// Platform Enumeration
// =============================================================================

enum class Platform {
    NVIDIA_WINDOWS,    // NVIDIA GPU on Windows (CUDA)
    NVIDIA_LINUX,      // NVIDIA GPU on Linux (CUDA)
    AMD_LINUX,         // AMD GPU on Linux (ROCm/HIP)
    AMD_WINDOWS,       // AMD GPU on Windows (DirectML/Vulkan)
    APPLE_MACOS,       // Apple Silicon on macOS (Metal)
    CPU_ONLY,          // No GPU, CPU inference only
    UNKNOWN
};

// =============================================================================
// Compute Backend
// =============================================================================

enum class ComputeBackend {
    CUDA,              // NVIDIA CUDA
    HIP,               // AMD ROCm/HIP
    METAL,             // Apple Metal
    VULKAN,            // Vulkan (cross-platform)
    DIRECTML,          // DirectML (Windows)
    CPU,               // CPU-only (no GPU acceleration)
    UNKNOWN
};

// =============================================================================
// Memory Architecture
// =============================================================================

enum class MemoryArchitecture {
    DISCRETE,          // Separate VRAM and RAM (NVIDIA, AMD discrete)
    UNIFIED,           // Shared memory pool (Apple Silicon, some APUs)
    UNKNOWN
};

// =============================================================================
// Platform Info
// =============================================================================

struct PlatformInfo {
    Platform platform = Platform::UNKNOWN;
    ComputeBackend backend = ComputeBackend::UNKNOWN;
    MemoryArchitecture memory_arch = MemoryArchitecture::UNKNOWN;
    
    std::string platform_name;          // "Windows", "Linux", "macOS"
    std::string gpu_vendor;             // "NVIDIA", "AMD", "Apple"
    std::string compute_api;            // "CUDA", "ROCm", "Metal"
    
    bool is_unified_memory() const { 
        return memory_arch == MemoryArchitecture::UNIFIED; 
    }
    
    bool has_gpu() const {
        return backend != ComputeBackend::CPU && backend != ComputeBackend::UNKNOWN;
    }
};

// =============================================================================
// Platform Detection (compile-time)
// =============================================================================

inline Platform detect_platform() {
#if defined(_WIN32) || defined(_WIN64)
    // Windows
    #if defined(GGML_CUDA)
        return Platform::NVIDIA_WINDOWS;
    #elif defined(GGML_VULKAN) || defined(GGML_DIRECTML)
        return Platform::AMD_WINDOWS;
    #else
        return Platform::CPU_ONLY;
    #endif
#elif defined(__APPLE__) && defined(__MACH__)
    // macOS
    #if defined(GGML_METAL)
        return Platform::APPLE_MACOS;
    #else
        return Platform::CPU_ONLY;
    #endif
#elif defined(__linux__)
    // Linux
    #if defined(GGML_CUDA)
        return Platform::NVIDIA_LINUX;
    #elif defined(GGML_HIPBLAS)
        return Platform::AMD_LINUX;
    #else
        return Platform::CPU_ONLY;
    #endif
#else
    return Platform::UNKNOWN;
#endif
}

inline ComputeBackend detect_backend() {
#if defined(GGML_CUDA)
    return ComputeBackend::CUDA;
#elif defined(GGML_HIPBLAS)
    return ComputeBackend::HIP;
#elif defined(GGML_METAL)
    return ComputeBackend::METAL;
#elif defined(GGML_VULKAN)
    return ComputeBackend::VULKAN;
#elif defined(GGML_DIRECTML)
    return ComputeBackend::DIRECTML;
#else
    return ComputeBackend::CPU;
#endif
}

inline MemoryArchitecture detect_memory_arch() {
#if defined(__APPLE__) && defined(__MACH__)
    // Apple Silicon uses unified memory
    return MemoryArchitecture::UNIFIED;
#else
    // NVIDIA and AMD discrete GPUs use separate VRAM/RAM
    return MemoryArchitecture::DISCRETE;
#endif
}

// =============================================================================
// Platform Info Factory
// =============================================================================

inline PlatformInfo get_platform_info() {
    PlatformInfo info;
    info.platform = detect_platform();
    info.backend = detect_backend();
    info.memory_arch = detect_memory_arch();
    
    switch (info.platform) {
        case Platform::NVIDIA_WINDOWS:
            info.platform_name = "Windows";
            info.gpu_vendor = "NVIDIA";
            info.compute_api = "CUDA";
            break;
        case Platform::NVIDIA_LINUX:
            info.platform_name = "Linux";
            info.gpu_vendor = "NVIDIA";
            info.compute_api = "CUDA";
            break;
        case Platform::AMD_LINUX:
            info.platform_name = "Linux";
            info.gpu_vendor = "AMD";
            info.compute_api = "ROCm";
            break;
        case Platform::AMD_WINDOWS:
            info.platform_name = "Windows";
            info.gpu_vendor = "AMD";
            info.compute_api = "Vulkan/DirectML";
            break;
        case Platform::APPLE_MACOS:
            info.platform_name = "macOS";
            info.gpu_vendor = "Apple";
            info.compute_api = "Metal";
            break;
        case Platform::CPU_ONLY:
            info.platform_name = "Unknown";
            info.gpu_vendor = "None";
            info.compute_api = "CPU";
            break;
        default:
            info.platform_name = "Unknown";
            info.gpu_vendor = "Unknown";
            info.compute_api = "Unknown";
            break;
    }
    
    return info;
}
