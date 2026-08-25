#pragma once

#include "platform_types.h"
#include "hardware_profiler_interface.h"
#include "execution_backend_interface.h"
#include <string>
#include <vector>
#include <memory>

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
// =============================================================================

// Create a profiler for the best available platform
// Tries NVIDIA → AMD → Apple → CPU-only in order
std::unique_ptr<IHardwareProfiler> create_platform_profiler_auto();

// Create a profiler for a specific platform (from --platform flag)
// Valid values: "cuda", "hip", "rocm", "metal", "apple", "cpu"
std::unique_ptr<IHardwareProfiler> create_platform_profiler_for_platform(const std::string& platform_name);

// Create an executor for the best available platform
std::unique_ptr<IExecutionBackend> create_platform_executor_auto();

// Create an executor for a specific platform
std::unique_ptr<IExecutionBackend> create_platform_executor_for_platform(const std::string& platform_name);

// =============================================================================
// Multi-GPU Detection
// =============================================================================

// Get the number of available GPUs on this platform
uint32_t get_gpu_count();

// GPU info for display
struct GpuInfo {
    std::string name;
    uint64_t vram_bytes;
    Platform platform;
    uint32_t index;
};

// Enumerate all available GPUs
std::vector<GpuInfo> enumerate_gpus();
