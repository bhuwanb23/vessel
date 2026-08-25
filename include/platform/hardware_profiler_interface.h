#pragma once

#include "platform_types.h"
#include "../types.h"
#include <string>
#include <memory>

// =============================================================================
// Hardware Profiler Interface (Step 11, Phase A)
// =============================================================================
// Abstract interface for platform-specific hardware profiling.
// Each platform (NVIDIA, AMD, Apple) implements this interface.
//
// The profiler is responsible for:
//   1. Detecting GPU hardware
//   2. Querying VRAM/RAM sizes and free space
//   3. Measuring GPU bandwidth and TFLOPS
//   4. Measuring disk I/O speeds
//   5. Generating a stable hardware fingerprint
//
// The HardwareSpec struct is platform-agnostic — the profiler fills it
// with platform-specific data.
// =============================================================================

class IHardwareProfiler {
public:
    virtual ~IHardwareProfiler() = default;
    
    // =========================================================================
    // Availability Check
    // =========================================================================
    
    // Check if this profiler's platform is available on this system
    // Returns false if the required GPU/driver/API is not present
    virtual bool isAvailable() const = 0;
    
    // Get the platform this profiler handles
    virtual Platform getPlatform() const = 0;
    
    // Get a human-readable name for this profiler
    virtual std::string getName() const = 0;
    
    // =========================================================================
    // Full Hardware Profile
    // =========================================================================
    
    // Profile all hardware subsystems and return a complete HardwareSpec
    // This is the main entry point — called once at startup
    virtual HardwareSpec profile(const std::string& model_path_for_disk_bench = "") = 0;
    
    // =========================================================================
    // Live GPU Queries (for execution monitoring)
    // =========================================================================
    
    // Get current free VRAM in bytes (called during inference)
    virtual uint64_t getFreeVRAM() const = 0;
    
    // Get current GPU temperature in Celsius
    virtual uint32_t getGPUTemp() const = 0;
    
    // Get current GPU clock speed in MHz
    virtual uint32_t getGPUClock() const = 0;
    
    // Get current GPU utilization percentage (0-100)
    virtual uint32_t getGPUUtilization() const = 0;
    
    // =========================================================================
    // Platform-Specific Capabilities
    // =========================================================================
    
    // Check if the platform supports unified memory (Apple Silicon)
    virtual bool supportsUnifiedMemory() const = 0;
    
    // Check if the platform supports multi-GPU
    virtual bool supportsMultiGPU() const = 0;
    
    // Get the number of available GPUs
    virtual uint32_t getGPUCount() const = 0;
    
    // Get the compute capability/capability string
    // NVIDIA: "sm_86", AMD: "gfx1100", Apple: "apple_m2"
    virtual std::string getComputeCapability() const = 0;
};

// =============================================================================
// Platform-Specific Profiler Factory
// =============================================================================

// Create a profiler for the current platform
// Returns nullptr if no GPU is detected (CPU-only mode)
std::unique_ptr<IHardwareProfiler> create_platform_profiler();

// Create a profiler for a specific platform
// Useful for testing or cross-platform development
std::unique_ptr<IHardwareProfiler> create_platform_profiler(Platform platform);

// =============================================================================
// Legacy Interface (for backward compatibility)
// =============================================================================
// These free functions delegate to the platform profiler.
// They maintain compatibility with existing code that calls profile_hardware().

// Profile all hardware subsystems (delegates to platform profiler)
HardwareSpec profile_hardware(const std::string& model_path_for_disk_bench = "");

// Profile RAM only
uint64_t profile_ram_total();
uint64_t profile_ram_free();

// Profile GPU only
std::string profile_gpu_name();
uint64_t profile_vram_total();
uint64_t profile_vram_free();
double profile_gpu_bandwidth();

// Profile disk only
double profile_disk_sequential(const std::string& file_path);
double profile_disk_random_4k(const std::string& file_path);
