#pragma once

#include "platform_types.h"
#include "../types.h"
#include <string>
#include <memory>
#include <functional>

// =============================================================================
// Forward Declarations (llama.cpp types)
// =============================================================================
struct llama_model;
struct llama_context;
struct llama_model_params;
struct llama_context_params;

// =============================================================================
// Execution Backend Interface (Step 11, Phase A)
// =============================================================================
// Abstract interface for platform-specific inference execution.
// Each platform (CUDA, ROCm, Metal, CPU) implements this interface.
//
// The backend is responsible for:
//   1. Initializing the compute backend (CUDA context, Metal device, etc.)
//   2. Configuring model loading parameters (n_gpu_layers, tensor split, etc.)
//   3. Configuring inference context parameters (n_ctx, n_batch, etc.)
//   4. Providing platform-specific optimizations (flash attention, etc.)
//   5. Shutting down cleanly
//
// The StrategyConfig is platform-agnostic — the backend translates it
// into platform-specific parameters.
// =============================================================================

class IExecutionBackend {
public:
    virtual ~IExecutionBackend() = default;
    
    // =========================================================================
    // Backend Identity
    // =========================================================================
    
    // Get the compute backend type
    virtual ComputeBackend getBackendType() const = 0;
    
    // Get a human-readable name for this backend
    // Examples: "CUDA", "ROCm", "Metal", "CPU"
    virtual std::string getName() const = 0;
    
    // Check if this backend is available on this system
    virtual bool isAvailable() const = 0;
    
    // =========================================================================
    // Lifecycle
    // =========================================================================
    
    // Initialize the backend (call once at program start)
    // Returns true on success, false if initialization fails
    virtual bool initialize() = 0;
    
    // Shutdown the backend (call once at program end)
    virtual void shutdown() = 0;
    
    // Check if the backend is initialized
    virtual bool isInitialized() const = 0;
    
    // =========================================================================
    // Model Loading Configuration
    // =========================================================================
    
    // Get model loading parameters for the given strategy
    // This translates StrategyConfig into platform-specific llama_model_params
    virtual llama_model_params getModelParams(const StrategyConfig& strategy) const = 0;
    
    // Get the number of GPU layers to offload for the given strategy
    virtual uint32_t getGPULayers(const StrategyConfig& strategy) const = 0;
    
    // Get tensor split ratio for multi-GPU or CPU/GPU split
    // Returns array of floats summing to 1.0, one per device
    virtual std::vector<float> getTensorSplit(const StrategyConfig& strategy) const = 0;
    
    // =========================================================================
    // Context Configuration
    // =========================================================================
    
    // Get inference context parameters for the given strategy
    // This translates StrategyConfig into platform-specific llama_context_params
    virtual llama_context_params getContextParams(const StrategyConfig& strategy) const = 0;
    
    // Get the number of CPU threads for this backend
    virtual uint32_t getThreadCount() const = 0;
    
    // Get the number of batch processing threads
    virtual uint32_t getBatchThreadCount() const = 0;
    
    // =========================================================================
    // Platform-Specific Features
    // =========================================================================
    
    // Check if flash attention is supported
    virtual bool supportsFlashAttention() const = 0;
    
    // Check if the backend supports KV cache quantization
    virtual bool supportsKVCacheQuant() const = 0;
    
    // Check if the backend supports tensor overrides (for MoE/hot-cold)
    virtual bool supportsTensorOverrides() const = 0;
    
    // Get the recommended batch size for prefill
    virtual uint32_t getRecommendedBatchSize() const = 0;
    
    // =========================================================================
    // Memory Queries (during execution)
    // =========================================================================
    
    // Get current VRAM usage in bytes
    virtual uint64_t getCurrentVRAMUsage() const = 0;
    
    // Get current RAM usage in bytes (for this backend's allocations)
    virtual uint64_t getCurrentRAMUsage() const = 0;
    
    // Check if the backend is memory-constrained
    virtual bool isMemoryConstrained() const = 0;
};

// =============================================================================
// Platform-Specific Backend Factory
// =============================================================================

// Create a backend for the current platform
// Returns the best available backend (CUDA > ROCm > Metal > CPU)
std::unique_ptr<IExecutionBackend> create_platform_backend();

// Create a backend for a specific platform
// Useful for testing or forcing a specific backend
std::unique_ptr<IExecutionBackend> create_platform_backend(ComputeBackend backend);

// =============================================================================
// Legacy Interface (for backward compatibility)
// =============================================================================
// These free functions delegate to the platform backend.
// They maintain compatibility with existing code that calls executor_init().

// Initialize the best available backend
bool executor_init();

// Shutdown the backend
void executor_shutdown();

// Get CPU thread count
int get_cpu_thread_count();
