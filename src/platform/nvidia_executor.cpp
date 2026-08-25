#include "platform/execution_backend_interface.h"

// =============================================================================
// NVIDIA CUDA Execution Backend
// =============================================================================
// Implements IExecutionBackend for NVIDIA GPUs using CUDA.
// =============================================================================

#if defined(GGML_CUDA)

#include <llama.h>
#include <cstdio>
#include <cstring>
#include <thread>
#include <algorithm>

// =============================================================================
// NVIDIA CUDA Execution Backend Class
// =============================================================================

class NvidiaCudaBackend : public IExecutionBackend {
public:
    NvidiaCudaBackend() : initialized_(false) {}
    
    ~NvidiaCudaBackend() override {
        if (initialized_) {
            shutdown();
        }
    }
    
    // =========================================================================
    // IExecutionBackend Interface
    // =========================================================================
    
    ComputeBackend getBackendType() const override {
        return ComputeBackend::CUDA;
    }
    
    std::string getName() const override {
        return "NVIDIA CUDA";
    }
    
    bool isAvailable() const override {
        // Check if CUDA runtime is available
        return true;  // Assume available if compiled with CUDA
    }
    
    bool initialize() override {
        if (initialized_) return true;
        
        fprintf(stderr, "[NvidiaCudaBackend] Initializing CUDA backend...\n");
        
        llama_backend_init();
        
        initialized_ = true;
        fprintf(stderr, "[NvidiaCudaBackend] CUDA backend initialized successfully\n");
        
        return true;
    }
    
    void shutdown() override {
        if (!initialized_) return;
        
        fprintf(stderr, "[NvidiaCudaBackend] Shutting down CUDA backend...\n");
        
        llama_backend_free();
        
        initialized_ = false;
        fprintf(stderr, "[NvidiaCudaBackend] CUDA backend shut down\n");
    }
    
    bool isInitialized() const override {
        return initialized_;
    }
    
    // =========================================================================
    // Model Loading Configuration
    // =========================================================================
    
    llama_model_params getModelParams(const StrategyConfig& strategy) const override {
        llama_model_params params = llama_model_default_params();
        
        params.n_gpu_layers = strategy.gpu_layers;
        params.main_gpu = 0;
        params.use_mmap = (strategy.gpu_layers == 0);
        params.use_mlock = false;
        
        return params;
    }
    
    uint32_t getGPULayers(const StrategyConfig& strategy) const override {
        return strategy.gpu_layers;
    }
    
    std::vector<float> getTensorSplit(const StrategyConfig& strategy) const override {
        if (strategy.gpu_layers == 0) {
            return {0.0f};
        }
        return {1.0f};
    }
    
    // =========================================================================
    // Context Configuration
    // =========================================================================
    
    llama_context_params getContextParams(const StrategyConfig& strategy) const override {
        llama_context_params params = llama_context_default_params();
        
        params.n_ctx = strategy.context_length;
        if (params.n_ctx == 0) params.n_ctx = 4096;
        
        params.n_batch = 512;
        params.n_ubatch = 512;
        
        params.n_threads = getThreadCount();
        params.n_threads_batch = getBatchThreadCount();
        
        params.flash_attn = true;
        
        switch (strategy.kv_quant_bits) {
            case 4:
                params.type_k = GGML_TYPE_Q4_0;
                params.type_v = GGML_TYPE_Q4_0;
                break;
            case 8:
                params.type_k = GGML_TYPE_Q8_0;
                params.type_v = GGML_TYPE_Q8_0;
                break;
            default:
                params.type_k = GGML_TYPE_F16;
                params.type_v = GGML_TYPE_F16;
                break;
        }
        
        return params;
    }
    
    uint32_t getThreadCount() const override {
        uint32_t cores = std::thread::hardware_concurrency();
        return (cores > 0) ? cores : 4;
    }
    
    uint32_t getBatchThreadCount() const override {
        return getThreadCount() * 2;
    }
    
    // =========================================================================
    // Platform-Specific Features
    // =========================================================================
    
    bool supportsFlashAttention() const override {
        return true;
    }
    
    bool supportsKVCacheQuant() const override {
        return true;
    }
    
    bool supportsTensorOverrides() const override {
        return true;
    }
    
    uint32_t getRecommendedBatchSize() const override {
        return 512;
    }
    
    // =========================================================================
    // Memory Queries (during execution)
    // =========================================================================
    
    uint64_t getCurrentVRAMUsage() const override {
        // Would need NVML integration
        return 0;
    }
    
    uint64_t getCurrentRAMUsage() const override {
        return 0;
    }
    
    bool isMemoryConstrained() const override {
        return false;
    }
    
private:
    bool initialized_;
};

// =============================================================================
// Factory Implementation
// =============================================================================

std::unique_ptr<IExecutionBackend> create_platform_backend(ComputeBackend backend) {
    if (backend == ComputeBackend::CUDA) {
        return std::make_unique<NvidiaCudaBackend>();
    }
    return nullptr;
}

std::unique_ptr<IExecutionBackend> create_platform_backend() {
    return std::make_unique<NvidiaCudaBackend>();
}

#endif  // GGML_CUDA

// =============================================================================
// Factory Fallback (when GGML_CUDA is not defined)
// =============================================================================
// Provides the create_platform_backend symbols when CUDA is not compiled.
// These are separate definitions that return nullptr.
// =============================================================================

#if !defined(GGML_CUDA)
std::unique_ptr<IExecutionBackend> create_platform_backend(ComputeBackend) {
    return nullptr;
}

std::unique_ptr<IExecutionBackend> create_platform_backend() {
    return nullptr;
}
#endif
