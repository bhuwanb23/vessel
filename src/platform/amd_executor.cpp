#include "platform/execution_backend_interface.h"

// =============================================================================
// AMD HIP Execution Backend (Linux)
// =============================================================================
// Implements IExecutionBackend for AMD GPUs using ROCm/HIP.
// This backend is only compiled on Linux with HIP support.
//
// Requirements:
//   - ROCm toolkit installed
//   - llama.cpp built with -DGGML_HIPBLAS=ON
//   - AMD GPU with ROCm support (RX 7900 series recommended)
//
// Build: -DGGML_HIPBLAS=ON (enables HIP backend)
// =============================================================================

#if defined(__linux__) && defined(GGML_HIPBLAS)

#include <llama.h>
#include <cstdio>
#include <cstring>
#include <thread>
#include <algorithm>

// =============================================================================
// AMD HIP Execution Backend Class
// =============================================================================

class AmdHipBackend : public IExecutionBackend {
public:
    AmdHipBackend() : initialized_(false), gpu_layers_(0) {}
    
    ~AmdHipBackend() override {
        if (initialized_) {
            shutdown();
        }
    }
    
    // =========================================================================
    // IExecutionBackend Interface
    // =========================================================================
    
    ComputeBackend getBackendType() const override {
        return ComputeBackend::HIP;
    }
    
    std::string getName() const override {
        return "AMD HIP";
    }
    
    bool isAvailable() const override {
        // Check if HIP runtime is available
        // In practice, this would check for rocminfo or HIP runtime
        return true;  // Assume available if compiled with HIP
    }
    
    bool initialize() override {
        if (initialized_) return true;
        
        fprintf(stderr, "[AmdHipBackend] Initializing HIP backend...\n");
        
        // llama_backend_init() will initialize the HIP backend
        // when compiled with GGML_HIPBLAS
        llama_backend_init();
        
        initialized_ = true;
        fprintf(stderr, "[AmdHipBackend] HIP backend initialized successfully\n");
        
        return true;
    }
    
    void shutdown() override {
        if (!initialized_) return;
        
        fprintf(stderr, "[AmdHipBackend] Shutting down HIP backend...\n");
        
        llama_backend_free();
        
        initialized_ = false;
        fprintf(stderr, "[AmdHipBackend] HIP backend shut down\n");
    }
    
    bool isInitialized() const override {
        return initialized_;
    }
    
    // =========================================================================
    // Model Loading Configuration
    // =========================================================================
    
    llama_model_params getModelParams(const StrategyConfig& strategy) const override {
        llama_model_params params = llama_model_default_params();
        
        // GPU layers
        params.n_gpu_layers = strategy.gpu_layers;
        
        // Main GPU (always 0 for single GPU)
        params.main_gpu = 0;
        
        // Memory mapping
        // For HIP, use mmap for CPU-only strategies
        params.use_mmap = (strategy.gpu_layers == 0);
        params.use_mlock = false;
        
        // Tensor split (for multi-GPU, not used in MVP)
        // params.tensor_split = nullptr;
        
        return params;
    }
    
    uint32_t getGPULayers(const StrategyConfig& strategy) const override {
        return strategy.gpu_layers;
    }
    
    std::vector<float> getTensorSplit(const StrategyConfig& strategy) const override {
        // Single GPU: [1.0]
        // CPU-only: [0.0]
        // Split: calculated based on GPU layers
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
        
        // Context length
        params.n_ctx = strategy.context_length;
        if (params.n_ctx == 0) params.n_ctx = 4096;
        
        // Batch size
        params.n_batch = 512;
        params.n_ubatch = 512;
        
        // Thread count
        params.n_threads = getThreadCount();
        params.n_threads_batch = getBatchThreadCount();
        
        // Flash attention (supported on RDNA3+)
        params.flash_attn = true;
        
        // KV cache quantization
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
        // Use physical cores, not hyperthreads
        uint32_t cores = std::thread::hardware_concurrency();
        return (cores > 0) ? cores : 4;
    }
    
    uint32_t getBatchThreadCount() const override {
        // Can use more threads for batch processing
        return getThreadCount() * 2;
    }
    
    // =========================================================================
    // Platform-Specific Features
    // =========================================================================
    
    bool supportsFlashAttention() const override {
        // RDNA3+ supports flash attention
        // For MVP, assume supported
        return true;
    }
    
    bool supportsKVCacheQuant() const override {
        // HIP supports KV cache quantization
        return true;
    }
    
    bool supportsTensorOverrides() const override {
        // HIP supports tensor overrides for MoE/hot-cold
        return true;
    }
    
    uint32_t getRecommendedBatchSize() const override {
        // AMD GPUs typically use larger batch sizes
        return 1024;
    }
    
    // =========================================================================
    // Memory Queries (during execution)
    // =========================================================================
    
    uint64_t getCurrentVRAMUsage() const override {
        // Would need ROCm-SMI integration
        // For now, return 0
        return 0;
    }
    
    uint64_t getCurrentRAMUsage() const override {
        // Would need /proc/self/status integration
        // For now, return 0
        return 0;
    }
    
    bool isMemoryConstrained() const override {
        // Check if VRAM is running low
        // For now, return false
        return false;
    }
    
private:
    bool initialized_;
    uint32_t gpu_layers_;
};

// =============================================================================
// Factory Implementation
// =============================================================================

std::unique_ptr<IExecutionBackend> create_platform_backend(ComputeBackend backend) {
    if (backend == ComputeBackend::HIP) {
        return std::make_unique<AmdHipBackend>();
    }
    return nullptr;
}

std::unique_ptr<IExecutionBackend> create_platform_backend() {
    // Auto-detect the best backend
    #if defined(GGML_HIPBLAS)
        return std::make_unique<AmdHipBackend>();
    #elif defined(GGML_CUDA)
        // Would return CUDA backend
        return nullptr;
    #else
        // CPU-only
        return nullptr;
    #endif
}

#endif  // __linux__ && GGML_HIPBLAS
