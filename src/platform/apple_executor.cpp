#include "platform/execution_backend_interface.h"

// =============================================================================
// Apple Silicon Metal Execution Backend (macOS)
// =============================================================================
// Implements IExecutionBackend for Apple Silicon using Metal.
// This backend is only compiled on macOS.
//
// Requirements:
//   - macOS 14+ (Sonoma or later)
//   - Apple Silicon (M1/M2/M3/M4)
//   - llama.cpp built with -DGGML_METAL=ON
//
// Build: -DGGML_METAL=ON (enables Metal backend)
// =============================================================================

#if defined(__APPLE__) && defined(__MACH__) && defined(GGML_METAL)

#include <llama.h>
#include <cstdio>
#include <cstring>
#include <thread>
#include <algorithm>

// =============================================================================
// Apple Silicon Metal Execution Backend Class
// =============================================================================

class AppleMetalBackend : public IExecutionBackend {
public:
    AppleMetalBackend() : initialized_(false) {}
    
    ~AppleMetalBackend() override {
        if (initialized_) {
            shutdown();
        }
    }
    
    // =========================================================================
    // IExecutionBackend Interface
    // =========================================================================
    
    ComputeBackend getBackendType() const override {
        return ComputeBackend::METAL;
    }
    
    std::string getName() const override {
        return "Apple Metal";
    }
    
    bool isAvailable() const override {
        // Check if Metal is available
        return true;  // Assume available if compiled with Metal
    }
    
    bool initialize() override {
        if (initialized_) return true;
        
        fprintf(stderr, "[AppleMetalBackend] Initializing Metal backend...\n");
        
        // llama_backend_init() will initialize the Metal backend
        // when compiled with GGML_METAL
        llama_backend_init();
        
        initialized_ = true;
        fprintf(stderr, "[AppleMetalBackend] Metal backend initialized successfully\n");
        
        return true;
    }
    
    void shutdown() override {
        if (!initialized_) return;
        
        fprintf(stderr, "[AppleMetalBackend] Shutting down Metal backend...\n");
        
        llama_backend_free();
        
        initialized_ = false;
        fprintf(stderr, "[AppleMetalBackend] Metal backend shut down\n");
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
        // On Apple Silicon, n_gpu_layers controls Metal compute, not memory placement
        // All layers use unified memory, but GPU compute is faster
        params.n_gpu_layers = strategy.gpu_layers;
        
        // Main GPU (always 0 for Apple Silicon)
        params.main_gpu = 0;
        
        // Memory mapping
        // On unified memory, mmap is less important but still useful
        params.use_mmap = true;
        params.use_mlock = false;
        
        return params;
    }
    
    uint32_t getGPULayers(const StrategyConfig& strategy) const override {
        return strategy.gpu_layers;
    }
    
    std::vector<float> getTensorSplit(const StrategyConfig& strategy) const override {
        // Apple Silicon has one GPU
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
        
        // Flash attention (supported on Apple Silicon)
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
        // Use performance cores, not efficiency cores
        // Apple Silicon has P-cores and E-cores
        // For inference, P-cores are preferred
        
        int perf_cores = 0;
        size_t size = sizeof(perf_cores);
        
        if (sysctlbyname("hw.perflevel0.physicalcpu", &perf_cores, &size, NULL, 0) == 0) {
            return perf_cores;
        }
        
        // Fallback: use all cores
        uint32_t cores = std::thread::hardware_concurrency();
        return (cores > 0) ? cores / 2 : 4;  // Assume half are P-cores
    }
    
    uint32_t getBatchThreadCount() const override {
        // Can use more threads for batch processing
        return getThreadCount() * 2;
    }
    
    // =========================================================================
    // Platform-Specific Features
    // =========================================================================
    
    bool supportsFlashAttention() const override {
        // All Apple Silicon supports flash attention
        return true;
    }
    
    bool supportsKVCacheQuant() const override {
        // Metal supports KV cache quantization
        return true;
    }
    
    bool supportsTensorOverrides() const override {
        // Metal supports tensor overrides for MoE/hot-cold
        return true;
    }
    
    uint32_t getRecommendedBatchSize() const override {
        // Apple Silicon typically uses moderate batch sizes
        return 512;
    }
    
    // =========================================================================
    // Memory Queries (during execution)
    // =========================================================================
    
    uint64_t getCurrentVRAMUsage() const override {
        // Unified memory: VRAM = RAM
        // Would need mach VM statistics
        return 0;
    }
    
    uint64_t getCurrentRAMUsage() const override {
        // Would need /proc/self/status equivalent
        return 0;
    }
    
    bool isMemoryConstrained() const override {
        // Check if unified memory is running low
        return false;
    }
    
private:
    bool initialized_;
};

// =============================================================================
// Factory Implementation
// =============================================================================

std::unique_ptr<IExecutionBackend> create_platform_backend(ComputeBackend backend) {
    if (backend == ComputeBackend::METAL) {
        return std::make_unique<AppleMetalBackend>();
    }
    return nullptr;
}

std::unique_ptr<IExecutionBackend> create_platform_backend() {
    #if defined(GGML_METAL)
        return std::make_unique<AppleMetalBackend>();
    #else
        return nullptr;
    #endif
}

#endif  // __APPLE__ && __MACH__ && GGML_METAL
