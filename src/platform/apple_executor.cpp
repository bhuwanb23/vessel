#include "platform/execution_backend_interface.h"

// =============================================================================
// Apple Silicon Metal Execution Backend (macOS)
// =============================================================================
// Implements IExecutionBackend for Apple Silicon using Metal.
// This backend is only compiled on macOS.
//
// Key differences from discrete GPU backends:
//   - Unified memory: VRAM = RAM, no PCIe transfer penalty
//   - n_gpu_layers controls compute location, not memory location
//   - Metal shader compilation on first run (~1-5 seconds)
//   - recommendedMaxWorkingSetSize × 0.9 as effective memory budget
//   - Split strategies are more attractive (no PCIe bottleneck)
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
#include <sys/sysctl.h>

// =============================================================================
// Apple Silicon Memory Budget Constants
// =============================================================================
// Exceeding recommendedMaxWorkingSetSize causes macOS to swap aggressively.
// Use 90% of recommended as the effective budget.
static constexpr double METAL_MEMORY_BUDGET_FRACTION = 0.90;

// Metal shader compilation can take 1-5 seconds on first run.
// Subsequent runs use cached shaders (~200ms).
static constexpr double METAL_FIRST_RUN_OVERHEAD_SEC = 3.0;

// =============================================================================
// Apple Silicon Metal Execution Backend Class
// =============================================================================

class AppleMetalBackend : public IExecutionBackend {
public:
    AppleMetalBackend() : initialized_(false), first_run_(true),
                          recommended_max_bytes_(0) {}
    
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
        
        // First run: Metal compiles shaders from .metal source files.
        // This takes 1-5 seconds. Subsequent runs use cached binaries.
        if (first_run_) {
            fprintf(stderr, "[AppleMetalBackend] ⚠️ First run: Metal shader compilation may take 3-5 seconds...\n");
            fprintf(stderr, "[AppleMetalBackend] Subsequent runs will use cached shaders (~200ms).\n");
        }
        
        // llama_backend_init() will initialize the Metal backend
        // when compiled with GGML_METAL
        llama_backend_init();
        
        // Get recommended max working set size
        // This is the maximum memory Metal recommends for GPU compute.
        // Exceeding it causes macOS to swap aggressively.
        // Note: We can't get this without a device, but we know the
        // total unified memory from the profiler. Use 90% of total.
        // The actual recommendedMaxWorkingSetSize is queried by the profiler.
        
        initialized_ = true;
        first_run_ = false;
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
        // On Apple Silicon, if available memory < 2GB, we're constrained
        return false;  // Would need mach VM statistics
    }
    
    // =========================================================================
    // Apple Silicon Specific: Memory Budget
    // =========================================================================
    
    // Get the effective memory budget for Metal compute.
    // Uses recommendedMaxWorkingSetSize × 0.9 to avoid swapping.
    uint64_t getEffectiveMemoryBudget() const {
        if (recommended_max_bytes_ > 0) {
            return static_cast<uint64_t>(recommended_max_bytes_ * METAL_MEMORY_BUDGET_FRACTION);
        }
        return 0;  // Unknown, caller should use total unified memory
    }
    
    // Get the Metal first-run overhead estimate.
    // Returns > 0 if this is likely the first run (shader compilation).
    double getFirstRunOverheadSec() const {
        return first_run_ ? METAL_FIRST_RUN_OVERHEAD_SEC : 0.2;
    }
    
private:
    bool initialized_;
    bool first_run_;
    uint64_t recommended_max_bytes_;
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
