#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// =============================================================================
// Metadata Source (where did the model info come from?)
// =============================================================================
enum class MetadataSource {
    GGUF_HEADER,    // Direct from GGUF header (high confidence)
    CONFIG_JSON,    // Fallback to HuggingFace config.json (lower confidence)
    UNKNOWN         // No metadata source
};

// =============================================================================
// Model Type (for confidence calculation)
// =============================================================================
enum class ModelType {
    DENSE,          // Standard transformer (all params active)
    MOE,            // Mixture of Experts (Phase 2)
    UNKNOWN
};

// =============================================================================
// Placement strategy for model layers
// =============================================================================
enum class PlacementStrategy {
    FULL_GPU,       // All layers on GPU (fastest, needs enough VRAM)
    GPU_CPU_SPLIT,  // Some layers on GPU, rest on CPU (flexible)
    CPU_ONLY        // All layers on CPU (slowest, needs enough RAM)
};

// =============================================================================
// Confidence level for the prediction
// =============================================================================
enum class PredictionConfidence {
    HIGH,       // Formula well-validated, inputs are accurate
    MEDIUM,     // Formula approximate, or some inputs estimated
    LOW         // Significant uncertainty in inputs or formula
};

// =============================================================================
// Hardware Specification (from Step 1 hardware profiler)
// =============================================================================
struct HardwareSpec {
    // System Memory
    uint64_t ram_total_bytes = 0;       // Total physical RAM (bytes)
    uint64_t ram_free_bytes = 0;        // Available RAM (bytes) - includes standby cache
    double ram_bandwidth_gbs = 0.0;     // RAM bandwidth (GB/s) - not measured in Step 1, estimate if unknown

    // GPU
    uint64_t vram_total_bytes = 0;      // Total GPU VRAM (bytes)
    uint64_t vram_free_bytes = 0;       // Free GPU VRAM (bytes)
    double gpu_bandwidth_gbs = 0.0;     // GPU memory bandwidth (GB/s) - derived from NVML
    double gpu_tflops_fp16 = 0.0;       // GPU FP16 TFLOPS - from spec or NVML

    // Storage
    double nvme_sequential_mbs = 0.0;   // NVMe sequential read (MB/s)
    double nvme_random_4k_mbs = 0.0;    // NVMe random 4K read (MB/s)

    // Hardware fingerprint (unique key for calibration log)
    std::string hardware_fingerprint;

    // GPU Info
    std::string gpu_name;               // GPU model name (for display)
    uint32_t gpu_compute_major = 0;     // CUDA compute capability major
    uint32_t gpu_compute_minor = 0;     // CUDA compute capability minor
};

// =============================================================================
// Model Metadata (from Step 2 metadata fetcher)
// =============================================================================
struct ModelSpec {
    // Identity
    std::string architecture;           // "llama", "qwen2", "phi3", etc.
    std::string name;                   // Model name
    std::string quant_type;             // "Q4_K_M", "Q8_0", etc.
    
    // Source tracking (for confidence calculation)
    MetadataSource source = MetadataSource::UNKNOWN;
    ModelType model_type = ModelType::DENSE;  // Default to dense

    // Dimensions
    uint64_t param_count = 0;           // Number of parameters
    uint32_t layers = 0;                // Number of transformer layers
    uint32_t embedding_dim = 0;         // Hidden/embedding dimension
    uint32_t attention_heads = 0;       // Number of attention heads
    uint32_t kv_heads = 0;              // Number of KV heads (for GQA)
    uint32_t head_dim = 0;              // Dimension per head (embedding_dim / attention_heads)
    uint32_t ffn_dim = 0;              // Feed-forward network dimension
    uint32_t context_length = 0;        // Maximum context length

    // Quantization
    double bits_per_weight = 0.0;       // Bits per weight (e.g., 4.85 for Q4_K_M)
    
    // MLA-specific fields (for DeepSeek/Kimi-class models)
    uint32_t kv_lora_rank = 0;           // MLA KV compression rank
    uint32_t qk_rope_head_dim = 0;      // MLA rope head dimension
    
    // MoE-specific fields
    bool is_moe = false;                 // true if architecture is MoE
    uint32_t expert_count = 0;           // Total routed experts per layer (N)
    uint32_t expert_used_count = 0;      // Active routed experts per token (k)
    uint32_t expert_shared_count = 0;    // Number of shared experts (always active)
    uint32_t expert_ffn_dim = 0;         // FFN intermediate dimension per expert
    float expert_weights_scale = 1.0f;   // Routing score scaling factor
    
    // MoE derived parameters (calculated from above)
    uint64_t params_per_routed_expert = 0;  // Parameters in one routed expert
    uint64_t params_routed_total = 0;        // Total routed expert parameters
    uint64_t params_shared = 0;              // Shared parameters (attention, embed, etc.)
    uint64_t params_active_per_token = 0;    // Active parameters per token
    
    // Raw GGUF metadata for debugging
    std::unordered_map<std::string, uint32_t> raw_kv_uint32;

    // Derived helpers
    uint32_t get_kv_ratio() const {
        return (kv_heads > 0) ? (attention_heads / kv_heads) : 1;
    }

    // Estimate parameter count if not available from GGUF
    void estimate_parameters() {
        if (param_count > 0 || layers == 0 || embedding_dim == 0) return;

        uint64_t emb = embedding_dim;
        uint64_t ffn = (ffn_dim > 0) ? ffn_dim : emb * 4;
        uint32_t kv_ratio = get_kv_ratio();

        uint64_t per_layer = (2 * emb * emb) + (2 * emb * emb / kv_ratio) + (3 * ffn * emb);
        param_count = static_cast<uint64_t>(layers) * per_layer + (2 * emb * emb);
    }

    // Calculate MoE parameter breakdown
    // Call after setting MoE fields from GGUF parser
    void calculate_moe_parameters() {
        if (!is_moe || expert_count == 0 || embedding_dim == 0 || layers == 0) return;
        
        // Each routed expert has Gate, Up, Down projections
        // params_per_expert = 3 × embedding_dim × expert_ffn_dim
        uint32_t ffn = (expert_ffn_dim > 0) ? expert_ffn_dim : embedding_dim * 4;
        params_per_routed_expert = 3ULL * embedding_dim * ffn;
        
        // Total routed expert parameters across all layers
        params_routed_total = static_cast<uint64_t>(expert_count) * params_per_routed_expert * layers;
        
        // Shared parameters: attention, embeddings, norms, router
        // Estimate from dense transformer formula (Q/K/V + O projections per layer)
        uint32_t kv_ratio = get_kv_ratio();
        uint64_t attention_per_layer = 2ULL * embedding_dim * embedding_dim  // Q, O
            + 2ULL * embedding_dim * embedding_dim / kv_ratio;               // K, V (GQA)
        uint64_t shared_per_layer = attention_per_layer;  // + norms (negligible)
        uint64_t embedding_params = 2ULL * embedding_dim * embedding_dim;    // input + output embeddings
        uint64_t estimated_shared = static_cast<uint64_t>(layers) * shared_per_layer + embedding_params;
        
        // Use GGUF param_count if it's larger than routed alone (reliable)
        // Otherwise use our estimate
        if (param_count > params_routed_total) {
            params_shared = param_count - params_routed_total;
        } else {
            // GGUF param_count may be wrong for MoE — use estimate
            params_shared = estimated_shared;
            // Recalculate total param_count from components
            param_count = params_shared + params_routed_total;
        }
        
        // Active parameters per token = shared + (k × per_expert × layers)
        params_active_per_token = params_shared +
            static_cast<uint64_t>(expert_used_count) * params_per_routed_expert * layers;
    }
};

// =============================================================================
// Strategy Configuration
// =============================================================================
struct StrategyConfig {
    PlacementStrategy placement = PlacementStrategy::FULL_GPU;
    uint32_t gpu_layers = 0;            // How many layers on GPU (0 = CPU-only, all = full GPU)
    uint32_t context_length = 0;        // Override context (0 = use model default)
    uint32_t batch_size = 1;            // Batch size (usually 1 for decode)
    uint32_t kv_quant_bits = 16;        // KV cache quantization (16, 8, or 4 bits)

    // Helpers
    bool is_full_gpu(uint32_t total_layers) const { return gpu_layers >= total_layers; }
    bool is_cpu_only() const { return gpu_layers == 0; }
    bool is_split(uint32_t total_layers) const { return gpu_layers > 0 && gpu_layers < total_layers; }
};

// =============================================================================
// Prediction Result
// =============================================================================
struct Prediction {
    // Memory predictions (bytes)
    uint64_t memory_total_bytes = 0;
    uint64_t memory_vram_bytes = 0;
    uint64_t memory_ram_bytes = 0;

    // Performance predictions
    double tokens_per_sec = 0.0;
    double ttft_ms = 0.0;
    double prompt_eval_tps = 0.0;
    
    // Context analysis
    uint32_t max_safe_context = 0;

    // Viability
    bool viable = false;
    PredictionConfidence confidence = PredictionConfidence::LOW;
    std::string confidence_reason;

    // Warnings
    std::string warnings;
};

// =============================================================================
// Strategy with Prediction (for method matrix output)
// =============================================================================
struct StrategyResult {
    StrategyConfig strategy;
    Prediction prediction;
    std::string description;  // Human-readable description
};

// =============================================================================
// BPW Lookup Table
// =============================================================================
inline double get_bits_per_weight(const std::string& quant_type) {
    if (quant_type == "F32") return 32.0;
    if (quant_type == "F16" || quant_type == "BF16") return 16.0;
    if (quant_type == "Q8_0") return 8.5;
    if (quant_type == "Q6_K") return 6.56;
    if (quant_type == "Q5_K_M") return 5.69;
    if (quant_type == "Q5_K_S") return 5.54;
    if (quant_type == "Q5_0") return 5.5;
    if (quant_type == "Q5_1") return 5.75;
    if (quant_type == "Q4_K_M") return 4.85;
    if (quant_type == "Q4_K_S") return 4.58;
    if (quant_type == "Q4_K_L") return 5.0;
    if (quant_type == "Q4_0") return 4.5;
    if (quant_type == "Q4_1") return 4.75;
    if (quant_type == "Q3_K_L") return 3.91;
    if (quant_type == "Q3_K_M") return 3.69;
    if (quant_type == "Q3_K_S") return 3.44;
    if (quant_type == "Q2_K") return 2.96;
    if (quant_type == "IQ4_NL") return 4.5;
    if (quant_type == "IQ4_XS") return 4.3;
    if (quant_type == "IQ3_XXS") return 3.06;
    if (quant_type == "IQ3_M") return 3.3;
    if (quant_type == "IQ2_XS") return 2.31;
    if (quant_type == "IQ2_XXS") return 2.06;
    if (quant_type == "IQ2_M") return 2.2;
    if (quant_type == "IQ1_S") return 1.5;
    return 0.0;
}

inline double get_bits_per_weight(uint32_t file_type) {
    switch (file_type) {
        case 0: return 32.0;    case 1: return 16.0;    case 2: return 4.5;
        case 3: return 4.75;    case 6: return 5.5;     case 7: return 5.75;
        case 8: return 8.5;     case 10: return 2.96;   case 11: return 3.44;
        case 12: return 3.69;   case 13: return 3.91;   case 14: return 4.58;
        case 15: return 4.85;   case 16: return 5.0;    case 17: return 5.54;
        case 18: return 5.69;   case 19: return 6.56;   case 20: return 2.06;
        case 21: return 2.31;   case 22: return 3.06;   case 23: return 1.5;
        case 24: return 4.5;    default: return 0.0;
    }
}
