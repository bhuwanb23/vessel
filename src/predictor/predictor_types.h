#pragma once

#include <cstdint>
#include <string>

// =============================================================================
// Input Struct 1: Hardware Specification (from Step 1 hardware profiler)
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

    // GPU Info
    std::string gpu_name;               // GPU model name (for display)
    uint32_t gpu_compute_major = 0;     // CUDA compute capability major
    uint32_t gpu_compute_minor = 0;     // CUDA compute capability minor
};

// =============================================================================
// Input Struct 2: Model Metadata (from Step 2 metadata fetcher)
// =============================================================================
// NOTE: This extends the existing ModelMetadata from gguf_parser.h
// with additional derived fields needed for prediction math.

struct ModelSpec {
    // Identity
    std::string architecture;           // "llama", "qwen2", "phi3", etc.
    std::string name;                   // Model name
    std::string quant_type;             // "Q4_K_M", "Q8_0", etc.

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
    double bits_per_weight = 0.0;       // Bits per weight (e.g., 4.5 for Q4_K_M)

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

        // Per-layer: Q/K/V projections + FFN gate/up/down
        // Q: emb x emb, K/V: emb x (emb/kv_ratio) each
        // FFN: 3 x emb x ffn (gate, up, down projections)
        uint64_t per_layer = (2 * emb * emb) + (2 * emb * emb / kv_ratio) + (3 * ffn * emb);
        param_count = static_cast<uint64_t>(layers) * per_layer + (2 * emb * emb);  // + embeddings
    }
};

// =============================================================================
// Input Struct 3: Strategy Configuration
// =============================================================================

// Placement strategy for model layers
enum class PlacementStrategy {
    FULL_GPU,       // All layers on GPU (fastest, needs enough VRAM)
    GPU_CPU_SPLIT,  // Some layers on GPU, rest on CPU (flexible)
    CPU_ONLY        // All layers on CPU (slowest, needs enough RAM)
};

// Configuration for a prediction run
struct StrategyConfig {
    PlacementStrategy placement = PlacementStrategy::FULL_GPU;
    uint32_t gpu_layers = 0;            // How many layers on GPU (0 = CPU-only, all = full GPU)
    uint32_t context_length = 0;        // Override context (0 = use model default)
    uint32_t batch_size = 1;            // Batch size (usually 1 for decode)
    uint32_t kv_quant_bits = 16;        // KV cache quantization (16, 8, or 4 bits)

    // Helper to determine actual placement
    bool is_full_gpu(uint32_t total_layers) const {
        return gpu_layers >= total_layers;
    }

    bool is_cpu_only() const {
        return gpu_layers == 0;
    }

    bool is_split(uint32_t total_layers) const {
        return gpu_layers > 0 && gpu_layers < total_layers;
    }
};

// =============================================================================
// Output Struct: Prediction Result
// =============================================================================

// Confidence level for the prediction
enum class PredictionConfidence {
    HIGH,       // Formula well-validated, inputs are accurate
    MEDIUM,     // Formula approximate, or some inputs estimated
    LOW         // Significant uncertainty in inputs or formula
};

// The prediction result
struct Prediction {
    // Memory predictions (bytes)
    uint64_t memory_total_bytes = 0;    // Total memory needed (weights + KV cache + overhead)
    uint64_t memory_vram_bytes = 0;     // VRAM needed
    uint64_t memory_ram_bytes = 0;      // RAM needed (for CPU-offloaded layers)

    // Performance predictions
    double tokens_per_sec = 0.0;        // Predicted decode speed (tokens/sec)
    double ttft_ms = 0.0;               // Predicted time to first token (milliseconds)
    double prompt_eval_tps = 0.0;       // Prompt evaluation speed (tokens/sec)

    // Viability
    bool viable = false;                // Does this strategy fit in available memory?
    PredictionConfidence confidence = PredictionConfidence::LOW;

    // Warnings
    std::string warnings;               // Any warnings about the prediction
};

// =============================================================================
// Convenience: Convert bits_per_weight from quant type string
// =============================================================================

// Bits per weight lookup table (empirically measured from file_size / param_count)
// These values account for scale factors, super-block metadata, and other overhead.
inline double get_bits_per_weight(const std::string& quant_type) {
    // Full precision
    if (quant_type == "F32") return 32.0;
    if (quant_type == "F16" || quant_type == "BF16") return 16.0;
    
    // 8-bit
    if (quant_type == "Q8_0") return 8.5;
    
    // 6-bit k-quant
    if (quant_type == "Q6_K") return 6.56;
    
    // 5-bit quants
    if (quant_type == "Q5_K_M") return 5.69;
    if (quant_type == "Q5_K_S") return 5.54;
    if (quant_type == "Q5_0") return 5.5;
    if (quant_type == "Q5_1") return 5.75;
    
    // 4-bit quants (most common)
    if (quant_type == "Q4_K_M") return 4.85;  // Most popular quantization
    if (quant_type == "Q4_K_S") return 4.58;
    if (quant_type == "Q4_K_L") return 5.0;
    if (quant_type == "Q4_0") return 4.5;
    if (quant_type == "Q4_1") return 4.75;
    
    // 3-bit quants
    if (quant_type == "Q3_K_L") return 3.91;
    if (quant_type == "Q3_K_M") return 3.69;
    if (quant_type == "Q3_K_S") return 3.44;
    
    // 2-bit quants
    if (quant_type == "Q2_K") return 2.96;
    
    // Imatrix quants
    if (quant_type == "IQ4_NL") return 4.5;
    if (quant_type == "IQ4_XS") return 4.3;
    if (quant_type == "IQ3_XXS") return 3.06;
    if (quant_type == "IQ3_M") return 3.3;
    if (quant_type == "IQ2_XS") return 2.31;
    if (quant_type == "IQ2_XXS") return 2.06;
    if (quant_type == "IQ2_M") return 2.2;
    if (quant_type == "IQ1_S") return 1.5;
    
    return 0.0;  // Unknown - return 0 to flag low confidence
}

// Get bits per weight from file_type integer (empirically measured values)
inline double get_bits_per_weight(uint32_t file_type) {
    switch (file_type) {
        case 0: return 32.0;    // F32
        case 1: return 16.0;    // F16
        case 2: return 4.5;     // Q4_0
        case 3: return 4.75;    // Q4_1
        case 6: return 5.5;     // Q5_0
        case 7: return 5.75;    // Q5_1
        case 8: return 8.5;     // Q8_0
        case 10: return 2.96;   // Q2_K
        case 11: return 3.44;   // Q3_K_S
        case 12: return 3.69;   // Q3_K_M
        case 13: return 3.91;   // Q3_K_L
        case 14: return 4.58;   // Q4_K_S
        case 15: return 4.85;   // Q4_K_M (most popular)
        case 16: return 5.0;    // Q4_K_L
        case 17: return 5.54;   // Q5_K_S
        case 18: return 5.69;   // Q5_K_M
        case 19: return 6.56;   // Q6_K
        case 20: return 2.06;   // IQ2_XXS
        case 21: return 2.31;   // IQ2_XS
        case 22: return 3.06;   // IQ3_XXS
        case 23: return 1.5;    // IQ1_S
        case 24: return 4.5;    // IQ4_NL
        default: return 0.0;    // Unknown - return 0 to flag low confidence
    }
}
