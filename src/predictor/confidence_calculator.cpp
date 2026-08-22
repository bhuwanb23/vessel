#include "confidence_calculator.h"

// =============================================================================
// Confidence Band Logic (Phase F)
// =============================================================================
// From spec §8.4:
// HIGH: GGUF header, ≥5 calibration records, dense model, validated context
// MEDIUM: GGUF header, <5 calibration records, dense model, or long context
// LOW: config.json fallback, MoE model, unknown quant, incomplete hardware
// =============================================================================

bool is_hardware_complete(const HardwareSpec& hw) {
    // Hardware is complete if we have all essential measurements
    return hw.gpu_bandwidth_gbs > 0 &&
           hw.vram_total_bytes > 0 &&
           hw.ram_total_bytes > 0;
}

bool is_context_validated(uint32_t context_length) {
    // For Step 3, we validate up to 32K context
    // Beyond that, KV cache dominates and errors compound
    return context_length <= 32768;
}

ConfidenceResult calculate_confidence(
    const ModelSpec& model,
    const HardwareSpec& hw,
    uint32_t context_length,
    uint32_t calibration_records) {
    
    ConfidenceResult result;
    result.level = PredictionConfidence::LOW;
    result.reason = "";
    
    // Track reasons for confidence level
    bool has_gguf_header = (model.source == MetadataSource::GGUF_HEADER);
    bool is_dense = (model.model_type == ModelType::DENSE);
    bool has_calibration = (calibration_records >= 5);
    bool hw_complete = is_hardware_complete(hw);
    bool ctx_validated = is_context_validated(context_length);
    bool known_quant = (model.bits_per_weight > 0);
    
    // =========================================================================
    // Determine confidence level
    // =========================================================================
    
    // Check for LOW confidence conditions first (any triggers LOW)
    if (!has_gguf_header) {
        result.level = PredictionConfidence::LOW;
        result.reason = "Metadata from config.json fallback (not GGUF header)";
        return result;
    }
    
    if (model.model_type == ModelType::MOE) {
        result.level = PredictionConfidence::LOW;
        result.reason = "MoE model - expert offload behavior is complex";
        return result;
    }
    
    if (!known_quant) {
        result.level = PredictionConfidence::LOW;
        result.reason = "Unknown quantization type";
        return result;
    }
    
    if (!hw_complete) {
        result.level = PredictionConfidence::LOW;
        result.reason = "Incomplete hardware profile (missing bandwidth or VRAM info)";
        return result;
    }
    
    // Check for HIGH confidence conditions
    if (has_gguf_header && is_dense && has_calibration && ctx_validated) {
        result.level = PredictionConfidence::HIGH;
        result.reason = "GGUF header, dense model, calibrated hardware, validated context";
        return result;
    }
    
    // Check for MEDIUM confidence (default for most cases)
    result.level = PredictionConfidence::MEDIUM;
    
    // Build reason based on what's missing for HIGH
    if (!has_calibration) {
        result.reason = "Fewer than 5 calibration records for this hardware";
    } else if (!ctx_validated) {
        result.reason = "Long context (>32K) - KV cache errors compound";
    } else {
        result.reason = "Standard prediction conditions";
    }
    
    return result;
}

const char* confidence_to_string(PredictionConfidence level) {
    switch (level) {
        case PredictionConfidence::HIGH: return "High";
        case PredictionConfidence::MEDIUM: return "Medium";
        case PredictionConfidence::LOW: return "Low";
        default: return "Unknown";
    }
}
