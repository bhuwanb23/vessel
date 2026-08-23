#include "moe_tensor_overrides.h"
#include <sstream>
#include <cmath>

// =============================================================================
// MoE Tensor Override Generation (Step 9, Phase C)
// =============================================================================
// For MoE models, we need to split expert tensors between GPU and CPU.
// llama.cpp's tensor_split parameter controls the ratio of weights on GPU vs CPU.
// For more precise control, we generate explicit tensor name patterns.
// =============================================================================

// GGUF tensor naming conventions for MoE models (llama-like arch):
//   blk.L.attn_*          — Attention (always GPU)
//   blk.L.ffn_norm*       — LayerNorm (always GPU)
//   blk.L.ffn_gate_inp.*  — Router/Gating (always GPU)
//   blk.L.ffn_gate_exps.* — Routed expert gate projections
//   blk.L.ffn_down_exps.* — Routed expert down projections
//   blk.L.ffn_up_exps.*   — Routed expert up projections
//   blk.L.ffn_gate.*      — Shared expert gate (if any)
//   blk.L.ffn_down.*      — Shared expert down (if any)
//   blk.L.ffn_up.*        — Shared expert up (if any)
//   token_embd.*          — Embeddings (always GPU)
//   output_norm.*         — Final norm (always GPU)
//   output.*              — Output projection (always GPU)

MoETensorOverrides generateMoETensorOverrides(const ModelSpec& model,
                                               const MoEPlacementPlan& plan) {
    MoETensorOverrides result;
    
    // Not needed for dense models
    if (!is_moe_model(model) || plan.variant_name.empty()) {
        return result;
    }
    
    result.needed = true;
    result.gpu_weight_bytes = plan.gpu_expert_vram_bytes + plan.shared_vram_bytes;
    result.cpu_weight_bytes = plan.cpu_expert_ram_bytes;
    
    // Calculate tensor_split ratio (GPU fraction of total weights)
    uint64_t total_weights = result.gpu_weight_bytes + result.cpu_weight_bytes;
    if (total_weights > 0) {
        result.tensor_split[0] = static_cast<float>(result.gpu_weight_bytes) / total_weights;
        result.tensor_split[1] = 1.0f - result.tensor_split[0];
    } else {
        result.tensor_split[0] = 0.0f;
        result.tensor_split[1] = 1.0f;
    }
    
    // Generate tensor override rules based on expert placement
    uint32_t experts_per_layer_gpu = plan.gpu_experts_per_layer;
    uint32_t experts_per_layer_cpu = plan.cpu_experts_per_layer;
    uint32_t total_experts = experts_per_layer_gpu + experts_per_layer_cpu;
    
    if (total_experts == 0) return result;
    
    // For each layer, the first N expert slots go to GPU, rest to CPU
    // GGUF expert tensors are indexed: blk.L.ffn_gate_exps.weight[0..N-1]
    // We can't do per-index overrides easily, so we use tensor_split as the
    // primary mechanism and add explicit patterns for clarity.
    
    // Override 1: Attention + Norms → always GPU
    result.overrides.push_back({
        "blk.\\d+\\.attn_.*",
        0,  // GPU
        "Attention projections"
    });
    
    result.overrides.push_back({
        "blk.\\d+\\.ffn_norm.*",
        0,  // GPU
        "FFN layer norms"
    });
    
    // Override 2: Router/Gating → always GPU (small, critical for routing)
    result.overrides.push_back({
        "blk.\\d+\\.ffn_gate_inp.*",
        0,  // GPU
        "Expert routing gate"
    });
    
    // Override 3: Embeddings + Output → always GPU
    result.overrides.push_back({
        "token_embd.*",
        0,  // GPU
        "Token embeddings"
    });
    
    result.overrides.push_back({
        "output_norm.*",
        0,  // GPU
        "Output norm"
    });
    
    result.overrides.push_back({
        "output.*",
        0,  // GPU
        "Output projection"
    });
    
    // Override 4: Routed experts → split based on plan
    // For expert-offload, the first E_gpu experts per layer go to GPU
    // The remaining experts go to CPU
    // 
    // NOTE: llama.cpp's tensor_split is the primary mechanism for this.
    // The per-tensor overrides below are for documentation and future
    // use with ggml_backend_sched_set_tensor_backend() if needed.
    
    if (experts_per_layer_gpu > 0 && experts_per_layer_gpu < total_experts) {
        // Partial offload: some experts on GPU, some on CPU
        // The tensor_split ratio handles the actual splitting
        result.overrides.push_back({
            "blk.\\d+\\.ffn_(gate|down|up)_exps.*",
            -1,  // Split (handled by tensor_split)
            "Routed experts (split: " + std::to_string(experts_per_layer_gpu) 
            + "/" + std::to_string(total_experts) + " on GPU)"
        });
    } else if (experts_per_layer_gpu == 0) {
        // All experts on CPU
        result.overrides.push_back({
            "blk.\\d+\\.ffn_(gate|down|up)_exps.*",
            1,  // CPU
            "Routed experts (all on CPU)"
        });
    } else {
        // All experts on GPU
        result.overrides.push_back({
            "blk.\\d+\\.ffn_(gate|down|up)_exps.*",
            0,  // GPU
            "Routed experts (all on GPU)"
        });
    }
    
    // Override 5: Shared experts (if any) → always GPU
    if (model.expert_shared_count > 0) {
        result.overrides.push_back({
            "blk.\\d+\\.ffn_shared.*",
            0,  // GPU
            "Shared experts"
        });
    }
    
    // Build summary
    std::ostringstream oss;
    oss << plan.variant_name << ": "
        << experts_per_layer_gpu << "/" << total_experts << " experts on GPU"
        << " (tensor_split=" << (result.tensor_split[0] * 100.0f) << "% GPU)";
    result.summary = oss.str();
    
    return result;
}

MoETensorOverrides generateMoETensorOverrides(const ModelSpec& model,
                                               const StrategyConfig& strategy,
                                               const HardwareSpec& hw) {
    // Compute placement plan
    MoEPlacementPlan plan = computeMoEPlacement(hw, model, strategy);
    
    // Generate overrides from plan
    return generateMoETensorOverrides(model, plan);
}
