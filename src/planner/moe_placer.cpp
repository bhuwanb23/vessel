#include "moe_placer.h"
#include "../predictor/memory_predictor.h"
#include <algorithm>
#include <cmath>

// =============================================================================
// MoE Expert Placement Engine (Step 9, Phase B)
// =============================================================================
// Core logic: given VRAM/RAM budget, determine how many routed experts
// can fit on GPU while keeping shared parameters + KV cache in VRAM.
// =============================================================================

static MoEPlacementPlan compute_moe_plan(const HardwareSpec& hw, const ModelSpec& model,
                                          uint32_t context_length, uint32_t kv_quant_bits,
                                          bool force_full_vram, bool force_cpu_only) {
    MoEPlacementPlan plan;
    
    // Guard: not an MoE model
    if (!is_moe_model(model)) {
        plan.viable = false;
        plan.reason = "Not an MoE model";
        return plan;
    }
    
    double bytes_per_param = model.bits_per_weight / 8.0;
    
    // 1. Calculate memory requirements
    // Shared parameters: attention, embeddings, norms, router
    uint64_t shared_weight_bytes = static_cast<uint64_t>(model.params_shared * bytes_per_param);
    
    // One routed expert
    uint64_t one_expert_bytes = model.params_per_routed_expert *
        static_cast<uint64_t>(std::ceil(bytes_per_param));
    // More precise: use actual bpw for expert weights
    one_expert_bytes = static_cast<uint64_t>(model.params_per_routed_expert * bytes_per_param);
    
    // KV cache for the given context
    uint64_t kv_cache_bytes = predict_kv_cache_memory(model, context_length, kv_quant_bits, 1);
    
    // CUDA overhead
    uint64_t cuda_overhead = 512ULL * 1024 * 1024;  // 512 MB
    
    // Store per-expert size
    plan.bytes_per_expert = one_expert_bytes;
    
    // 2. CPU-Only: everything in RAM
    if (force_cpu_only) {
        plan.viable = (shared_weight_bytes + kv_cache_bytes + cuda_overhead +
                       static_cast<uint64_t>(model.expert_count) * one_expert_bytes * model.layers)
                      <= hw.ram_free_bytes;
        plan.reason = plan.viable ? "" : "RAM insufficient for all weights + KV cache";
        plan.shared_vram_bytes = 0;
        plan.kv_vram_bytes = 0;
        plan.kv_ram_bytes = kv_cache_bytes;
        plan.gpu_experts_per_layer = 0;
        plan.cpu_experts_per_layer = model.expert_count;
        plan.gpu_expert_vram_bytes = 0;
        plan.cpu_expert_ram_bytes = static_cast<uint64_t>(model.expert_count) * one_expert_bytes * model.layers;
        plan.total_vram_bytes = 0;
        plan.total_ram_bytes = shared_weight_bytes + kv_cache_bytes + plan.cpu_expert_ram_bytes;
        plan.variant_name = "MoE-CPU-Only";
        return plan;
    }
    
    // 3. Check if Shared Parameters + KV Cache fit in VRAM
    uint64_t min_vram_required = shared_weight_bytes + kv_cache_bytes + cuda_overhead;
    
    if (hw.vram_free_bytes < min_vram_required) {
        plan.viable = false;
        plan.reason = "Shared weights + KV Cache exceed VRAM";
        plan.variant_name = "MoE-Expert-Offload";
        return plan;
    }
    
    // 4. Compute remaining VRAM available for routed experts
    uint64_t vram_for_experts = hw.vram_free_bytes - min_vram_required;
    
    // 5. Calculate how many expert instances can fit in GPU VRAM
    //    Total expert slots across all layers = layers × experts_per_layer
    uint64_t total_expert_slots = static_cast<uint64_t>(model.layers) * model.expert_count;
    
    if (one_expert_bytes == 0) {
        plan.viable = false;
        plan.reason = "Expert size is zero (missing FFN dim?)";
        plan.variant_name = "MoE-Expert-Offload";
        return plan;
    }
    
    uint64_t gpu_expert_capacity = vram_for_experts / one_expert_bytes;
    
    // Full VRAM: try to fit ALL experts
    if (force_full_vram) {
        if (gpu_expert_capacity >= total_expert_slots) {
            // All experts fit on GPU
            plan.viable = true;
            plan.shared_vram_bytes = shared_weight_bytes;
            plan.kv_vram_bytes = kv_cache_bytes;
            plan.kv_ram_bytes = 0;
            plan.gpu_experts_per_layer = model.expert_count;
            plan.cpu_experts_per_layer = 0;
            plan.gpu_expert_vram_bytes = static_cast<uint64_t>(model.expert_count) * one_expert_bytes * model.layers;
            plan.cpu_expert_ram_bytes = 0;
            plan.total_vram_bytes = min_vram_required + plan.gpu_expert_vram_bytes;
            plan.total_ram_bytes = 0;
            plan.variant_name = "MoE-Full-VRAM";
        } else {
            // Not enough VRAM for all experts — fall back to expert offload
            plan.viable = false;
            plan.reason = "Not enough VRAM for all routed experts";
            plan.variant_name = "MoE-Full-VRAM";
        }
        return plan;
    }
    
    // 6. Expert-Offload (flagship): fit as many experts as possible
    uint64_t gpu_expert_count = std::min(total_expert_slots, gpu_expert_capacity);
    
    // Distribute GPU experts evenly across layers
    uint32_t experts_per_layer_gpu = static_cast<uint32_t>(gpu_expert_count / model.layers);
    
    // Ensure at least 0 and at most expert_count
    experts_per_layer_gpu = std::min(experts_per_layer_gpu, model.expert_count);
    
    uint32_t experts_per_layer_cpu = model.expert_count - experts_per_layer_gpu;
    
    // Calculate actual memory usage
    uint64_t actual_gpu_experts = static_cast<uint64_t>(experts_per_layer_gpu) * model.layers;
    uint64_t actual_cpu_experts = static_cast<uint64_t>(experts_per_layer_cpu) * model.layers;
    
    plan.viable = true;
    plan.shared_vram_bytes = shared_weight_bytes;
    plan.kv_vram_bytes = kv_cache_bytes;
    plan.kv_ram_bytes = 0;  // KV cache stays in VRAM
    plan.gpu_experts_per_layer = experts_per_layer_gpu;
    plan.cpu_experts_per_layer = experts_per_layer_cpu;
    plan.gpu_expert_vram_bytes = actual_gpu_experts * one_expert_bytes;
    plan.cpu_expert_ram_bytes = actual_cpu_experts * one_expert_bytes;
    plan.total_vram_bytes = min_vram_required + plan.gpu_expert_vram_bytes;
    plan.total_ram_bytes = shared_weight_bytes + plan.cpu_expert_ram_bytes;  // shared also in RAM for CPU experts
    plan.variant_name = "MoE-Expert-Offload";
    
    return plan;
}

// =============================================================================
// Public API
// =============================================================================

MoEPlacementPlan computeMoEPlacement(const HardwareSpec& hw, const ModelSpec& model,
                                      const StrategyConfig& strategy) {
    uint32_t ctx = strategy.context_length > 0 ? strategy.context_length : model.context_length;
    if (ctx == 0) ctx = 4096;
    
    bool force_full_vram = (strategy.placement == PlacementStrategy::FULL_GPU);
    bool force_cpu_only = (strategy.placement == PlacementStrategy::CPU_ONLY);
    
    return compute_moe_plan(hw, model, ctx, strategy.kv_quant_bits,
                            force_full_vram, force_cpu_only);
}

MoEPlacementPlan computeMoEFullVRAM(const HardwareSpec& hw, const ModelSpec& model,
                                     uint32_t context_length, uint32_t kv_quant_bits) {
    return compute_moe_plan(hw, model, context_length, kv_quant_bits, true, false);
}

MoEPlacementPlan computeMoEExpertOffload(const HardwareSpec& hw, const ModelSpec& model,
                                          uint32_t context_length, uint32_t kv_quant_bits) {
    return compute_moe_plan(hw, model, context_length, kv_quant_bits, false, false);
}

MoEPlacementPlan computeMoECPUOnly(const HardwareSpec& hw, const ModelSpec& model,
                                    uint32_t context_length, uint32_t kv_quant_bits) {
    return compute_moe_plan(hw, model, context_length, kv_quant_bits, false, true);
}
