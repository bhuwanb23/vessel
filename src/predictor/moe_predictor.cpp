#include "moe_predictor.h"
#include "../../include/moe_placer.h"
#include "memory_predictor.h"
#include <algorithm>
#include <cmath>

// =============================================================================
// MoE Predictor — Range-Based Speed Prediction (Step 9, Phase D)
// =============================================================================
// Formulas from spec:
//
// B_active = (params_shared + k × params_routed_expert_one) × bpw / 8
//
// Best case (100% GPU hit):
//   tok/s_best = 1 / (B_active / (BW_vram × 1e9))
//
// Worst case (0% GPU hit):
//   tok/s_worst = 1 / (B_gpu_part / (BW_vram × 1e9) + B_cpu_part / (BW_ram × 1e9))
//
// Expected (uniform routing):
//   p_gpu = E_gpu / N
//   tok/s_expected = 1 / (B_gpu_avg / (BW_vram × 1e9) + B_cpu_avg / (BW_ram × 1e9))
// =============================================================================

MoEPrediction predictMoERange(const HardwareSpec& hw, const ModelSpec& model,
                               const MoEPlacementPlan& plan, uint32_t kv_quant_bits) {
    MoEPrediction result;
    
    // Guard: not an MoE model or no placement plan
    if (!is_moe_model(model) || plan.variant_name.empty() || !plan.viable) {
        return result;
    }
    
    result.valid = true;
    
    double bpw = model.bits_per_weight;
    double bytes_per_param = bpw / 8.0;
    
    // Active parameters per token
    double params_shared = static_cast<double>(model.params_shared);
    double params_expert_one = static_cast<double>(model.params_per_routed_expert);
    uint32_t k = model.expert_used_count;
    uint32_t N = model.expert_count;
    uint32_t E_gpu = plan.gpu_experts_per_layer;
    
    // B_active: total bytes read per token (weights only)
    result.active_bytes_per_token = (params_shared + k * params_expert_one) * bytes_per_param;
    
    // Add KV cache bytes per token
    double kv_bytes_per_token = predict_kv_bytes_per_token(model, kv_quant_bits);
    result.active_bytes_per_token += kv_bytes_per_token;
    
    // =========================================================================
    // Routing Probability: P(single token hits GPU expert)
    // =========================================================================
    // Phase 1 (Step 9): Assume uniform routing (E_gpu / N)
    // Phase 2: Integrate calibration log hit-rate data to weigh lower-indexed
    //          experts as "hotter" (asymmetric routing from training data).
    // =========================================================================
    result.p_gpu = (N > 0) ? static_cast<double>(E_gpu) / N : 0.0;
    result.k_gpu_expected = k * result.p_gpu;
    result.k_cpu_expected = k * (1.0 - result.p_gpu);
    
    // Bandwidth values
    double bw_vram_gbs = hw.gpu_bandwidth_gbs;
    double bw_ram_gbs = hw.ram_bandwidth_gbs;
    if (bw_vram_gbs <= 0) bw_vram_gbs = 448.0;  // RTX 5060 default
    if (bw_ram_gbs <= 0) bw_ram_gbs = 40.0;     // DDR5-5600 default
    
    // Efficiency factors (calibrated for dense, apply to MoE)
    double gpu_efficiency = 0.27;  // GPU memory-bandwidth efficiency
    double cpu_efficiency = 0.80;  // CPU near-theoretical efficiency
    
    // =========================================================================
    // Best Case: All k active experts hit GPU (100% hit rate)
    // =========================================================================
    // All weights read from GPU at full bandwidth
    {
        double time_per_token_sec = result.active_bytes_per_token / 
            (bw_vram_gbs * 1e9 * gpu_efficiency);
        result.time_per_token_best_ms = time_per_token_sec * 1000.0;
        result.tok_s_best = (time_per_token_sec > 0) ? (1.0 / time_per_token_sec) : 0.0;
    }
    
    // =========================================================================
    // Worst Case: All k active experts hit CPU (0% hit rate)
    // =========================================================================
    // GPU reads: shared params only
    // CPU reads: k × expert params
    {
        double B_gpu_part = params_shared * bytes_per_param + kv_bytes_per_token;
        double B_cpu_part = k * params_expert_one * bytes_per_param;
        
        result.gpu_part_bytes = B_gpu_part;
        result.cpu_part_bytes = B_cpu_part;
        
        double time_gpu = B_gpu_part / (bw_vram_gbs * 1e9 * gpu_efficiency);
        double time_cpu = B_cpu_part / (bw_ram_gbs * 1e9 * cpu_efficiency);
        double time_per_token_sec = time_gpu + time_cpu;
        
        result.time_per_token_worst_ms = time_per_token_sec * 1000.0;
        result.tok_s_worst = (time_per_token_sec > 0) ? (1.0 / time_per_token_sec) : 0.0;
    }
    
    // =========================================================================
    // Expected: Uniform routing probability across N experts
    // =========================================================================
    // Expected active GPU experts: k_gpu = k × p_gpu
    // Expected active CPU experts: k_cpu = k × (1 - p_gpu)
    {
        double k_gpu = result.k_gpu_expected;
        double k_cpu = result.k_cpu_expected;
        
        double B_gpu_avg = (params_shared + k_gpu * params_expert_one) * bytes_per_param 
                          + kv_bytes_per_token;
        double B_cpu_avg = (k_cpu * params_expert_one) * bytes_per_param;
        
        double time_gpu = B_gpu_avg / (bw_vram_gbs * 1e9 * gpu_efficiency);
        double time_cpu = B_cpu_avg / (bw_ram_gbs * 1e9 * cpu_efficiency);
        double time_per_token_sec = time_gpu + time_cpu;
        
        result.time_per_token_expected_ms = time_per_token_sec * 1000.0;
        result.tok_s_expected = (time_per_token_sec > 0) ? (1.0 / time_per_token_sec) : 0.0;
    }
    
    return result;
}
