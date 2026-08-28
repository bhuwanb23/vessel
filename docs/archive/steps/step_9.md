# Step 9 — MoE Expert-Offload Integration: Full Detailed Plan

---

## Goal of Step 9
Extend the **Metadata Fetcher**, **Predictor**, **Ranker**, and **Executor** to natively support Mixture-of-Experts (MoE) architectures (Mixtral 8x7B, Qwen1.5/2.5-MoE, DeepSeek-V2/V3/R1, DBRX). Instead of blindly offloading whole layers, the tool dynamically partitions model weights: keeping all **shared parameters** (attention, embeddings, norms, shared experts) and a subset of **"hot" routed experts** in GPU VRAM, while offloading cold routed experts to system RAM (or NVMe). The Predictor is upgraded to output **uncertainty-bounded performance ranges** (best-case hit vs. worst-case miss), providing a true competitive edge over standard layer-offloading tools.

---

## What You Need Before Starting

### From Steps 1–8 (already done and working)
- **Step 1:** Hardware Profiler supplying live free VRAM, free RAM, GPU memory bandwidth ($\text{BW}_{\text{vram}}$), and system RAM bandwidth ($\text{BW}_{\text{ram}}$).
- **Step 2:** Metadata Fetcher reading GGUF header KV metadata via range requests.
- **Step 3 & 4:** Predictor and Method Matrix Generator producing strategy grids.
- **Step 6:** Executor linking directly against `llama.cpp` C API libraries (`llama.h`, `ggml.h`).
- **Step 8:** Download Manager capable of fetching multi-shard MoE GGUF models.

### Theoretical Foundation
In dense Transformers, every parameter is executed for every token. In MoE Transformers:
- Total parameters ($P_{\text{total}}$) are large (e.g., 47B for Mixtral 8x7B, 671B for DeepSeek-V3).
- Active parameters per token ($P_{\text{active}}$) are small (e.g., 13B for Mixtral, 37B for DeepSeek-V3).
- **Shared Parameters ($P_{\text{shared}}$):** Embeddings, Attention, LayerNorms, Router/Gating networks, and Shared Experts (executed for 100% of tokens).
- **Routed Experts ($P_{\text{routed}}$):** $N$ total experts per layer, of which only $k$ are activated per token ($k \ll N$).

Because routing varies per token, keeping all $N$ routed experts in VRAM is unnecessary if VRAM is constrained. Offloading cold routed experts to system RAM avoids saturating PCIe with non-active weight transfers, provided shared parameters remain GPU-resident.

---

## Phase A — MoE Metadata Extraction

### What It Does
Extend `fetcher.cpp` (Step 2) to extract MoE-specific architecture parameters directly from the GGUF header without downloading tensor data.

### GGUF Metadata Keys to Parse

For a detected architecture (`general.architecture` $\in$ `{"mixtral", "qwen2_moe", "dbrx", "deepseek2", "deepseek3"}`):

| Key | Type | Description | Example (Mixtral 8x7B) | Example (DeepSeek-V3) |
|---|---|---|---|---|
| `{arch}.expert_count` | `UINT32` | Total routed experts per layer ($N$) | `8` | `256` |
| `{arch}.expert_used_count` | `UINT32` | Active routed experts per token ($k$) | `2` | `8` |
| `{arch}.expert_shared_count` | `UINT32` | Number of shared experts (always active) | `0` | `1` |
| `{arch}.expert_weights_scale` | `FLOAT32` | Routing score scaling factor | `1.0` | `1.0` |
| `{arch}.expert_feed_forward_length` | `UINT32` | FFN intermediate dimension per expert | `14336` | `2048` |

### Derived Parameter Calculation

From these fields, calculate the parameter breakdown:

$$\text{params}_{\text{routed\_expert\_one}} = 3 \times \text{layers} \times \text{embedding\_dim} \times \text{expert\_feed\_forward\_length}$$
*(Accounts for Gate, Up, and Down projection matrices per expert)*

$$\text{params}_{\text{routed\_total}} = N \times \text{params}_{\text{routed\_expert\_one}}$$

$$\text{params}_{\text{shared}} = P_{\text{total}} - \text{params}_{\text{routed\_total}}$$

$$\text{params}_{\text{active\_per\_token}} = \text{params}_{\text{shared}} + (k \times \text{params}_{\text{routed\_expert\_one}})$$

---

## Phase B — The Expert Placement Engine

### What It Does
Given a hardware profile (VRAM free, RAM free) and an MoE model, compute the exact tensor-by-tensor placement configuration that maximizes inference throughput.

### The Placement Logic Algorithm

```
function computeMoEPlacement(HardwareSpec hw, ModelMetadata model, StrategyConfig strategy):
    
    // 1. Calculate memory requirements
    bytes_per_param = model.bits_per_weight / 8.0
    
    shared_weight_bytes = model.params_shared * bytes_per_param
    one_expert_bytes    = model.params_routed_expert_one * bytes_per_param
    kv_cache_bytes      = calculateKVCacheBytes(model, strategy.context, strategy.kv_quant_bits)
    cuda_overhead       = 512 * 1024 * 1024 // 512 MB base CUDA overhead
    
    // 2. Check if Shared Parameters + KV Cache fit in VRAM
    min_vram_required = shared_weight_bytes + kv_cache_bytes + cuda_overhead
    
    if hw.vram_free < min_vram_required:
        // Hard constraint failure: Cannot run Expert-Offload strategy.
        // Fall back to CPU-Only MoE or Layer-wise Offload.
        return PlacementPlan{ viable: false, reason: "Shared weights + KV Cache exceed VRAM" }
    
    // 3. Compute remaining VRAM available for routed experts
    vram_for_experts = hw.vram_free - min_vram_required
    
    // 4. Calculate total expert slots across all layers that can fit in GPU VRAM
    total_expert_slots = model.layers * model.expert_count
    gpu_expert_capacity = floor(vram_for_experts / one_expert_bytes)
    gpu_expert_count    = min(total_expert_slots, gpu_expert_capacity)
    
    // 5. Distribute GPU experts evenly across layers
    experts_per_layer_gpu = floor(gpu_expert_count / model.layers)
    
    return PlacementPlan{
        viable: true,
        shared_in_vram: true,
        gpu_experts_per_layer: experts_per_layer_gpu, // e.g., 2 out of 8 experts on GPU
        cpu_experts_per_layer: model.expert_count - experts_per_layer_gpu,
        vram_used: min_vram_required + (experts_per_layer_gpu * model.layers * one_expert_bytes),
        ram_used: (model.expert_count - experts_per_layer_gpu) * model.layers * one_expert_bytes
    }
```

### Strategy Variants Generated for the Method Matrix

Step 4's matrix generator is extended to yield three specific MoE placement strategies:

1. **MoE-Full-VRAM:** All shared parameters + all $N$ routed experts in VRAM. (Fastest, highest memory requirement).
2. **MoE-Expert-Offload (Flagship):** All shared parameters + $E_{\text{gpu}}$ experts in VRAM; remaining $N - E_{\text{gpu}}$ experts in system RAM.
3. **MoE-CPU-Only:** All weights in system RAM. (Slowest, minimum VRAM requirement).

---

## Phase C — Programmatic Tensor Override Generation

### How `llama.cpp` Handles Tensor Placement
`llama.cpp` assigns default backends (CUDA or CPU) to tensors based on `n_gpu_layers`. To override placement on a per-tensor level (e.g., forcing routed expert #3 to CPU while keeping attention on GPU), we use `llama.cpp`'s tensor placement controls via `llama_model_params`.

### GGUF Tensor Naming Conventions

The placement engine constructs regular expression rules or explicit tensor override definitions based on the GGUF spec:

| Tensor Target | GGUF Tensor Name Pattern | Assigned Backend |
|---|---|---|
| Attention & Norms | `blk.L.attn_*`, `blk.L.ffn_norm*` | `GPU` |
| Shared Experts | `blk.L.ffn_gate_inp`, `blk.L.ffn_shared*` | `GPU` |
| GPU-Resident Experts | `blk.L.ffn_gate_exps.weight` (Indices $0 \dots E_{\text{gpu}}-1$) | `GPU` |
| CPU-Resident Experts | `blk.L.ffn_gate_exps.weight` (Indices $E_{\text{gpu}} \dots N-1$) | `CPU` |

### Interfacing with `llama.cpp` C API

In `executor.cpp`, configure `llama_model_params` during model loading:

```cpp
llama_model_params model_params = llama_model_default_params();

// Offload non-expert layers (attention, norms, shared experts) to GPU
model_params.n_gpu_layers = model_metadata.layers;

// Use tensor_buft_overrides or custom buffer predicate to split expert tensors
// Define tensor split ratio across backends:
// Index 0 = GPU, Index 1 = CPU
float tensor_split[LLAMA_MAX_DEVICES] = {0};
// Calculate percentage of weights residing on GPU
tensor_split[0] = (float)gpu_weight_bytes / (float)total_weight_bytes;

model_params.tensor_split = tensor_split;
```

If exact per-tensor override is required, pass tensor override rules into `llama.cpp`'s backend scheduler setup (`ggml_backend_sched`).

---

## Phase D — MoE Predictor Math (Range-Based Output)

### Why Point Estimates Fail for MoE
In dense models, every token requires reading $100\%$ of active weights. In MoE expert-offload:
- If a token routes to GPU-resident experts $\rightarrow$ throughput is bounded by GPU bandwidth ($\text{BW}_{\text{vram}}$).
- If a token routes to CPU-resident experts $\rightarrow$ throughput is bounded by RAM bandwidth ($\text{BW}_{\text{ram}}$) or PCIe transfer rate.

Because prompt content determines routing, token generation speed varies dynamically. The Predictor MUST report a **[Worst Case, Best Case] range**.

### Decode Speed Formulas

#### 1. Active Bytes Transferred Per Token ($B_{\text{active}}$)
$$B_{\text{active}} = \left( \text{params}_{\text{shared}} + k \times \text{params}_{\text{routed\_expert\_one}} \right) \times \frac{\text{bits\_per\_weight}}{8}$$

#### 2. Best-Case Speed ($\text{tok/s}_{\text{best}}$)
Occurs when all $k$ active tokens hit GPU-resident experts (Hit Rate = 100%):
$$t_{\text{best}} = \frac{B_{\text{active}}}{\text{BW}_{\text{vram}} \times 10^9}$$
$$\text{tok/s}_{\text{best}} = \frac{1}{t_{\text{best}}}$$

#### 3. Worst-Case Speed ($\text{tok/s}_{\text{worst}}$)
Occurs when all $k$ active tokens hit CPU-resident experts (Hit Rate = 0%):
$$B_{\text{gpu\_part}} = \text{params}_{\text{shared}} \times \frac{\text{bits\_per\_weight}}{8}$$
$$B_{\text{cpu\_part}} = \left( k \times \text{params}_{\text{routed\_expert\_one}} \right) \times \frac{\text{bits\_per\_weight}}{8}$$

$$t_{\text{worst}} = \frac{B_{\text{gpu\_part}}}{\text{BW}_{\text{vram}} \times 10^9} + \frac{B_{\text{cpu\_part}}}{\text{BW}_{\text{ram}} \times 10^9}$$
$$\text{tok/s}_{\text{worst}} = \frac{1}{t_{\text{worst}}}$$

#### 4. Expected Average Speed ($\text{tok/s}_{\text{expected}}$)
Assuming uniform routing probability across all $N$ experts, the probability $p_{\text{gpu}}$ of a single routed token hitting a GPU expert is:
$$p_{\text{gpu}} = \frac{E_{\text{gpu}}}{N}$$

Expected active GPU expert params: $k_{\text{gpu}} = k \times p_{\text{gpu}}$
Expected active CPU expert params: $k_{\text{cpu}} = k \times (1 - p_{\text{gpu}})$

$$B_{\text{gpu\_avg}} = \left( \text{params}_{\text{shared}} + k_{\text{gpu}} \times \text{params}_{\text{routed\_expert\_one}} \right) \times \frac{\text{bits\_per\_weight}}{8}$$
$$B_{\text{cpu\_avg}} = \left( k_{\text{cpu}} \times \text{params}_{\text{routed\_expert\_one}} \right) \times \frac{\text{bits\_per\_weight}}{8}$$

$$t_{\text{expected}} = \frac{B_{\text{gpu\_avg}}}{\text{BW}_{\text{vram}} \times 10^9} + \frac{B_{\text{cpu\_avg}}}{\text{BW}_{\text{ram}} \times 10^9}$$
$$\text{tok/s}_{\text{expected}} = \frac{1}{t_{\text{expected}}}$$

### Ranker Display Format for MoE
In `ranker.cpp`, MoE predictions format the output column as a range with an explicit low-confidence indicator:

```
 #  Placement      GPU Experts  Context  VRAM     RAM     tok/s (Range)    Confidence
─── ────────────── ─────────── ──────── ──────── ─────── ─────────────── ───────────
 1  MoE-Full-VRAM  8/8         4K       24.2 GB  0.8 GB  ~112 tok/s      MEDIUM
 2  MoE-Offload    3/8         4K       11.4 GB  14.2GB  ~38 - 84 tok/s  LOW (MoE)
 3  MoE-CPU-Only   0/8         4K       0.2 GB   28.1GB  ~12 tok/s       MEDIUM
```

---

## Phase E — Execution & Telemetry Integration

### Live Sampling Modifications
During Step 6 execution of an MoE model, `executor.cpp` tracks additional metrics via `NVML` and host memory counters:

1. **VRAM Allocation Verification:** Query `nvmlDeviceGetMemoryInfo()` immediately post-load to verify that actual VRAM usage matches $B_{\text{shared}} + B_{\text{gpu\_experts}}$.
2. **PCIe Throughput Monitoring:** Sample PCIe RX/TX throughput via `nvmlDeviceGetPcieThroughput()`. High PCIe throughput during decode confirms active expert streaming from system RAM.
3. **Execution Sampling:** Log variance in token-to-token generation time ($\sigma_{\text{tok/s}}$). Higher variance indicates dynamic switching between GPU hits and CPU misses.

### Calibration Log Record (Step 7 Compatibility)
Appended log entries include MoE specific metadata:

```json
{
  "hardware_fingerprint": "i7-12700K|RTX 3080|32GB|Samsung 980 PRO",
  "model_id": "mistralai/Mixtral-8x7B-Instruct-v0.1-GGUF/Q4_K_M",
  "strategy": {
    "backend": "llama.cpp",
    "placement": "MoE-Expert-Offload",
    "gpu_experts_per_layer": 3,
    "total_experts_per_layer": 8,
    "context": 4096
  },
  "predicted": {
    "tokens_per_sec_min": 38.2,
    "tokens_per_sec_max": 84.1,
    "tokens_per_sec_expected": 56.4,
    "confidence": "LOW_MOE"
  },
  "actual": {
    "tokens_per_sec_avg": 51.8,
    "tokens_per_sec_min": 32.1,
    "tokens_per_sec_max": 79.4,
    "pcie_throughput_mbs": 3200,
    "peak_vram_bytes": 12240656384,
    "peak_ram_bytes": 15247132800
  }
}
```

---

## Phase F — Validation & Testing Protocol

### Test Setup
- **Target Model:** `Mixtral-8x7B-Instruct-v0.1-GGUF` (Q4_K_M, ~26 GB total size).
- **Target Hardware:** Single GPU with 10 GB or 12 GB VRAM (e.g., RTX 3080 / RTX 4070) + 32 GB System RAM.
  *(This setup forces expert offloading, as the model cannot fit entirely in VRAM)*.

### Test Cases

| ID | Scenario | Expected Behavior |
|---|---|---|
| **T9.1** | Metadata Extraction | Parser extracts $N=8$, $k=2$, and distinguishes routed vs shared params from GGUF header. |
| **T9.2** | Placement Calculation | On a 10 GB VRAM GPU, placement engine selects 2 or 3 experts on GPU, remaining 5 or 6 on CPU. |
| **T9.3** | Tensor Override | `llama.cpp` loads without OOM. `nvidia-smi` confirms ~9.2 GB VRAM usage. |
| **T9.4** | Prediction Range | Observed actual average tok/s falls strictly between predicted `tok/s_worst` and `tok/s_best`. |
| **T9.5** | Comparison vs Naive Split | MoE-Expert-Offload yields $\ge 1.8\times$ higher tok/s compared to naive layer-wise split using equivalent VRAM. |

---

## Phase G — Edge Cases & Safety Rules

### 1. DeepSeek-V3 / R1 Class Architectures (256 Experts, 8 Active)
- **Problem:** DeepSeek-V3 uses 256 routed experts per layer + 1 shared expert. Parameter scale is massive (671B total).
- **Rule:** If total experts $N > 32$, group experts into block chunks (e.g., 16 experts per chunk) for VRAM placement calculations to avoid exceeding `llama.cpp` tensor command line string limits.

### 2. Extreme Memory Constraint (Shared Params Exceed VRAM)
- **Problem:** Shared parameters alone (e.g., 30 GB for massive models) exceed total GPU VRAM.
- **Rule:** Set `viable = false` for MoE-Expert-Offload. Force strategy generation to standard **CPU-Only** or **MoE-Layer-Streaming**. Do not attempt partial expert offloading if shared parameters cannot sit in VRAM.

### 3. Asymmetric Expert Routing (Non-Uniform Activation)
- **Problem:** Certain experts (e.g., Expert 0, Expert 1) are activated more frequently by common English tokens.
- **Rule:** For Phase 1 of Step 9, assume uniform routing ($E_{\text{gpu}} / N$). In Phase 2, integrate calibration log hit-rate data to weigh lower-indexed experts as "hotter".

---

## Step 9 — Done Checklist

- [ ] `fetcher.cpp` extracts $N$ (`expert_count`), $k$ (`expert_used_count`), and shared expert parameters from GGUF header.
- [ ] `matrix.cpp` implements `computeMoEPlacement()` algorithm.
- [ ] Predictor math calculates $B_{\text{active}}$, $\text{tok/s}_{\text{worst}}$, $\text{tok/s}_{\text{best}}$, and $\text{tok/s}_{\text{expected}}$.
- [ ] Ranker outputs MoE speed predictions as explicit ranges (`min - max tok/s`).
- [ ] `executor.cpp` constructs tensor split parameters and passes them to `llama_model_load_from_file()`.
- [ ] Live sampler records PCIe throughput via NVML during MoE runs.
- [ ] Calibration log correctly appends MoE strategy fields and actual range metrics.
- [ ] Validated on Mixtral-8x7B: MoE-Expert-Offload runs without OOM on 10GB/12GB GPU.
- [ ] MoE-Expert-Offload demonstrably beats naive layer-offloading throughput on identical hardware.

---

## Common Failure Points

| Failure Mode | Cause | Fix |
|---|---|---|
| **CUDA OOM on Model Load** | CUDA overhead or KV cache calculation ignored shared expert size. | Include `params_shared` and 512 MB CUDA buffer in baseline VRAM allocation *before* computing available expert slots. |
| **Zero Performance Gain over CPU** | All attention layers fell back to CPU due to incorrect layer offload count. | Ensure `n_gpu_layers` is set to `total_layers` so all Attention, Norm, and Shared FFN layers remain on GPU. |
| **`llama.cpp` Parse Failure** | Misformed tensor override string or invalid `tensor_split` array bounds. | Validate `tensor_split` array size against `LLAMA_MAX_DEVICES` (typically 16). |

---

## Time Estimate

| Sub-Task | Description | Duration |
|---|---|---|
| **Phase A** | GGUF MoE Header Parsing & Parameter Derivation | 1 Day |
| **Phase B & C** | Placement Engine & Tensor Override Generator | 2–3 Days |
| **Phase D** | MoE Range Predictor Math & Ranker UI Updates | 1 Day |
| **Phase E** | Executor Integration & PCIe Telemetry | 1–2 Days |
| **Phase F & G** | Mixtral Testing, Validation, and Edge-Case Hardening | 1–2 Days |

**Total Estimated Duration:** **6–9 Days**