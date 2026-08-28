# Step 3 — Predictor Math: Full Detailed Plan

---

## Goal of Step 3
Build a set of pure mathematical functions that take hardware specs and model metadata as input and produce predicted memory footprint, tokens/sec, and time-to-first-token as output. No hardware calls. No network requests. No file I/O. Just math. You will feed these functions hardcoded numbers from your Step 1 and Step 2 outputs, then validate the predictions against what llama.cpp actually reports when you run the same model manually. This is where the spec doc's formulas meet reality.

---

## What You Need Before Starting

### From Steps 0–2 (already done)
- Working build toolchain
- Hardware profiler that reports real numbers (Step 1)
- Metadata fetcher that extracts real model dimensions (Step 2)
- `baseline.txt` from Step 0 with your first manual llama.cpp run
- At least 2–3 GGUF models downloaded and tested

### What You Do NOT Need
- Any connection between Step 1 and Step 2 code (you are hardcoding inputs)
- Any execution of models from within your tool (you run llama.cpp manually for validation)
- Any networking or hardware APIs in this step's code

---

## Phase A — Architecture of the Predictor Module

### The Pure Function Contract
Your predictor is a single function (or a small set of functions) with this contract:

**Inputs:**
- A `HardwareSpec` struct (total RAM, free RAM, GPU VRAM, GPU bandwidth, RAM bandwidth, NVMe sequential/random speed)
- A `ModelMetadata` struct (param count, layers, embedding dim, attention heads, KV heads, head dim, FFN dim, context length, quant type, bits per weight)
- A `StrategyConfig` struct (placement: full-GPU / split / CPU-only, context length override, batch size)

**Outputs:**
- A `Prediction` struct containing:
  - `memory_total_bytes` (predicted total memory usage)
  - `memory_vram_bytes` (predicted VRAM usage)
  - `memory_ram_bytes` (predicted RAM usage)
  - `tokens_per_sec` (predicted decode speed)
  - `ttft_ms` (predicted time to first token)
  - `confidence` (high / medium / low enum)
  - `viable` (boolean — does this strategy fit at all?)

**Side effects:** None. No global state. No I/O. Given the same inputs, it always returns the same outputs. This makes it trivially testable.

### The Struct Definitions (Design These First)
Before writing any math, define the data structures. This forces you to think about exactly what information flows in and out.

**HardwareSpec:**
| Field | Type | Source (Step 1) | Unit |
|---|---|---|---|
| `ram_total_bytes` | uint64 | GlobalMemoryStatusEx | bytes |
| `ram_free_bytes` | uint64 | GlobalMemoryStatusEx | bytes |
| `ram_bandwidth_gbs` | double | memcpy benchmark | GB/s |
| `vram_total_bytes` | uint64 | NVML | bytes |
| `vram_free_bytes` | uint64 | NVML | bytes |
| `gpu_bandwidth_gbs` | double | NVML derived | GB/s |
| `gpu_tflops_fp16` | double | NVML or spec | TFLOPS |
| `nvme_sequential_mbs` | double | disk benchmark | MB/s |
| `nvme_random_4k_mbs` | double | disk benchmark | MB/s |

**ModelMetadata:**
| Field | Type | Source (Step 2) | Unit |
|---|---|---|---|
| `param_count` | uint64 | GGUF header | count |
| `layers` | uint32 | GGUF `block_count` | count |
| `embedding_dim` | uint32 | GGUF `embedding_length` | count |
| `attention_heads` | uint32 | GGUF `head_count` | count |
| `kv_heads` | uint32 | GGUF `head_count_kv` | count |
| `head_dim` | uint32 | derived: `embedding_dim / attention_heads` | count |
| `ffn_dim` | uint32 | GGUF `feed_forward_length` | count |
| `context_length` | uint32 | GGUF `context_length` | count |
| `bits_per_weight` | double | derived from quant type | bits |
| `quant_type` | string | GGUF `file_type` lookup | label |
| `architecture` | string | GGUF `general.architecture` | label |

**StrategyConfig:**
| Field | Type | Description |
|---|---|---|
| `placement` | enum | `FULL_GPU`, `GPU_CPU_SPLIT`, `CPU_ONLY` |
| `gpu_layers` | uint32 | How many layers on GPU (0 for CPU-only, all for full-GPU) |
| `context_length` | uint32 | Override context (e.g., 4096 or max-safe) |
| `batch_size` | uint32 | Usually 1 for decode |
| `kv_quant_bits` | uint32 | KV cache quantization (16, 8, or 4 — llama.cpp supports this) |

### Why This Structure Matters
By separating hardware, model, and strategy into three distinct inputs, you make it trivial to generate the "Method Matrix" in Step 4. You will loop over multiple `StrategyConfig` values for the same hardware + model combination, calling the predictor each time. The pure function design makes this loop clean and predictable.

---

## Phase B — Bits-Per-Weight Lookup

### The Mapping
Before you can calculate memory footprint, you need to convert the quantization type label (e.g., "Q4_K_M") into a numeric bits-per-weight value. This is not always a clean integer — k-quants use mixed precision across different tensor types, so the effective bits-per-weight is a fractional number.

### The Lookup Table
| Quant Type | Effective Bits/Weight | Notes |
|---|---|---|
| F32 | 32.0 | Full precision |
| F16 | 16.0 | Half precision |
| BF16 | 16.0 | Brain float |
| Q8_0 | 8.5 | Slightly above 8 due to scale factors |
| Q6_K | 6.56 | K-quant, mixed precision |
| Q5_K_M | 5.69 | K-quant, mixed precision |
| Q5_K_S | 5.54 | K-quant, slightly smaller |
| Q5_0 | 5.5 | Older quant |
| Q5_1 | 5.75 | Older quant with min |
| Q4_K_M | 4.85 | **Most common.** Higher than 4.0 due to scale/metadata overhead |
| Q4_K_S | 4.58 | Smaller variant |
| Q4_0 | 4.5 | Older quant |
| Q4_1 | 4.75 | Older quant with min |
| Q3_K_L | 3.91 | K-quant |
| Q3_K_M | 3.69 | K-quant |
| Q3_K_S | 3.44 | K-quant |
| Q2_K | 2.96 | K-quant |
| IQ4_NL | 4.5 | Imatrix quant |
| IQ3_XXS | 3.06 | Imatrix quant |
| IQ2_XS | 2.31 | Imatrix quant |
| IQ2_XXS | 2.06 | Imatrix quant |

### Why These Are Not Clean Integers
A "Q4" quantization does not mean every weight is exactly 4 bits. The quantization scheme stores:
- The quantized weights themselves (4 bits each)
- Per-block scale factors (typically FP16, adding overhead)
- Per-block minimum values (for some quants)
- Super-block metadata (for k-quants)

The effective bits-per-weight accounts for all of this overhead. The numbers above are empirically measured by dividing the actual GGUF file size by the parameter count.

### How to Derive These Yourself (Validation Check)
For any GGUF file you have downloaded:
`effective_bpw = (file_size_bytes × 8) / param_count`

This gives you the ground-truth bits-per-weight for that specific file. Compare it against the table above — they should match within ~5%. If they don't, the table value is wrong for that specific model (some models have more or fewer non-quantized tensors like embeddings and norms).

### Implementation
A simple `std::map<std::string, double>` or a `switch` on the `file_type` integer. Return 0.0 for unknown types and flag the prediction as low confidence.

---

## Phase C — Memory Footprint Formula

This is the most important formula and the easiest to validate, because llama.cpp reports actual memory usage when it loads a model.

### The Three Components

**Component 1: Weight Memory**

```
weight_bytes = param_count × bits_per_weight / 8
```

This is straightforward. For a 3.2B parameter model at Q4_K_M (4.85 bpw):
`weight_bytes = 3,212,749,824 × 4.85 / 8 = 1,947,739,581 bytes ≈ 1.81 GB`

**Component 2: KV Cache Memory**

```
kv_cache_bytes = 2 × layers × kv_heads × head_dim × context_length × batch_size × bytes_per_kv_element
```

Breaking this down:
- `2` — because you store both Keys and Values
- `layers` — each layer has its own KV cache
- `kv_heads` — number of KV heads (NOT attention heads — GQA means kv_heads < attention_heads)
- `head_dim` — dimension of each head
- `context_length` — number of tokens in the context window
- `batch_size` — usually 1 for interactive use
- `bytes_per_kv_element` — depends on KV cache precision:
  - FP16 KV cache: 2 bytes
  - Q8 KV cache: 1 byte
  - Q4 KV cache: 0.5 bytes

**Example:** Llama 3.2 3B at 4K context, FP16 KV cache:
`kv_cache_bytes = 2 × 28 × 8 × 128 × 4096 × 1 × 2 = 469,762,048 bytes ≈ 448 MB`

**Same model at 128K context:**
`kv_cache_bytes = 2 × 28 × 8 × 128 × 131072 × 1 × 2 = 15,032,385,536 bytes ≈ 14.0 GB`

This shows why context length matters enormously — the KV cache at 128K is 32× larger than at 4K, and can easily exceed the weight memory.

**MLA Attention Branch (DeepSeek/Kimi-class models):**
From §8.1: "MLA-style attention compresses KV cache to a small latent dimension instead of full per-head KV."

For MLA models, the KV cache formula changes to:
```
kv_cache_bytes = 2 × layers × kv_lora_rank × context_length × batch_size × bytes_per_kv_element
               + layers × qk_rope_head_dim × context_length × batch_size × bytes_per_kv_element
```

Where `kv_lora_rank` and `qk_rope_head_dim` are MLA-specific dimensions stored in the GGUF metadata (keys like `deepseek2.kv_lora_rank`).

**For MVP:** Detect MLA by checking if the architecture is `"deepseek2"` or `"deepseek_v2"`. If so, use the MLA formula. Otherwise, use the standard formula. If the MLA-specific metadata keys are missing, flag the prediction as low confidence.

**Component 3: Runtime Overhead**

```
runtime_overhead_bytes = backend_constant
```

This is the fudge factor. It accounts for:
- CUDA context initialization (~200-500 MB on NVIDIA)
- ggml compute buffers (scales with batch size and context)
- Driver allocations
- Fragmentation waste

**This constant is NOT derivable analytically.** You calibrate it empirically from your validation runs.

**Starting estimate:** 512 MB for CUDA backend, 128 MB for CPU-only. You will refine this in the validation phase.

### The Total

```
total_memory_bytes = weight_bytes + kv_cache_bytes + runtime_overhead_bytes
```

### The Placement Split (VRAM vs RAM)

For the `FULL_GPU` strategy:
- All weights and KV cache go to VRAM
- `vram_usage = total_memory_bytes`
- `ram_usage = runtime_overhead_bytes` (minimal — just the CPU-side runtime)
- `viable = (vram_usage <= vram_free_bytes)`

For the `GPU_CPU_SPLIT` strategy:
- `gpu_layers` layers go to VRAM, the rest stay in RAM
- `vram_usage = (gpu_layers / total_layers) × weight_bytes + (gpu_layers / total_layers) × kv_cache_bytes + gpu_overhead`
- `ram_usage = total_memory_bytes - vram_usage + cpu_overhead`
- `viable = (vram_usage <= vram_free_bytes) AND (ram_usage <= ram_free_bytes)`

For the `CPU_ONLY` strategy:
- Everything in RAM
- `vram_usage = 0`
- `ram_usage = total_memory_bytes`
- `viable = (ram_usage <= ram_free_bytes)`

### The "Max Safe Context" Calculation
Given a strategy and its memory budget, you can reverse the KV cache formula to find the maximum context length that fits:

```
available_for_kv = memory_budget - weight_bytes - runtime_overhead_bytes
max_context = available_for_kv / (2 × layers × kv_heads × head_dim × batch_size × bytes_per_kv_element)
```

This is the "safe default" context length mentioned in §3 item 6. Report it alongside the user-requested context length.

---

## Phase D — Decode Speed Formula (Tokens/Sec)

### The Core Insight
From §8.2: "Decode at batch=1 is memory-bandwidth-bound: every token requires reading all active weights once."

This means the GPU's compute units are mostly idle during decode. The bottleneck is how fast you can stream the model weights from memory (VRAM or RAM) into the compute units. Each token generation requires reading the entire active weight set once.

### The Formula

```
bytes_per_token = active_params × bits_per_weight / 8
tokens_per_sec = effective_bandwidth_gbs × 1,000,000,000 / bytes_per_token
```

Where `active_params` equals `param_count` for dense models (all parameters are active for every token). For MoE models, it would be smaller, but MoE is Phase 2.

### Effective Bandwidth by Placement Strategy

This is where the placement strategy dramatically affects speed:

**FULL_GPU:**
```
effective_bandwidth = gpu_bandwidth_gbs
```
Simplest case. You're reading weights from VRAM at GPU memory bandwidth.

**Example:** RTX 3080 at 760 GB/s, 3.2B model at Q4_K_M:
- `bytes_per_token = 3,212,749,824 × 4.85 / 8 = 1,947,739,581 bytes ≈ 1.95 GB`
- `tokens_per_sec = 760 / 1.95 ≈ 390 tokens/sec`

**CPU_ONLY:**
```
effective_bandwidth = ram_bandwidth_gbs
```
Reading from system RAM. Much slower than VRAM.

**Example:** DDR4-3200 dual channel ≈ 40 GB/s:
- `tokens_per_sec = 40 / 1.95 ≈ 20.5 tokens/sec`

**GPU_CPU_SPLIT (The Complex Case):**
From §8.2: "Sequential dependency — slower side dominates total time."

When some layers are on GPU and some on CPU, each token must pass through ALL layers sequentially. The GPU layers process at GPU bandwidth, the CPU layers at RAM bandwidth. The total time per token is the sum:

```
gpu_fraction = gpu_layers / total_layers
cpu_fraction = 1.0 - gpu_fraction

bytes_gpu = gpu_fraction × weight_bytes
bytes_cpu = cpu_fraction × weight_bytes

time_gpu_sec = bytes_gpu / (gpu_bandwidth_gbs × 1e9)
time_cpu_sec = bytes_cpu / (ram_bandwidth_gbs × 1e9)

tokens_per_sec = 1.0 / (time_gpu_sec + time_cpu_sec)
```

**Why this is not a simple weighted average:** Because the layers are sequential, not parallel. The CPU layers become a bottleneck that the GPU has to wait for. Even a few layers on CPU can dramatically reduce throughput.

**Example:** 3.2B model, 20 of 28 layers on GPU (RTX 3080), 8 on CPU (DDR4-3200):
- `gpu_fraction = 20/28 = 0.714`
- `cpu_fraction = 8/28 = 0.286`
- `bytes_gpu = 0.714 × 1.95 GB = 1.39 GB`
- `bytes_cpu = 0.286 × 1.95 GB = 0.56 GB`
- `time_gpu = 1.39 / 760 = 0.00183 sec`
- `time_cpu = 0.56 / 40 = 0.0140 sec`
- `tokens_per_sec = 1.0 / (0.00183 + 0.0140) = 63.2 tokens/sec`

Notice: even though 71% of layers are on the fast GPU, the speed is only 63 tokens/sec vs 390 for full GPU. The CPU layers dominate the time. This is the kind of insight your tool will surface for users.

### The KV Cache Bandwidth Addition (Minor Correction)
The formula above only accounts for reading weights. In reality, each token also requires reading and writing the KV cache. For short contexts, this is negligible. For long contexts (32K+), it becomes significant.

**Refined formula:**
```
kv_bytes_per_token = 2 × layers × kv_heads × head_dim × current_context × bytes_per_kv_element
total_bytes_per_token = weight_bytes + kv_bytes_per_token
tokens_per_sec = effective_bandwidth / total_bytes_per_token
```

For MVP, you can skip this refinement and add it later. The weight-only formula is accurate enough for contexts under ~8K. Flag predictions at longer contexts as slightly optimistic.

---

## Phase E — Prefill / TTFT Formula

### The Core Insight
From §8.3: "Prefill is compute-bound (whole prompt processed in parallel), not bandwidth-bound."

During prefill, the model processes all prompt tokens simultaneously in a large matrix multiplication. The bottleneck shifts from memory bandwidth to raw compute throughput (FLOPS).

### The Formula

```
flops_per_token = 2 × active_params   (the "2" accounts for multiply-accumulate)
total_flops = flops_per_token × prompt_tokens
ttft_seconds = total_flops / (device_compute_throughput × 1e12)
ttft_ms = ttft_seconds × 1000
```

### Device Compute Throughput

**For GPU (full offload or split with most layers on GPU):**
```
device_compute_throughput = gpu_tflops_fp16 × efficiency_factor
```

The `efficiency_factor` accounts for the fact that real-world throughput never reaches peak spec FLOPS. Typical values:
- Large batch, large model: 0.4–0.6 (40-60% of peak)
- Small batch, small model: 0.2–0.4 (20-40% of peak)
- **Starting estimate for MVP: 0.3** (calibrate later)

**For CPU-only:**
```
device_compute_throughput = cpu_vector_throughput_tflops
```

This depends on the CPU's SIMD capabilities:
- AVX2 (most modern CPUs): ~0.5–1.5 TFLOPS FP16 equivalent
- AVX-512 (Xeon, some Ryzen): ~1.5–3.0 TFLOPS
- AMX (Intel 4th gen Xeon): higher, but rare in consumer hardware
- **Starting estimate for MVP: 0.8 TFLOPS** for a modern desktop CPU (calibrate later)

### Why TTFT Is Harder to Predict Than Tokens/Sec
- It depends on **prompt length**, which varies per query (tokens/sec doesn't — it's per-token)
- It depends on **compute throughput**, which is harder to measure accurately than memory bandwidth
- It depends on **batch scheduling** inside llama.cpp, which has internal heuristics you don't control
- The efficiency factor is a bigger fudge factor than the runtime overhead in memory

**For MVP:** Report TTFT with a wider confidence band than tokens/sec. It's a rougher estimate.

---

## Phase F — Confidence Band Logic

### The Three Levels (From §8.4)

**High Confidence:**
- GGUF header was available (not config.json fallback)
- At least 5 calibration records exist for this hardware fingerprint (not applicable in Step 3 since calibration log is Step 7, but design the field now)
- Dense model (not MoE)
- Context length is within the range you've validated

**Medium Confidence:**
- GGUF header was available
- Fewer than 5 calibration records (first runs on new hardware)
- Dense model
- Context length is very long (>32K) where KV cache dominates and small errors compound

**Low Confidence:**
- config.json fallback (no GGUF header)
- MoE model (Phase 2)
- Unknown quantization type
- Hardware profile incomplete (e.g., bandwidth unknown)

### Implementation
A simple function that takes the metadata source flag and calibration count and returns an enum. For Step 3, since you have no calibration log yet, everything will be "medium" at best. That's fine — the mechanism exists and will activate in Step 7.

---

## Phase G — Hardcoding Inputs and First Validation

### The Validation Workflow

This is the most important part of Step 3. Do not skip it.

**Step G1: Pick a model you already have downloaded.** Use the same model from your Step 0 baseline.

**Step G2: Write down the hardware numbers from Step 1.** Copy them from your profiler's output into hardcoded constants in your test code.

**Step G3: Write down the model metadata from Step 2.** Copy them from your fetcher's output into hardcoded constants.

**Step G4: Call your predictor function with these hardcoded inputs.** Get the predicted memory, tokens/sec, and TTFT.

**Step G5: Run the model manually through llama.cpp.** Use the exact same configuration (same number of GPU layers, same context length, same quant). Note the actual reported numbers.

**Step G6: Compare.** Fill in the validation table below.

### Validation Table (Fill This Out for Each Model)

| Metric | Predicted | Actual (llama.cpp) | Delta | Acceptable? |
|---|---|---|---|---|
| Total memory (MB) | | | | Within 10% |
| VRAM usage (MB) | | | | Within 10% |
| RAM usage (MB) | | | | Within 15% |
| Tokens/sec (decode) | | | | Within 20% |
| TTFT (ms) | | | | Within 40% |

### What to Do When Numbers Don't Match

**Memory is off by a constant amount:**
- Adjust the `runtime_overhead_bytes` constant. This is exactly what it's for.
- If the delta is proportional to model size, your bits-per-weight value might be slightly wrong. Re-derive it from the actual file size.

**Memory is off by a percentage that scales with context:**
- Your KV cache formula has a bug. Check the `kv_heads` vs `attention_heads` distinction — using the wrong one is the most common error.

**Tokens/sec is way off (>30%):**
- Check your bandwidth number. Is the GPU bandwidth derived correctly? Is the RAM bandwidth from the benchmark realistic?
- Check if llama.cpp is actually using the number of GPU layers you think it is. Look at its startup output — it prints "offloaded X/Y layers to GPU."

**TTFT is way off (>50%):**
- Adjust the `efficiency_factor`. This is expected to need calibration.
- Check if your prompt length matches what you told the predictor.

### Iterate
Run this validation for at least 2–3 different models of different sizes. Adjust the constants until the predictions are consistently within the acceptable ranges. **Don't chase perfection** — the calibration log in Step 7 will handle ongoing refinement.

---

## Phase H — The Method Matrix Preview

### What This Looks Like (You Build the Full Version in Step 4)
Even though you're hardcoding inputs in Step 3, structure your test code to call the predictor multiple times with different `StrategyConfig` values. This previews what Step 4 will do automatically.

**Example test for a 7B Q4_K_M model on an RTX 3080 (10GB VRAM):**

| Strategy | GPU Layers | Context | Predicted Memory | Predicted tok/s | Viable? |
|---|---|---|---|---|---|
| Full GPU | 32 | 4K | 5.2 GB VRAM | 180 tok/s | ✅ |
| Full GPU | 32 | 32K | 9.1 GB VRAM | 145 tok/s | ✅ |
| Full GPU | 32 | 128K | 22.4 GB VRAM | — | ❌ (exceeds VRAM) |
| Split | 24 | 4K | 3.9 GB VRAM + 1.8 GB RAM | 95 tok/s | ✅ |
| Split | 24 | 128K | 3.9 GB VRAM + 19.0 GB RAM | 78 tok/s | ✅ |
| CPU Only | 0 | 4K | 5.7 GB RAM | 18 tok/s | ✅ |
| CPU Only | 0 | 128K | 22.9 GB RAM | 12 tok/s | ✅ (if 32GB RAM) |

This table is the core output of your entire tool. Step 4 will generate it automatically. Step 3 proves the numbers in it are roughly correct.

---

## Step 3 — Done Checklist

Before moving to Step 4, confirm every item:

- [ ] `HardwareSpec`, `ModelMetadata`, `StrategyConfig`, and `Prediction` structs are defined
- [ ] Bits-per-weight lookup table covers at least 15 common quant types
- [ ] Effective bits-per-weight values validated against actual GGUF file sizes (within 5%)
- [ ] Memory footprint formula implemented (weight + KV cache + overhead)
- [ ] KV cache formula uses `kv_heads`, not `attention_heads`
- [ ] MLA attention branch exists (even if basic) for DeepSeek-class models
- [ ] Placement split logic correctly divides memory between VRAM and RAM
- [ ] Max safe context calculation works (reverse KV cache formula)
- [ ] Decode speed formula implemented for all three placement strategies
- [ ] GPU+CPU split uses sequential time composition, not weighted average
- [ ] TTFT formula implemented with efficiency factor
- [ ] Confidence band logic returns correct levels based on input quality
- [ ] Predictor is a pure function with no I/O or global state
- [ ] Predicted memory matches llama.cpp actual within 10% for at least 2 models
- [ ] Predicted tokens/sec matches llama.cpp actual within 20% for at least 2 models
- [ ] Predicted TTFT matches llama.cpp actual within 40% for at least 2 models
- [ ] `runtime_overhead_bytes` constant calibrated from validation runs
- [ ] `efficiency_factor` constant calibrated from validation runs
- [ ] Non-viable strategies (memory exceeds hardware) are correctly flagged as `viable = false`
- [ ] Method matrix preview generates sensible-looking output for at least one model

---

## Common Failure Points at Step 3

| Problem | Likely Cause | Fix |
|---|---|---|
| Predicted memory is consistently 200-500 MB too low | `runtime_overhead_bytes` not accounting for CUDA context | Increase the overhead constant. Start at 512 MB for CUDA, calibrate from there. |
| Predicted memory is way too high for long contexts | Using `attention_heads` instead of `kv_heads` in KV cache formula | GQA models have fewer KV heads than attention heads. Double-check which field you're using. |
| Predicted memory doesn't scale with context | KV cache formula is using a hardcoded context instead of the strategy's context override | Ensure `StrategyConfig.context_length` is what's fed into the formula, not `ModelMetadata.context_length`. |
| Predicted tokens/sec is impossibly high | Using spec-sheet peak bandwidth instead of measured bandwidth | Use the bandwidth from your Step 1 profiler, not the NVIDIA spec page. |
| Predicted tokens/sec for split is nearly the same as full GPU | Using weighted average instead of sequential time composition | The split formula must sum the times, not average the bandwidths. The slow side dominates. |
| Predicted tokens/sec doesn't change with quantization | Bits-per-weight lookup returning the same value for all quants | Check that the lookup table maps each quant type to a distinct bpw value. |
| TTFT prediction is always near zero | Efficiency factor too high, or using peak TFLOPS | Reduce efficiency factor to 0.2-0.4 range. Use measured/calibrated TFLOPS, not spec sheet. |
| TTFT prediction is extremely high (>10 seconds for short prompts) | Efficiency factor too low, or TFLOPS value wrong | Check units — TFLOPS is 10^12 FLOPS. Make sure you're not mixing FLOPS and TFLOPS. |
| All strategies show `viable = false` | Using total VRAM/RAM instead of free VRAM/RAM | The viability check must compare against `vram_free_bytes`, not `vram_total_bytes`. |
| Bits-per-weight doesn't match file size | Model has large non-quantized tensors (embeddings, norms) | Some models keep embeddings in F16 even when the rest is Q4. The effective bpw will be higher than the table value. Derive from actual file size. |
| Overflow in memory calculation | Using 32-bit integers for byte counts | All memory calculations must use `uint64_t` or `double`. A 70B model at F16 is 140 GB — exceeds 32-bit range. |

---

## Time Estimate for Step 3
- Phase A (Struct design + architecture): **2–3 hours**
- Phase B (Bits-per-weight lookup): **1–2 hours** (including file-size validation)
- Phase C (Memory footprint formula): **3–4 hours** (including KV cache, MLA branch, placement split)
- Phase D (Decode speed formula): **3–4 hours** (especially the split strategy sequential composition)
- Phase E (TTFT formula): **2–3 hours**
- Phase F (Confidence bands): **1 hour**
- Phase G (Hardcoded validation against 2-3 models): **4–6 hours** (this is the iterative calibration loop — run llama.cpp, compare, adjust, repeat)
- Phase H (Method matrix preview): **1–2 hours**

**Total: 2–3 days, as originally estimated. The validation loop in Phase G is where most of the time goes — getting the constants right requires multiple rounds of compare-and-adjust.**