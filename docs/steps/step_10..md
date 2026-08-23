# Step 10 — Hot/Cold CPU-GPU Offload for Dense Models (+ Layer-Streaming Fallback): Full Detailed Plan

---

## Goal of Step 10
For dense models too large to fit on GPU, introduce two new placement strategies that go beyond the naive layer-split from Steps 3–6:

1. **Hot/Cold Neuron Offload (PowerInfer-style):** Profile which neurons activate frequently across diverse inputs, keep those "hot" neurons on GPU, compute the rarely-activated "cold" neurons on CPU. This exploits the power-law distribution of neuron activations to keep the critical path on fast hardware while the long tail runs on slower hardware — without shuttling the entire model across PCIe every token.

2. **Layer-Streaming Fallback (AirLLM-style):** When even hot/cold splitting doesn't fit (extreme model-to-hardware mismatch), stream one transformer layer from disk at a time, compute it, discard it, load the next. This makes an otherwise-impossible model technically runnable with minimal resident memory, at the cost of extreme slowness.

These are the final placement strategies in the method matrix. After this step, the tool genuinely "finds a way" for every model-hardware combination, with honest labeling of the performance cost.

---

## Why This Step Is Different From Everything Before It

Steps 1–9 operated at the **layer** or **expert** granularity. You moved whole layers or whole expert blocks between GPU and CPU. The mechanisms were llama.cpp flags (`n_gpu_layers`, `--override-tensor`) that already existed.

Step 10 operates at the **neuron** granularity — within a single layer's FFN, some neurons are hot and some are cold. No llama.cpp flag does this. You are entering the territory of custom operator dispatch, which means modifying or extending the compute path inside ggml/llama.cpp. This is why the time estimate is 2–3 weeks and why it was explicitly deferred to the end of Phase 2.

---

## What You Need Before Starting

### From Steps 1–9 (must be solid and validated)
- **Step 1:** Hardware profiler with VRAM, RAM, RAM bandwidth, NVMe sequential and random-read speeds. All four bandwidth numbers feed placement decisions in this step.
- **Step 3:** Predictor formulas for dense models. You will add new formulas, not replace existing ones.
- **Step 4:** Method matrix generator. You will add new placement enum values.
- **Step 6:** Executor linked against llama.cpp as a library. You understand the full inference lifecycle (load → context → prefill → decode → cleanup). You will be extending this lifecycle, not replacing it.
- **Step 7:** Calibration log. Hot/cold and layer-streaming runs need their own calibration entries, isolated from dense GPU and MoE records.
- **Step 9:** MoE expert-offload. Completed and validated. Do not start Step 10 until Step 9 is stable — they touch different model classes (dense vs MoE) but both modify the executor's compute path, and debugging two new compute paths simultaneously is a recipe for weeks of confusion.

### New Test Assets (Hard Blockers)
You need **one dense model that does NOT fit on your GPU**:

| Scenario | Example | Why |
|---|---|---|
| **Moderate overfit** (model 1.5–2× VRAM) | Llama-3.1-8B Q8 on 8GB VRAM, or 13B Q4 on 10GB | Exercises hot/cold split with a meaningful hot set |
| **Extreme overfit** (model 3–5× VRAM) | 30B+ Q4 on 8GB VRAM | Exercises layer-streaming fallback |
| **No meaningful GPU** (integrated graphics or very old card) | 7B Q4 on 2GB VRAM | Edge case: hot set is tiny, most work on CPU |

If your test machine has a 24GB GPU (RTX 3090/4090), you may need to artificially constrain VRAM or test with a very large model (70B Q4 ≈ 40GB). The hot/cold path is meaningless if everything fits.

### What You Are NOT Building
- **A new inference engine.** You are extending llama.cpp's compute path with sparse neuron operators, not replacing ggml.
- **Dynamic runtime neuron caching.** PowerInfer's original paper describes a predictor that dynamically decides hot/cold per token. The practical implementations (including PowerInfer's own fork) use a **static** hot set determined by offline profiling. You will use the static approach. Dynamic prediction adds enormous complexity for marginal speedup on most workloads.
- **A general-purpose sparse compute library.** You need exactly one sparse operation: sparse FFN (GELU/SiLU activation with neuron-level masking). Don't generalize beyond what the model needs.

---

## Phase A — Deep Understanding of PowerInfer's Mechanism

### A1. The Core Insight (Read the Paper Before Writing Code)

**Paper:** arXiv:2312.12456 (PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU)

**Key findings you need to internalize:**

1. **Neuron activation follows a power law.** In a typical FFN layer with 11,008 neurons (Llama-7B), roughly 10–20% of neurons activate on nearly every input ("hot"). The remaining 80–90% activate only for specific inputs ("cold"). This distribution is consistent across diverse prompts.

2. **Hot neurons dominate the output.** Because they activate on almost every token, hot neurons contribute the majority of the FFN output magnitude. Cold neurons provide refinement for specific inputs.

3. **The split is stable.** The same neurons are hot across different prompts, topics, and languages. This means a one-time offline profiling pass produces a hot set that generalizes. You do not need to re-profile per conversation.

4. **The speedup comes from avoiding PCIe transfer of the full model.** In a naive layer-split, every token requires reading the CPU-resident layers across PCIe. In hot/cold split, the hot neurons (which do most of the work) are already on GPU. The cold neurons are computed on CPU using data already in RAM, and only their small contribution crosses PCIe back to GPU.

5. **Reported performance:** ~8 tok/s for 30B–70B class models on i9-13900K + RTX 4090. This is usable for interactive chat, not just batch processing. The speedup over naive CPU-only is 5–10×.

### A2. How PowerInfer's Fork Actually Works (Study Before Implementing)

PowerInfer is a fork of llama.cpp. The key modifications are:

**Offline Profiling Pass:**
- Run the model on a diverse prompt set (e.g., 1,000+ prompts from general corpora)
- For each FFN layer, record which neurons activate (output > 0 after ReLU/SiLU) per token
- Aggregate across all tokens and prompts
- Rank neurons by activation frequency
- The top N neurons (sized to fit VRAM budget) become the "hot" set
- Store the hot neuron indices per layer as a binary mask file

**Online Sparse FFN Computation:**
- During inference, the FFN computation is split:
  - **GPU path:** Compute only the hot neurons using the GPU-resident hot weight subset. This is a smaller GEMM (general matrix multiply) because the weight matrix has fewer rows.
  - **CPU path:** Compute only the cold neurons that actually activated for this specific token (determined by a lightweight predictor or by checking the pre-activation values). This is a sparse operation on CPU-resident cold weights.
  - **Combine:** Sum the GPU and CPU partial results to get the full FFN output.

**The Critical Detail — Neuron-Level Sparsity in FFN:**
A standard FFN in a transformer is:
```
output = down_proj( activation( up_proj(x) ) * gate_proj(x) )
```

Where `up_proj` and `gate_proj` are weight matrices of shape `[hidden_dim, ffn_dim]` and `down_proj` is `[ffn_dim, hidden_dim]`. The `ffn_dim` is the neuron count (e.g., 11,008 for Llama-7B).

Hot/cold splitting means:
- `up_proj` is split into `up_proj_hot` (shape `[hidden_dim, n_hot]`) on GPU and `up_proj_cold` (shape `[hidden_dim, n_cold]`) on CPU
- Same for `gate_proj` and `down_proj`
- The activation function (SiLU/GELU) naturally zeros out non-activated neurons, so the CPU only needs to compute neurons that will actually be non-zero

### A3. What You Must Decide Before Coding

| Decision | Options | Recommendation |
|---|---|---|
| **Integration approach** | (a) Merge PowerInfer's fork patches into your llama.cpp build, or (b) implement sparse FFN from scratch against ggml | **(a) Study and adapt the fork.** The original authors spent months tuning the sparse operators. Reimplementing from the paper alone will take longer and produce slower code. |
| **Profiling prompt set** | (a) Bundled general-purpose prompts, or (b) user-provided prompts | **(a) Bundled.** The hot set is stable across prompts. Ship 500–1,000 diverse prompts (Wikipedia snippets, code, conversation, Q&A). Allow user override for domain-specific optimization. |
| **Hot set storage** | (a) Per-model binary mask file, or (b) embedded in the GGUF | **(a) Separate file.** Don't modify the GGUF. Store `model_name.hot_neurons.bin` alongside the model. |
| **Granularity** | (a) Per-layer hot sets, or (b) global hot set | **(a) Per-layer.** Different layers have different activation patterns. A global set wastes VRAM on layers where the hot neurons are different. |
| **Activation function awareness** | (a) ReLU (sparse by nature), or (b) SiLU/Swish (Llama, most modern models) | **Handle both.** ReLU zeros are exact. SiLU is never exactly zero but is near-zero for negative inputs — you need a threshold. |

### A4. The Honest Scope Assessment

This is the hardest engineering in the project. The difficulty breakdown:

| Component | Difficulty | Why |
|---|---|---|
| Profiling infrastructure | Medium | Run model on many prompts, record activations. Conceptually simple, logistically tedious. |
| Hot/cold weight splitting | Medium | Slice weight matrices by neuron index. Data manipulation, not algorithmic. |
| Sparse FFN operator (GPU) | **High** | Custom GEMM with reduced dimension. Must be fast enough to beat the naive baseline. |
| Sparse FFN operator (CPU) | **High** | Sparse activation + GEMM on CPU. Must skip cold neurons efficiently, not compute-and-discard. |
| GPU-CPU result combination | Medium | Sum two partial FFN outputs per layer per token. Synchronization overhead. |
| Layer-streaming fallback | Medium | Sequential layer loading. Conceptually simple, I/O-bound. |
| Predictor formulas | Low-Medium | New bandwidth/compute model for hot/cold split. |
| Integration with existing pipeline | Low | New enum values, new matrix rows, new calibration entries. |

**Total honest assessment:** 2–3 weeks of focused work. Do not treat this as a "quick add-on." If you are time-constrained, ship Steps 8–9 first and leave Step 10 as a future release.

---

## Phase B — Offline Neuron Activation Profiling Infrastructure

### B1. The Profiling Pipeline

This is a one-time per-model process that produces the hot neuron mask.

**Input:**
- A GGUF model file (must be loadable by your Executor from Step 6)
- A diverse prompt set (500–1,000 prompts, ~50–200 tokens each)
- A target hot-set VRAM budget (from Step 1's profiler: how much VRAM is available for hot neurons after attention + KV + overhead)

**Output:**
- A binary mask file per model: `model_name.hot_neurons.bin`
- Contents: for each FFN layer, a bit vector of length `ffn_dim` where 1 = hot, 0 = cold
- Metadata: profiling prompt count, total tokens processed, hot neuron count per layer, activation threshold used

**The Profiling Sequence:**

1. **Load the model** using your existing Executor (Step 6). For profiling, load the entire model on CPU (or GPU+CPU split if it partially fits) — you need all neurons accessible to record their activations.

2. **Instrument the FFN layers.** This is the invasive part. You need to hook into the FFN computation to record per-neuron activation values. Two approaches:
   - **(Preferred):** Modify the ggml compute graph to insert a "recording" node after the activation function in each FFN layer. This node copies the post-activation values to a host buffer without affecting computation.
   - **(Alternative):** Run the model layer-by-layer in a custom loop, extracting the intermediate tensor after activation at each FFN. Slower but less invasive.

3. **Run all profiling prompts.** For each prompt:
   - Tokenize and run prefill
   - Record the post-activation values for every FFN layer, every token
   - Accumulate into a per-layer, per-neuron activation frequency counter

4. **Aggregate.** After all prompts:
   - For each layer, for each neuron index `i`:
     - `activation_freq[i] = (number of tokens where neuron i activated) / total_tokens`
   - "Activated" means: post-activation value > threshold
     - For ReLU: threshold = 0.0 (exact)
     - For SiLU: threshold = 0.01 or similar (near-zero). The paper uses a small positive threshold because SiLU never reaches exactly zero.

5. **Rank and select.** For each layer:
   - Sort neurons by `activation_freq` descending
   - Select the top `n_hot[layer]` neurons such that the total hot weight size fits in the VRAM budget
   - The VRAM budget per layer: `vram_for_hot / n_layers` (approximately, with adjustment for layers that have more or fewer hot neurons)

6. **Write the mask file.** Binary format:
   ```
   Header: magic (4 bytes) + version (4 bytes) + n_layers (4 bytes) + ffn_dim (4 bytes)
   Per layer: n_hot (4 bytes) + bit vector (ffn_dim / 8 bytes, rounded up)
   ```

### B2. The Profiling Prompt Set

**Where to get prompts:**
- Wikipedia random article summaries (diverse topics)
- OpenWebText or similar general corpus snippets
- Code snippets (Python, JavaScript, C++)
- Conversational turns (Q&A pairs)
- Instruction-following prompts

**Key requirement:** Diversity, not domain specificity. PowerInfer's finding is that the hot set generalizes across domains. If you profile only on code, the hot set will be code-biased and perform worse on prose.

**Size:** 500–1,000 prompts, each 50–200 tokens. Total: ~50,000–200,000 tokens. This is enough for stable activation statistics without taking hours to profile.

**Profiling time expectation:** For a 13B model on CPU, ~1–4 hours. For a 70B model, potentially 8–12 hours. This is a one-time cost per model. Display a progress bar and estimated completion time.

### B3. Hot Set Stability Validation

Before trusting the mask, verify PowerInfer's stability claim:

1. Profile on 500 prompts → get hot set A
2. Profile on a different 500 prompts → get hot set B
3. Compute overlap: `|A ∩ B| / |A|`
4. **Pass criterion:** overlap > 85%. If the hot set changes wildly between prompt sets, something is wrong with your profiling methodology (threshold too low, prompts too narrow, or the model architecture doesn't exhibit power-law activation).

### B4. The Profiling CLI Command

Add a new subcommand to your tool:

```
llm-planner --profile-neurons --model-path C:\models\llama-13b-q4.gguf --prompts bundled --vram-budget 6GB
```

This runs the profiling pipeline and produces the mask file. It is separate from the normal prediction/execution flow.

### B5. When to Re-Profile

| Event | Re-profile? |
|---|---|
| Same model, same quant | No |
| Same model, different quant | Yes — activation patterns shift slightly with quantization |
| Different model | Yes |
| New llama.cpp version | No (unless the model architecture handling changed) |
| User wants domain-specific optimization | Optional — profile on domain-specific prompts for a tailored hot set |

---

## Phase C — Hot/Cold Weight Splitting

### C1. What Gets Split

Only the FFN weight matrices. Attention, embeddings, norms, and the output head stay intact (and on GPU if possible).

For each FFN layer, three weight matrices are split by neuron index:

| Matrix | Shape (standard) | Hot subset | Cold subset |
|---|---|---|---|
| `gate_proj` (W_gate) | `[ffn_dim, hidden_dim]` | `[n_hot, hidden_dim]` | `[n_cold, hidden_dim]` |
| `up_proj` (W_up) | `[ffn_dim, hidden_dim]` | `[n_hot, hidden_dim]` | `[n_cold, hidden_dim]` |
| `down_proj` (W_down) | `[hidden_dim, ffn_dim]` | `[hidden_dim, n_hot]` | `[hidden_dim, n_cold]` |

**Note the asymmetry:** `gate_proj` and `up_proj` are split along the output dimension (rows), while `down_proj` is split along the input dimension (columns). This matters for the GEMM implementation.

### C2. How to Split

**Option A: Pre-split at load time (Recommended for MVP)**
When loading the model, read the full weight tensors from the GGUF, then slice them into hot and cold subsets based on the mask file. Store hot subsets in GPU memory and cold subsets in CPU memory.

**Pros:** Simple, clean separation. The inference loop sees two smaller matrices instead of one large one.
**Cons:** Requires loading the full model into RAM first, then splitting. For a 40GB model on a 32GB RAM machine, this is a problem.

**Option B: Split at the GGUF level (Advanced)**
Create a modified GGUF file where the FFN tensors are already split into hot and cold tensors. The Executor loads them directly into the correct memory space.

**Pros:** No intermediate full-model load. Works on memory-constrained machines.
**Cons:** Requires a GGUF rewriting tool. Adds a preprocessing step. The modified file is model-specific and hot-set-specific.

**Recommendation for MVP:** Option A for models that fit in RAM. Option B as a Phase 3 optimization for extreme cases. If the model doesn't fit in RAM for splitting, the layer-streaming fallback (Phase E) is the appropriate path anyway.

### C3. Memory Layout After Split

**GPU memory (VRAM):**
- All attention weights (all layers)
- All norm weights
- Embedding + output head
- Hot FFN weights (gate_hot, up_hot, down_hot per layer)
- KV cache
- CUDA context + runtime overhead

**CPU memory (RAM):**
- Cold FFN weights (gate_cold, up_cold, down_cold per layer)
- CPU compute buffers

**The VRAM savings:** If 15% of neurons are hot and FFN is ~60% of total model parameters, then hot FFN ≈ 9% of total params. The GPU holds attention (~30%) + hot FFN (~9%) + KV + overhead ≈ 45–50% of the model, instead of 100%. This is how a 13B model fits on an 8GB GPU.

---

## Phase D — Sparse FFN Execution (The Hardest Part)

### D1. The Standard FFN Computation (Baseline)

For reference, the standard SwiGLU FFN (Llama-style) for a single token:

```
x: [hidden_dim]                          // input to FFN
g = x @ W_gate.T                         // [ffn_dim]  — gate projection
u = x @ W_up.T                           // [ffn_dim]  — up projection
a = SiLU(g) * u                          // [ffn_dim]  — gated activation
y = a @ W_down.T                         // [hidden_dim] — down projection
```

In a naive layer-split, this entire computation happens on either GPU or CPU for each layer.

### D2. The Hot/Cold FFN Computation

With hot/cold splitting, the FFN is decomposed:

```
x: [hidden_dim]

// GPU path (hot neurons)
g_hot = x @ W_gate_hot.T                 // [n_hot]
u_hot = x @ W_up_hot.T                   // [n_hot]
a_hot = SiLU(g_hot) * u_hot              // [n_hot]
y_hot = a_hot @ W_down_hot.T             // [hidden_dim]  ← partial result

// CPU path (cold neurons)
g_cold = x @ W_gate_cold.T               // [n_cold]
u_cold = x @ W_up_cold.T                 // [n_cold]
a_cold = SiLU(g_cold) * u_cold           // [n_cold]
// Sparsity optimization: zero out neurons where |a_cold[i]| < threshold
// This avoids computing the down_proj contribution for near-zero neurons
y_cold = a_cold_sparse @ W_down_cold.T   // [hidden_dim]  ← partial result

// Combine
y = y_hot + y_cold                       // [hidden_dim]  ← full FFN output
```

### D3. The Sparsity Optimization (Why This Is Fast)

The key insight: after activation, many cold neurons have near-zero values. You don't need to multiply them through `W_down_cold`.

**For ReLU:** Exact zeros. Skip those rows of `W_down_cold` entirely. The GEMM becomes a sparse-dense multiply.

**For SiLU:** Near-zeros. Apply a threshold (e.g., `|a_cold[i]| < 0.01 → treat as zero`). The number of surviving neurons varies per token, but for cold neurons, it's typically 5–20% of `n_cold`.

**Implementation on CPU:**
- After computing `a_cold`, scan for non-zero entries
- Build an index list of surviving neurons
- Gather the corresponding rows from `W_down_cold`
- Perform a small dense GEMM with only the surviving rows

This is the operation that must be genuinely fast. If you compute all `n_cold` neurons and then discard the zeros, you've gained nothing over the naive approach.

### D4. Integration with ggml (The Practical Path)

**Option A: Custom ggml operator (Recommended)**
Create a new ggml operation `ggml_ffn_sparse_cold()` that:
1. Takes `x`, `W_gate_cold`, `W_up_cold`, `W_down_cold`, and the activation threshold
2. Computes the gate and up projections
3. Applies activation and thresholding
4. Performs the sparse down projection
5. Returns the partial result `y_cold`

Register this operator with ggml's compute backend system. The CPU backend implements the sparse logic. The GPU backend is not needed for this operator (cold neurons are always on CPU).

**Option B: External compute + ggml tensor copy**
Compute the cold FFN path outside ggml (raw C++ with BLAS or manual loops), then copy the result back into a ggml tensor for the combination step.

**Pros of A:** Cleaner integration, ggml handles memory and threading.
**Pros of B:** Easier to prototype, doesn't require modifying ggml internals.

**Recommendation:** Start with Option B for prototyping. Once the logic is correct and validated, migrate to Option A for performance. The prototype phase is where you'll discover the real bottlenecks.

### D5. GPU-CPU Synchronization

The hot and cold paths can run **in parallel** for each layer:
- GPU computes `y_hot` (small GEMM, fast)
- CPU computes `y_cold` (sparse GEMM, slower but parallel)
- Synchronize at the combination point: `y = y_hot + y_cold`

**The bottleneck:** The CPU path is almost always slower than the GPU path. The GPU will finish `y_hot` and wait for the CPU's `y_cold`. This means the effective per-layer time is `max(t_gpu_hot, t_cpu_cold)`, not the sum.

**Optimization:** Overlap the CPU cold computation of layer N with the GPU hot computation of layer N+1 (pipeline parallelism). This is advanced and should be deferred to post-MVP optimization. For the initial implementation, accept the `max()` bottleneck.

### D6. Batch Size Consideration

PowerInfer's reported numbers are for batch=1 (interactive decode). For batch > 1 (prefill or batched inference), the sparsity patterns differ across tokens in the batch, complicating the sparse GEMM. For MVP, support batch=1 only for the hot/cold path. Fall back to naive computation for batch > 1.

---

## Phase E — Layer-Streaming Fallback (AirLLM-Style)

### E1. When This Triggers

The layer-streaming fallback activates when:
1. The model is dense (not MoE — MoE has its own offload from Step 9)
2. Even the hot/cold split's minimum GPU-resident set (attention + KV + minimal hot neurons) exceeds available VRAM
3. The full model exceeds available RAM (so CPU-only with mmap is also not viable without extreme swap pressure)

In practice, this means: very large model, very constrained hardware. Example: 70B Q4 on a machine with 8GB VRAM and 16GB RAM.

### E2. The Mechanism

Instead of loading the entire model into memory, load **one layer at a time**:

```
for each token to generate:
    hidden_state = embedding(input_token)    // small, always resident
    
    for layer in 0..n_layers-1:
        load layer weights from disk         // read ~1-2 GB from NVMe
        hidden_state = layer(hidden_state)   // compute on CPU (or GPU if layer fits)
        free layer weights                   // release memory
    
    logits = output_head(hidden_state)       // small, always resident
    next_token = sample(logits)
```

**Memory footprint:** Only one layer's weights + the hidden state + KV cache (which must stay resident across layers). For a 70B model with 80 layers, each layer is ~500MB–1GB. Total resident memory: ~1–2GB + KV cache.

**The cost:** Every token requires reading ALL layer weights from disk. For a 70B model at ~1GB/layer and 80 layers, that's ~80GB of disk reads per token. At 3GB/s sequential NVMe speed, that's ~27 seconds per token. At random-read speeds, much worse.

### E3. Implementation

**Resident components (always in memory):**
- Embedding table (or mmap it — it's accessed once per token)
- Output head / LM head
- KV cache (must persist across layers within a token, and across tokens)
- One layer's worth of compute buffers

**Streaming components (loaded and freed per layer per token):**
- Attention weights (W_q, W_k, W_v, W_o)
- FFN weights (W_gate, W_up, W_down)
- Norm weights (tiny, could be resident)

**The loading mechanism:**
- Use `mmap` with `MADV_SEQUENTIAL` hint (or Windows equivalent `PrefetchVirtualMemory`) to tell the OS you'll read the layer sequentially
- Or use direct `ReadFile` with `FILE_FLAG_SEQUENTIAL_SCAN` for explicit control
- After computing the layer, call `VirtualUnlock` / `madvise(MADV_DONTNEED)` to release the pages

**KV cache handling:** The KV cache must be updated by each layer and persist across the layer loop. It stays resident in RAM (or VRAM if it fits). For long contexts, the KV cache itself may exceed RAM — in that case, the model simply cannot run, and the tool should say so honestly.

### E4. Predictor Formula for Layer-Streaming

```
bytes_per_layer = total_weight_bytes / n_layers
disk_reads_per_token = n_layers × bytes_per_layer
time_per_token = disk_reads_per_token / nvme_sequential_bandwidth
                 + n_layers × compute_time_per_layer  // compute is negligible vs I/O
tokens_per_sec = 1 / time_per_token
```

**Example:** 70B Q4 (40GB), 80 layers, NVMe 3GB/s:
- `bytes_per_layer = 40GB / 80 = 500MB`
- `disk_reads_per_token = 80 × 500MB = 40GB`
- `time_per_token = 40GB / 3GB/s = 13.3 seconds`
- `tokens_per_sec ≈ 0.075` (roughly one token per 13 seconds)

This is honest. Don't round it up.

### E5. The "Is This Even Worth It?" Threshold

Set a minimum acceptable speed threshold. If the predicted layer-streaming speed is below 0.01 tok/s (one token per 100 seconds), the tool should say:

```
❌ Layer-streaming fallback would run at ~0.008 tok/s (one token per ~2 minutes).
   This is technically possible but not practically useful.
   Consider: a smaller model, a cloud API, or hardware upgrade.
```

Let the user override this threshold if they really want to try it (e.g., for a one-off batch job where time doesn't matter).

---

## Phase F — Predictor Updates

### F1. New Placement Enum Values

Add to the existing `Placement` enum:

| Value | Description |
|---|---|
| `HOT_COLD_SPLIT` | PowerInfer-style neuron-level split. Requires pre-profiled mask. |
| `LAYER_STREAM` | AirLLM-style sequential layer loading from disk. |

### F2. Hot/Cold Split Prediction Formulas

**Memory:**
```
vram_usage = bytes(attention_all_layers)
           + bytes(norms_all_layers)
           + bytes(embedding + output_head)
           + bytes(hot_ffn_all_layers)
           + kv_cache_bytes
           + cuda_overhead

ram_usage = bytes(cold_ffn_all_layers)
          + cpu_overhead
```

**Decode speed:**
```
// Per layer, per token:
t_gpu_hot = bytes(hot_ffn_one_layer) / gpu_bandwidth
          + bytes(attention_one_layer) / gpu_bandwidth
t_cpu_cold = effective_sparse_bytes(cold_ffn_one_layer) / ram_bandwidth

// effective_sparse_bytes accounts for the sparsity optimization:
// only ~10-20% of cold neurons actually activate per token
effective_sparse_bytes = bytes(cold_ffn_one_layer) × cold_activation_rate
// cold_activation_rate ≈ 0.10–0.20 (from profiling data)

t_layer = max(t_gpu_hot, t_cpu_cold)   // parallel execution, slower side dominates
tokens_per_sec = 1 / (n_layers × t_layer)
```

**Key difference from naive split:** In naive split, CPU layers process the ENTIRE layer (attention + FFN) at RAM bandwidth. In hot/cold split, only the cold FFN fraction runs on CPU, and it's sparse. The attention stays on GPU. This is why hot/cold is faster.

**Prefill / TTFT:**
```
flops_per_token ≈ 2 × params_total   // prefill activates all neurons, not just hot
ttft_ms = (prompt_tokens × flops_per_token) / (effective_compute_throughput × 1e12)
```

Prefill is less optimized by hot/cold splitting because all neurons activate during the large-batch prefill. The speedup is primarily in decode. Flag TTFT predictions for hot/cold as lower confidence.

### F3. Layer-Streaming Prediction Formulas

Already covered in Phase E4. Simple I/O-bound formula.

### F4. Confidence Bands

| Strategy | Band |
|---|---|
| `HOT_COLD_SPLIT` with profiling mask | MEDIUM (new technique, less calibration data) |
| `HOT_COLD_SPLIT` without profiling (using default power-law estimate) | LOW |
| `LAYER_STREAM` | HIGH for speed (the formula is simple and I/O-bound), LOW for "is this useful" |

### F5. The "No Mask Available" Case

If the user hasn't run `--profile-neurons` for the model, the hot/cold strategy should either:
- Offer to run profiling now (adds 1–4 hours to the pipeline)
- Use a default power-law estimate (top 15% of neurons per layer, uniformly) and flag as LOW confidence
- Not offer the strategy at all

**Recommendation:** Offer with a default estimate and a clear label:
```
⚠️ Hot/cold split using estimated neuron distribution (no profiling data).
   Run --profile-neurons for accurate placement.
```

---

## Phase G — Method Matrix and Ranker Integration

### G1. New Matrix Rows

For dense models that don't fit on GPU:

| Placement | When it appears | Context values |
|---|---|---|
| `FULL_GPU` | Always (may be NO FIT) | 4K, max-safe |
| `GPU_CPU_SPLIT` (naive) | Always | 4K, max-safe |
| `CPU_ONLY` | Always | 4K, max-safe |
| **`HOT_COLD_SPLIT`** | When model > VRAM but model fits in RAM, and profiling mask exists or default estimate used | 4K, max-safe |
| **`LAYER_STREAM`** | When model > RAM or hot/cold minimum GPU set > VRAM | 4K only (long context KV cache would exceed RAM) |

### G2. Ranker Behavior

**Speed priority:**
- `HOT_COLD_SPLIT` typically ranks above `GPU_CPU_SPLIT` (naive) and `CPU_ONLY` for models 1.5–3× VRAM
- `LAYER_STREAM` ranks last (slowest)

**Safety priority:**
- `HOT_COLD_SPLIT` often wins because VRAM usage is moderate (attention + hot FFN only)
- `LAYER_STREAM` has the lowest memory footprint but highest I/O wear

**Quality priority:**
- All strategies use the same weights (no quantization change), so quality ranking is determined by KV cache precision, same as dense

### G3. Honest Labeling in the Table

```
 #  Placement        VRAM    RAM     tok/s       Status
 1  Hot/Cold Split   6.2 GB  18 GB   ~7–12       ✅ VIABLE (profiled)
 2  Split (naive)    7.9 GB  18 GB   ~4          ✅ VIABLE
 3  CPU Only         0 GB    26 GB   ~2          ✅ VIABLE
 4  Full GPU         26 GB   0.5 GB  —           ❌ NO FIT
 5  Layer Stream     0.5 GB  3 GB    ~0.07       ⚠️ VIABLE (~14s/token)
```

The layer-streaming row must show the per-token time, not just tok/s. "0.07 tok/s" is abstract. "~14s/token" is visceral.

---

## Phase H — Calibration Log Extensions

### H1. New Fields

```
strategy.placement = "HOT_COLD_SPLIT" | "LAYER_STREAM"
strategy.hot_neuron_pct = 0.15          // fraction of neurons on GPU
strategy.profiled = true | false        // was a real mask used?
strategy.cold_activation_rate = 0.12    // observed sparsity during decode
predicted.tok_s_range = [7.0, 12.0]    // for hot/cold
actual.tok_s = 9.3
actual.cold_compute_pct = 0.35          // fraction of time spent on CPU cold path
```

### H2. Isolation

Hot/cold and layer-stream calibration records must NOT mix with dense GPU, dense split, or MoE records. Filter by `placement` in the aggregation logic.

### H3. Useful Derived Constants

After ≥5 hot/cold runs:
- `cold_activation_rate` — how many cold neurons actually fire per token (refines the sparsity estimate)
- `gpu_cpu_sync_overhead` — the gap between `max(t_hot, t_cold)` and actual per-layer time (synchronization cost)
- `layer_stream_actual_io_speed` — real-world sequential read speed during streaming (may differ from the micro-benchmark)

---

## Phase I — Validation

### I1. Profiling Validation

| Check | Pass |
|---|---|
| Hot set stability (two independent prompt sets) | >85% overlap |
| Hot set size matches VRAM budget | Within 5% |
| Profiling completes without OOM | Even for models larger than RAM (use mmap during profiling) |
| Mask file is valid and loadable | Round-trip test |

### I2. Hot/Cold Execution Validation

| Check | Pass |
|---|---|
| VRAM usage matches predicted GPU-resident set | Within 10% |
| Hot/cold tok/s > naive split tok/s | Clear improvement (>20%), same model, same hardware |
| Hot/cold tok/s > CPU-only tok/s | Clear improvement (>2×) |
| Generated text is coherent | Compare against naive split output for the same prompt |
| No NaN/Inf in activations | Monitor during first few runs |
| Sparsity optimization actually skips neurons | CPU cold path time < naive full-FFN time |

### I3. Layer-Streaming Validation

| Check | Pass |
|---|---|
| Model runs to completion (even slowly) | Generates 10+ tokens |
| Memory footprint matches prediction (one layer + KV) | Within 20% |
| tok/s matches I/O-bound prediction | Within 30% |
| No memory leak across layers | RAM usage stable over 10+ tokens |

### I4. Regression Pack

- All dense models from Steps 3–6: unchanged predictions and performance
- All MoE models from Step 9: unchanged
- Hot/cold only appears for models that don't fit on GPU
- Layer-streaming only appears as last resort

---

## Phase J — User Interaction Flow

### J1. First-Time Hot/Cold Use

```
Model: Llama-3.1-13B Q4_K_M | 13B params | 40 layers | 128K context
Hardware: RTX 3080 (10GB VRAM, 8.7GB free) | 32GB RAM

This model does not fit entirely on your GPU. Available strategies:

 #  Placement        VRAM    RAM     tok/s       Status
 1  Hot/Cold Split   6.8 GB  12 GB   ~7–11       ✅ VIABLE
 2  Split (naive)    7.9 GB  8 GB    ~4          ✅ VIABLE
 3  CPU Only         0 GB    14 GB   ~2          ✅ VIABLE
 4  Full GPU         14 GB   0.5 GB  —           ❌ NO FIT

⚠️ Hot/cold split requires a one-time neuron profiling step (~2 hours for this model).
   Run: llm-planner --profile-neurons --model-path <path> --vram-budget 6GB
   After profiling, re-run this command for optimized placement.

   Currently using estimated neuron distribution (lower accuracy).
```

### J2. After Profiling

```
 #  Placement        VRAM    RAM     tok/s       Status
 1  Hot/Cold Split   6.4 GB  12 GB   ~8–12       ✅ VIABLE (profiled ✅)
 ...
```

### J3. Layer-Streaming Last Resort

```
 #  Placement        VRAM    RAM     tok/s        Status
 5  Layer Stream     0.5 GB  3 GB    ~0.07        ⚠️ ~14s/token
    ↑ This mode streams one layer at a time from disk.
      Usable for batch jobs where time is not critical.
      Not recommended for interactive use.
```

---

## Step 10 — Done Checklist

- [ ] PowerInfer paper read and mechanism understood
- [ ] PowerInfer fork studied; integration approach decided (adapt vs reimplement)
- [ ] Profiling infrastructure loads model, instruments FFN, records activations
- [ ] Profiling prompt set is diverse (500+ prompts, multiple domains)
- [ ] Hot set stability validated (>85% overlap across prompt sets)
- [ ] Mask file format defined and round-trips correctly
- [ ] `--profile-neurons` CLI command works end-to-end
- [ ] FFN weight splitting produces correct hot/cold subsets
- [ ] Sparse FFN CPU path genuinely skips non-activated neurons (not compute-and-discard)
- [ ] GPU hot path produces correct partial FFN output
- [ ] GPU + CPU partial results combine correctly (sum matches full FFN)
- [ ] Hot/cold tok/s measurably beats naive split on the same hardware (>20%)
- [ ] Hot/cold tok/s measurably beats CPU-only (>2×)
- [ ] Generated text is coherent (matches naive split quality)
- [ ] Layer-streaming fallback triggers automatically when hot/cold doesn't fit
- [ ] Layer-streaming completes generation (even at ~10s/token)
- [ ] Layer-streaming memory footprint matches prediction (one layer + KV)
- [ ] Predictor formulas for hot/cold and layer-stream produce reasonable numbers
- [ ] Confidence bands correctly reflect profiling status and technique maturity
- [ ] Method matrix includes hot/cold and layer-stream rows only when appropriate
- [ ] Ranker sorts hot/cold above naive split for speed priority
- [ ] Layer-streaming row shows per-token time, not just tok/s
- [ ] "Not practically useful" warning for extremely slow layer-streaming
- [ ] Calibration log extended with hot/cold and layer-stream fields
- [ ] Calibration records isolated from dense GPU and MoE records
- [ ] No regressions in dense or MoE predictions
- [ ] Tested with at least one moderate-overfit model (hot/cold) and one extreme-overfit model (layer-stream)

---

## Common Failure Points at Step 10

| Problem | Likely Cause | Fix |
|---|---|---|
| Hot/cold is slower than naive split | Sparse CPU path not actually skipping neurons; computing all cold neurons and discarding | Verify the sparse GEMM only touches surviving neuron rows. Profile the CPU path in isolation. |
| Hot/cold is slower than naive split | GPU-CPU synchronization overhead dominates | The GPU finishes hot FFN in microseconds and waits milliseconds for CPU cold FFN. Reduce cold set size or optimize CPU sparse path. |
| Hot set is unstable across prompts | Profiling prompts too narrow, or activation threshold too low | Broaden prompt set. Raise SiLU threshold. Verify against the paper's reported stability. |
| Hot set doesn't fit in VRAM | Attention + KV + hot FFN exceeds budget | Reduce hot neuron count. Reduce context length. Use Q8 KV cache. |
| Generated text is garbage | Hot/cold split corrupts the FFN output (wrong indices, wrong combination) | Compare layer-by-layer outputs against a full-GPU baseline for the first few tokens. The error will be obvious. |
| NaN/Inf in activations | Quantized weights + sparse computation produces numerical instability | Check that the sparse down_proj handles the reduced dimension correctly. Ensure no division by zero in normalization. |
| Layer-streaming OOM on KV cache | KV cache at the requested context exceeds RAM | Layer-streaming should only be offered at short contexts (4K). Refuse long-context layer-streaming. |
| Layer-streaming never triggers | Predictor's viability check has a bug | Add an explicit test case: 70B model on 8GB VRAM + 16GB RAM. Confirm layer-stream appears. |
| Profiling takes too long | Running on CPU with a very large model | Use mmap for the profiling run. Accept that 70B profiling takes hours. Display ETA. |
| Profiling OOM | Model doesn't fit in RAM for profiling | Use mmap + sequential layer loading during profiling (ironically, use the layer-streaming technique to profile for hot/cold). |
| Mask file is model-version-specific | User updates the GGUF file but keeps the old mask | Include the GGUF file hash in the mask metadata. If the hash doesn't match, refuse to use the mask and re-profile. |
| ggml integration breaks existing paths | Custom operator interferes with dense or MoE compute | Gate the sparse operator behind the `HOT_COLD_SPLIT` placement flag. Dense and MoE paths must not call it. |
| Batch > 1 crashes | Sparse patterns differ across tokens in the batch | For MVP, enforce batch=1 for hot/cold. Fall back to naive for batch > 1. |

---

## Time Estimate for Step 10

| Phase | Work | Time |
|---|---|---|
| A | Paper study + fork analysis + architecture decisions | 2–3 days |
| B | Profiling infrastructure (instrumentation, prompt set, aggregation, mask file) | 3–4 days |
| C | Weight splitting logic (load-time slicing, memory layout) | 1–2 days |
| D | Sparse FFN execution (GPU hot path, CPU sparse path, combination, ggml integration) | 5–8 days |
| E | Layer-streaming fallback (sequential loading, memory management, I/O optimization) | 2–3 days |
| F | Predictor formulas (hot/cold speed, layer-stream speed, confidence) | 1–2 days |
| G | Matrix + ranker integration (new rows, honest labeling) | 1 day |
| H | Calibration log extensions | 0.5 day |
| I | Validation (profiling stability, speedup vs naive, layer-stream completion, regression) | 2–3 days |
| J | UX flow (first-time profiling prompt, honest warnings) | 1 day |

**Total: 2–3 weeks, as estimated. Phase D (sparse FFN execution) is the single hardest piece of engineering in the entire project. Size your expectations accordingly. If Phase D takes longer than expected, ship the layer-streaming fallback (Phase E) first — it's simpler and still delivers the "always find a way" promise, even if the speed is modest.**

---

## Final Note: When to Stop

Step 10 is complete when:
1. A dense model that doesn't fit on GPU runs faster with hot/cold than with naive split
2. A model that doesn't fit in RAM runs (slowly) via layer-streaming
3. The tool honestly labels the performance cost of each approach
4. No regressions in Steps 1–9

Do not chase PowerInfer's exact reported numbers. Their fork has months of operator tuning. Your initial implementation will be slower. The calibration log will close the gap over time. Ship when the feature works and is measurably better than the naive baseline, not when it matches a research paper's optimized numbers.