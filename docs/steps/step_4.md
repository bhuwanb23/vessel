# Step 4 — Wire It Together: Full Detailed Plan

---

## Goal of Step 4
Connect the three standalone components from Steps 1–3 into a single end-to-end pipeline. The user provides a model URL (and optionally a priority flag). The tool profiles the hardware, fetches the model metadata, generates every viable deployment strategy, predicts performance for each, and prints a comparison table. No model execution happens. This is the "advisor" milestone — the first version of the tool that is genuinely useful to someone other than you.

---

## What You Need Before Starting

### From Steps 1–3 (already done)
- **Step 1:** Hardware profiler binary that outputs a `HardwareSpec` struct (or equivalent)
- **Step 2:** Metadata fetcher binary that outputs a `ModelMetadata` struct (or equivalent)
- **Step 3:** Predictor function that takes `HardwareSpec` + `ModelMetadata` + `StrategyConfig` and returns a `Prediction` struct
- All three components tested and validated independently

### What Changes in This Step
- The three components stop being standalone binaries and become **modules within a single binary**
- A new component appears: the **Method Matrix Generator**, which enumerates strategy combinations
- A new component appears: the **Pipeline Orchestrator**, which sequences the flow
- A command-line interface is added for user input
- The output changes from three separate reports to a single unified prediction table

---

## Phase A — Refactor from Binaries to Modules

### The Structural Shift
In Steps 1–3, each component was its own `main()` function in its own binary. Now they become callable functions within a single program.

**Before (Steps 1–3):**
```
llm-planner/
├── src/
│   ├── profiler_main.cpp    ← standalone binary
│   ├── fetcher_main.cpp     ← standalone binary
│   └── predictor_test.cpp   ← standalone test binary
```

**After (Step 4):**
```
llm-planner/
├── src/
│   ├── main.cpp             ← single entry point, CLI parsing, orchestration
│   ├── profiler.cpp         ← HardwareSpec profileHardware()
│   ├── fetcher.cpp          ← ModelMetadata fetchMetadata(string url)
│   ├── predictor.cpp        ← Prediction predict(HardwareSpec, ModelMetadata, StrategyConfig)
│   └── matrix.cpp           ← vector<StrategyConfig> generateMatrix(HardwareSpec, ModelMetadata)
├── include/
│   ├── types.h              ← all shared structs (HardwareSpec, ModelMetadata, etc.)
│   ├── profiler.h
│   ├── fetcher.h
│   ├── predictor.h
│   └── matrix.h
```

### The Shared Types Header
This is the most important file in the project. Every module includes it. It defines the structs that flow between modules. If the struct definitions are wrong or inconsistent, nothing works.

**Key design decision:** Put ALL shared types in a single `types.h` file. Don't scatter them across module headers. The reason is that the `Prediction` struct references fields from `HardwareSpec` and `ModelMetadata`, and the `StrategyConfig` struct references fields from both. A single header avoids circular dependency headaches.

### What to Preserve
- Keep the standalone test binaries if you want (rename them to `test_profiler.cpp`, etc.) — they're useful for debugging individual modules later
- Don't delete the Step 1–3 validation code — you'll want to re-run it if something breaks during integration

---

## Phase B — The Method Matrix Generator (New Component)

### What It Does
Given a `HardwareSpec` and `ModelMetadata`, it produces a `std::vector<StrategyConfig>` containing every deployment strategy worth evaluating. This is the "enumeration" step that no existing tool does — it's the core differentiator of your project.

### The Three Dimensions of the Matrix

For MVP, the matrix has three dimensions with constrained values (per the plan's decision to keep the demo manageable):

| Dimension | Values | Count |
|---|---|---|
| **Placement** | FULL_GPU, GPU_CPU_SPLIT (multiple split points), CPU_ONLY | 3–6 depending on model size |
| **Context Length** | 4K (4096), max-safe (calculated) | 2 |
| **KV Cache Precision** | FP16 (2 bytes), Q8 (1 byte) | 2 |

**Total matrix size:** roughly 12–24 entries for a typical model. This is small enough to evaluate instantly and large enough to show meaningful tradeoffs.

### How to Enumerate Placement Strategies

**1. FULL_GPU:**
- `gpu_layers = total_layers`
- Only viable if total memory fits in free VRAM
- Always include in the matrix — if it's not viable, the predictor will flag it as `viable = false`, which is useful information ("this model doesn't fit entirely on your GPU")

**2. CPU_ONLY:**
- `gpu_layers = 0`
- Only viable if total memory fits in free RAM
- Always include — shows the user the "it works but it's slow" option

**3. GPU_CPU_SPLIT (the interesting part):**
You need to generate a few meaningful split points. Don't enumerate every possible layer count from 1 to N-1 — that's too many and most are redundant. Instead, generate these specific splits:

| Split Point | How to Calculate | Why It Matters |
|---|---|---|
| **Max-fit split** | Maximum layers that fit in VRAM with the given context | The fastest split option — puts as much as possible on GPU |
| **Half split** | `total_layers / 2` | A balanced middle ground |
| **Minimal GPU** | 1–2 layers on GPU, rest on CPU | Shows whether even a small GPU contribution helps |
| **KV-only on GPU** | All weights on CPU, KV cache on GPU (if VRAM allows) | An advanced strategy that can help with long contexts |

**How to calculate the max-fit split:**

```
per_layer_weight_bytes = weight_bytes / total_layers
per_layer_kv_bytes = kv_cache_bytes / total_layers
per_layer_total = per_layer_weight_bytes + per_layer_kv_bytes

gpu_overhead = 512 MB  (CUDA context, from Step 3 calibration)
available_for_layers = vram_free_bytes - gpu_overhead

max_gpu_layers = available_for_layers / per_layer_total
max_gpu_layers = min(max_gpu_layers, total_layers)  (can't exceed total)
max_gpu_layers = max(max_gpu_layers, 0)  (can't be negative)
```

**Important edge case:** If `max_gpu_layers == total_layers`, then the full-GPU strategy already fits and the max-fit split is redundant. Skip it. If `max_gpu_layers == 0`, then no layers fit on GPU and the split strategies are all equivalent to CPU-only. Skip them.

### How to Calculate Max-Safe Context

From Step 3's Phase C, you already have the reverse KV cache formula:

```
available_for_kv = memory_budget - weight_bytes - runtime_overhead_bytes
max_context = available_for_kv / (2 × layers × kv_heads × head_dim × batch_size × bytes_per_kv_element)
```

Calculate this for the FULL_GPU strategy (using VRAM budget) and the CPU_ONLY strategy (using RAM budget). Use the larger of the two as the "max-safe" context for the matrix.

**Cap it:** Don't let max-safe exceed the model's advertised `context_length`. A model trained on 8K context won't produce good output at 128K even if the memory technically fits.

**Floor it:** If max-safe is less than 4096, set it to 4096 and let the predictor flag the strategy as non-viable. The user should see that even 4K doesn't fit, rather than seeing a confusing "max-safe = 1,200" number.

### The Generation Algorithm (Pseudocode)

```
function generateMatrix(hardware, model):
    strategies = []
    
    // Calculate key values
    weight_bytes = model.param_count × model.bpw / 8
    max_safe_ctx = calculateMaxSafeContext(hardware, model, weight_bytes)
    contexts = [4096, max_safe_ctx]  // deduplicate if max_safe == 4096
    kv_quants = [16, 8]  // FP16 and Q8
    
    // Calculate split points
    max_gpu_layers = calculateMaxFitLayers(hardware, model, weight_bytes, context=max(contexts))
    split_points = deduplicate([0, max_gpu_layers/2, max_gpu_layers, model.layers])
    
    for gpu_layers in split_points:
        for ctx in contexts:
            for kv_bits in kv_quants:
                placement = classifyPlacement(gpu_layers, model.layers)
                config = StrategyConfig(placement, gpu_layers, ctx, batch=1, kv_bits)
                strategies.append(config)
    
    return strategies
```

### Deduplication
After generating all combinations, remove duplicates. For example, if `max_gpu_layers == total_layers`, the "max-fit split" and "full GPU" are the same strategy. Don't show the user two identical rows.

---

## Phase C — The Pipeline Orchestrator

### The End-to-End Flow

```
User runs: llm-planner --model <url> [--priority speed|quality|safety]

Step 1: Profile hardware
    → Call profileHardware()
    → Get HardwareSpec
    → Print brief hardware summary (1-2 lines, not the full Step 1 report)

Step 2: Fetch model metadata
    → Call fetchMetadata(url)
    → Get ModelMetadata
    → Print brief model summary (1-2 lines)
    → If fetch fails, print error and exit

Step 3: Generate method matrix
    → Call generateMatrix(hardware, model)
    → Get vector<StrategyConfig>
    → Print "Evaluating N strategies..."

Step 4: Predict each strategy
    → Loop over matrix
    → Call predict(hardware, model, strategy) for each
    → Get vector<Prediction>
    → Filter out non-viable strategies (or keep them grayed out — design decision)

Step 5: Rank and display
    → Sort predictions by user's priority
    → Print the comparison table
    → Print the recommendation
```

### Timing Expectations
- Hardware profile: <1 second (the disk benchmark takes 2-3 seconds, everything else is instant)
- Metadata fetch: 1-5 seconds (network latency to Hugging Face)
- Matrix generation: <1 millisecond (pure math)
- Prediction loop: <1 millisecond (pure math, even for 24 strategies)
- **Total wall time: ~3-7 seconds**, dominated by the disk benchmark and network fetch

### The "Brief Summary" Design Decision
In Steps 1 and 2, each component printed a detailed multi-line report. In the integrated tool, printing both full reports before the prediction table would overwhelm the user. Instead:

- Print a **one-line hardware summary**: `Hardware: RTX 3080 (10GB VRAM, 8.7GB free) | 32GB RAM (24GB free) | NVMe 4.2GB/s`
- Print a **one-line model summary**: `Model: Llama 3.2 3B Instruct Q4_K_M | 3.2B params | 28 layers | 128K context`
- Print the **full prediction table** (this is the main output)
- Offer a `--verbose` flag that prints the full Step 1 and Step 2 reports for debugging

---

## Phase D — Command-Line Interface

### The CLI Design
Keep it simple for MVP. Don't pull in a CLI framework — just parse `argc/argv` directly.

**Required arguments:**
- `--model <url>` — the Hugging Face GGUF URL

**Optional arguments:**
- `--priority <speed|quality|safety>` — how to rank strategies (default: `speed`)
- `--context <4k|max|both>` — which context lengths to evaluate (default: `both`)
- `--verbose` — print full hardware and model reports
- `--help` — print usage

**Example invocations:**
```
llm-planner --model https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf

llm-planner --model <url> --priority safety

llm-planner --model <url> --context 4k --verbose
```

### Argument Parsing Implementation
For MVP, a simple loop over `argv` is sufficient:

```
for i = 1 to argc-1:
    if argv[i] == "--model" and i+1 < argc:
        model_url = argv[i+1]
        i++
    else if argv[i] == "--priority" and i+1 < argc:
        priority = argv[i+1]
        i++
    // ... etc
```

Don't spend time on a CLI library. This will take 30 minutes to implement and is not the point of the project.

### Error Messages
If the user provides no arguments or an invalid URL, print a clear error message and the usage help. Don't crash with a segfault.

---

## Phase E — Output Formatting (The Prediction Table)

### The Table Design
This is the most important UX element of the entire tool. It must be readable at a glance and convey the tradeoffs between strategies clearly.

**Example output:**

```
=== LLM Deployment Planner — Strategy Comparison ===
Hardware: RTX 3080 (10GB VRAM, 8.7GB free) | 32GB RAM (24GB free) | NVMe 4.2/0.13 GB/s
Model:    Llama 3.2 3B Instruct Q4_K_M | 3.2B params | 28 layers | 128K max context

Ranked by: speed (use --priority to change)

 #  Placement      GPU Layers  Context  KV Cache  VRAM     RAM     tok/s    TTFT    Status
─── ────────────── ────────── ──────── ──────── ──────── ─────── ──────── ─────── ──────────
 1  Full GPU       28/28      4K       FP16     2.4 GB   0.5 GB  ~385     ~45ms   ✅ VIABLE
 2  Full GPU       28/28      4K       Q8       2.1 GB   0.5 GB  ~390     ~45ms   ✅ VIABLE
 3  Full GPU       28/28      128K     Q8       9.6 GB   0.5 GB  ~290     ~1.2s   ✅ VIABLE
 4  Split          20/28      128K     Q8       6.9 GB   3.2 GB  ~62      ~1.8s   ✅ VIABLE
 5  Split          14/28      128K     Q8       4.8 GB   5.3 GB  ~35      ~2.4s   ✅ VIABLE
 6  CPU Only       0/28       4K       FP16     0 GB     2.9 GB  ~21      ~380ms  ✅ VIABLE
 7  CPU Only       0/28       128K     Q8       0 GB     9.1 GB  ~14      ~5.1s   ✅ VIABLE
 8  Full GPU       28/28      128K     FP16     17.1 GB  0.5 GB  —        —       ❌ NO FIT

💡 Recommendation: Strategy #1 (Full GPU, 4K, FP16) for fastest generation.
   For long documents, Strategy #3 (Full GPU, 128K, Q8) keeps everything on GPU.
   Q8 KV cache saves ~0.3GB VRAM at negligible quality cost for this model size.
```

### Key Formatting Decisions

**Memory columns:** Show in GB to 1 decimal place. Don't show raw bytes — nobody thinks in bytes.

**Speed columns:** Prefix with `~` to signal these are predictions, not measurements. Round to whole numbers for tokens/sec, 1 decimal for TTFT.

**Status column:** Use clear visual indicators:
- `✅ VIABLE` — fits in memory, predicted to work
- `⚠️ TIGHT` — fits but uses >90% of available memory (risky if background apps allocate more)
- `❌ NO FIT` — exceeds available memory
- `❓ LOW CONF` — viable but prediction is low confidence (config.json fallback, unknown quant, etc.)

**Non-viable strategies:** Show them grayed out or at the bottom of the table with `—` for speed numbers. Don't hide them entirely — the user should see that "Full GPU at 128K FP16 doesn't fit" rather than wondering why that option is missing.

**The recommendation line:** A one-line suggestion based on the user's priority. This is the "so what?" that turns a data table into actionable advice.

### The "Tight" Threshold
A strategy is "tight" when predicted memory usage exceeds 90% of available VRAM or RAM. This is the "safety margin" concept from §3 item 9. The 10% buffer accounts for:
- OS memory allocation fluctuations between planning and execution
- Background apps that might start during the run
- CUDA context growth during long runs
- Prediction error margin

---

## Phase F — Error Handling and Edge Cases

### The Pipeline Can Fail at Multiple Points

| Failure Point | What Happens | How to Handle |
|---|---|---|
| Hardware profile fails (NVML error) | No GPU data | Print error, suggest checking NVIDIA driver. Exit — can't predict without hardware data. |
| Metadata fetch fails (network) | No model data | Print error with HTTP status code. Suggest checking URL and internet connection. Exit. |
| Metadata fetch returns non-GGUF | Parser fails | Print "URL does not point to a GGUF file." If it looks like a repo URL, suggest appending the specific filename. |
| GGUF version unsupported | Parser fails | Print "GGUF version X not supported (this tool supports v2-v3)." |
| Model too large for any strategy | All strategies non-viable | Show the table with all `❌ NO FIT` entries. Print a clear message: "This model does not fit on your hardware in any configuration. Consider a smaller quantization or a smaller model." |
| Model fits but only on CPU | Only CPU strategies viable | Show the table normally. The recommendation should note that GPU offload is not possible and suggest a smaller model or quant for GPU use. |
| Disk benchmark abnormally slow | NVMe speed < 100 MB/s | Print a warning: "Storage read speed is unusually low. This may affect model loading time and MoE performance." Don't block — it doesn't affect the core predictions much for dense models. |
| Free VRAM is very low (<1GB) | Most GPU strategies non-viable | Print a warning: "Available VRAM is very low (X GB). Consider closing other GPU applications." |

### Graceful Degradation
The tool should never crash. If one subsystem fails, print the error and continue with what you have. For example:
- If the disk benchmark fails, still show the prediction table (disk speed only matters for MoE, which is Phase 2 anyway)
- If the GPU bandwidth derivation fails (unknown bus width), use a conservative default and flag predictions as medium confidence

---

## Phase G — End-to-End Testing

### Test Scenarios

**Test 1: The Happy Path**
- Run with a model that fits fully on your GPU at 4K context
- Expected: Full GPU strategy ranked #1, all strategies viable at 4K, some non-viable at max context
- Validate: Numbers match your Step 3 manual predictions

**Test 2: The Tight Fit**
- Run with a model that barely fits in VRAM (e.g., a 7B model on an 8GB card)
- Expected: Full GPU viable only at 4K with Q8 KV cache, split strategies viable at longer contexts
- Validate: The "tight" warning appears for strategies using >90% VRAM

**Test 3: The Too-Large Model**
- Run with a model that doesn't fit on GPU at all (e.g., a 13B model on an 8GB card)
- Expected: Full GPU shows `❌ NO FIT`, split and CPU-only strategies are viable
- Validate: The recommendation suggests CPU-only or a smaller model

**Test 4: The CPU-Only Machine (Simulated)**
- Temporarily set `vram_free_bytes = 0` in the hardware spec (hardcode it for testing)
- Expected: All GPU strategies non-viable, CPU-only strategies ranked
- Validate: The tool handles zero-VRAM gracefully without division-by-zero errors

**Test 5: Network Failure**
- Disconnect from the internet and run the tool
- Expected: Clear error message about network failure, not a crash or hang
- Validate: The timeout works (doesn't hang for 60+ seconds)

**Test 6: Invalid URL**
- Run with a nonsense URL or a URL pointing to a non-GGUF file
- Expected: Clear error message, not a segfault
- Validate: The magic number check catches non-GGUF responses

---

## Phase H — The "Pause and Evaluate" Milestone

### Why This Is a Good Stopping Point
From the original plan: "This is your first real milestone — a working 'advisor' tool, useful on its own even with nothing past this point."

At this point, you have a tool that:
1. Reads your real hardware specs (not spec-sheet numbers)
2. Fetches real model metadata without downloading the full file
3. Tells you every viable way to run the model
4. Predicts speed and memory for each option
5. Ranks them by what you care about
6. Warns you about strategies that won't fit

**This is already more useful than every existing VRAM calculator on the market.** You could stop here, publish this, and it would be a valuable tool.

### What's Missing (That Steps 5–7 Add)
- **Step 5 (Ranker):** The sorting logic is partially implemented in Step 4's output formatting. Step 5 formalizes the user-weighted priority system.
- **Step 6 (Executor):** The tool predicts but doesn't run. The user still has to manually construct the llama.cpp command.
- **Step 7 (Calibration Log):** Predictions are based on initial constants. Without the feedback loop, accuracy doesn't improve over time.

### Decision Point
After completing Step 4, evaluate:
- Are the predictions accurate enough to be useful? (Compare against manual llama.cpp runs)
- Is the output format clear and actionable?
- Are there edge cases you didn't anticipate?
- Do you want to refine the predictor formulas before adding execution?

If the answer to any of these is "no," spend time iterating on Steps 1–4 before proceeding. The advisor is the foundation — if it's wrong, the executor will just execute wrong predictions faster.

---

## Step 4 — Done Checklist

Before moving to Step 5, confirm every item:

- [ ] All three modules (profiler, fetcher, predictor) are compiled into a single binary
- [ ] Shared types header (`types.h`) defines all structs consistently
- [ ] Method matrix generator produces 8–24 strategies per model
- [ ] Matrix includes FULL_GPU, GPU_CPU_SPLIT (multiple points), and CPU_ONLY
- [ ] Matrix includes two context lengths (4K and max-safe)
- [ ] Matrix includes two KV cache precisions (FP16 and Q8)
- [ ] Max-safe context is calculated correctly and capped at model's advertised max
- [ ] Non-viable strategies are flagged but still shown in the table
- [ ] "Tight" warning appears for strategies using >90% of available memory
- [ ] CLI accepts `--model` URL and `--priority` flag
- [ ] Pipeline runs end-to-end in under 10 seconds
- [ ] Output table is readable and correctly formatted
- [ ] Recommendation line provides actionable advice
- [ ] Network failure produces a clear error message, not a crash
- [ ] Invalid URL produces a clear error message, not a segfault
- [ ] Model-too-large scenario shows all `❌ NO FIT` with helpful guidance
- [ ] Predictions match Step 3's manual validation (within the same margins)
- [ ] `--verbose` flag prints full hardware and model reports
- [ ] Tested against at least 2 different model sizes

---

## Common Failure Points at Step 4

| Problem | Likely Cause | Fix |
|---|---|---|
| Struct fields don't match between modules | Step 1–3 were built independently with slightly different field names or types | Unify everything in `types.h`. This may require renaming fields in the original modules. |
| Linker errors (unresolved symbols) | CMake not linking all source files into the single target | Update `CMakeLists.txt` to include all `.cpp` files in the `add_executable()` call. |
| Matrix generates too many strategies | Not deduplicating split points or context values | Add deduplication after generation. If max-safe == 4K, don't create duplicate rows. |
| Matrix generates zero strategies | Viability filter is too aggressive, or hardware spec reads 0 for free memory | Check that the profiler is reading live free memory correctly. Don't filter out non-viable strategies — show them as `❌ NO FIT`. |
| Division by zero in predictor | GPU layers = 0 in split formula, or bandwidth = 0 | Add guards for zero values. If bandwidth is 0, the strategy is non-viable, not infinite-speed. |
| Table columns misaligned | Model names or strategy labels of varying length | Use fixed-width formatting (`printf`-style `%12s` or `std::setw()`). |
| Pipeline hangs on metadata fetch | libcurl timeout not set, or DNS resolution failing | Confirm `CURLOPT_TIMEOUT` is set to 30 seconds. Test with a known-good URL first. |
| Predictions differ from Step 3 | Hardcoded test values in Step 3 don't match live profiler output | This is expected — live free memory changes between runs. The predictions should be close, not identical. |
| Max-safe context is unreasonably large (>1M tokens) | KV cache formula bug, or bytes_per_kv_element set to 0 | Check the KV cache formula. Ensure bytes_per_kv_element is 2 for FP16, 1 for Q8. |
| Max-safe context is unreasonably small (<1K tokens) | Runtime overhead constant too large, eating all available memory | Re-check the overhead calibration from Step 3. |

---

## Time Estimate for Step 4
- Phase A (Refactor to modules + shared types): **2–3 hours** (mostly header reorganization and CMake updates)
- Phase B (Method matrix generator): **3–4 hours** (the split-point calculation and deduplication logic)
- Phase C (Pipeline orchestrator): **2–3 hours** (sequencing the flow, error handling between stages)
- Phase D (CLI): **30 minutes** (simple argv parsing)
- Phase E (Output formatting): **2–3 hours** (getting the table alignment right takes more time than you'd expect)
- Phase F (Error handling): **1–2 hours** (edge cases and graceful degradation)
- Phase G (End-to-end testing): **2–3 hours** (running all test scenarios, comparing against Step 3)
- Phase H (Milestone evaluation): **1 hour** (honest assessment of prediction quality)

**Total: 1 day, as originally estimated. The integration is straightforward because the hard work was done in Steps 1–3. The main risk is struct mismatches between independently developed modules.**