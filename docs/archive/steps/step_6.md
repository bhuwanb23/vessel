# Step 6 — Executor: Full Detailed Plan

---

## Goal of Step 6
Transform the advisor into a controller. After the user selects a strategy from the ranked table, the tool loads the model, configures it exactly as predicted, runs inference, samples hardware state live during the run, and reports actual performance compared to predictions. This is the hardest step in the entire project because it requires deep integration with llama.cpp's internal C API, multi-threaded hardware polling, and careful memory lifecycle management.

---

## What You Need Before Starting

### From Steps 1–5 (already done and solid)
- The full advisor pipeline works end-to-end
- Predictions are validated and reasonably accurate
- The user can select a strategy from the ranked table
- Hardware profiler, metadata fetcher, predictor, matrix generator, and ranker are all stable modules

### From Step 0 (revisit this)
- llama.cpp was cloned and built as a **standalone binary** in Step 0
- You ran models through `llama-cli` manually and recorded baseline numbers
- You understand how llama.cpp takes a GGUF file and generates tokens from the command line

### What Changes in This Step
- llama.cpp transitions from an **external binary you call** to an **internal library you link against**
- A new module (`executor.cpp`) appears that manages the full inference lifecycle
- A background thread is spawned for live hardware sampling during inference
- The output expands to include a predicted-vs-actual comparison report
- The model file must actually be downloaded (the header-only fetch from Step 2 is no longer sufficient)

---

## Phase A — Understand the llama.cpp C API

### Why the C API and Not C++
llama.cpp exposes a **C API** (declared in `llama.h`), not a C++ class hierarchy. This is deliberate — it provides a stable ABI that doesn't break across compiler versions. Even though you are writing C++, you will call C functions.

### The Core API Objects

llama.cpp's API revolves around four opaque pointer types:

| Type | What It Represents | Lifecycle |
|---|---|---|
| `llama_model*` | The loaded model weights in memory | Created once, shared across contexts, freed at shutdown |
| `llama_context*` | A specific inference configuration (context length, batch size, GPU layers) | Created per-run, freed after generation |
| `llama_sampler*` | The token sampling strategy (temperature, top-k, top-p) | Created per-run, freed after generation |
| `llama_batch` | A batch of tokens to process (input prompt or generated tokens) | Allocated per-run, freed after generation |

### The Lifecycle Sequence

Every inference run follows this exact sequence:

```
1. llama_backend_init()           ← once at program start
2. llama_model_load_from_file()   ← loads weights from GGUF
3. llama_new_context_with_model() ← creates inference context with your config
4. llama_sampler_chain_init()     ← sets up token sampling
5. [inference loop]               ← prompt processing + token generation
6. llama_sampler_free()           ← cleanup
7. llama_free()                   ← free context
8. llama_model_free()             ← free model (or keep for next run)
9. llama_backend_free()           ← once at program shutdown
```

### The Key Configuration Struct

When you call `llama_new_context_with_model()`, you pass a `llama_context_params` struct. This is where your strategy configuration translates into llama.cpp settings:

| Field | Type | Maps To (Your StrategyConfig) | Notes |
|---|---|---|---|
| `n_ctx` | uint32 | `context_length` | The context window size |
| `n_batch` | uint32 | `batch_size` (for prefill) | How many prompt tokens to process at once. Higher = faster prefill but more VRAM. Typical: 512–2048. |
| `n_ubatch` | uint32 | Related to batch | Physical batch size. Usually set equal to `n_batch`. |
| `n_threads` | uint32 | Derived from hardware | CPU threads for computation. Set to physical core count (not hyperthreads). |
| `n_threads_batch` | uint32 | Derived from hardware | CPU threads for batch processing (prefill). Can be higher than decode threads. |
| `offload_kqv` | bool | Always `true` for GPU strategies | Whether to offload KQV computation to GPU |
| `flash_attn` | bool | Optional optimization | Flash attention — saves VRAM on long contexts. Enable if available. |
| `type_k` | enum | `kv_quant_bits` | KV cache key precision: `GGML_TYPE_F16`, `GGML_TYPE_Q8_0`, `GGML_TYPE_Q4_0` |
| `type_v` | enum | `kv_quant_bits` | KV cache value precision (same options) |

### The Model Loading Configuration

When you call `llama_model_load_from_file()`, you pass a `llama_model_params` struct:

| Field | Type | Maps To | Notes |
|---|---|---|---|
| `n_gpu_layers` | int32 | `gpu_layers` | **The most critical field.** How many layers to offload to GPU. -1 = all layers. 0 = CPU only. |
| `main_gpu` | int32 | 0 (single GPU) | Which GPU to use. For MVP with one GPU, always 0. |
| `vocab_only` | bool | `false` | Set to `true` only if you just want the tokenizer (not relevant here). |
| `use_mmap` | bool | `true` for CPU strategies | Memory-map the model file instead of reading it into RAM. Critical for large models on limited RAM. |
| `use_mlock` | bool | `false` | Lock model in RAM to prevent swapping. Only use if you have enough RAM and want to guarantee no swap. |

### The Inference Loop (The Complex Part)

The inference loop has two phases:

**Phase 1: Prefill (Prompt Processing)**
- Take the user's prompt, tokenize it into token IDs
- Feed all prompt tokens to the model in one (or a few) batches
- The model processes them in parallel (compute-bound)
- This produces the TTFT measurement

**Phase 2: Decode (Token Generation)**
- Sample the next token from the model's output logits
- Feed that single token back into the model
- Repeat until: max tokens reached, stop token generated, or user interrupts
- Each iteration produces one token (bandwidth-bound)
- This produces the tokens/sec measurement

---

## Phase B — Build llama.cpp as a Library

### The Shift from Binary to Library
In Step 0, you built llama.cpp as a standalone executable (`llama-cli`). Now you need to build it as a **library** that your project links against.

### What llama.cpp Provides
The llama.cpp CMake build system produces several targets:
- `llama` — the core library (this is what you want)
- `ggml` — the low-level tensor computation library (llama depends on this)
- `llama-cli`, `llama-server`, etc. — the executables (you don't need these)

### CMake Integration Strategy

You have two options for integrating llama.cpp into your project:

**Option A: add_subdirectory (Recommended for MVP)**
Add the llama.cpp source tree as a subdirectory of your project. CMake builds it alongside your code.

```
llm-planner/
├── CMakeLists.txt
├── external/
│   └── llama.cpp/          ← git clone or git submodule
├── src/
│   ├── main.cpp
│   ├── profiler.cpp
│   ├── fetcher.cpp
│   ├── predictor.cpp
│   ├── matrix.cpp
│   ├── ranker.cpp
│   └── executor.cpp        ← new
```

In your `CMakeLists.txt`:
- Add `add_subdirectory(external/llama.cpp)`
- Link your executable against `llama` and `ggml` targets
- Set the CUDA-related CMake variables before the `add_subdirectory` call (the same flags you used in Step 0)

**Option B: Pre-built library**
Build llama.cpp separately, install it to a local directory, and use `find_package()` or manual `target_link_libraries()` to link against it.

**Recommendation:** Option A for MVP. It's simpler, ensures version consistency, and avoids "it works on my machine" issues. The downside is longer build times (llama.cpp recompiles every time you clean your project), but for MVP this is acceptable.

### CMake Configuration Details

Before calling `add_subdirectory(external/llama.cpp)`, set these variables:

| Variable | Value | Why |
|---|---|---|
| `GGML_CUDA` | `ON` | Enables CUDA backend (same as Step 0) |
| `CMAKE_CUDA_ARCHITECTURES` | Your GPU's arch (e.g., `86`) | Same as Step 0 |
| `LLAMA_BUILD_EXAMPLES` | `OFF` | Don't build llama-cli, llama-server, etc. Saves build time. |
| `LLAMA_BUILD_TESTS` | `OFF` | Don't build llama.cpp's test suite |
| `LLAMA_BUILD_SERVER` | `OFF` | Don't build the HTTP server |
| `BUILD_SHARED_LIBS` | `OFF` | Build static libraries. Avoids DLL deployment headaches on Windows. |

### Build Verification
After configuring CMake and building:
- Confirm that `llama.lib` and `ggml.lib` (or `.a` files) are produced in the build directory
- Confirm that your executable links against them without unresolved symbol errors
- Confirm that the CUDA backend is included (look for `ggml-cuda.lib` or similar in the build output)

### Common Linking Issues on Windows

| Error | Cause | Fix |
|---|---|---|
| Unresolved `cudaMalloc`, `cudaMemcpy`, etc. | CUDA runtime library not linked | Add `cudart` to `target_link_libraries` |
| Unresolved `cublasCreate`, etc. | cuBLAS not linked | Add `cublas` to `target_link_libraries` |
| Unresolved `WSAStartup`, etc. | Winsock not linked (llama.cpp's networking code) | Add `ws2_32` to `target_link_libraries` |
| Duplicate symbol errors | Both your code and llama.cpp define `main()` | Ensure you're linking against the `llama` library target, not the `llama-cli` executable target |
| Runtime crash on `llama_backend_init()` | CUDA DLLs not in PATH | Ensure CUDA toolkit `bin` directory is in system PATH, or copy required DLLs next to your executable |

---

## Phase C — The Model Download Problem

### The New Requirement
Steps 1–5 never needed the full model file. Step 2 fetched only the 64KB header. Step 6 needs the **entire GGUF file** on disk because llama.cpp must load the actual weights.

### The Download Strategy

**Option A: Assume the model is already downloaded (Recommended for MVP)**
The user provides a local file path instead of (or in addition to) a Hugging Face URL. The tool checks if the file exists locally. If it does, proceed. If it doesn't, print an error telling the user to download it first.

**Why this is the right MVP choice:**
- Downloading multi-GB files is a solved problem (browser, `huggingface-cli`, `wget`)
- Adding a download manager to your tool is a significant feature that distracts from the core value
- The user already downloaded the model in Step 0 for testing

**Option B: Download if missing (Phase 2 feature)**
If the file doesn't exist locally, use libcurl to download it with a progress bar. This is straightforward (you already have libcurl) but adds complexity around:
- Resume support (partial downloads)
- Disk space checking before starting
- Progress reporting to the user
- Handling download interruptions

**For MVP:** Go with Option A. Extend the CLI to accept `--model-path <local_path>` in addition to `--model <url>`. If both are provided, use the local path for execution and the URL for metadata. If only the URL is provided, check if the file exists in a default models directory (e.g., `C:\dev\models\`) by matching the filename from the URL.

### File Path Resolution Logic

```
1. If --model-path is provided:
   → Check if file exists
   → If yes, use it
   → If no, error: "File not found: <path>"

2. If only --model URL is provided:
   → Extract filename from URL (e.g., "Llama-3.2-3B-Q4_K_M.gguf")
   → Check if C:\dev\models\<filename> exists
   → If yes, use it
   → If no, error: "Model file not found locally. Download it first:
      huggingface-cli download <repo> <filename> --local-dir C:\dev\models\"
```

---

## Phase D — The Executor Module Implementation

### The Executor's Interface

The executor is a single function (or a small class) that takes the selected strategy and runs it:

**Input:**
- `ModelMetadata` (from Step 2)
- `HardwareSpec` (from Step 1)
- `StrategyConfig` (the user's selected strategy)
- `Prediction` (the predicted performance for this strategy)
- `model_file_path` (local path to the GGUF file)
- `prompt` (the user's input text, or a default benchmark prompt)
- `max_tokens` (how many tokens to generate, e.g., 100 for benchmarking)

**Output:**
- `ExecutionResult` struct containing:
  - `actual_tokens_per_sec` (measured decode speed)
  - `actual_ttft_ms` (measured time to first token)
  - `actual_peak_vram_bytes` (highest VRAM usage observed during run)
  - `actual_peak_ram_bytes` (highest RAM usage observed during run)
  - `throttled` (boolean — did GPU thermal throttle during the run?)
  - `tokens_generated` (total count)
  - `generation_text` (the actual generated text, for verification)
  - `error_message` (empty if successful)

### The Execution Sequence (Detailed)

**Step D1: Initialize Backend**
```
llama_backend_init()
```
This initializes ggml's compute backends (CUDA, CPU, etc.). Call this once at program start, not per-run. If you plan to run multiple models in a session, keep this alive.

**Step D2: Load the Model**
```
llama_model_params model_params = llama_model_default_params()
model_params.n_gpu_layers = strategy.gpu_layers  // THE KEY SETTING
model_params.main_gpu = 0
model_params.use_mmap = (strategy.placement == CPU_ONLY)  // mmap for CPU strategies
model_params.use_mlock = false

llama_model* model = llama_model_load_from_file(model_file_path, model_params)
```

**Error handling:** If `model` is `nullptr`, the load failed. Common causes:
- File not found or corrupted
- Not enough memory (VRAM or RAM) for the requested `n_gpu_layers`
- GGUF version mismatch with the llama.cpp build

Print the error and return an `ExecutionResult` with `error_message` set.

**Step D3: Create the Context**
```
llama_context_params ctx_params = llama_context_default_params()
ctx_params.n_ctx = strategy.context_length
ctx_params.n_batch = 512  // reasonable default for prefill
ctx_params.n_ubatch = 512
ctx_params.n_threads = getPhysicalCoreCount()  // from hwloc or std::thread::hardware_concurrency() / 2
ctx_params.n_threads_batch = getPhysicalCoreCount()
ctx_params.offload_kqv = (strategy.gpu_layers > 0)
ctx_params.flash_attn = true  // enable if supported

// KV cache precision
if (strategy.kv_quant_bits == 8) {
    ctx_params.type_k = GGML_TYPE_Q8_0
    ctx_params.type_v = GGML_TYPE_Q8_0
} else if (strategy.kv_quant_bits == 4) {
    ctx_params.type_k = GGML_TYPE_Q4_0
    ctx_params.type_v = GGML_TYPE_Q4_0
} else {
    ctx_params.type_k = GGML_TYPE_F16
    ctx_params.type_v = GGML_TYPE_F16
}

llama_context* ctx = llama_new_context_with_model(model, ctx_params)
```

**Error handling:** If `ctx` is `nullptr`, context creation failed. Most common cause: not enough VRAM/RAM for the requested context length. This is exactly the kind of failure your predictor should have caught — if it happens here, your memory formula needs recalibration.

**Step D4: Tokenize the Prompt**
```
// Allocate token buffer
std::vector<llama_token> prompt_tokens(prompt.length() + 16)  // over-allocate
int n_tokens = llama_tokenize(
    model,
    prompt.c_str(),
    prompt.length(),
    prompt_tokens.data(),
    prompt_tokens.size(),
    true,   // add special tokens (BOS)
    false   // don't parse special tokens in the prompt text
)
prompt_tokens.resize(n_tokens)
```

**Step D5: Start the Live Hardware Sampler**
Before starting inference, spawn a background thread that polls hardware state every 500ms. This is described in detail in Phase E below.

**Step D6: Prefill (Prompt Processing)**
```
// Create a batch for the prompt
llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size())

// Record start time
auto t_start = high_resolution_clock::now()

// Process the prompt
int result = llama_decode(ctx, batch)
if (result != 0) {
    // Error: prompt processing failed (usually OOM)
}

// Record TTFT
auto t_ttft = high_resolution_clock::now()
actual_ttft_ms = duration_cast<milliseconds>(t_ttft - t_start).count()
```

**Important:** For long prompts that exceed `n_batch`, you need to process the prompt in chunks. For MVP, keep the benchmark prompt short (under 512 tokens) to avoid this complexity.

**Step D7: Decode Loop (Token Generation)**
```
auto t_decode_start = high_resolution_clock::now()
int tokens_generated = 0
std::string generated_text = ""

for (int i = 0; i < max_tokens; i++) {
    // Sample the next token
    llama_token new_token = llama_sampler_sample(sampler, ctx, -1)
    
    // Check for end-of-generation
    if (llama_token_is_eog(model, new_token)) {
        break
    }
    
    // Convert token to text and append
    char buf[256]
    int n = llama_token_to_piece(model, new_token, buf, sizeof(buf), 0, true)
    generated_text += std::string(buf, n)
    
    // Feed the token back for the next iteration
    llama_batch batch = llama_batch_get_one(&new_token, 1)
    int result = llama_decode(ctx, batch)
    if (result != 0) {
        // Error during generation
        break
    }
    
    tokens_generated++
}

auto t_decode_end = high_resolution_clock::now()
double decode_seconds = duration_cast<microseconds>(t_decode_end - t_decode_start).count() / 1e6
actual_tokens_per_sec = tokens_generated / decode_seconds
```

**Step D8: Stop the Live Sampler**
Signal the background thread to stop. Collect the peak VRAM, peak RAM, and throttle data.

**Step D9: Cleanup**
```
llama_sampler_free(sampler)
llama_free(ctx)
llama_model_free(model)  // or keep alive for subsequent runs
// llama_backend_free() — only at program exit
```

**Step D10: Return ExecutionResult**
Package all measured values into the result struct and return.

---

## Phase E — Live Hardware Sampler (Background Thread)

### Why a Background Thread
The inference loop is synchronous — it blocks the main thread while generating tokens. You can't poll NVML in the middle of `llama_decode()` because you don't control when it returns. The solution is a separate thread that samples hardware state at regular intervals.

### The Sampler Thread Design

**Thread lifecycle:**
1. Main thread spawns the sampler thread just before prefill (Step D5)
2. Sampler thread loops: sleep 500ms → poll NVML → poll RAM → store readings
3. Main thread signals the sampler to stop after decode completes (Step D8)
4. Sampler thread exits and returns its collected data

**Data collected per sample:**
| Metric | Source | Frequency |
|---|---|---|
| VRAM used | `nvmlDeviceGetMemoryInfo()` | Every 500ms |
| GPU temperature | `nvmlDeviceGetTemperature()` | Every 500ms |
| GPU clock speed | `nvmlDeviceGetClockInfo()` | Every 500ms |
| RAM used | `GlobalMemoryStatusEx()` | Every 500ms |

**Thermal throttle detection:**
Compare the current GPU clock speed against the max clock speed (from Step 1). If the current clock drops below 85% of max for two consecutive samples, flag `throttled = true`. This indicates the GPU is thermal-throttling, which means your predicted tokens/sec will be higher than actual.

**Implementation details:**
- Use `std::thread` for the sampler thread
- Use `std::atomic<bool>` for the stop signal (thread-safe, no mutex needed)
- Store readings in a `std::vector<HardwareSample>` protected by a `std::mutex` (or just track running max values to avoid storing the full history)
- The sampler thread should handle NVML errors gracefully — if a poll fails, skip that sample, don't crash

### The HardwareSample Struct
```
struct HardwareSample {
    uint64_t vram_used_bytes
    uint64_t ram_used_bytes
    uint32_t gpu_temp_celsius
    uint32_t gpu_clock_mhz
    uint64_t timestamp_ms
}
```

### Post-Run Aggregation
After the sampler thread exits, aggregate the samples:
- `peak_vram = max(all_samples.vram_used_bytes)`
- `peak_ram = max(all_samples.ram_used_bytes)`
- `max_temp = max(all_samples.gpu_temp_celsius)`
- `throttled = any sample where gpu_clock < 0.85 × max_clock`

---

## Phase F — The Predicted-vs-Actual Report

### The Comparison Output
After the run completes, print a side-by-side comparison:

```
=== Execution Complete — Predicted vs Actual ===
Strategy: Full GPU, 28/28 layers, 4K context, FP16 KV

Metric              Predicted    Actual       Delta     Status
─────────────────── ────────── ────────── ────────── ──────────
Tokens/sec          ~385         371          -3.6%     ✅ Close
TTFT                ~45ms        52ms         +15.6%    ✅ Close
Peak VRAM           2.4 GB       2.6 GB       +8.3%     ✅ Close
Peak RAM            0.5 GB       0.7 GB       +40.0%    ⚠️  Off
Thermal Throttle    No           No           —         ✅ Match

Generated 100 tokens in 0.27 seconds.
First token in 52ms.

💡 The RAM prediction was off by 40%. This will be recalibrated
   for future predictions on this hardware.
```

### The Delta Thresholds
| Delta | Status | Meaning |
|---|---|---|
| <10% | ✅ Close | Prediction is accurate |
| 10-25% | ⚠️ Off | Prediction needs calibration |
| >25% | ❌ Wrong | Formula or constant is significantly wrong |

### What to Do With the Delta
For MVP, just display it. The calibration log in Step 7 will use these deltas to adjust the runtime overhead and efficiency constants. Don't try to auto-correct in Step 6 — that's Step 7's job.

---

## Phase G — The User Interaction Flow

### How the User Selects a Strategy
After the ranked table is printed (Step 5), the tool prompts the user:

```
Select a strategy to execute (1-8), or 'q' to quit: _
```

The user types a number. The tool validates it (is it in range? is the strategy viable?) and proceeds with execution.

**If the user selects a non-viable strategy:**
```
⚠️  Strategy #8 is not viable (exceeds VRAM by 7.1 GB).
   The model will likely fail to load. Execute anyway? (y/n): _
```

This gives the user the option to try it anyway (maybe they closed some apps since the prediction was made), but warns them clearly.

### The Benchmark Prompt
For consistent measurement, use a fixed benchmark prompt rather than asking the user to type something. A good default:

```
"The following is a detailed explanation of how quantum computing works, 
step by step:"
```

This prompt is:
- Long enough to produce a measurable TTFT (~20 tokens)
- Open-ended enough that the model will generate many tokens without stopping early
- Content-neutral (doesn't trigger safety filters)

Allow the user to override with `--prompt "your text"` if they want.

### The Token Limit
Default to 100 tokens for benchmarking. This is enough to get a stable tokens/sec measurement without making the user wait too long. Allow override with `--max-tokens N`.

---

## Phase H — Error Handling and Failure Modes

### The Executor Can Fail in Many Ways

| Failure | When | Cause | How to Handle |
|---|---|---|---|
| Model file not found | Load | User didn't download the model | Clear error with download instructions |
| Model load OOM | Load | `n_gpu_layers` too high for available VRAM | Catch the error, suggest reducing GPU layers. This means the predictor's memory formula was too optimistic — flag for calibration. |
| Context creation OOM | Context | Context length too large for remaining memory | Catch the error, suggest reducing context. Same calibration flag. |
| Tokenization failure | Prefill | Prompt contains unsupported characters | Unlikely with modern models, but handle gracefully |
| Decode returns error code | Decode | Memory corruption, CUDA error, or hardware fault | Stop generation, report partial results, print the CUDA error string |
| GPU falls off the bus | Decode | Hardware failure, driver crash, or extreme thermal event | The sampler thread will detect VRAM reading failures. Stop immediately. This is a hardware problem, not a software problem. |
| Thermal throttle detected | Decode | Sustained load causes GPU to reduce clock speed | Don't stop — just flag it in the report. The user should know their prediction was based on non-throttled performance. |
| OS starts swapping | Decode | RAM oversubscribed, OS writes to page file | The sampler thread detects this via `GlobalMemoryStatusEx()` — if `ullAvailPageFile` drops sharply, flag it. This is the "severe slowdown" failure mode from §3 item 7. |
| Generation produces garbage | Decode | Model is corrupted, or quantization is too aggressive | The user will see the output. Don't try to auto-detect quality — just display it. |

### The "Graceful Abort" Mechanism
The user should be able to stop generation at any time by pressing Ctrl+C. Register a signal handler that sets a global `std::atomic<bool> abort_requested` flag. The decode loop checks this flag every iteration and breaks if set.

```
// In the decode loop:
if (abort_requested.load()) {
    break  // stop generation, report partial results
}
```

Don't just `exit()` on Ctrl+C — clean up the llama.cpp context and model first, or you'll leak GPU memory.

---

## Phase I — Integration with the Existing Pipeline

### The Extended Pipeline

```
Before Step 6:
    Profile → Fetch → Matrix → Predict → Rank → Print Table → Exit

After Step 6:
    Profile → Fetch → Matrix → Predict → Rank → Print Table → 
    User Selects → Download Check → Load Model → Create Context → 
    Start Sampler → Prefill → Decode → Stop Sampler → Cleanup → 
    Print Predicted-vs-Actual → (Feed to Calibration Log in Step 7)
```

### The `--execute` Flag
For MVP, add an `--execute` flag to the CLI. Without it, the tool behaves exactly as in Steps 1–5 (advisor only). With it, the tool proceeds to execution after printing the table.

```
llm-planner --model <url> --execute
llm-planner --model <url> --execute --prompt "Explain relativity" --max-tokens 200
```

**Why a flag instead of always executing:** Because the advisor mode (Steps 1–5) is useful on its own. Users might want to compare strategies without actually running anything. Don't force execution on every invocation.

---

## Phase J — Testing the Executor

### Test Scenarios

**Test 1: The Happy Path**
- Select a full-GPU strategy for a small model (3B Q4_K_M)
- Execute with 100 tokens
- Verify: Generation completes, output text is coherent, predicted-vs-actual deltas are within 20%

**Test 2: The Split Strategy**
- Select a GPU+CPU split strategy
- Verify: llama.cpp's startup output shows the correct number of layers on GPU
- Verify: Tokens/sec is lower than full GPU (as predicted)
- Verify: The split speed prediction is within 25% of actual

**Test 3: The CPU-Only Strategy**
- Select CPU-only
- Verify: All layers on CPU (llama.cpp output shows "0 layers offloaded")
- Verify: Tokens/sec matches the RAM-bandwidth-based prediction
- Verify: VRAM usage stays near zero

**Test 4: The Tight Fit**
- Select a strategy that uses >90% VRAM
- Verify: The model loads successfully (or fails gracefully with a clear error)
- Verify: The sampler captures peak VRAM accurately

**Test 5: The OOM Failure**
- Select a strategy that exceeds VRAM (a non-viable one, if you chose to override the warning)
- Verify: The tool catches the OOM error and prints a clear message
- Verify: No crash, no GPU memory leak (check with nvidia-smi after the error)

**Test 6: The Thermal Throttle**
- Run a long generation (500+ tokens) on a laptop or poorly-cooled GPU
- Verify: The sampler detects clock speed reduction
- Verify: The report flags `throttled = true`
- Verify: Actual tokens/sec is lower than predicted (because prediction assumed non-throttled)

**Test 7: Ctrl+C Abort**
- Start a long generation and press Ctrl+C mid-run
- Verify: Generation stops cleanly
- Verify: Partial results are reported
- Verify: GPU memory is freed (check nvidia-smi)
- Verify: The tool exits without crashing

**Test 8: Memory Leak Check**
- Run 5 consecutive generations with different strategies
- Verify: VRAM usage returns to baseline after each run (check nvidia-smi)
- Verify: RAM usage doesn't grow monotonically

---

## Step 6 — Done Checklist

Before moving to Step 7, confirm every item:

- [ ] llama.cpp is built as a static library and linked into your executable
- [ ] CMake configuration includes CUDA backend flags
- [ ] `llama_backend_init()` is called once at program start
- [ ] Model loads successfully from a local GGUF file with the correct `n_gpu_layers`
- [ ] Context is created with the correct `n_ctx`, KV cache type, and thread count
- [ ] Prompt tokenization works correctly
- [ ] Prefill completes and TTFT is measured accurately
- [ ] Decode loop generates tokens at the predicted speed (within 20%)
- [ ] End-of-generation is detected correctly (stop tokens, max tokens)
- [ ] Live sampler thread starts before inference and stops after
- [ ] Sampler captures peak VRAM, peak RAM, and thermal state
- [ ] Thermal throttle detection works (clock speed comparison)
- [ ] Predicted-vs-actual report is printed after each run
- [ ] Delta percentages are calculated correctly
- [ ] Non-viable strategy execution is warned but allowed
- [ ] OOM errors during model load are caught and reported clearly
- [ ] OOM errors during context creation are caught and reported clearly
- [ ] Ctrl+C aborts generation cleanly without crashing or leaking memory
- [ ] GPU memory is fully freed after each run (verified with nvidia-smi)
- [ ] `--execute` flag controls whether execution happens
- [ ] `--prompt` and `--max-tokens` flags work correctly
- [ ] Model file not found produces a clear error with download instructions
- [ ] Tested with at least 2 different placement strategies (full GPU + split or CPU)
- [ ] Tested with at least 2 different context lengths

---

## Common Failure Points at Step 6

| Problem | Likely Cause | Fix |
|---|---|---|
| Linker errors when building with llama.cpp | Missing CUDA libraries or wrong target name | Link against `llama`, `ggml`, `ggml-cuda`, `cudart`, `cublas`. Check llama.cpp's CMake targets. |
| `llama_model_load_from_file()` returns nullptr | File path wrong, or not enough memory for requested GPU layers | Verify the file path. Try with `n_gpu_layers = 0` first to confirm the file is valid, then increase. |
| `llama_new_context_with_model()` returns nullptr | Context length too large for available memory | Reduce `n_ctx`. Check if the KV cache at the requested context fits in remaining VRAM/RAM. |
| `llama_decode()` returns non-zero | Batch too large, or CUDA out of memory during computation | Reduce `n_batch`. Check if flash attention is enabled (it reduces VRAM usage). |
| Generated text is garbage | Model file corrupted, or wrong tokenizer | Verify the model loads correctly in vanilla `llama-cli` first. If it works there but not in your tool, your API usage is wrong. |
| Tokens/sec is much lower than predicted | Thermal throttling, or background GPU processes | Check the sampler's throttle flag. Close other GPU apps. Verify the bandwidth number from Step 1 is still accurate. |
| Tokens/sec is much higher than predicted | Prediction used wrong bandwidth, or model is smaller than metadata suggests | Re-check the predictor's bandwidth input. Verify the model's actual parameter count matches the metadata. |
| VRAM not freed after run | Missing `llama_free()` or `llama_model_free()` call | Ensure cleanup happens in all code paths, including error paths. Use RAII wrappers if possible. |
| Sampler thread crashes | NVML call fails because GPU is busy with inference | NVML calls are generally safe during inference, but wrap them in try-catch or check return codes. Skip failed samples. |
| Program hangs on exit | Sampler thread not stopped, or llama.cpp backend not freed | Ensure the stop signal is sent to the sampler thread and `join()` is called. Call `llama_backend_free()` at the very end. |
| Build time explodes | llama.cpp recompiles every time | This is the cost of `add_subdirectory`. Consider Option B (pre-built library) if build times become unbearable. |
| Runtime DLL errors on Windows | CUDA DLLs not found at runtime | Add CUDA toolkit `bin` to PATH, or use `set(CMAKE_INSTALL_RPATH)` in CMake, or copy DLLs to the output directory. |

---

## Time Estimate for Step 6
- Phase A (Understanding the llama.cpp C API): **3–4 hours** (reading `llama.h`, studying the example code in `llama.cpp/examples/`)
- Phase B (Building llama.cpp as a library + CMake integration): **3–5 hours** (this is where most people get stuck — CMake linking issues on Windows with CUDA)
- Phase C (Model download logic): **1–2 hours** (file path resolution and existence checking)
- Phase D (Executor implementation): **6–10 hours** (the inference loop is the most complex code in the entire project)
- Phase E (Live hardware sampler thread): **3–4 hours** (threading + NVML polling + throttle detection)
- Phase F (Predicted-vs-actual report): **1–2 hours**
- Phase G (User interaction flow): **1–2 hours**
- Phase H (Error handling): **3–4 hours** (OOM handling, graceful abort, cleanup)
- Phase I (Pipeline integration): **1–2 hours**
- Phase J (Testing): **4–6 hours** (all test scenarios, memory leak checks, edge cases)

**Total: 3–5 days, as originally estimated. This is the hardest step by far. The CMake linking (Phase B) and the inference loop (Phase D) are where most of the debugging time goes. Don't rush this step — if Steps 1–5 are solid, you have a working advisor even if Step 6 takes longer than expected.**