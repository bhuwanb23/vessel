# Phase 3 — Making It Capable of Everything: Full Roadmap

---

## Where You Stand Right Now

You have built a tool that, for a given model URL and a Windows+NVIDIA machine, can:

- Profile real hardware (RAM, VRAM, NVMe, bandwidth)
- Fetch model metadata without downloading weights (64KB–2MB header)
- Enumerate every viable deployment strategy (full GPU, split, CPU, MoE expert-offload, hot/cold neuron split, layer-streaming)
- Predict memory, speed, and latency for each strategy with confidence bands
- Rank strategies by user priority (speed/quality/safety)
- Download the model safely (resumable, verified)
- Execute the chosen strategy via llama.cpp
- Self-calibrate from real runs

**This is already the most comprehensive local LLM deployment tool in existence.** No other tool does even half of this.

But "capable of everything almost" means closing the remaining gaps. Here is what is still missing, organized by impact and dependency order.

---

## The Gap Analysis

| Gap | Current State | User Impact | Effort |
|---|---|---|---|
| Only NVIDIA GPUs | AMD, Intel, Apple users get nothing | **Massive** — ~40% of the market | High |
| CLI only | Non-technical users can't use it | **High** — limits adoption | Medium |
| One model at a time | Can't compare models side-by-side | **Medium** — users want "which model should I run?" | Low |
| No API server mode | Can't integrate with other tools (Open WebUI, SillyTavern, etc.) | **High** — the ecosystem expects OpenAI-compatible APIs | Medium |
| Local-only calibration | Your machine learns, others don't | **Medium** — cold-start problem for new users | High |
| No quality scoring | Quality ranking uses proxy metrics, not real perplexity | **Medium** — users care about output quality | Medium |
| No auto-recommendation | User must know which model URL to paste | **High** — "just tell me what to run" | Low |
| No live monitoring | Can't watch a long session's health | **Low** — nice-to-have | Low |
| No multi-model serving | Can't run two models simultaneously | **Low** — niche use case | High |
| No distributed inference | Single machine only | **Low** — power users only | Very High |
| No training/fine-tuning | Inference only | **Out of scope** — different tool | N/A |

---

## Phase 3 Build Order (12 Steps, Prioritized)

### TIER 1 — Maximum Impact, Builds Directly on Existing Code

---

### Step 11 — Platform Expansion: AMD (ROCm) + Apple (Metal)

**The gap it fills:** Your `IHardwareProfiler` and `IExecutionBackend` interfaces from §6 were designed for exactly this. Right now they have one implementation each (NVML + CUDA). Adding AMD and Apple means filling in the other implementations behind the same interfaces.

**What changes:**
- **AMD Profiler:** Replace NVML calls with `rocm-smi` library calls (`librocm_smi64`). Same data: VRAM, bandwidth, temperature, compute capability. The API is structurally similar to NVML.
- **AMD Executor:** llama.cpp already supports AMD via the `GGML_HIPBLAS` backend. Your CMake config adds `-DGGML_HIPBLAS=ON` instead of `-DGGML_CUDA=ON`. The C API is identical — `llama_model_params`, `llama_context_params`, tensor overrides all work the same.
- **Apple Profiler:** Use `IOKit` / `Metal` API to read unified memory size, GPU core count, and memory bandwidth. Apple Silicon has no separate VRAM — everything is unified memory, which simplifies the placement matrix (no GPU/CPU split, just "how much unified memory to allocate").
- **Apple Executor:** llama.cpp's `GGML_METAL` backend. Same C API. The key difference: no `n_gpu_layers` concept in the traditional sense — Metal uses unified memory, so the placement strategies collapse to "how much memory to reserve for the model vs the OS."

**What doesn't change:**
- Predictor formulas (memory, speed, TTFT) — same math, different bandwidth/TFLOPS constants
- Method matrix structure — same strategies, some may be NO FIT or N/A on specific platforms
- Ranker, calibration log, download manager — platform-agnostic
- All Steps 1–10 logic — untouched

**Key architectural decision:** The `IHardwareProfiler` interface now has three implementations:
```
NvidiaProfiler : IHardwareProfiler   // NVML
AmdProfiler : IHardwareProfiler      // ROCm-SMI
AppleProfiler : IHardwareProfiler    // Metal/IOKit
```
Runtime detection: check which GPU driver is present, instantiate the correct profiler. The rest of the pipeline doesn't know or care.

**Time estimate:** 1–2 weeks (AMD: 3–5 days, Apple: 4–7 days, testing: 2–3 days)

---

### Step 12 — Auto-Recommendation Engine ("What Should I Run?")

**The gap it fills:** Right now the user must know which model URL to paste. Most users don't know whether they should run Llama 3.2 3B or Qwen 2.5 7B or Mistral 7B on their specific hardware. The tool should answer "what's the best model for my machine?" not just "how should I run this specific model?"

**What it does:**
1. Profile the hardware (Step 1)
2. Query a curated model catalog (a local JSON file or a remote API) containing the most popular models with their GGUF variants
3. For each model in the catalog, run the predictor (Step 3) against the user's hardware
4. Rank all model×strategy combinations by the user's priority
5. Present the top 5–10 recommendations

**Example output:**
```
=== Recommended Models for Your Hardware ===
Hardware: RTX 3060 (12GB VRAM) | 16GB RAM

 #  Model                    Quant    Strategy    tok/s    Quality   VRAM
 1  Qwen 2.5 7B Instruct    Q4_K_M   Full GPU    ~45      ★★★★☆    5.2 GB
 2  Llama 3.2 8B Instruct   Q4_K_M   Full GPU    ~38      ★★★★☆    5.8 GB
 3  Mistral 7B v0.3         Q5_K_M   Full GPU    ~32      ★★★★☆    6.1 GB
 4  Qwen 2.5 14B Instruct   Q3_K_M   Split       ~18      ★★★☆☆    8.9 GB
 5  Llama 3.2 3B Instruct   Q8_0     Full GPU    ~95      ★★★☆☆    3.8 GB

💡 Best balance of speed and quality: #1 (Qwen 2.5 7B Q4_K_M).
   For maximum speed: #5 (Llama 3.2 3B Q8_0) at 95 tok/s.
   For maximum quality: #3 (Mistral 7B Q5_K_M) — slower but higher quant.
```

**The model catalog:** A JSON file shipped with the tool, updated periodically. Contains:
- Model name, architecture, parameter count
- Available GGUF quants with file sizes and download URLs
- Published quality scores (from Open LLM Leaderboard or similar)
- Minimum hardware requirements

**Time estimate:** 3–5 days (catalog format + recommendation logic + CLI integration)

---

### Step 13 — GUI / Web Dashboard

**The gap it fills:** CLI is powerful but limits adoption. A web dashboard makes the tool accessible to non-technical users and provides a visual interface for the prediction table, hardware monitoring, and model management.

**Architecture:**
- **Backend:** Your existing C++ binary, extended with a lightweight HTTP server (use `cpp-httplib` — single header, no dependencies). Exposes the prediction pipeline as REST endpoints.
- **Frontend:** A single-page web app (HTML + JavaScript + CSS). No framework needed for MVP — vanilla JS is sufficient for a dashboard. Served by the same C++ backend.

**Key endpoints:**
```
GET  /api/hardware          → HardwareSpec JSON
POST /api/predict           → { model_url, priority } → Prediction table JSON
POST /api/download          → { model_url, target_dir } → Download progress stream
POST /api/execute           → { model_path, strategy } → Execution results JSON
GET  /api/calibration       → Calibration stats JSON
GET  /api/recommendations   → Auto-recommendation JSON
```

**Dashboard views:**
1. **Hardware Overview:** Real-time gauges for VRAM, RAM, GPU temp, disk speed
2. **Model Predictor:** Paste a URL, see the strategy table with visual bars for speed/memory
3. **Model Browser:** Browse the catalog, click to predict, click to download
4. **Execution Monitor:** Live tok/s, VRAM usage, temperature during a run
5. **Calibration History:** Charts showing prediction accuracy improving over time

**Time estimate:** 1–2 weeks (backend API: 2–3 days, frontend: 5–7 days, integration testing: 2–3 days)

---

### Step 14 — OpenAI-Compatible API Server Mode

**The gap it fills:** The local LLM ecosystem (Open WebUI, SillyTavern, Continue, Cursor, etc.) expects an OpenAI-compatible API endpoint (`/v1/chat/completions`, `/v1/models`, etc.). Right now your tool is a planner + executor, not a server. Adding API server mode makes it a drop-in replacement for Ollama or LM Studio's server.

**What it does:**
- `llm-planner --serve` starts an HTTP server on `localhost:8080`
- Exposes `/v1/models` — returns the list of locally available models with metadata
- Exposes `/v1/chat/completions` — accepts a chat request, selects the best strategy (or uses the user's cached preference), executes via llama.cpp, streams the response
- Exposes `/v1/completions` — same for raw text completion
- The planner runs automatically: when a new model is loaded, it profiles hardware, predicts the best strategy, and configures llama.cpp optimally

**Why this is powerful:** Every other API server (Ollama, llama.cpp server, LM Studio) uses a fixed or user-manual configuration. Yours auto-optimizes the placement based on real hardware profiling and calibrated predictions. The user just points their frontend at `localhost:8080` and gets the best possible performance without tuning anything.

**Implementation:**
- Use `cpp-httplib` (same as Step 13) for the HTTP server
- Implement the OpenAI API schema (well-documented, ~10 endpoints)
- Reuse your Executor (Step 6) for the actual inference
- Add streaming support (Server-Sent Events for token-by-token output)
- Add model hot-swapping (load/unload models based on API requests)

**Time estimate:** 1–2 weeks (API schema: 3–4 days, streaming: 2–3 days, model management: 2–3 days, testing with real frontends: 2–3 days)

---

### TIER 2 — Significant New Capabilities

---

### Step 15 — Cross-User Calibration Aggregation

**The gap it fills:** Right now every user starts with default constants and calibrates from scratch. A new user with an RTX 4070 gets mediocre predictions until they've run 10+ models. If you aggregate calibration data across users with the same hardware, new users get accurate predictions from day one.

**Architecture:**
- **Client side:** After each run, optionally upload the calibration entry (anonymized — hardware fingerprint + model + strategy + predicted vs actual, no personal data) to a central server
- **Server side:** A lightweight backend (can be a simple Python/Node.js service, or even a Cloudflare Worker + D1 database) that aggregates entries by hardware fingerprint
- **Query:** When the tool starts a prediction, it checks the server for existing calibration data matching the user's hardware fingerprint. If found, downloads the aggregated constants and uses them as the starting point.

**Privacy:** The hardware fingerprint is already somewhat identifying (specific GPU + CPU + RAM combo). Make upload opt-in. Never upload model names or generated text. The fingerprint can be hashed before upload to add a layer of anonymity.

**Time estimate:** 2–3 weeks (client upload: 2–3 days, server backend: 5–7 days, aggregation logic: 3–4 days, privacy review: 2–3 days)

---

### Step 16 — Real Quality Scoring (Perplexity-Based)

**The gap it fills:** Step 5's quality ranking uses proxy metrics (bits-per-weight, KV precision). Real quality scoring requires perplexity measurements — how well the model predicts held-out text at each quantization level.

**What it does:**
- Ship a small benchmark corpus (~10,000 tokens from WikiText or similar)
- For each model×quant the user downloads, optionally run a quick perplexity evaluation (1–5 minutes)
- Store the perplexity score in the calibration log
- Use real perplexity deltas in the quality ranking instead of proxy metrics

**The honest limitation:** Perplexity doesn't capture everything (instruction-following quality, reasoning ability, etc.). But it's the best automated metric available and is far better than "Q4 is lower quality than Q5" without numbers.

**Time estimate:** 3–5 days (perplexity evaluation loop: 2–3 days, integration with ranker: 1–2 days)

---

### Step 17 — Model Comparison Mode

**The gap it fills:** Users want to compare two or more models side-by-side on their hardware. "Should I run Llama 3.2 8B or Qwen 2.5 7B?" Right now they have to run the predictor twice and mentally compare.

**What it does:**
- `llm-planner --compare <url1> <url2> <url3>`
- Fetches metadata for all models (64KB each, no downloads)
- Runs the predictor for each model with the optimal strategy
- Prints a side-by-side comparison table

**Example:**
```
=== Model Comparison on RTX 3080 (10GB VRAM) ===

Metric          Llama 3.2 8B Q4    Qwen 2.5 7B Q4    Mistral 7B Q5
─────────────── ────────────────── ───────────────── ─────────────────
Params          8.0B               7.6B              7.2B
Quant           Q4_K_M (4.85 bpw)  Q4_K_M (4.85 bpw) Q5_K_M (5.69 bpw)
Best Strategy   Full GPU, 4K       Full GPU, 4K      Full GPU, 4K
VRAM            5.8 GB             5.2 GB            6.1 GB
tok/s           ~38                ~45               ~32
TTFT            ~65ms              ~55ms             ~72ms
Quality (PPL)   8.2                7.9               7.5
Max Context     32K                64K               16K

💡 Best speed: Qwen 2.5 7B. Best quality: Mistral 7B Q5.
   Best balance: Qwen 2.5 7B (fastest AND lowest perplexity).
```

**Time estimate:** 2–3 days (mostly output formatting and multi-model pipeline orchestration)

---

### Step 18 — Live Session Monitoring Dashboard

**The gap it fills:** During long inference sessions (hours of chat, batch processing), the user has no visibility into hardware health, memory pressure, or performance degradation over time.

**What it does:**
- During execution (Step 6), the live sampler thread (already exists) feeds data to a real-time dashboard
- Shows: tok/s over time (line chart), VRAM/RAM usage (gauges), GPU temperature (thermometer), throttle events (alerts)
- Detects anomalies: sudden speed drops (thermal throttle), memory creep (KV cache growing), swap activation (OS memory pressure)
- Alerts the user: "⚠️ GPU temperature reached 85°C, clock speed reduced by 15%. Consider reducing context length or improving cooling."

**Implementation:** The sampler thread already collects this data every 500ms. The dashboard is a web page (from Step 13) that polls or receives WebSocket updates.

**Time estimate:** 2–3 days (data pipeline already exists, just need the visualization)

---

### TIER 3 — Advanced / Niche Features

---

### Step 19 — Multi-Model Serving

**The gap it fills:** Running two models simultaneously (e.g., a small fast model for chat + a large slow model for reasoning). The tool manages VRAM/RAM allocation across models.

**Complexity:** High. Requires:
- VRAM partitioning (model A gets 6GB, model B gets 4GB)
- Dynamic model loading/unloading based on request patterns
- Shared KV cache management
- Request routing (which model handles which request)

**Time estimate:** 3–4 weeks

---

### Step 20 — Dynamic Expert Caching (MoE Enhancement)

**The gap it fills:** Step 9 uses static expert placement. Dynamic caching loads/unloads experts on demand based on routing patterns, keeping frequently-used experts in GPU memory and evicting cold ones.

**Complexity:** Very high. Requires:
- Runtime expert LRU cache
- Asynchronous expert loading (prefetch predicted experts while current token computes)
- PCIe bandwidth management
- Integration with llama.cpp's compute graph at a deep level

**Time estimate:** 4–6 weeks

---

### Step 21 — Distributed Multi-Node Inference

**The gap it fills:** Split a model across multiple machines on a network. Machine A holds layers 1–16, Machine B holds layers 17–32.

**Complexity:** Extremely high. Requires:
- Network communication layer (gRPC or custom)
- Tensor parallelism or pipeline parallelism
- Latency-aware placement (network bandwidth vs PCIe bandwidth)
- Fault tolerance

**Time estimate:** 2–3 months. This is essentially building a new distributed inference engine. Consider whether this is worth the effort vs. pointing users to existing solutions (vLLM, Ray Serve).

---

## The "Capable of Everything" Checklist

After completing Steps 11–18 (the practical tiers), your tool will:

| Capability | Status |
|---|---|
| Predict performance before download | ✅ Steps 1–5 |
| Support all major GPU vendors | ✅ Step 11 (NVIDIA + AMD + Apple) |
| Handle dense models | ✅ Steps 3–6, 10 |
| Handle MoE models | ✅ Step 9 |
| Handle extreme hardware constraints | ✅ Step 10 (hot/cold + layer-stream) |
| Download models safely | ✅ Step 8 |
| Execute and self-calibrate | ✅ Steps 6–7 |
| Recommend the best model for the hardware | ✅ Step 12 |
| Provide a visual interface | ✅ Step 13 |
| Serve as an OpenAI-compatible API | ✅ Step 14 |
| Leverage community calibration data | ✅ Step 15 |
| Score quality with real perplexity | ✅ Step 16 |
| Compare models side-by-side | ✅ Step 17 |
| Monitor long sessions | ✅ Step 18 |
| Multi-model serving | 🔲 Step 19 (advanced) |
| Dynamic expert caching | 🔲 Step 20 (advanced) |
| Distributed inference | 🔲 Step 21 (very advanced) |

---

## Recommended Build Order Summary

| Priority | Step | What | Time | Cumulative |
|---|---|---|---|---|
| 🔴 Now | 11 | AMD + Apple support | 1–2 weeks | 2 weeks |
| 🔴 Now | 12 | Auto-recommendation | 3–5 days | ~3 weeks |
| 🟡 Next | 13 | GUI / Web dashboard | 1–2 weeks | ~5 weeks |
| 🟡 Next | 14 | OpenAI API server | 1–2 weeks | ~7 weeks |
| 🟢 Later | 15 | Cross-user calibration | 2–3 weeks | ~10 weeks |
| 🟢 Later | 16 | Perplexity quality scoring | 3–5 days | ~11 weeks |
| 🟢 Later | 17 | Model comparison | 2–3 days | ~11 weeks |
| 🟢 Later | 18 | Live monitoring | 2–3 days | ~12 weeks |
| ⚪ Optional | 19–21 | Advanced features | 2–4 months | ~6 months |

---

## The Honest Assessment

After Steps 11–14, you will have a tool that:
- Works on **every major desktop platform** (Windows/Linux/Mac, NVIDIA/AMD/Apple)
- Tells users **what model to run** and **how to run it** before downloading anything
- Provides a **visual interface** for non-technical users
- Serves as a **drop-in API backend** for the entire local LLM ecosystem
- **Self-improves** with every run

At that point, the remaining gaps (Steps 15–21) are incremental improvements, not fundamental missing capabilities. The tool will be, for all practical purposes, **capable of everything a local LLM deployment tool needs to be.**

The only thing it won't do is fine-tuning/training — and that's a fundamentally different tool (Unsloth, Axolotl, etc.). Your tool's identity is **inference planning and execution**, and within that domain, it will be the most comprehensive solution available.

**Start with Step 11 (platform expansion). The interfaces are already designed. The code is already structured for it. It's the single highest-impact next step because it doubles your potential user base.**