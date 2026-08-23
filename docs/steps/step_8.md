# Vessel — Phase 2 Features: Full Detailed Plans

**Covers:** Step 8 (Model Download Manager), Step 9 (MoE Expert-Offload Integration), Step 10 (Hot/Cold CPU-GPU Offload for Dense Models + Layer-Streaming Fallback)

Each section follows the same format as Steps 0–7: goal, prerequisites, phases, formulas, done-checklist, common failure points, time estimate. Research grounding is cited by project/paper name inline; treat every performance number below as *the source's* reported number, not a promise about your hardware — your own calibration log (Step 7) is what turns these into numbers you can trust for your machine.

---

# Step 8 — Model Download Manager

## Goal
Given a model selected from the ranked strategy table, download the actual weight file(s) needed for that strategy — resumable, integrity-checked, disk-space-safe — and hand off a ready-to-load path to the Executor (Step 6). This is a prerequisite for Steps 9 and 10: neither MoE expert-offload nor hot/cold CPU offload can run against a model that isn't on disk yet.

## Why This Wasn't Needed Until Now
Steps 1–7 only ever needed the GGUF *header* (a few KB via range request). The Executor in Step 6 assumed the file was already manually placed in your models folder. That assumption breaks the moment a user picks a strategy for a model they haven't downloaded yet — which is the normal case.

## What You Need Before Starting
- Step 2's metadata fetcher (already parses GGUF headers via range request — you're extending this, not replacing it)
- Step 1's hardware profiler (you need live free disk space before committing to a download)
- `libcurl` (already in your stack from Step 2)

## Phase A — Pre-Download Safety Check
Before starting any transfer:
1. Take the predicted memory footprint for the **chosen strategy** from Step 3/4's output
2. Query live free disk space on the target drive (Windows: `GetDiskFreeSpaceExW`)
3. Require free space ≥ file size × 1.15 (safety margin — some quantization/conversion workflows need scratch space; downloads can also be interrupted and resumed, leaving partial+final files briefly coexisting)
4. If insufficient: **refuse to start**, tell the user the shortfall in GB, suggest a smaller quant from the same method matrix instead of just failing

## Phase B — Resumable Download
1. Use HTTP `Range` headers (same mechanism as Step 2's header fetch, now for the full file) so an interrupted download can resume from the last confirmed byte rather than restarting
2. Write to a `.partial` filename; only rename to the final filename after integrity check passes (Phase C) — this prevents a half-downloaded file from ever being mistaken for a complete one by the Executor
3. Report progress (bytes done / total, current speed, ETA) — `libcurl`'s progress callback (`CURLOPT_XFERINFOFUNCTION`) gives you this without polling
4. Handle interruption gracefully: Ctrl+C, network drop, and disk-full-mid-download should all leave a resumable `.partial` file, not a corrupt one

## Phase C — Integrity Verification
1. Hugging Face repos publish file hashes (SHA256) in their metadata API — fetch this alongside the download
2. After download completes, hash the local file and compare
3. Only rename `.partial` → final filename on match. On mismatch: delete and report a clear error (corrupted transfer, not a code bug) — this is one of the "silent failure" traps the original spec's stress-test warned about, applied to a new component

## Phase D — Multi-File Models
Some repos split a model across multiple GGUF shards (common for large models). Detect this from the repo's file listing (filenames following a `-00001-of-00005` pattern), download all shards, verify each independently, and only report "ready" once the complete set is present and verified.

## Design Decision: Where Does This Fit Architecturally?
This is a new module (`download_manager.cpp`) sitting between the Ranker (Step 5) and the Executor (Step 6) — the user picks a strategy, the download manager ensures the file exists and is verified, then the Executor takes over exactly as it does today. No changes needed to Steps 1–7's internals.

## Done Checklist
- [ ] Pre-download disk-space check refuses cleanly when space is insufficient, with an actionable suggestion
- [ ] A killed download (Ctrl+C mid-transfer) resumes correctly on re-run rather than restarting from zero
- [ ] SHA256 verification catches a deliberately corrupted test file
- [ ] Multi-shard models download and verify all shards before being marked ready
- [ ] Progress reporting shows sane speed/ETA numbers during a real download

## Common Failure Points
| Problem | Cause | Fix |
|---|---|---|
| Resume produces a corrupted file | Range request off-by-one on the resume offset | Resume from `bytes_downloaded`, not `bytes_downloaded - 1`; verify hash regardless |
| Disk fills up mid-download | Safety margin check used stale free-space number | Re-check free space periodically during long downloads, not just once at start |
| Hash mismatch on every download | Comparing against wrong hash field (some HF APIs expose both git-LFS SHA256 and a legacy SHA1) | Confirm which hash field the file actually corresponds to before wiring up comparison |

## Time Estimate
Phase A: 2–3 hours · Phase B: 1–2 days (resumability + progress reporting is the bulk of it) · Phase C: 3–4 hours · Phase D: 3–4 hours. **Total: 2–3 days.**

---

# Step 9 — MoE Expert-Offload Integration

## Goal
Extend the Executor and Predictor to support Mixture-of-Experts models (DeepSeek-V3-class, Kimi K2-class, Mixtral-class) by placing shared/hot experts on GPU and routed/cold experts on CPU or NVMe — your flagship differentiating feature.

## Research Grounding (know this before building)
- **This capability already exists in `llama.cpp`** via the `--override-tensor` flag, which lets you pin specific tensors (typically the per-expert FFN weights) to CPU while the rest runs on GPU. You are not building a new inference engine — you are building the **intelligence layer that decides which tensors to override**, which today a human has to figure out by trial and error.
- The academic grounding for *why* this works: MoE models route each token to only a handful of experts out of many (Eliseev & Mazur, 2023, the foundational MoE-offloading paper). Most experts sit idle for any given token, so keeping all of them in fast memory is wasteful — you only need the *active* ones fast.
- **ktransformers** (Tsinghua, SOSP '25) is the more sophisticated version of this idea: place shared experts (used by every token) on GPU, offload routed experts (used sometimes) to CPU, with specialized AMX kernels for CPU-side expert computation. You don't need to integrate ktransformers itself — the placement *strategy* is the valuable, portable idea; your own C++ code applying it via `llama.cpp`'s existing override mechanism keeps you in one codebase and one language.

## What You Need Before Starting
- Step 6's Executor, working and linked against `llama.cpp`
- Step 3's predictor, specifically the MoE row from the original spec's §8.2 (range-based prediction, not point estimate)
- A real MoE GGUF model for testing (Mixtral 8x7B is the most accessible size for local testing; DeepSeek/Kimi-class models are large enough that even testing requires serious hardware)

## Phase A — Detect MoE Architecture from Metadata
Your Step 2 metadata fetcher already reads the GGUF header. Extend the parser to also read: number of experts, experts-active-per-token (top-k), and per-expert parameter count. GGUF's metadata schema includes these fields for MoE architectures — if they're absent, the model is dense and none of this step applies.

## Phase B — Decide Expert Placement
This is the actual new intelligence, not just a flag pass-through:
1. Compute **shared/always-active parameter size** (attention layers + any shared experts) vs **total routed-expert parameter size**
2. Compare shared-params size against free VRAM (from your Step 1 profiler) — if it fits, that's your GPU placement baseline
3. Compute how many *additional* experts' worth of parameters fit in remaining VRAM — these become GPU-resident "hot" experts (an approximation of ktransformers' shared-expert insight, without needing per-model offline profiling data you don't have)
4. Everything else stays CPU-resident (RAM) or, if RAM is also insufficient, NVMe-streamed at inference time

## Phase C — Generate the `--override-tensor` Configuration
Translate Phase B's placement decision into the actual tensor-name patterns `llama.cpp` expects (expert FFN tensors typically follow a predictable naming pattern per layer/expert index in GGUF). Generate this configuration programmatically rather than requiring the user to hand-write regex patterns — this is the actual product value over using `llama.cpp` directly.

## Phase D — Predictor Update
Wire the MoE-specific formula from your original spec (§8.2) into the method matrix: report tokens/sec as a **range** (best case = requested experts are the GPU-resident hot set, worst case = cold miss requiring CPU or NVMe read every time), and mark confidence as inherently lower than dense-model predictions — this was locked in your original design stress-test (item #4) and now has a real implementation to attach to.

## Phase E — Validation
1. Run the same MoE model and prompt 5+ times, log actual tokens/sec each time
2. Confirm real variance falls within your predicted range (not necessarily centered on it yet — that's what calibration is for)
3. Compare against a naive "everything on GPU that fits, rest on CPU sequentially" baseline (i.e., not expert-aware) to confirm your placement strategy is actually faster — this is your evidence that the feature adds real value, not just complexity

## Done Checklist
- [ ] MoE metadata (expert count, top-k, per-expert size) parsed correctly from GGUF header
- [ ] Placement algorithm produces a valid `--override-tensor` configuration automatically
- [ ] Predicted range brackets real observed tokens/sec across 5+ runs
- [ ] Expert-aware placement measurably outperforms naive placement on the same hardware
- [ ] Confidence band correctly shows "lower confidence" for MoE vs dense predictions in the ranker output

## Common Failure Points
| Problem | Cause | Fix |
|---|---|---|
| `--override-tensor` pattern matches wrong tensors | GGUF tensor naming varies slightly by model family/converter version | Validate the generated pattern against the model's actual tensor list before launching, not just against a template |
| Predicted range never brackets reality | Expert cache-hit-rate assumption too optimistic/pessimistic | Widen the range initially; let the Step 7 calibration log narrow it per hardware+model combination over real runs |
| No speed improvement over naive placement | Shared/hot experts misidentified — CPU is still doing most of the work | Re-check Phase B's shared-vs-routed size calculation against the model's actual architecture, not assumed defaults |

## Time Estimate
Phase A: 1 day · Phase B: 2–3 days (the real engineering) · Phase C: 1–2 days · Phase D: 1 day · Phase E: 1–2 days. **Total: 6–9 days.**

---

# Step 10 — Hot/Cold CPU-GPU Offload for Dense Models (+ Layer-Streaming Fallback)

## Goal
For dense (non-MoE) models too large to fit on GPU, apply a PowerInfer-style hot/cold split — frequently-activated neurons on GPU, rarely-activated ones computed on CPU — as the primary no/low-GPU strategy. Where even that doesn't fit (extreme model-to-hardware ratio, e.g. a large dense model on a machine with no meaningful GPU at all), fall back to AirLLM-style layer-by-layer disk streaming, clearly labeled as slow-but-possible.

## Research Grounding
- **PowerInfer** (arXiv:2312.12456) is the primary technique here. Its core finding: LLM neuron activation follows a power-law distribution — a small set of neurons activate on nearly every input ("hot"), the rest activate only for specific inputs ("cold"). Keeping hot neurons GPU-resident and computing cold ones on CPU avoids constantly shuttling the *whole* model across PCIe. Reported real numbers: averaging roughly 8 tokens/sec across OPT-30B/66B, Falcon-40B, and Llama-70B-class models on a consumer i9-13900K + RTX 4090 pairing — a usable speed, not a demo-only curiosity. **Critically, PowerInfer is itself a `llama.cpp` fork in C++** — meaning the technique's implementation approach is directly compatible with your stack, not a foreign paradigm you'd need to bridge from Python.
- **AirLLM** is the fallback technique: stream one transformer layer from disk at a time, discard it after use, load the next. This makes an otherwise-impossible model *technically run* with minimal resident memory. Be honest with yourself and your users about the cost: community-reported numbers for this approach include cases as slow as roughly one token per minute on weak hardware. This is a "last resort, proof that it's at least possible" mode, not a mode you'd want as anyone's default.

## What You Need Before Starting
- Step 1's profiler (RAM/VRAM/disk numbers feed the placement decision)
- Step 6's Executor (you're adding a new placement mode, not replacing the lifecycle management)
- A dense model too large to fully fit on your test GPU, to actually exercise this path

## Phase A — Offline Neuron Activation Profiling
PowerInfer's technique requires knowing *which* neurons are hot before inference starts. This means a one-time profiling pass per model:
1. Run the model against a diverse general-purpose prompt set (not your final use-case prompts specifically — the hot-neuron set should generalize, per PowerInfer's own finding that hot neurons are consistent across varied inputs)
2. Record per-neuron activation frequency
3. Rank neurons by activation frequency; the top N (sized to fit available VRAM) become the GPU-resident set
4. Cache this profiling result per model — it's expensive to redo and doesn't need to be repeated per run, only per model (and arguably per major quantization change)

## Phase B — Placement Decision
1. Given available VRAM (from Step 1's profiler) minus KV cache and runtime overhead, compute how many "hot" neurons fit
2. Everything else is "cold" — resident in system RAM, computed on CPU during inference
3. This is architecturally a **new placement strategy** in your method matrix, alongside full-GPU, split, and CPU-only — reported with its own predicted tokens/sec using PowerInfer's published throughput class as a starting reference point, refined by your Step 7 calibration log per real hardware

## Phase C — Execution Integration
This is the hardest part of Step 10: hot/cold splitting isn't a llama.cpp flag — it requires operator-level awareness of which neurons are being computed where, mid-inference. Two honest paths:
- **(Preferred, lower risk):** study PowerInfer's own fork of llama.cpp for the specific mechanism (neuron-aware sparse operators, predictor integration) and adapt the relevant pieces into your executor rather than reimplementing from the paper alone
- **(Higher risk, more original):** implement the sparse hot/cold operator dispatch yourself directly against `ggml` primitives, guided by the paper's described architecture

Given your project's existing depth of `ggml`/`llama.cpp` familiarity from Step 6, studying the working fork first is the pragmatic choice — this is a case where "don't reinvent it, adapt the proven implementation" beats "build from the paper alone," especially given how much operator-level tuning the original authors already did.

## Phase D — Layer-Streaming Fallback Mode
For cases where even hot/cold splitting doesn't produce an acceptable placement (extremely large dense model, extremely constrained hardware):
1. Detect this case in the Predictor: if hot/cold placement's *worst-case* VRAM requirement still exceeds available VRAM even at minimum hot-set size, flag layer-streaming as the only viable path
2. Implement layer-wise loading: read one layer's weights from disk, run it, free it, load the next — the AirLLM mechanism
3. **Label this mode honestly and prominently in the ranker output** — this is a direct application of your original spec's confidence-band and quality-vs-speed tradeoff philosophy (§3, items 5 and 9): don't hide that this mode may run at a pace measured in seconds-per-token or worse; let the user decide if "technically possible" is worth it for their use case

## Phase E — Validation
1. Confirm hot-neuron profiling produces a stable set across different prompts (per PowerInfer's own consistency claim) — if the "hot" set changes wildly per prompt, something's wrong with the profiling methodology
2. Compare hot/cold placement's real tokens/sec against your CPU-only baseline from earlier steps — it should be a clear improvement, not a wash
3. Confirm layer-streaming fallback actually completes a generation (even slowly) on a model that fails every other placement strategy — proving the "always find *a* way" promise from your original idea

## Done Checklist
- [ ] Neuron activation profiling completes and produces a consistent hot-set across varied prompts
- [ ] Hot/cold placement strategy appears correctly in the method matrix with its own prediction
- [ ] Real hot/cold throughput measurably beats CPU-only baseline on the same model+hardware
- [ ] Layer-streaming fallback triggers automatically when hot/cold placement doesn't fit, not left as a manual-only option
- [ ] Ranker output clearly and honestly labels layer-streaming's expected slowness before the user commits to it

## Common Failure Points
| Problem | Cause | Fix |
|---|---|---|
| Hot-neuron set differs wildly per prompt | Profiling prompt set too narrow/specific | Broaden profiling prompts to general, diverse text, matching PowerInfer's own methodology |
| Hot/cold placement no faster than CPU-only | GPU-resident set too small to matter, or CPU-side sparse computation not actually skipping cold neurons | Verify the CPU-side operator is genuinely skipping non-activated neurons, not computing everything and discarding |
| Layer-streaming never triggers even when needed | Predictor's "doesn't fit" threshold check has a bug | Add an explicit test case: a deliberately oversized model, confirm the fallback path is selected |

## Time Estimate
Phase A: 3–4 days (neuron profiling infra + validation is nontrivial) · Phase B: 1–2 days · Phase C: 5–8 days (the hardest engineering in the whole project, comparable in difficulty to Step 6 itself) · Phase D: 2–3 days · Phase E: 2 days. **Total: 2–3 weeks.** This is, honestly, the single largest chunk of remaining work in the project — size your expectations accordingly and don't treat it as a quick add-on next to Steps 8 and 9.

---

## Suggested Build Order for Phase 2

| Order | Step | Why here |
|---|---|---|
| 1 | Step 8 — Download Manager | Small, unblocks the other two |
| 2 | Step 9 — MoE Expert-Offload | Reuses existing `llama.cpp` mechanism (`--override-tensor`); real differentiator, moderate effort |
| 3 | Step 10 — Hot/Cold Offload + Layer-Streaming | Largest effort, most original engineering; do this once 8 and 9 are solid and validated |

Don't start Step 10 in parallel with Step 9 — they touch different model classes (dense vs MoE) but both compete for your attention on the hardest parts of the project, and Step 10 in particular deserves undivided focus given its size.