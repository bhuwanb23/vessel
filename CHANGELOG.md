# Changelog

All notable changes to Vessel will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-08-28

### Initial Release

Vessel is a local LLM deployment tool that predicts model performance before download, enumerates deployment strategies, and executes models via llama.cpp.

### What's Included

**Hardware Profiling (Step 1)**
- Live VRAM, RAM, GPU bandwidth, NVMe speed, temperature
- NVIDIA (NVML), AMD (ROCm-SMI/ADL), Apple (Metal/IOKit), CPU-only
- Stable hardware fingerprint for calibration

**Model Metadata Fetching (Step 2)**
- GGUF header parsing via 64KB–2MB range requests
- Architecture, params, layers, attention heads, context, quantization
- MoE metadata: expert count, top-k, shared experts
- Config.json fallback for non-GGUF models

**Performance Prediction (Steps 3–4)**
- Memory footprint: weight + KV cache + runtime overhead
- Decode speed: bandwidth-bound formula with per-platform efficiency
- TTFT: hybrid memory/compute-bound model
- Method matrix: 8–24 strategies per model (Full GPU, Split, CPU, MoE, Hot/Cold, Layer-stream)

**Strategy Ranking (Step 5)**
- Speed, quality, safety priorities with confidence weighting
- Normalized scoring with tradeoff callouts

**Model Execution (Step 6)**
- llama.cpp library integration (not subprocess)
- Live hardware sampling (VRAM, temp, clock, PCIe throughput)
- Thermal throttle detection
- Predicted vs. actual comparison report

**Self-Calibration (Step 7)**
- JSONL log per hardware fingerprint
- Confidence upgrade after 5+ matching runs
- `--calibration-info` and `--calibration-reset` CLI commands

**Download Manager (Step 8)**
- Pre-download disk space check (1.15× safety margin)
- Resumable downloads via HTTP Range
- SHA256 verification (Windows CNG API)
- Multi-shard model support
- Progress bar with speed and ETA

**MoE Expert-Offload (Step 9)**
- GGUF header MoE detection
- Shared vs routed expert placement
- Tensor override generation
- Range-based tok/s predictions (best/worst/expected)

**Hot/Cold Neuron Split (Step 10)**
- Offline neuron profiling (500+ prompts)
- Sparse FFN computation
- Layer-streaming fallback

**Multi-Platform (Step 11)**
- NVIDIA: Windows + Linux, CUDA, NVML
- AMD: Linux (ROCm/HIP), Windows (Vulkan)
- Apple Silicon: macOS, Metal, unified memory
- CPU-only fallback

**Auto-Recommendation (Step 12)**
- Curated catalog: 23 variants across 7 families
- Cross-model ranking with speed/quality/balanced priorities
- Use case filtering, download size limits
- Copy-pasteable run commands

**CLI Interface**
- 25+ flags covering all features
- `--recommend`, `--model`, `--execute`, `--verbose`
- `--priority`, `--use-case`, `--max-download`, `--top`, `--catalog`

**Infrastructure**
- GitHub Actions CI/CD (6 platform builds)
- 88 test cases across 8 test suites
- MIT License

### Test Results

| Suite | Tests | Status |
|-------|-------|--------|
| E2E | 12 | ✅ All pass |
| Recommendation | 32 | ✅ All pass |
| Catalog | 22 | ✅ All pass |
| Step 11 | 22 | ✅ All pass |
| **Total** | **88** | **✅ All pass** |

### Hardware Validated

- NVIDIA GeForce RTX 5060 (8.5 GB VRAM) — all scenarios
- Simulated 24 GB VRAM — recommendation engine
- Simulated 2 GB VRAM — constrained hardware

### Known Limitations

- MoE models from gated Hugging Face repos require authentication
- Intel Arc GPU support deferred (llama.cpp SYCL backend immature)
- No real-time chat UI (CLI only)
- No OpenAI-compatible API server
- Predictions are estimates (improve with calibration)
