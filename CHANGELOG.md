# Changelog

All notable changes to Vessel will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-08-28

### Added

#### Core Pipeline (Steps 1–7)
- **Hardware Profiler** — Live VRAM, RAM, GPU bandwidth, NVMe speed, temperature
- **Metadata Fetcher** — GGUF header parsing (64KB–2MB, no full download)
- **Performance Predictor** — Bandwidth-bound decode speed, compute-bound TTFT
- **Method Matrix Generator** — 8–24 strategies per model (Full GPU, Split, CPU)
- **Strategy Ranker** — Speed/quality/safety priorities with confidence weighting
- **Model Executor** — llama.cpp library integration with live hardware sampling
- **Self-Calibration** — Predicted vs. actual comparison, JSONL log, confidence upgrade

#### Download Manager (Step 8)
- Pre-download disk space check with 1.15× safety margin
- Resumable downloads via HTTP Range (.partial files)
- SHA256 integrity verification (Windows CNG API)
- Multi-shard model support (-00001-of-00005 patterns)
- Progress bar with speed and ETA

#### MoE Expert-Offload (Step 9)
- GGUF header MoE detection (Mixtral, DeepSeek, Qwen-MoE)
- Shared vs routed expert placement computation
- Tensor override pattern generation for llama.cpp
- Range-based tok/s predictions (best/worst/expected)

#### Hot/Cold Neuron Split (Step 10)
- Offline neuron activation profiling (500+ prompts)
- Power-law hot neuron selection sized to VRAM budget
- Sparse FFN computation (genuinely skips non-activated neurons)
- Layer-streaming fallback for extreme cases

#### Multi-Platform (Step 11)
- **NVIDIA**: Windows + Linux, CUDA, NVML profiling
- **AMD**: Linux (ROCm/HIP), Windows (Vulkan), ROCm-SMI/ADL
- **Apple Silicon**: macOS, Metal, unified memory model
- **CPU-only**: Universal fallback
- Runtime auto-detection with `--platform` override

#### Auto-Recommendation (Step 12)
- Curated catalog of 23 model variants across 7 families
- Cross-model ranking with speed/quality/balanced priorities
- Use case filtering (chat, coding, reasoning)
- Download size limits, top-N control
- Copy-pasteable run commands

#### CLI Interface
- `--recommend` — Model recommendation mode
- `--priority` — Speed/quality/safety ranking
- `--use-case` — Filter by model tags
- `--max-download` — Limit download size
- `--top` — Control output count
- `--catalog` — Custom catalog file
- `--verbose` — Full hardware and model reports
- `--calibration-info` / `--calibration-reset` — Calibration management

#### Infrastructure
- GitHub Actions CI/CD (6 platform builds)
- Automated release packaging
- 88+ test cases across 8 test suites
- MIT License

### Known Limitations
- MoE models from gated Hugging Face repos require authentication
- Intel Arc GPU support deferred (llama.cpp SYCL backend immature)
- No real-time chat UI (CLI only)
- No OpenAI-compatible API server
- Predictions are estimates with confidence bands (improve with calibration)

---

## [0.9.0] - 2026-08-15

### Added
- Initial development build
- Hardware profiling (NVIDIA only)
- GGUF metadata fetching
- Basic predictor (memory + speed)
- Method matrix generation
- Strategy ranking
- llama.cpp executor integration
- Self-calibration log

---

## [0.1.0] - 2026-07-01

### Added
- Project initialization
- Architecture design and spec documentation
- C++17 project structure with CMake
