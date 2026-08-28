# Vessel

**The local LLM deployment tool that tells you exactly how to run any model on your hardware — before you download a single gigabyte.**

No other tool does this. Existing calculators answer "will it fit?" Vessel answers "what should I run, how should I run it, how fast will it be, and can you just do it for me?"

```
$ vessel --recommend

=== Vessel — Model Recommendations ===

Hardware: RTX 4070 (12GB VRAM, 10.2 free) | 32GB RAM | NVMe 5.1/0.3 GB/s
Priority: balanced (use --priority to change)

 #  Model                     Quant    Strategy       VRAM     tok/s   Quality  Download
 1  🏆 Qwen 2.5 7B Instruct   Q4_K_M   Full GPU, 4K   5.2 GB   ~62     ★★★★☆   4.5 GB
 2  ⚡ Llama 3.2 3B Instruct  Q8_0     Full GPU, 4K   3.4 GB   ~120    ★★★☆☆   3.4 GB
 3  🧠 Llama 3.1 8B Instruct  Q5_K_M   Full GPU, 4K   6.1 GB   ~48     ★★★★☆   5.7 GB

💡 Top pick: #1 Qwen 2.5 7B — best balance of speed and quality.
   Run it: vessel --model <url> --execute
```

## Features

| Feature | Description |
|---------|-------------|
| **Zero-download prediction** | Fetches only the GGUF header (64KB–2MB) to predict memory, speed, and latency |
| **Strategy enumeration** | Generates 8–24 deployment strategies per model (Full GPU, Split, CPU, MoE offload, Hot/Cold, Layer-streaming) |
| **Auto-recommendation** | "What model should I run?" — curated catalog of 23 model variants across 7 families |
| **MoE expert-offload** | Native support for Mixtral, DeepSeek, Qwen-MoE with tensor-level placement |
| **Hot/cold neuron split** | PowerInfer-style sparse FFN for oversized dense models |
| **Layer-streaming fallback** | AirLLM-style sequential layer loading for extreme cases |
| **Self-calibration** | Predicted vs. actual comparison after each run, gets smarter over time |
| **Multi-platform** | NVIDIA (CUDA), AMD (ROCm/Vulkan), Apple Silicon (Metal), CPU-only |
| **Confidence bands** | HIGH / MEDIUM / LOW confidence with honest uncertainty |
| **Resumable downloads** | HTTP Range, SHA256 verification, multi-shard support |
| **Thermal monitoring** | Live GPU temp, clock speed, PCIe throughput, OS swap detection |

## Quick Start

### Install

**Windows (NVIDIA):**
```bash
# Download the latest release
curl -L -o vessel.exe https://github.com/bhuwanb23/vessel/releases/latest/download/vessel-windows-cuda.exe
```

**Linux (NVIDIA):**
```bash
curl -L -o vessel https://github.com/bhuwanb23/vessel/releases/latest/download/vessel-linux-cuda
chmod +x vessel
```

**macOS (Apple Silicon):**
```bash
curl -L -o vessel https://github.com/bhuwanb23/vessel/releases/latest/download/vessel-macos-metal
chmod +x vessel
```

### First Run

```bash
# See what models fit your hardware
vessel --recommend

# Check a specific model
vessel --model https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf

# Download and run it
vessel --model <url> --execute
```

## CLI Reference

```
Usage: vessel --model <url_or_path> [options]

Required:
  --model <url_or_path>               Hugging Face GGUF URL or local file

Options:
  --model-path <path>                 Local GGUF file (for --model URL)
  --download-dir <path>               Directory to download models (default: ~/models)
  --priority <speed|quality|safety>   Rank by (default: speed)
  --context <4k|max|both>             Contexts to evaluate (default: both)
  --execute                           Run inference after planning
  --skip-verify                       Skip SHA256 verification on download
  --prompt <text>                     Prompt for inference (default: benchmark)
  --max-tokens <N>                    Max tokens to generate (default: 100)
  --platform <cuda|hip|metal|cpu>     Force specific platform (default: auto-detect)
  --gpu <index>                       Select GPU by index (for multi-GPU systems)
  --gpu-name <pattern>                Select GPU by name pattern (e.g., 'RTX 4090')
  --verbose                           Full hardware & model reports
  --calibration-info                  Show calibration log stats
  --calibration-reset                 Delete calibration log (with confirm)
  --profile-neurons                   Run neuron activation profiling
  --hot-ratio <0.0-1.0>              Target hot neuron ratio (default: 0.15)
  --vram-budget <GB>                  VRAM budget for hot neurons (default: auto)
  --recommend                         Show model recommendations for your hardware
  --use-case <chat|coding|...>        Filter recommendations by use case
  --max-download <GB>                 Max download size in GB (e.g., 5)
  --top <N>                           Number of recommendations to show (default: 8)
  --catalog <path>                    Path to custom catalog JSON file
  --help                              Show this help
```

### Examples

```bash
# What model should I run?
vessel --recommend

# Best model for coding
vessel --recommend --use-case coding

# Fastest model for my GPU
vessel --recommend --priority speed

# Best model under 5GB download
vessel --recommend --max-download 5

# Check a specific model
vessel --model https://huggingface.co/.../model.gguf

# Compare strategies for a model
vessel --model <url> --priority safety

# Download and execute
vessel --model <url> --execute --prompt "What is AI?"

# Apple Silicon
vessel --model <url> --platform metal
```

## How It Works

1. **Profile hardware** — Reads live VRAM, RAM, GPU bandwidth, NVMe speed, temperature
2. **Fetch metadata** — Downloads only the GGUF header (64KB–2MB), extracts architecture, params, layers
3. **Predict performance** — Bandwidth-bound formulas for decode speed, compute-bound for TTFT
4. **Enumerate strategies** — Full GPU, Split, CPU, MoE offload, Hot/Cold, Layer-streaming
5. **Rank by priority** — Speed, quality, or safety with confidence-weighted scoring
6. **Download & execute** — Resumable download, SHA256 verification, llama.cpp execution
7. **Self-calibrate** — Compares predicted vs. actual, updates constants for future predictions

## Supported Models

| Family | Sizes | Variants |
|--------|-------|----------|
| **Llama 3.2** | 1B, 3B | Q4_K_M, Q8_0 |
| **Llama 3.1** | 8B | Q3_K_M, Q4_K_M, Q5_K_M, Q8_0 |
| **Qwen 2.5** | 3B, 7B, 14B | Q4_K_M, Q5_K_M |
| **Mistral** | 7B | Q4_K_M, Q5_K_M |
| **Phi 3.5** | Mini (3.8B) | Q4_K_M, Q8_0 |
| **Gemma 2** | 2B, 9B | Q4_K_M |
| **DeepSeek V2** | Lite (16B MoE) | Q4_K_M |
| **Mixtral** | 8×7B (MoE) | Q3_K_M, Q4_K_M |

Any GGUF model from Hugging Face works with `--model <url>`, even if not in the catalog.

## Platform Support

| Platform | Backend | Status |
|----------|---------|--------|
| Windows + NVIDIA | CUDA | ✅ Production |
| Linux + NVIDIA | CUDA | ✅ Production |
| Linux + AMD | ROCm/HIP | ✅ Good |
| Windows + AMD | Vulkan | ⚠️ Experimental |
| macOS + Apple Silicon | Metal | ✅ Good |
| CPU-only | CPU | ✅ Production |

## Building from Source

### Prerequisites

- CMake 3.20+
- C++17 compiler (MSVC 17+, GCC 11+, Clang 14+)
- CUDA Toolkit 12+ (for NVIDIA builds)
- ROCm 6+ (for AMD Linux builds)
- Xcode 15+ (for macOS builds)
- libcurl (usually pre-installed)

### Build

```bash
git clone https://github.com/bhuwanb23/vessel.git
cd vessel

# NVIDIA Windows
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release

# NVIDIA Linux
cmake -B build -DGGML_CUDA=ON
cmake --build build

# AMD Linux (ROCm)
cmake -B build -DGGML_HIPBLAS=ON -DCMAKE_HIP_ARCHITECTURES="gfx1100"
cmake --build build

# Apple Silicon
cmake -B build -DGGML_METAL=ON
cmake --build build

# CPU-only
cmake -B build
cmake --build build
```

### Run Tests

```bash
cd build/bin/Debug  # or build/bin/Release
./e2e_test
./recommend_test
./catalog_test
./step11_test
```

## Architecture

```
vessel/
├── src/
│   ├── planner/          # Main pipeline (profiler, fetcher, predictor, ranker, executor)
│   ├── profiler/         # Hardware profiling (GPU, RAM, disk)
│   ├── predictor/        # Performance prediction (speed, memory, TTFT, confidence)
│   ├── fetcher/          # Model metadata fetching (GGUF parser, HTTP)
│   ├── platform/         # Platform-specific profilers and executors
│   ├── recommend/        # Auto-recommendation engine (catalog, scoring)
│   ├── hotcold/          # Hot/cold neuron split (profiler, splitter, sparse FFN)
│   └── moe/              # MoE expert-offload (placer, tensor overrides)
├── include/              # Headers
├── data/                 # Model catalog JSON
├── docs/                 # Documentation
└── .github/              # CI/CD workflows
```

## Calibration

Vessel learns from every execution. After running a model, it logs the actual performance and uses it to improve future predictions on the same hardware.

```bash
# View calibration stats
vessel --calibration-info

# Reset calibration (start fresh)
vessel --calibration-reset
```

Calibration data is stored in:
- Windows: `%APPDATA%\vessel\calibration.jsonl`
- Linux/macOS: `~/.config/vessel/calibration.jsonl`

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgments

- [llama.cpp](https://github.com/ggerganov/llama.cpp) — Inference engine
- [PowerInfer](https://arxiv.org/abs/2312.12456) — Hot/cold neuron split research
- [AirLLM](https://github.com/lyogavin/airllm) — Layer-streaming research
- [Hugging Face](https://huggingface.co) — Model hosting and GGUF format
