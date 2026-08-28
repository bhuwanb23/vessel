# Quick Start Guide

Get Vessel running in 3 commands.

## 1. Install

Download the latest release for your platform from [GitHub Releases](https://github.com/bhuwanb23/vessel/releases).

**Windows (NVIDIA):**
```bash
# Download
curl -L -o vessel.exe https://github.com/bhuwanb23/vessel/releases/latest/download/vessel-windows-cuda.exe

# Or with PowerShell
Invoke-WebRequest -Uri "https://github.com/bhuwanb23/vessel/releases/latest/download/vessel-windows-cuda.exe" -OutFile "vessel.exe"
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

## 2. Discover Models

```bash
# See what models fit your hardware
vessel --recommend

# Output:
# === Vessel — Model Recommendations ===
# Hardware: RTX 4070 (12GB VRAM, 10.2 free) | 32GB RAM
#  #  Model                     Quant    Strategy       VRAM     tok/s   Quality
#  1  🏆 Qwen 2.5 7B Instruct   Q4_K_M   Full GPU, 4K   5.2 GB   ~62     ★★★★☆
#  2  ⚡ Llama 3.2 3B Instruct  Q8_0     Full GPU, 4K   3.4 GB   ~120    ★★★☆☆
```

## 3. Run a Model

```bash
# Option A: Run the top recommendation
vessel --model https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf --execute

# Option B: Just predict (no download)
vessel --model https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf
```

## What Just Happened?

1. **Vessel profiled your hardware** — GPU model, VRAM, RAM, disk speed
2. **Fetched model metadata** — Downloaded only the GGUF header (64KB)
3. **Predicted performance** — Memory usage, tokens/sec, time-to-first-token
4. **Ranked strategies** — Full GPU, Split, CPU-only, etc.
5. **(If --execute)** Downloaded the model, verified integrity, ran inference

## Common Tasks

### Check Multiple Strategies

```bash
vessel --model <url> --priority speed     # Fastest strategies first
vessel --model <url> --priority quality   # Highest quality first
vessel --model <url> --priority safety    # Most memory headroom first
```

### Filter by Use Case

```bash
vessel --recommend --use-case coding      # Best for coding
vessel --recommend --use-case chat        # Best for chat
vessel --recommend --use-case reasoning   # Best for reasoning
```

### Limit Download Size

```bash
vessel --recommend --max-download 5       # Only models under 5GB
vessel --recommend --top 3                # Show only top 3
```

### Use Custom Catalog

```bash
vessel --recommend --catalog my_models.json
```

### Apple Silicon

```bash
vessel --model <url> --platform metal
```

### Verbose Output

```bash
vessel --model <url> --verbose            # Full hardware & model details
```

## Troubleshooting

### "No NVIDIA GPU detected"

- Install NVIDIA drivers: https://www.nvidia.com/drivers
- For CUDA: Install CUDA Toolkit 12+: https://developer.nvidia.com/cuda-downloads

### "HTTP 401" Error

- The model URL may require authentication
- Try a different GGUF repo (e.g., bartowski/ instead of the original)

### Predictions Seem Off

- Run `vessel --calibration-info` to check calibration data
- Run `vessel --calibration-reset` to start fresh
- Predictions improve after 5+ executions on the same hardware

### "Model does not fit"

- Try a smaller quantization (Q3_K_M or Q2_K)
- Try a smaller model (3B instead of 7B+)
- Close GPU-intensive applications to free VRAM

## Next Steps

- Read the [Full CLI Reference](cli-reference.md)
- Learn about [How the Predictor Works](predictor.md)
- Contribute to the [Model Catalog](contributing.md#adding-a-new-model-to-the-catalog)
