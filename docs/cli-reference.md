# CLI Reference

Complete reference for all Vessel command-line options.

## Usage

```
vessel --model <url_or_path> [options]
vessel --recommend [options]
```

## Required Options

| Option | Description |
|--------|-------------|
| `--model <url_or_path>` | Hugging Face GGUF URL or local file path |

## Model Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model-path <path>` | — | Local GGUF file path (used with `--model` URL) |
| `--download-dir <path>` | `~/models` | Directory to download models |

## Prediction Options

| Option | Default | Description |
|--------|---------|-------------|
| `--priority <speed\|quality\|safety>` | `speed` | Ranking priority for strategies |
| `--context <4k\|max\|both>` | `both` | Context lengths to evaluate |
| `--verbose` | `false` | Show full hardware and model reports |

## Execution Options

| Option | Default | Description |
|--------|---------|-------------|
| `--execute` | `false` | Run inference after planning |
| `--skip-verify` | `false` | Skip SHA256 verification on download |
| `--prompt <text>` | benchmark | Prompt for inference |
| `--max-tokens <N>` | `100` | Maximum tokens to generate |

## Platform Options

| Option | Default | Description |
|--------|---------|-------------|
| `--platform <cuda\|hip\|metal\|cpu>` | auto-detect | Force specific platform |
| `--gpu <index>` | `0` | Select GPU by index (multi-GPU) |
| `--gpu-name <pattern>` | — | Select GPU by name (e.g., "RTX 4090") |

## Recommendation Options

| Option | Default | Description |
|--------|---------|-------------|
| `--recommend` | `false` | Activate recommendation mode |
| `--use-case <chat\|coding\|reasoning\|all>` | `all` | Filter by use case |
| `--max-download <GB>` | — | Maximum download size in GB |
| `--top <N>` | `8` | Number of recommendations to show |
| `--catalog <path>` | built-in | Path to custom catalog JSON |

## Calibration Options

| Option | Default | Description |
|--------|---------|-------------|
| `--calibration-info` | `false` | Show calibration log statistics |
| `--calibration-reset` | `false` | Delete calibration log (with confirm) |

## Advanced Options

| Option | Default | Description |
|--------|---------|-------------|
| `--profile-neurons` | `false` | Run neuron activation profiling |
| `--hot-ratio <0.0-1.0>` | `0.15` | Target hot neuron ratio |
| `--vram-budget <GB>` | auto | VRAM budget for hot neurons |

## Other Options

| Option | Description |
|--------|-------------|
| `--help` | Show help message |

---

## Examples

### Basic Usage

```bash
# Check a model
vessel --model https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf

# Download and run
vessel --model <url> --execute

# Run with custom prompt
vessel --model <url> --execute --prompt "Explain quantum computing"
```

### Recommendation Mode

```bash
# Basic recommendation
vessel --recommend

# Speed priority
vessel --recommend --priority speed

# Quality priority
vessel --recommend --priority quality

# Coding models only
vessel --recommend --use-case coding

# Small downloads only
vessel --recommend --max-download 3

# Top 3 only
vessel --recommend --top 3
```

### Platform-Specific

```bash
# Force CUDA
vessel --model <url> --platform cuda

# Force Metal (Apple Silicon)
vessel --model <url> --platform metal

# Force CPU-only
vessel --model <url> --platform cpu

# Select specific GPU
vessel --model <url> --gpu 1
vessel --model <url> --gpu-name "RTX 4090"
```

### Calibration

```bash
# View calibration stats
vessel --calibration-info

# Reset calibration
vessel --calibration-reset
```

### Verbose Output

```bash
# Full hardware and model details
vessel --model <url> --verbose

# Example output:
# --- Hardware Profile (Full) ---
# GPU:              NVIDIA GeForce RTX 4070
# VRAM:             12.00 GB total, 10.20 GB free
# GPU Bandwidth:    504.0 GB/s
# GPU TFLOPS:       20.0 TFLOPS (FP16)
# RAM:              32.00 GB total, 24.50 GB free
# NVMe:             5100 MB/s seq, 320 MB/s random 4K
# Compute:          sm_89
```

---

## Output Format

### Strategy Comparison Table

```
=== Vessel — Strategy Comparison ===

Ranked by: speed (use --priority to change)

 #  Placement    GPU Layers   Context  KV Cache VRAM      RAM       tok/s    TTFT     Status
─── ──────────── ──────────── ──────── ──────── ───────── ───────── ──────── ──────── ──────────
 1  Full GPU     32          4K       Q8       5.8 GB    -         ~55      ~52ms    ✅ VIABLE
 2  Full GPU     32          32K      Q8       9.4 GB    -         ~42      ~380ms   ✅ VIABLE
 3  Split        24          128K     Q8       7.2 GB    3.1 GB    ~28      ~520ms   ✅ VIABLE
 4  CPU Only     0           4K       FP16     0 GB      5.8 GB    ~8       ~450ms   ✅ VIABLE
```

### Recommendation Table

```
=== Vessel — Model Recommendations ===

 #  Model                     Quant    Strategy       VRAM     tok/s   Quality  Download
 1  🏆 Qwen 2.5 7B Instruct   Q4_K_M   Full GPU, 4K   5.2 GB   ~62     ★★★★☆   4.5 GB
 2  ⚡ Llama 3.2 3B Instruct  Q8_0     Full GPU, 4K   3.4 GB   ~120    ★★★☆☆   3.4 GB
```

### Status Icons

| Icon | Status | Meaning |
|------|--------|---------|
| ✅ | VIABLE | Strategy works with comfortable headroom |
| ⚠️ | TIGHT | Strategy works but memory is >90% utilized |
| ❌ | NO FIT | Strategy exceeds available memory |
| ❓ | LOW CONF | Prediction has low confidence |

### Labels

| Label | Meaning |
|-------|---------|
| 🏆 Best Overall | Highest balanced score |
| ⚡ Fastest | Highest tokens/sec |
| 🧠 Highest Quality | Highest quality score |
| 💾 Smallest | Smallest download size |
| 📄 Longest Context | Largest context length |
