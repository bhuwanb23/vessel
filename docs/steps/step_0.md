# Step 0 — Environment Setup: Full Detailed Plan

---

## Goal of Step 0
By the end of this step, you will have confirmed that your entire toolchain works end-to-end. You will have run one real model through vanilla `llama.cpp` and watched it generate tokens. Nothing you build in Steps 1–7 is trustworthy until this baseline is confirmed.

---

## What You Need Before Starting

### Hardware Confirmation
- A Windows machine with an NVIDIA GPU
- At least 8GB VRAM (recommended for comfortable testing)
- At least 16GB System RAM
- Sufficient disk space — budget at least **20GB free** for model files, build artifacts, and toolchain installs

### Accounts Needed
- A **Hugging Face account** — you will be downloading GGUF model files from HuggingFace repositories
- That's it

---

## Phase A — Install the Toolchain (in this exact order)

### A1 — NVIDIA Driver
**What:** The GPU driver that exposes NVML and CUDA to your system.

**Check first:** Open Device Manager → Display Adapters → confirm your NVIDIA GPU is listed with no warning signs. Then open a terminal and type `nvidia-smi`. If it runs and shows your GPU model, VRAM, and driver version, **this is already done**.

**If not installed:** Download from [nvidia.com/drivers](https://www.nvidia.com/drivers), select your exact GPU model and Windows version, install, reboot.

**What to confirm after:**
- `nvidia-smi` shows your GPU name
- Driver version is shown (note it down)
- VRAM total is shown and matches your GPU's spec

---

### A2 — CUDA Toolkit
**What:** The compiler and runtime libraries that let C++ code talk to the GPU. This is what lets `llama.cpp` offload layers to VRAM.

**Important:** The CUDA Toolkit version must be **compatible with your driver version**. NVIDIA publishes a compatibility table. General rule: newer drivers support older CUDA toolkit versions, but not the reverse. A safe choice is **CUDA Toolkit 12.x** if your driver is recent (which it should be).

**Download from:** [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads) — select Windows → x86_64 → your Windows version → exe (local installer).

**What the installer does:**
- Installs `nvcc` (the CUDA compiler)
- Installs CUDA libraries (`cublas`, `cudart`, etc.)
- Adds itself to your system PATH

**What to confirm after:**
- Open a **new** terminal (important — PATH updates require a new terminal)
- Run `nvcc --version` — it should print the CUDA toolkit version
- Run `nvidia-smi` again — it now shows both driver version and CUDA version in the top right corner

---

### A3 — Visual Studio (MSVC) or MinGW — Pick One Now

This is the build toolchain decision. You need a C++ compiler. Two options:

#### Option A: MSVC (Visual Studio)
- **Download:** Visual Studio Community (free) from [visualstudio.microsoft.com](https://visualstudio.microsoft.com)
- **During install:** Select the workload **"Desktop development with C++"** — this is the only required workload
- **Also check:** Inside that workload, confirm **"C++ CMake tools for Windows"** is selected
- **Why this option:** MSVC is what most Windows C++ projects assume. Better debugger integration. llama.cpp's documentation primarily references this path.

#### Option B: MinGW (GCC on Windows)
- **Download:** via [MSYS2](https://www.msys2.org/) — this is the cleanest way to get MinGW on Windows
- **Why this option:** If you are more comfortable with GCC-style builds and Linux-like terminal workflow
- **Caveat:** Some llama.cpp CUDA build configurations have historically been smoother with MSVC. Not a dealbreaker, but worth knowing.

**Recommendation for this project:** Go with **MSVC**. The reason is that llama.cpp's CUDA backend on Windows is most tested with MSVC + CUDA toolkit combination. Fewer surprises in Step 6 when you link it as a library.

**What to confirm after:**
- Open the **"Developer Command Prompt for VS"** (installed with Visual Studio)
- Run `cl` — it should print the MSVC compiler version
- Run `cmake --version` — CMake comes with Visual Studio, confirm it's present

---

### A4 — CMake (Standalone, if not using Visual Studio)
If you chose MSVC + Visual Studio, CMake is already installed. Skip this.

If you chose MinGW: Download CMake from [cmake.org/download](https://cmake.org/download) — Windows x64 installer. During install, select **"Add CMake to system PATH"**.

**Confirm:** `cmake --version` in a terminal.

---

### A5 — Git
**What:** You need Git to clone llama.cpp and later your own project repository.

**Check first:** Run `git --version` in a terminal. If it prints a version, skip this.

**If not installed:** Download from [git-scm.com](https://git-scm.com/download/win) — use all default settings during install.

**Confirm:** `git --version` in a terminal.

---

## Phase B — Clone and Build llama.cpp

### B1 — Create Your Working Directory
Decide where your projects will live. A clean, simple path with **no spaces** is important — CMake and CUDA build tools sometimes choke on paths with spaces.

**Good example:** `C:\dev\` or `C:\projects\`
**Avoid:** `C:\Users\Your Name\Documents\My Projects\` (spaces cause issues)

Create the folder, then also create `C:\dev\models\` — this is where GGUF files will live.

---

### B2 — Clone llama.cpp
Open a terminal, navigate to your working directory, and clone the repository.

**Repository:** `https://github.com/ggerganov/llama.cpp`

This pulls the full source code. The repository is large — expect a few minutes depending on connection speed.

---

### B3 — Understand What You Are Building (Before You Build)
Before running any build commands, spend 10 minutes reading the `llama.cpp` README on GitHub, specifically the **"Building"** section and the **"CUDA"** subsection.

**Why this matters:** You need to understand the CMake flags you are about to use, not just copy-paste them. The key flags for your setup:
- The flag that enables CUDA backend (so layers are offloaded to your GPU)
- The flag that tells it which CUDA architecture your GPU uses (e.g., `sm_86` for RTX 30 series, `sm_89` for RTX 40 series)

**How to find your GPU's compute capability / architecture:**
- Go to [developer.nvidia.com/cuda-gpus](https://developer.nvidia.com/cuda-gpus)
- Find your exact GPU model
- Note the "Compute Capability" number — this maps directly to the `sm_XX` flag

---

### B4 — Build llama.cpp with CUDA Support
Using CMake, configure and build the project with CUDA enabled.

**Build type:** Use `Release` mode, not `Debug`. Debug builds are significantly slower and will give you misleading performance numbers when you validate your predictor formulas in Step 3.

**Where outputs go:** CMake will produce the compiled binaries in a `build` subdirectory. The key binary you care about is `llama-cli` (or `main` in older versions) — this is the interactive inference binary.

**Build time expectation:** First build will take **5–15 minutes** depending on your CPU. This is normal — it is compiling CUDA kernels for your specific GPU architecture.

**What to confirm after build:**
- The build completed without errors (warnings are acceptable)
- The binary `llama-cli.exe` exists in the build output directory
- Run it with `--help` flag and confirm it prints usage information

---

## Phase C — Download a Test Model

### C1 — Choose Your Test Model Carefully
For Step 0, you want a model that is:
- **Small enough** to load quickly (not a 70B model — that would take 30+ minutes to download)
- **Popular enough** that its reported specs are well-documented (so you can verify your later predictor math)
- **Available as GGUF** on Hugging Face

**Good choices for Step 0:**
- `Llama-3.2-3B` in Q4_K_M quantization — small, well-documented, fast to load
- `Qwen2.5-7B` in Q4_K_M — slightly larger, still manageable, widely used for benchmarking

**Why Q4_K_M specifically:** It is the most commonly benchmarked quantization level. There is more published data to compare against, which helps you validate your Step 3 formulas later.

---

### C2 — Download the Model
Go to the model's Hugging Face page, find the GGUF file, and download it directly to `C:\dev\models\` (or your equivalent models folder).

**File size expectation:**
- 3B Q4_K_M: approximately 2GB
- 7B Q4_K_M: approximately 4.5GB

**Naming note:** Write down the exact filename. You will need it for the run command.

---

## Phase D — Run the Model and Confirm Everything Works

### D1 — Run llama.cpp With the Downloaded Model
From the terminal, navigate to the llama.cpp build output directory and run the model.

**Key parameters to set on your first run:**
- The path to your GGUF file
- Number of GPU layers to offload (start with a number that fits in your VRAM — for a 7B Q4_K_M model, 32–35 layers is typically full offload on an 8GB card)
- A simple prompt (e.g., "The capital of France is")
- A token limit (e.g., 50 tokens — enough to confirm generation without waiting long)

**What to watch for in the output:**
- It prints the model's architecture information (layer count, context length, embedding dimensions) — **write these down**, you will compare them against your Step 2 metadata fetcher later
- It reports memory usage — **write this down**, you will compare it against your Step 3 predictor formulas
- It prints tokens/sec at the end — **write this down**, same reason
- You see actual generated text appear

---

### D2 — Record Your Baseline Numbers
This is the most important part of Step 0 and most people skip it. **Don't skip it.**

Create a simple text file called `baseline.txt` in your working directory and record:

| What | Value |
|---|---|
| GPU model | (e.g., RTX 3080) |
| VRAM total | (from nvidia-smi) |
| RAM total | (from Task Manager) |
| Model used | (filename + quant) |
| Layers offloaded to GPU | (the number you set) |
| Reported memory usage by llama.cpp | (from its output) |
| Reported tokens/sec | (from its output) |
| Prompt used | (exact text) |

**Why this matters:** In Step 3, your predictor will calculate what these numbers *should* be. You will compare your formula's output against these real numbers. If you don't write them down now, you will have to re-run everything later.

---

## Phase E — Set Up Your Own Project Skeleton

### E1 — Create the Project Structure
Create a new directory for your own tool (separate from llama.cpp). A clean starting structure:

```
llm-planner/
├── CMakeLists.txt        ← your build file
├── README.md             ← even one line is fine
├── src/
│   └── main.cpp          ← entry point, prints "hello" for now
└── external/             ← where llama.cpp will be referenced from
```

---

### E2 — Write a Minimal CMakeLists.txt
At this stage, the CMake file just needs to:
- Declare the project name
- Set the C++ standard (C++17 minimum)
- Define one executable target from `src/main.cpp`

**Nothing else.** No llama.cpp linking yet. No NVML yet. That comes in Steps 1 and 6 respectively. The goal here is just confirming your own project builds cleanly.

---

### E3 — Confirm Your Project Builds
Build your skeleton project with CMake. If `main.cpp` prints "hello world" and the build succeeds, Step 0 is complete.

---

## Step 0 — Done Checklist

Before moving to Step 1, confirm every item:

- [ ] `nvidia-smi` shows your GPU, VRAM, and driver version
- [ ] `nvcc --version` shows the CUDA toolkit version
- [ ] `cmake --version` works
- [ ] `git --version` works
- [ ] llama.cpp cloned and built successfully with CUDA enabled
- [ ] At least one GGUF model downloaded to your models folder
- [ ] Model ran successfully through `llama-cli` and generated tokens
- [ ] Baseline numbers recorded in `baseline.txt`
- [ ] Your own CMake project skeleton exists and builds cleanly

---

## Common Failure Points at Step 0 (Know These Before You Hit Them)

| Problem | Likely Cause | Fix |
|---|---|---|
| `nvcc` not found after CUDA install | PATH not updated | Open a new terminal, or manually add CUDA bin to PATH |
| llama.cpp build fails with CUDA errors | Wrong `sm_XX` architecture flag | Check your GPU's compute capability and correct the flag |
| Model runs but 0 layers on GPU | CUDA backend not enabled in build | Rebuild llama.cpp with the CUDA flag explicitly set |
| Build fails with "path not found" | Spaces in directory path | Move project to a path with no spaces |
| Slow generation despite GPU | All layers falling back to CPU | Check llama.cpp output — it reports where each layer runs |
| Windows Defender quarantines build output | Compiled binaries flagged | Add your dev folder to Defender exclusions |

---

## Time Estimate for Step 0
- Phase A (Toolchain install): **2–4 hours** (mostly download + install time, not active work)
- Phase B (Build llama.cpp): **30–60 minutes**
- Phase C (Download model): **10–30 minutes** depending on connection
- Phase D (Run + record baseline): **20–30 minutes**
- Phase E (Project skeleton): **20–30 minutes**

**Total: Half a day, as originally estimated. Mostly waiting, not working.**