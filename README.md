# Local LLM Deployment Planner

**Status:** Step 0 - Environment Setup
**License:** TBD

A deployment planner + execution engine for local LLMs. Given a model and your hardware, it:
1. Predicts every viable execution strategy
2. Ranks them by your priorities (speed / quality / safety)
3. Executes the chosen strategy
4. Calibrates predictions against actual performance

## Architecture

```
local_llm/
├── CMakeLists.txt          # Root build file
├── src/                    # Planner source code
│   ├── main.cpp            # Entry point (Step 0 placeholder)
│   ├── profiler/           # Hardware profiling (NVML, NVMe, RAM)
│   ├── predictor/          # Memory/speed prediction formulas
│   ├── executor/           # llama.cpp integration & execution
│   └── calibration/        # Predicted vs actual logging
├── external/               # External dependencies
│   └── llama.cpp/          # (git-ignored, built separately)
├── docs/                   # Specifications and step-by-step plans
│   ├── local_llm_planner_spec.md
│   └── steps/
│       └── step_0.md
└── models/                 # GGUF model files (git-ignored)
```

## Prerequisites

- Windows with NVIDIA GPU (8GB+ VRAM)
- CUDA Toolkit 12.x
- Visual Studio 2022 (MSVC)
- CMake 3.14+
- Git

## Quick Start (Step 0)

```bash
# 1. Build llama.cpp with CUDA
./build_llama.ps1

# 2. Build the planner project
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release

# 3. Verify
./build/bin/Release/planner.exe
```

## Build Status

| Component | Status |
|-----------|--------|
| Project skeleton | ✅ Done |
| llama.cpp build | ⏳ Pending |
| Hardware profiler | 📋 Step 1 |
| Predictor | 📋 Step 3 |
| Executor | 📋 Step 6 |
| Calibration | 📋 Step 7 |

## References

- [MVP Specification](docs/local_llm_planner_spec.md)
- [Step 0 Setup Guide](docs/steps/step_0.md)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
