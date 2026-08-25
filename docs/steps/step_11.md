# Step 11 — Platform Expansion: AMD (ROCm) + Apple (Metal): Full Detailed Plan

---

## Goal of Step 11
Extend the tool to work on AMD GPUs (via ROCm/HIP on Linux, DirectML/Vulkan on Windows) and Apple Silicon (via Metal on macOS), using the interface-driven architecture already locked into the spec doc (§6). After this step, the tool profiles hardware, predicts performance, and executes models on **any major desktop platform** — not just NVIDIA. The predictor formulas, ranker, calibration log, and download manager are platform-agnostic and require zero changes. Only the hardware profiler and executor backends are platform-specific.

---

## Why This Step Exists

From the original spec §3 item 10: "MVP = Linux + NVIDIA + CPU/RAM/NVMe only. Other platforms are follow-on modules behind the same interface."

That interface (`IHardwareProfiler`, `IExecutionBackend`) was designed in §6 specifically to make this step additive, not a rewrite. You are now cashing in on that architectural decision.

**Market reality:** NVIDIA dominates the high-end local LLM market, but AMD holds ~30% of the discrete GPU market (RX 7900 XTX, RX 7800 XT), and Apple Silicon (M1/M2/M3/M4) is the dominant platform for Mac users who want local inference. Ignoring these platforms leaves ~40% of potential users without a tool.

---

## What You Need Before Starting

### From Steps 1–10 (must be solid)
- The full pipeline works on Windows + NVIDIA
- `IHardwareProfiler` and `IExecutionBackend` interfaces are defined (even if informally as function signatures)
- Predictor formulas are validated and calibrated for NVIDIA
- Calibration log isolates records by hardware fingerprint

### New Hardware Required (Hard Blockers)
You **cannot** develop and test platform support without the actual hardware. Emulators and CI runners are insufficient for performance profiling.

| Platform | Minimum Test Hardware | Why |
|---|---|---|
| **AMD Linux** | RX 7800 XT or better, Ubuntu 22.04+ with ROCm 6.x | ROCm is Linux-only for consumer GPUs. Windows AMD is a different path (DirectML). |
| **AMD Windows** | RX 7800 XT or better, Windows 11 | Uses DirectML or Vulkan backend, not ROCm. Different code path. |
| **Apple Silicon** | M2 Pro or better, macOS 14+ | Metal backend. Unified memory changes the placement model fundamentally. |

**If you don't have all three:** Prioritize. Recommended order:
1. **Apple Silicon** (if you have a Mac) — largest user base after NVIDIA, most mature llama.cpp backend after CUDA
2. **AMD Linux** — ROCm is well-supported in llama.cpp, and the API is structurally similar to NVML
3. **AMD Windows** — least mature path, most likely to have llama.cpp backend issues

### What You Are NOT Doing
- **Rewriting the predictor.** The formulas are the same math. Only the input constants (bandwidth, TFLOPS, overhead) change per platform.
- **Rewriting the ranker, matrix, download manager, or calibration log.** These are platform-agnostic.
- **Supporting Intel Arc GPUs.** Too early — llama.cpp's SYCL/oneAPI backend is immature. Defer to Phase 4.
- **Supporting mobile (Android/iOS).** Different tool entirely. Out of scope.

---

## Phase A — Platform Architecture Overview

### A1. The Interface Pattern (From §6, Now Realized)

Your existing code has implicit interfaces. Step 11 makes them explicit.

**IHardwareProfiler:**
```cpp
struct HardwareSpec {
    // Platform-agnostic fields (same struct for all platforms)
    std::string platform;          // "nvidia", "amd", "apple"
    std::string gpu_name;
    uint64_t vram_total_bytes;     // unified memory total on Apple
    uint64_t vram_free_bytes;      // unified memory available on Apple
    uint64_t ram_total_bytes;
    uint64_t ram_free_bytes;
    double gpu_bandwidth_gbs;
    double ram_bandwidth_gbs;
    double gpu_tflops_fp16;
    double nvme_sequential_mbs;
    double nvme_random_4k_mbs;
    uint32_t gpu_temp_celsius;
    std::string compute_capability; // "sm_86" / "gfx1100" / "apple_m2"
    std::string hardware_fingerprint;
    bool is_unified_memory;        // true for Apple, false for discrete GPUs
};

class IHardwareProfiler {
public:
    virtual ~IHardwareProfiler() = default;
    virtual bool isAvailable() = 0;
    virtual HardwareSpec profile() = 0;
    virtual uint64_t getFreeVRAM() = 0;  // live polling during execution
    virtual uint32_t getGPUTemp() = 0;
    virtual uint32_t getGPUClock() = 0;
};
```

**IExecutionBackend:**
```cpp
class IExecutionBackend {
public:
    virtual ~IExecutionBackend() = default;
    virtual bool isAvailable() = 0;
    virtual std::string name() = 0;  // "cuda", "hip", "metal", "vulkan", "directml"
    virtual void initBackend() = 0;
    virtual void shutdownBackend() = 0;
    virtual llama_model_params getModelParams(const StrategyConfig& strategy) = 0;
    virtual llama_context_params getContextParams(const StrategyConfig& strategy) = 0;
    // Tensor overrides, buffer types, etc.
};
```

### A2. The Platform Matrix

| Platform | OS | Profiler API | Executor Backend | llama.cpp Flag | Maturity |
|---|---|---|---|---|---|
| NVIDIA | Windows | NVML (`nvml.dll`) | CUDA | `-DGGML_CUDA=ON` | ✅ Production |
| NVIDIA | Linux | NVML (`libnvidia-ml.so`) | CUDA | `-DGGML_CUDA=ON` | ✅ Production |
| AMD | Linux | ROCm-SMI (`librocm_smi64.so`) | HIP | `-DGGML_HIPBLAS=ON` | ✅ Good |
| AMD | Windows | ADL / WMI | DirectML or Vulkan | `-DGGML_VULKAN=ON` | ⚠️ Experimental |
| Apple | macOS | IOKit / Metal | Metal | `-DGGML_METAL=ON` | ✅ Good |
| CPU-only | Any | OS APIs | CPU | (default) | ✅ Production |

### A3. Build System Implications

Your CMake configuration currently builds for one platform. Now it needs to support conditional compilation:

```cmake
# Detect platform
if(APPLE)
    set(GGML_METAL ON)
    set(PLATFORM_BACKEND "metal")
elseif(UNIX)
    # Check for ROCm
    find_package(hip QUIET)
    if(hip_FOUND)
        set(GGML_HIPBLAS ON)
        set(PLATFORM_BACKEND "hip")
    else()
        # Check for CUDA
        find_package(CUDAToolkit QUIET)
        if(CUDAToolkit_FOUND)
            set(GGML_CUDA ON)
            set(PLATFORM_BACKEND "cuda")
        endif()
    endif()
elseif(WIN32)
    # Check for CUDA first, then Vulkan
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        set(GGML_CUDA ON)
        set(PLATFORM_BACKEND "cuda")
    else()
        set(GGML_VULKAN ON)
        set(PLATFORM_BACKEND "vulkan")
    endif()
endif()
```

**Key decision:** Build for the **current** platform, not all platforms. The binary is platform-specific. You don't ship one binary that works everywhere — you ship three builds (Windows, Linux, macOS), each with the appropriate backend compiled in.

**Alternative (advanced):** Compile multiple backends into one binary and select at runtime. llama.cpp supports this (e.g., CUDA + CPU in the same binary). For MVP, keep it simple: one backend per build.

---

## Phase B — AMD Profiler (Linux: ROCm-SMI)

### B1. ROCm-SMI Overview

ROCm-SMI (System Management Interface) is AMD's equivalent of NVIDIA's NVML. It provides GPU telemetry via a C library.

**Library:** `librocm_smi64.so` (ships with the ROCm toolkit)
**Header:** `rocm_smi/rocm_smi.h`
**Install:** `sudo apt install rocm-smi-lib` (Ubuntu) or included with ROCm toolkit

### B2. ROCm-SMI Initialization

```cpp
#include <rocm_smi/rocm_smi.h>

// Initialize
rsmi_status_t status = rsmi_init(0);
if (status != RSMI_STATUS_SUCCESS) {
    // AMD GPU not available or driver not loaded
}

// Get device count
uint32_t num_devices = 0;
rsmi_num_monitor_devices(&num_devices);
```

### B3. The Queries You Need

| What | ROCm-SMI Function | Notes |
|---|---|---|
| GPU name | `rsmi_dev_name_get()` | Returns e.g., "AMD Radeon RX 7900 XTX" |
| VRAM total | `rsmi_dev_memory_total_get(device, RSMI_MEM_TYPE_VRAM, &total)` | In bytes |
| VRAM used | `rsmi_dev_memory_usage_get(device, RSMI_MEM_TYPE_VRAM, &used)` | `free = total - used` |
| Memory clock | `rsmi_dev_gpu_clk_freq_get(device, RSMI_CLK_TYPE_MEM, &freq)` | Returns multiple frequency levels; use the current active one |
| Memory bus width | `rsmi_dev_memory_bus_width_get(device, &width)` | In bits (e.g., 384 for RX 7900 XTX) |
| GPU temperature | `rsmi_dev_temp_metric_get(device, RSMI_TEMP_TYPE_JUNCTION, RSMI_TEMP_CURRENT, &temp)` | In millidegrees Celsius (divide by 1000) |
| GPU compute capability | `rsmi_dev_gpu_id_get()` or parse from `rsmi_dev_name_get()` | AMD uses "gfx" codes (e.g., `gfx1100` for RDNA3), not "sm_XX" |
| GPU utilization | `rsmi_dev_busy_percent_get(device, &busy)` | 0–100% |

### B4. Deriving Memory Bandwidth (AMD)

Same formula as NVIDIA, different inputs:

```
bandwidth_GB_per_sec = (memory_clock_MHz × 2 × bus_width_bits) / 8 / 1000
```

**Example:** RX 7900 XTX:
- Memory clock: 2500 MHz (effective, GDDR6)
- Bus width: 384-bit
- Bandwidth: `(2500 × 2 × 384) / 8 / 1000 = 960 GB/s`

**AMD-specific caveat:** AMD's memory clock reporting can be confusing. `rsmi_dev_gpu_clk_freq_get()` returns a list of supported frequencies with an "active" indicator. The memory clock may be reported as the base clock (half the effective rate for GDDR6). Verify against the spec sheet. If the derived bandwidth is 2× or 0.5× the expected value, you have a clock doubling issue.

### B5. AMD Compute Capability Mapping

AMD doesn't use NVIDIA's `sm_XX` naming. The equivalent is the "gfx" target:

| GPU Family | gfx Code | ROCm Target | Notes |
|---|---|---|---|
| RDNA 1 (RX 5000) | gfx1010 | `gfx1010` | Not well-supported by ROCm |
| RDNA 2 (RX 6000) | gfx1030 | `gfx1030` | Partial ROCm support |
| RDNA 3 (RX 7000) | gfx1100, gfx1101, gfx1102 | `gfx1100` | Best ROCm support for consumer AMD |
| CDNA 2 (MI250) | gfx90a | `gfx90a` | Data center, excellent support |
| CDNA 3 (MI300) | gfx942 | `gfx942` | Data center, best support |

For the `hardware_fingerprint`, use the gfx code instead of sm_XX:
```
"ryzen-9-7950x|AMD Radeon RX 7900 XTX|64GB|Samsung 990 PRO"
```

The `compute_capability` field stores `"gfx1100"` instead of `"sm_89"`. The predictor doesn't use this field for math — it's for calibration log grouping.

### B6. AMD-Specific Profiler Challenges

| Challenge | Detail | Mitigation |
|---|---|---|
| ROCm not installed | Many AMD users have the display driver but not the ROCm toolkit | Check for `librocm_smi64.so` at runtime. If missing, fall back to parsing `rocm-smi` CLI output (less reliable but works without the library). |
| Consumer GPU support | ROCm officially supports only a subset of consumer GPUs (RX 7900 series). Older RDNA GPUs may not be recognized. | Check `rsmi_init()` return code. If it fails, fall back to Vulkan backend (Phase B-Alt). |
| VRAM reporting | AMD's VRAM reporting includes "invisible" memory used by the display driver. Free VRAM may be lower than expected. | Use `rsmi_dev_memory_usage_get()` which reports actual usage, not just driver reservation. |
| Temperature sensor naming | AMD has multiple temperature sensors (junction, edge, memory). Junction is the most relevant for throttling. | Use `RSMI_TEMP_TYPE_JUNCTION`. |

### B7. AMD Windows Profiler (DirectML/Vulkan Path)

On Windows, AMD GPUs do not use ROCm. The profiling options are:

**Option A: AMD Display Library (ADL)**
- `atiadlxx.dll` / `atiadlxy.dll` — ships with AMD display drivers
- Provides GPU name, VRAM, temperature, clock speeds
- API is Windows-specific and poorly documented
- Requires dynamic loading (`LoadLibrary`) because the DLL name varies by driver version

**Option B: WMI + DXGI**
- `Win32_VideoController` WMI class for basic info (name, VRAM)
- `IDXGIAdapter` for VRAM details
- No temperature or clock speed access
- Simpler but less complete

**Recommendation for AMD Windows:** Use Option A (ADL) for full profiling, with Option B as fallback. The ADL API is messy but functional. Wrap it in the same `IHardwareProfiler` interface.

**Critical note:** The executor backend for AMD Windows is Vulkan or DirectML, not HIP. This means the performance characteristics are different (Vulkan is generally slower than HIP for LLM inference). The predictor should use a lower efficiency factor for Vulkan.

---

## Phase C — AMD Executor (Linux: HIP Backend)

### C1. Building llama.cpp with HIP

The CMake flag is `-DGGML_HIPBLAS=ON`. The build process:

```bash
cmake -B build -DGGML_HIPBLAS=ON -DCMAKE_HIP_ARCHITECTURES="gfx1100"
cmake --build build --config Release
```

**Key differences from CUDA build:**
- `CMAKE_HIP_ARCHITECTURES` replaces `CMAKE_CUDA_ARCHITECTURES`
- The HIP compiler (`hipcc`) replaces `nvcc`
- The output library is `libggml-hipblas.so` instead of `libggml-cuda.so`
- The C API (`llama.h`) is **identical** — no code changes in your executor

### C2. Runtime Differences

| Aspect | CUDA (NVIDIA) | HIP (AMD) | Impact |
|---|---|---|---|
| `n_gpu_layers` | Works identically | Works identically | None |
| Tensor overrides | Works identically | Works identically | None |
| KV cache quant | Works identically | Works identically | None |
| Flash attention | Supported | Supported on RDNA3+ | Check at runtime |
| Memory bandwidth | 760–1008 GB/s (consumer) | 576–960 GB/s (consumer) | Different predictor constants |
| FP16 TFLOPS | 30–83 TFLOPS (consumer) | 25–61 TFLOPS (consumer) | Different TTFT constants |
| CUDA context overhead | ~200–500 MB | ~150–400 MB | Slightly lower overhead |
| Warm-up time | 1–3 seconds | 2–5 seconds | First token may be slower |

### C3. The Executor Code Changes

**Almost none.** The executor from Step 6 calls `llama_model_load_from_file()`, `llama_new_context_with_model()`, `llama_decode()`, etc. These are the same C functions regardless of backend. The backend is selected at build time (and optionally at runtime if multiple backends are compiled in).

**The only changes:**
1. `llama_backend_init()` initializes the correct backend (HIP instead of CUDA)
2. The `IExecutionBackend` implementation returns HIP-specific default parameters
3. The live sampler reads VRAM via ROCm-SMI instead of NVML

### C4. AMD-Specific Execution Issues

| Issue | Detail | Mitigation |
|---|---|---|
| HIP kernel compilation at runtime | First run may JIT-compile kernels, causing a 10–30 second delay | Warn the user: "First run on AMD may take 30 seconds to compile GPU kernels." |
| GDDR6 bandwidth variability | AMD's GDDR6 controllers have more bandwidth variance under load than NVIDIA's GDDR6X | Use the measured bandwidth from the profiler, not the spec sheet. The micro-benchmark from Step 1 is even more important on AMD. |
| ROCm driver crashes | Some ROCm versions have stability issues with long-running workloads | Add a watchdog in the sampler thread: if VRAM reads fail, abort gracefully. |
| Multi-GPU | ROCm multi-GPU support is less mature than CUDA | For MVP, support single GPU only. Same as NVIDIA MVP. |

---

## Phase D — Apple Silicon Profiler (Metal)

### D1. The Fundamental Difference: Unified Memory

Apple Silicon (M1/M2/M3/M4) has **unified memory architecture (UMA)**. There is no separate VRAM and RAM — the CPU and GPU share the same physical memory pool. This changes the placement model fundamentally:

| Concept | Discrete GPU (NVIDIA/AMD) | Apple Silicon |
|---|---|---|
| VRAM | Separate physical memory on the GPU card | Shared with system RAM |
| RAM | Separate physical memory on the motherboard | Same as VRAM |
| `n_gpu_layers` | Controls which layers go to GPU VRAM vs system RAM | Controls which layers use Metal (GPU) vs CPU compute, but all data is in the same memory |
| Memory bandwidth | GPU bandwidth (fast) vs RAM bandwidth (slow) | Single bandwidth number, but GPU access is faster due to proximity |
| Placement strategies | Full GPU / Split / CPU-only | "Full Metal" / "Partial Metal" / "CPU-only" — the memory is the same, but compute location differs |
| KV cache location | VRAM or RAM | Unified memory — always "in RAM" but accessed by GPU or CPU |

**What this means for the predictor:**
- `vram_total_bytes` = total unified memory (e.g., 32GB for M2 Pro)
- `vram_free_bytes` = available unified memory (total minus OS and app usage)
- `ram_total_bytes` = same as `vram_total_bytes`
- `ram_free_bytes` = same as `vram_free_bytes`
- `is_unified_memory = true`
- The memory footprint formula is the same, but the **viability check** changes: instead of checking VRAM and RAM separately, check the single unified memory pool

### D2. Apple Hardware Profiling APIs

**Memory:**
```objc
#include <mach/mach.h>
#include <sys/sysctl.h>

// Total memory
int64_t total_memory;
size_t size = sizeof(total_memory);
sysctlbyname("hw.memsize", &total_memory, &size, NULL, 0);

// Available memory (approximate)
mach_port_t host = mach_host_self();
vm_statistics64_data_t vm_stats;
mach_msg_type_number_t info_count = HOST_VM_INFO64_COUNT;
host_statistics64(host, HOST_VM_INFO64, (host_info64_t)&vm_stats, &info_count);

int64_t page_size;
sysctlbyname("hw.pagesize", &page_size, &size, NULL, 0);
int64_t available = (int64_t)(vm_stats.free_count + vm_stats.inactive_count) * page_size;
```

**GPU Info (Metal):**
```objc
#import <Metal/Metal.h>

id<MTLDevice> device = MTLCreateSystemDefaultDevice();
NSString* gpu_name = device.name;                    // e.g., "Apple M2 Pro"
uint64_t recommended_max = device.recommendedMaxWorkingSetSize;  // max memory Metal can use
BOOL has_unified_memory = device.hasUnifiedMemory;    // always YES on Apple Silicon
```

**CPU Info:**
```objc
#include <sys/sysctl.h>

// CPU model
char cpu_name[256];
size_t size = sizeof(cpu_name);
sysctlbyname("machdep.cpu.brand_string", cpu_name, &size, NULL, 0);

// Core count
int ncpu;
sysctlbyname("hw.ncpu", &ncpu, &size, NULL, 0);

// Performance cores vs efficiency cores (Apple-specific)
int perf_cores, eff_cores;
sysctlbyname("hw.perflevel0.physicalcpu", &perf_cores, &size, NULL, 0);
sysctlbyname("hw.perflevel1.physicalcpu", &eff_cores, &size, NULL, 0);
```

**Memory Bandwidth:**
Apple doesn't expose memory bandwidth via an API. You must either:
- Use a lookup table based on the chip model
- Run a memcpy micro-benchmark (same as Step 1, but on unified memory)

**Lookup table for common Apple Silicon:**

| Chip | Memory Bandwidth | GPU Cores | Notes |
|---|---|---|---|
| M1 | 68.25 GB/s | 7–8 | Base model |
| M1 Pro | 200 GB/s | 14–16 | |
| M1 Max | 400 GB/s | 24–32 | |
| M1 Ultra | 800 GB/s | 48–64 | |
| M2 | 100 GB/s | 8–10 | |
| M2 Pro | 200 GB/s | 16–19 | |
| M2 Max | 400 GB/s | 30–38 | |
| M2 Ultra | 800 GB/s | 60–76 | |
| M3 | 100 GB/s | 8–10 | |
| M3 Pro | 150 GB/s | 14–18 | |
| M3 Max | 300–400 GB/s | 30–40 | |
| M4 | 120 GB/s | 10 | |
| M4 Pro | 273 GB/s | 16–20 | |
| M4 Max | 546 GB/s | 32–40 | |

**Recommendation:** Use the lookup table for the initial bandwidth estimate, then refine with the memcpy micro-benchmark from Step 1. The micro-benchmark measures actual achievable bandwidth, which is typically 70–85% of the theoretical maximum.

### D3. Apple-Specific Profiler Challenges

| Challenge | Detail | Mitigation |
|---|---|---|
| No temperature sensor | Apple Silicon doesn't expose GPU temperature via public APIs | Skip thermal throttle detection on Apple. Use CPU frequency throttling as a proxy (if available via `sysctl`). |
| Memory pressure | macOS aggressively compresses and pages memory. "Available" memory is misleading. | Use `vm_stats.compressor_count` to detect memory compression pressure. If compressor is active, the system is under memory stress. |
| GPU memory limit | `recommendedMaxWorkingSetSize` is the maximum memory Metal recommends using, not the total unified memory. Exceeding it causes performance degradation. | Use `recommendedMaxWorkingSetSize` as the effective VRAM limit for Metal compute, not the total unified memory. |
| No NVMe benchmark equivalent | Apple's SSD is soldered and uses a proprietary controller. The random-read benchmark still works but the numbers are different. | Run the same Step 1 disk benchmark. Apple SSDs typically achieve 2–7 GB/s sequential and 100–500 MB/s random 4K. |

---

## Phase E — Apple Executor (Metal Backend)

### E1. Building llama.cpp with Metal

```bash
cmake -B build -DGGML_METAL=ON
cmake --build build --config Release
```

**Key differences from CUDA/HIP build:**
- No GPU architecture flag needed (Metal handles device selection)
- The Metal backend compiles GPU kernels at runtime from `.metal` source files (shipped with llama.cpp)
- The output includes `libggml-metal.dylib` and a `ggml-metal.metal` shader file that must be accessible at runtime
- The C API is **identical** — no code changes in your executor

### E2. Metal-Specific Execution Behavior

| Aspect | CUDA/HIP | Metal | Impact |
|---|---|---|---|
| `n_gpu_layers` | Layers on GPU VRAM vs CPU RAM | Layers computed by Metal GPU vs CPU, but all data in unified memory | Same flag works, but the meaning is "compute location" not "data location" |
| Memory transfers | PCIe bottleneck between RAM and VRAM | No PCIe transfer — data is already in unified memory | **Major advantage:** split strategies don't have the PCIe penalty |
| Flash attention | Supported | Supported on M3+ | Check at runtime |
| KV cache | In VRAM or RAM | In unified memory, accessed by GPU or CPU | Same formula, single memory pool |
| Batch size | Large batches benefit from GPU | Metal has smaller optimal batch sizes | May need to tune `n_batch` for Metal |
| First-run overhead | CUDA context init (~200ms) | Metal shader compilation (~1–5 seconds) | First token slower on first run |

### E3. The Unified Memory Placement Model

For Apple Silicon, the placement strategies from Steps 3–6 need reinterpretation:

| Strategy | Discrete GPU Meaning | Apple Silicon Meaning |
|---|---|---|
| `FULL_GPU` | All layers in VRAM | All layers computed by Metal GPU |
| `GPU_CPU_SPLIT` | Some layers in VRAM, some in RAM | Some layers computed by Metal, some by CPU. **No memory transfer penalty.** |
| `CPU_ONLY` | All layers in RAM | All layers computed by CPU |
| `HOT_COLD_SPLIT` | Hot neurons in VRAM, cold in RAM | Hot neurons computed by Metal, cold by CPU. **Less beneficial** because there's no PCIe bottleneck to avoid. |
| `LAYER_STREAM` | Load layers from disk one at a time | Same mechanism, but less likely to be needed because unified memory is large (up to 192GB on M2 Ultra) |

**Key insight for the predictor:** On Apple Silicon, the GPU/CPU split doesn't have the sequential-time penalty from PCIe transfers. The penalty is purely compute-speed difference (Metal GPU is faster than CPU for matrix ops, but the data doesn't need to move). This means:

```
// Discrete GPU split (from Step 3):
t_layer = t_gpu + t_cpu  // sequential, PCIe transfer dominates

// Apple Silicon split:
t_layer = max(t_metal, t_cpu)  // parallel, no transfer penalty
// Actually more nuanced: Metal and CPU can process different layers simultaneously
```

This makes split strategies **much more attractive** on Apple Silicon than on discrete GPUs. The predictor needs a platform-specific branch for the split speed formula.

### E4. Metal-Specific Executor Issues

| Issue | Detail | Mitigation |
|---|---|---|
| Metal shader compilation | First run compiles `.metal` shaders, causing a 1–5 second delay | Cache compiled shaders (llama.cpp does this automatically in `~/.cache/`). Warn user on first run. |
| Memory limit enforcement | Exceeding `recommendedMaxWorkingSetSize` causes macOS to swap aggressively | Use `recommendedMaxWorkingSetSize` × 0.9 as the effective memory budget for the predictor. |
| Thermal throttling | Apple Silicon throttles silently under sustained load | Monitor CPU/GPU frequency via `sysctl` if available. Flag performance degradation in the sampler. |
| Multi-GPU | Apple Silicon has one GPU per SoC (except Ultra, which has two) | For M1/M2/M3/M4 (non-Ultra), single GPU. For Ultra, llama.cpp may use both GPU cores automatically. |
| ANE (Neural Engine) | Apple's Neural Engine is not used by llama.cpp | Don't factor ANE into predictions. It's not accessible for LLM inference via Metal. |

---

## Phase F — Platform Auto-Detection

### F1. Runtime Detection Logic

At program startup, detect which platform is available:

```cpp
std::unique_ptr<IHardwareProfiler> createProfiler() {
    // Try NVIDIA first (most common for LLM workloads)
    auto nvidia = std::make_unique<NvidiaProfiler>();
    if (nvidia->isAvailable()) return nvidia;
    
    // Try AMD
    auto amd = std::make_unique<AmdProfiler>();
    if (amd->isAvailable()) return amd;
    
    // Try Apple
    auto apple = std::make_unique<AppleProfiler>();
    if (apple->isAvailable()) return apple;
    
    // Fall back to CPU-only
    return std::make_unique<CpuOnlyProfiler>();
}
```

**Detection methods:**
- NVIDIA: `nvmlInit()` succeeds
- AMD Linux: `rsmi_init()` succeeds
- AMD Windows: `LoadLibrary("atiadlxx.dll")` succeeds or DXGI finds an AMD adapter
- Apple: `MTLCreateSystemDefaultDevice()` returns non-nil and `hasUnifiedMemory == YES`
- CPU-only: always available (fallback)

### F2. Multi-GPU Detection

If the system has both an NVIDIA and an AMD GPU (rare but possible), or an integrated GPU + discrete GPU:
- Prefer the discrete GPU with the most VRAM
- Allow the user to override with `--gpu <index>` or `--gpu-name <pattern>`
- For Apple, there's only one GPU (the SoC's integrated GPU)

### F3. The `--platform` Flag

Allow manual override for testing:
```
llm-planner --model <url> --platform cuda
llm-planner --model <url> --platform hip
llm-planner --model <url> --platform metal
llm-planner --model <url> --platform cpu
```

---

## Phase G — Predictor Adjustments Per Platform

### G1. What Changes

The predictor formulas from Step 3 are the same math. What changes are the **input constants**:

| Constant | NVIDIA | AMD | Apple | Source |
|---|---|---|---|---|
| `gpu_bandwidth_gbs` | 500–1008 | 500–960 | 68–800 | Profiler measurement |
| `gpu_tflops_fp16` | 20–83 | 25–61 | 3.6–21 | Profiler or lookup |
| `runtime_overhead_gpu` | 200–500 MB | 150–400 MB | 50–200 MB | Calibrated |
| `efficiency_factor` | 0.25–0.40 | 0.20–0.35 | 0.30–0.50 | Calibrated |
| `split_pcie_penalty` | Significant | Significant | **Negligible** | Platform architecture |

### G2. The Split Speed Formula Branch

**Discrete GPU (NVIDIA/AMD):**
```
t_layer = bytes_gpu / gpu_bandwidth + bytes_cpu / ram_bandwidth
// Sequential: PCIe transfer + compute
```

**Apple Silicon (Unified Memory):**
```
t_layer = max(bytes_metal / metal_bandwidth, bytes_cpu / cpu_bandwidth)
// Parallel: no transfer penalty, compute is the bottleneck
// metal_bandwidth ≈ 0.7 × unified_memory_bandwidth (GPU gets ~70% of total bandwidth)
// cpu_bandwidth ≈ 0.3 × unified_memory_bandwidth (CPU gets ~30%)
```

This is the most important predictor change. On Apple Silicon, a 50/50 Metal/CPU split is much faster than the same split on a discrete GPU because there's no PCIe bottleneck.

### G3. Confidence Bands Per Platform

| Platform | Initial Confidence | After 5+ Cal Runs |
|---|---|---|
| NVIDIA | MEDIUM | HIGH |
| AMD Linux (HIP) | MEDIUM | HIGH |
| AMD Windows (Vulkan) | LOW | MEDIUM |
| Apple Metal | MEDIUM | HIGH |
| CPU-only | HIGH | HIGH |

AMD Windows Vulkan gets lower confidence because the Vulkan backend in llama.cpp is less mature and has more performance variance.

---

## Phase H — Method Matrix Adjustments Per Platform

### H1. Platform-Specific Strategy Filtering

| Strategy | NVIDIA | AMD | Apple |
|---|---|---|---|
| FULL_GPU | ✅ | ✅ | ✅ (Full Metal) |
| GPU_CPU_SPLIT | ✅ | ✅ | ✅ (Metal/CPU, no PCIe penalty) |
| CPU_ONLY | ✅ | ✅ | ✅ |
| MOE_EXPERT_OFFLOAD | ✅ | ✅ (if HIP supports overrides) | ✅ |
| HOT_COLD_SPLIT | ✅ | ✅ | ⚠️ Less beneficial (no PCIe) |
| LAYER_STREAM | ✅ | ✅ | Rarely needed (large unified memory) |

### H2. Apple-Specific Matrix Additions

For Apple Silicon, add a strategy that doesn't exist on discrete GPUs:

**`METAL_MAX`**: Use `recommendedMaxWorkingSetSize` as the memory budget, put as many layers on Metal as possible, rest on CPU. This is the Apple equivalent of "full GPU" but respects the Metal memory limit rather than a VRAM limit.

### H3. The Output Table

The table format stays the same. The only visible change is the platform label:

```
=== LLM Deployment Planner — Strategy Comparison ===
Platform: Apple M2 Pro (Metal) | 32GB Unified Memory | NVMe 5.1 GB/s
Model:    Llama 3.2 3B Instruct Q4_K_M | 3.2B params | 28 layers

 #  Placement      Metal Layers  Context  Memory   tok/s    TTFT    Status
 1  Full Metal     28/28        4K       2.4 GB   ~85      ~35ms   ✅ VIABLE
 2  Full Metal     28/28        32K      5.1 GB   ~72      ~120ms  ✅ VIABLE
 3  Metal/CPU      20/28        128K     9.8 GB   ~55      ~280ms  ✅ VIABLE
 4  CPU Only       0/28         4K       2.4 GB   ~22      ~180ms  ✅ VIABLE
```

---

## Phase I — Testing and Validation

### I1. AMD Testing (Linux)

| Test | Method | Pass |
|---|---|---|
| Profiler reads GPU name | Compare vs `rocm-smi` output | Exact match |
| VRAM total matches | Compare vs `rocm-smi --showmeminfo vram` | Within 1% |
| VRAM free matches | Compare vs `rocm-smi` | Within 200MB |
| Bandwidth derivation | Compare vs spec sheet | 50–100% of spec |
| Temperature reading | Compare vs `rocm-smi --showtemp` | Within 5°C |
| Model loads on HIP | Run 3B Q4 model via executor | Generates coherent text |
| tok/s prediction accuracy | Compare predicted vs actual | Within 25% |
| Memory prediction accuracy | Compare predicted vs actual | Within 15% |
| Split strategy works | Run with partial GPU offload | tok/s between full-GPU and CPU-only |
| Calibration log records | Check JSONL entry | Platform = "amd", gfx code in fingerprint |

### I2. AMD Testing (Windows)

Same tests as Linux, plus:
- ADL library loads correctly
- Vulkan backend initializes
- Performance is lower than Linux HIP (expected — Vulkan is less optimized)
- Predictor uses lower efficiency factor for Vulkan

### I3. Apple Testing

| Test | Method | Pass |
|---|---|---|
| Profiler reads chip name | Compare vs "About This Mac" | Exact match |
| Unified memory total | Compare vs System Report | Exact match |
| Available memory | Compare vs Activity Manager | Within 1GB |
| Metal device detected | `MTLCreateSystemDefaultDevice()` non-nil | Yes |
| `recommendedMaxWorkingSetSize` | Reasonable value (60–80% of total) | Within expected range |
| Model loads on Metal | Run 3B Q4 model via executor | Generates coherent text |
| Full Metal speed | Compare vs CPU-only | Metal should be 2–5× faster |
| Metal/CPU split speed | Compare vs Full Metal | Split should be close to Full Metal (no PCIe penalty) |
| Memory prediction | Compare predicted vs Activity Monitor | Within 15% |
| No thermal throttle flag | Apple doesn't expose temp | Throttle detection gracefully disabled |
| Calibration log records | Check JSONL entry | Platform = "apple", chip name in fingerprint |

### I4. Cross-Platform Regression

| Test | Pass |
|---|---|
| NVIDIA predictions unchanged after adding AMD/Apple code | Exact match with pre-Step-11 numbers |
| Calibration log isolation | AMD records don't affect NVIDIA predictions |
| Dense model predictions | Same accuracy on all platforms |
| MoE predictions | Work on AMD and Apple (if llama.cpp backend supports overrides) |
| Download manager | Platform-agnostic, works everywhere |
| CLI interface | Same flags, same output format |

---

## Phase J — Integration with Existing Pipeline

### J1. The Build Matrix

You now ship three builds:

| Build | OS | Backend | CMake Flags |
|---|---|---|---|
| `llm-planner-windows.exe` | Windows | CUDA or Vulkan | `-DGGML_CUDA=ON` or `-DGGML_VULKAN=ON` |
| `llm-planner-linux` | Linux | CUDA or HIP | `-DGGML_CUDA=ON` or `-DGGML_HIPBLAS=ON` |
| `llm-planner-macos` | macOS | Metal | `-DGGML_METAL=ON` |

Each build auto-detects the available GPU at runtime and selects the correct profiler.

### J2. The User Experience

The user experience is identical across platforms. The only visible difference is the platform label in the output header:

```
NVIDIA:  Hardware: RTX 3080 (10GB VRAM) | 32GB RAM | NVMe 4.2 GB/s
AMD:     Hardware: RX 7900 XTX (24GB VRAM) | 64GB RAM | NVMe 5.1 GB/s
Apple:   Hardware: Apple M2 Pro (32GB Unified) | NVMe 5.1 GB/s
```

### J3. Calibration Log Isolation

The hardware fingerprint naturally isolates calibration records by platform:
- `"i7-12700K|NVIDIA GeForce RTX 3080|32GB|Samsung 980 PRO"` — NVIDIA
- `"ryzen-9-7950x|AMD Radeon RX 7900 XTX|64GB|WD SN850X"` — AMD
- `"Apple M2 Pro|Apple M2 Pro|32GB|Apple SSD"` — Apple

No cross-contamination is possible because the GPU name is part of the fingerprint.

---

## Step 11 — Done Checklist

- [ ] `IHardwareProfiler` interface defined with `isAvailable()`, `profile()`, `getFreeVRAM()`, `getGPUTemp()`, `getGPUClock()`
- [ ] `IExecutionBackend` interface defined with `isAvailable()`, `name()`, `initBackend()`, `getModelParams()`, `getContextParams()`
- [ ] NVIDIA profiler refactored behind `IHardwareProfiler` (no behavior change)
- [ ] AMD Linux profiler implemented via ROCm-SMI
- [ ] AMD Windows profiler implemented via ADL/DXGI (or Vulkan fallback)
- [ ] Apple profiler implemented via Metal/IOKit/sysctl
- [ ] CPU-only profiler implemented as universal fallback
- [ ] Platform auto-detection selects correct profiler at startup
- [ ] `--platform` flag allows manual override
- [ ] AMD Linux executor builds with `-DGGML_HIPBLAS=ON`
- [ ] AMD Windows executor builds with `-DGGML_VULKAN=ON`
- [ ] Apple executor builds with `-DGGML_METAL=ON`
- [ ] llama.cpp C API calls are identical across all backends
- [ ] Unified memory model correctly handled for Apple (single memory pool, `is_unified_memory = true`)
- [ ] Split speed formula branches for discrete GPU (PCIe penalty) vs Apple (no penalty)
- [ ] `recommendedMaxWorkingSetSize` used as Metal memory budget
- [ ] Thermal throttle detection gracefully disabled on Apple
- [ ] Predictor confidence bands reflect platform maturity
- [ ] Method matrix filters strategies by platform capabilities
- [ ] Calibration log isolates records by platform (via hardware fingerprint)
- [ ] NVIDIA predictions unchanged after adding AMD/Apple code
- [ ] Tested on at least one AMD GPU (Linux or Windows)
- [ ] Tested on at least one Apple Silicon Mac
- [ ] Three separate builds compile and run correctly
- [ ] User experience is consistent across all platforms

---

## Common Failure Points at Step 11

| Problem | Likely Cause | Fix |
|---|---|---|
| ROCm-SMI `rsmi_init()` fails | ROCm not installed, or consumer GPU not supported | Fall back to parsing `rocm-smi` CLI output. Or fall back to Vulkan backend. |
| AMD VRAM shows 0 | Wrong memory type enum | Use `RSMI_MEM_TYPE_VRAM`, not `RSMI_MEM_TYPE_GTT` |
| AMD bandwidth is 2× expected | Memory clock reported as base, not effective | Multiply by 2 for GDDR6, or check if the API already returns effective clock |
| Metal `recommendedMaxWorkingSetSize` is very small | macOS is under memory pressure | This is dynamic. Re-check at prediction time, not just at startup. |
| Apple unified memory shows as 0 VRAM | Profiler treating unified memory as "no GPU" | Set `vram_total = unified_memory_total` and `is_unified_memory = true` |
| Split strategy is slower on Apple than Full Metal | Metal/CPU bandwidth split is wrong | On Apple, Metal gets ~70% of bandwidth, CPU gets ~30%. Don't use the discrete GPU formula. |
| Vulkan backend crashes on AMD Windows | Driver version too old | Require AMD Adrenalin 23.x+. Print driver version in profiler output. |
| llama.cpp Metal backend not found at runtime | `ggml-metal.metal` shader file not in the expected location | Bundle the shader file with the executable, or set the search path explicitly. |
| HIP kernel compilation timeout | First run on AMD takes too long | Increase timeout. Cache compiled kernels (llama.cpp does this). |
| Cross-platform calibration contamination | Hardware fingerprint not including platform identifier | The GPU name is already in the fingerprint, which is platform-specific. Verify no collisions. |
| Build fails on macOS | Xcode command-line tools not installed | `xcode-select --install`. Also ensure Metal framework is linked. |
| Build fails on AMD Linux | ROCm toolkit not in CMake search path | Set `CMAKE_PREFIX_PATH=/opt/rocm` or use `hip-config.cmake`. |

---

## Time Estimate for Step 11

| Phase | Work | Time |
|---|---|---|
| A | Interface design + platform matrix + CMake restructuring | 1–2 days |
| B | AMD profiler (ROCm-SMI on Linux + ADL on Windows) | 2–3 days |
| C | AMD executor (HIP build + runtime validation) | 1–2 days |
| D | Apple profiler (Metal/IOKit/sysctl + unified memory model) | 2–3 days |
| E | Apple executor (Metal build + unified memory placement) | 1–2 days |
| F | Platform auto-detection + `--platform` flag | 1 day |
| G | Predictor adjustments (split formula branch, platform constants) | 1–2 days |
| H | Method matrix adjustments (platform-specific filtering) | 0.5 day |
| I | Testing on all three platforms | 2–3 days |
| J | Pipeline integration + build matrix | 1 day |

**Total: 1–2 weeks, as estimated. The AMD path (Phases B–C) and Apple path (Phases D–E) can be developed in parallel if you have access to both hardware types simultaneously. The predictor adjustments (Phase G) are the most intellectually demanding part — getting the unified memory split formula right for Apple is critical.**

---

## What "Done" Looks Like

A user on any major desktop platform can:
1. Download the appropriate build for their OS
2. Run `llm-planner --model <url>`
3. See a prediction table optimized for their specific GPU
4. Execute the best strategy
5. Get calibrated predictions that improve over time

The tool no longer says "NVIDIA only." It says "your hardware, your model, your best strategy" — regardless of what's inside the machine.