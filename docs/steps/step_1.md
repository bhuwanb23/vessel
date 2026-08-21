# Step 1 — Hardware Profiler: Full Detailed Plan

---

## Goal of Step 1
Build a standalone C++ binary that, when run, prints a complete hardware profile of your machine: system RAM, GPU details, and storage throughput. No model loading, no inference, no networking. This binary is the foundation that Steps 2–7 will feed into. By the end of this step, you will trust the numbers it reports because you have validated them against independent tools.

---

## What You Need Before Starting

### From Step 0 (already done)
- Working MSVC + CMake toolchain
- CUDA Toolkit installed (needed for NVML headers and libraries)
- Your project skeleton (`llm-planner/`) that builds cleanly
- At least one large GGUF file sitting in your models folder (you will use it as the disk benchmark target)

### New Dependencies to Acquire During This Step
- **NVML headers and library** — these ship with the CUDA Toolkit you already installed. No separate download needed. You just need to know where they are.
- **Windows API** — already available via MSVC. `GlobalMemoryStatusEx()` lives in `<windows.h>`, which is part of the Windows SDK installed with Visual Studio.

---

## Phase A — Understand the Three Subsystems

Before writing anything, understand that your profiler has **three independent subsystems** that do not talk to each other. Each reads a different piece of hardware. Structure your code accordingly from day one — separate functions or modules for each — because in Phase 2 of the overall project, when you add AMD or Apple support, you will replace individual subsystems without touching the others.

### The Three Subsystems
| Subsystem | What It Reads | Windows API | External Library |
|---|---|---|---|
| RAM Profiler | Total and available system memory | `GlobalMemoryStatusEx()` | None (built into Windows) |
| GPU Profiler | GPU name, VRAM total/free, memory bandwidth, temperature | NVML C API | `nvml.dll` / `nvml.lib` |
| Disk Profiler | Sequential and random read throughput | `CreateFile()` + `ReadFile()` | None (built into Windows) |

### Design Decision to Make Now
**How will you structure the output?** The simplest approach for MVP: three functions, each returning a struct, and a `main()` that calls all three and prints the results. Don't over-engineer the interface yet — the `IHardwareProfiler` abstraction from the spec doc can be formalized later when you actually have a second platform to support. For now, clean separation of concerns is enough.

---

## Phase B — RAM Profiler

### What You Are Measuring
- **Total physical RAM** installed in the machine
- **Available physical RAM** right now (at the moment the profiler runs)

### The Windows API Call
The function is `GlobalMemoryStatusEx()`. It takes a single argument: a pointer to a `MEMORYSTATUSEX` struct that you must initialize by setting its `dwLength` field to `sizeof(MEMORYSTATUSEX)` before calling.

### Key Fields You Will Read
| Field | What It Means | Why It Matters |
|---|---|---|
| `ullTotalPhys` | Total physical RAM in bytes | Tells the predictor the absolute ceiling for CPU-offloaded models |
| `ullAvailPhys` | Available physical RAM in bytes | Tells the predictor how much room is actually left for model weights after the OS and background apps |
| `ullTotalPageFile` | Total commit charge limit | Useful for detecting swap risk — if available RAM is low but page file is large, the OS will swap instead of crashing, which is the "severe slowdown" failure mode from §3 item 7 |
| `ullAvailPageFile` | Available commit charge | Combined with `ullAvailPhys`, tells you whether the system is close to swapping |

### Critical Distinction: "Available" vs "Free"
Windows does not have a simple "free RAM" number the way Linux's `/proc/meminfo` does. Windows aggressively caches disk data in RAM (called "Standby" memory). This cached memory is technically "in use" but can be **instantly reclaimed** when an application needs it.

- `ullAvailPhys` includes this reclaimable standby cache — **this is the number you want**
- Task Manager's "Free" number is smaller because it excludes standby cache
- Task Manager's "Available" number matches `ullAvailPhys` — **this is what you validate against**

**Why this matters for LLMs:** When llama.cpp loads a model into RAM, Windows will evict standby cache to make room. So the relevant question is "how much RAM can the OS give me right now?" — which is `ullAvailPhys`, not the smaller "free" number.

### Unit Conversion
The API returns bytes as `DWORDLONG` (unsigned 64-bit). Convert to gigabytes for display by dividing by `1024 * 1024 * 1024`. Display to 2 decimal places.

### What to Confirm After Building
- Run your profiler and note the total RAM number
- Open Task Manager → Performance → Memory
- The "Total" number in Task Manager should match your profiler's number within ~1% (tiny differences are normal due to hardware-reserved memory)
- The "Available" number in Task Manager should be close to your profiler's number (exact match is impossible because background processes allocate and free memory every second — being within ~200MB is good enough)

---

## Phase C — GPU Profiler (NVML)

### What You Are Measuring
- GPU model name
- Total VRAM
- Free VRAM (right now)
- GPU memory bandwidth (derived, not directly reported)
- GPU temperature (nice-to-have, useful for the thermal throttling warning in §3 item 8)
- CUDA compute capability (useful later for the TTFT formula)

### Setting Up NVML on Windows
NVML ships with the NVIDIA driver. On Windows, the library is `nvml.dll` and it lives in `C:\Windows\System32\`. The header file (`nvml.h`) and the import library (`nvml.lib`) ship with the CUDA Toolkit.

**Where to find them:**
- Header: `<CUDA_TOOLKIT_PATH>\include\nvml.h` (typically `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\include\`)
- Library: `<CUDA_TOOLKIT_PATH>\lib\x64\nvml.lib`

**CMake configuration needed:**
- Add the CUDA include directory to your include paths
- Link against `nvml.lib`
- At runtime, `nvml.dll` is found automatically because it's in System32

### NVML Initialization Sequence
NVML requires an explicit init/shutdown lifecycle:
1. Call `nvmlInit()` at the start of your GPU profiler function
2. Do all your queries
3. Call `nvmlShutdown()` at the end
4. Every NVML function returns an `nvmlReturn_t` — check it. If it's not `NVML_SUCCESS`, print the error string via `nvmlErrorString()` and handle gracefully. Don't crash.

### The Queries You Need

**Device Count:**
- Call `nvmlDeviceGetCount()` to find out how many NVIDIA GPUs are in the system
- For MVP, assume one GPU — but loop over all devices anyway, it costs nothing and makes the code future-proof

**For Each Device:**

| What | NVML Function | Notes |
|---|---|---|
| GPU name | `nvmlDeviceGetName()` | Returns a string like "NVIDIA GeForce RTX 3080" |
| VRAM total + free | `nvmlDeviceGetMemoryInfo()` | Fills an `nvmlMemory_t` struct with `total`, `free`, and `used` fields (in bytes) |
| Memory clock speed | `nvmlDeviceGetMaxClockInfo(device, NVML_CLOCK_MEM, &clock_mhz)` | Returns the max memory clock in MHz — needed to derive bandwidth |
| Temperature | `nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temp_c)` | Returns current GPU temp in Celsius |
| Compute capability | `nvmlDeviceGetCudaComputeCapability(device, &major, &minor)` | Returns e.g. 8 and 6 for sm_86 |

### Deriving Memory Bandwidth (The Tricky Part)
NVML does **not** directly report memory bandwidth in GB/s. You have to calculate it.

**The formula:**
`bandwidth_GB_per_sec = (memory_clock_MHz × 2 × bus_width_bits) / 8 / 1000`

- `memory_clock_MHz`: from `nvmlDeviceGetMaxClockInfo()` — the "× 2" accounts for DDR (double data rate)
- `bus_width_bits`: this is the memory bus width (e.g., 256-bit, 320-bit, 384-bit) — **NVML does not expose this directly**

**How to get bus width — three options, pick one:**

| Option | How | Pros | Cons |
|---|---|---|---|
| A: Hardcoded lookup table | Map GPU model names to known bus widths | Simple, reliable for common GPUs | Breaks for new/unusual GPUs |
| B: CUDA device properties | Initialize a CUDA context, call `cudaGetDeviceProperties()`, read `memoryBusWidth` | Accurate, works for any GPU | Requires linking CUDA runtime, adds a dependency |
| C: Published spec fallback | Use memory clock from NVML + bus width from a public spec table | No extra dependencies | Manual maintenance |

**Recommendation for MVP:** Option A — a small hardcoded table covering the most common consumer GPUs (RTX 3060 through 4090, maybe 10-15 entries). If the GPU isn't in the table, print "bandwidth unknown — using memory clock only" and move on. You can always add Option B in a later step. This keeps Step 1 simple.

**Example table entries:**
| GPU | Bus Width |
|---|---|
| RTX 3060 | 192-bit |
| RTX 3070 | 256-bit |
| RTX 3080 | 320-bit |
| RTX 3090 | 384-bit |
| RTX 4060 | 128-bit |
| RTX 4070 | 192-bit |
| RTX 4080 | 256-bit |
| RTX 4090 | 384-bit |

### What to Confirm After Building
- Run your profiler and note the GPU name and VRAM numbers
- Open a terminal and run `nvidia-smi`
- GPU name should match exactly
- VRAM total should match exactly (to the MB)
- VRAM free should be within ~100-200MB of nvidia-smi (it fluctuates as the display driver allocates and frees memory)
- Temperature should be within a few degrees of what nvidia-smi reports (it changes every second)

---

## Phase D — Disk Benchmark (The Most Complex Part)

### What You Are Measuring
Two separate throughput numbers:
1. **Sequential read speed** — how fast you can read a large file from start to finish. Relevant for initial model loading.
2. **Random read speed** — how fast you can read small blocks from random locations in a large file. Relevant for MoE expert-offload (Phase 2) and mmap-based weight streaming.

### Why This Matters (From the Spec)
From §3 item 2: "Advertised NVMe sequential-read and real random-read-under-load speed can differ 3–5×. MoE expert-offload is random access, not sequential — the exact case where spec-sheet numbers are most wrong."

Your profiler will expose this gap. On a typical consumer NVMe drive, you might see:
- Sequential: 3,000–7,000 MB/s
- Random 4K: 50–200 MB/s

That 20-50× difference is why you cannot trust spec sheets.

### The Target File
Use one of the GGUF files you downloaded in Step 0. It needs to be:
- **Large enough** that you don't read the entire file within the 2-3 second benchmark window (a 2GB+ file is ideal)
- **On the drive you actually want to measure** (your NVMe, not a secondary HDD)
- **In your models folder** (which you should have already excluded from Windows Defender real-time scanning in Step 0)

### Windows API for Unbuffered Reads
To get a true measurement of disk speed (not OS cache speed), you must bypass the Windows file cache.

**The key flag:** `FILE_FLAG_NO_BUFFERING` passed to `CreateFile()`.

**Critical constraints when using `FILE_FLAG_NO_BUFFERING`:**
- Read sizes must be **sector-aligned** (multiples of the disk's sector size, typically 4096 bytes on modern NVMe drives)
- Buffer memory must be **page-aligned** (use `_aligned_malloc()` to allocate the read buffer)
- File offsets for `SetFilePointerEx()` must also be sector-aligned

**If you violate these constraints, `ReadFile()` will fail silently or return an error.** This is the most common bug in Windows disk benchmarks.

### Sequential Read Benchmark

**Procedure:**
1. Open the GGUF file with `CreateFile()` using `GENERIC_READ`, `FILE_SHARE_READ`, and `FILE_FLAG_NO_BUFFERING`
2. Allocate a page-aligned read buffer — use a large block size for sequential reads: **1MB or 4MB** (must be a multiple of 4096)
3. Record the start time using `QueryPerformanceCounter()` (high-resolution timer, much better than `GetTickCount()`)
4. Loop: call `ReadFile()` to fill the buffer, accumulating total bytes read
5. Check elapsed time after each read — stop when you hit the **2-3 second cap**
6. Calculate: `throughput_MB_per_sec = total_bytes_read / elapsed_seconds / (1024 * 1024)`
7. Close the file handle

**Expected result:** For a Gen4 NVMe drive, somewhere in the 2,000–6,000 MB/s range. For Gen3, 1,000–3,000 MB/s. If you see numbers above 10,000 MB/s, your cache bypass isn't working.

### Random Read Benchmark

**Procedure:**
1. Open the same file with the same flags
2. Determine the file size (use `GetFileSizeEx()`)
3. Allocate a page-aligned read buffer — use a small block size for random reads: **4KB** (one sector, the most pessimistic and most relevant case for MoE expert loading)
4. Record start time
5. Loop:
   - Generate a random offset within the file (must be aligned to 4096 bytes)
   - Seek to that offset with `SetFilePointerEx()`
   - Read one 4KB block with `ReadFile()`
   - Increment a counter
6. Stop after the **2-3 second cap**
7. Calculate:
   - `throughput_MB_per_sec = (num_reads × 4096) / elapsed_seconds / (1024 * 1024)`
   - `iops = num_reads / elapsed_seconds` (I/O operations per second)
8. Close the file handle

**Expected result:** For a consumer NVMe, random 4K reads typically yield 50–200 MB/s and 10,000–50,000 IOPS. This will be dramatically lower than sequential — **that is correct and expected**. If random and sequential numbers are similar, your benchmark is measuring cached reads, not disk reads.

### Warm-Up Consideration
The first few reads after opening a file may be slower due to OS-level metadata loading and page table setup. Consider doing a brief "warm-up" read (1-2 seconds, untimed) before starting the actual timed benchmark. This ensures you are measuring steady-state throughput, not cold-start overhead.

### Windows Defender Reminder
If you did not exclude your models folder from Windows Defender real-time scanning in Step 0, **do it now**. Defender intercepts every `ReadFile()` call to scan the data, which will tank your benchmark numbers unpredictably. Add `C:\dev\models\` (or your equivalent) to the Defender exclusion list via Windows Security → Virus & threat protection → Manage settings → Exclusions.

### What to Confirm After Building
- Sequential read number is in the plausible range for your drive (check your drive's spec sheet — your number should be 50-80% of the advertised sequential read, since real-world is always lower than spec)
- Random 4K read number is **dramatically lower** than sequential (at least 10× lower, often 30-50× lower)
- Running the benchmark twice gives similar numbers (within ~10%) — if numbers vary wildly between runs, something is wrong (likely Defender interference or thermal throttling on the drive)

---

## Phase E — Combining the Report

### Output Format
Design a clean, readable console report. Something like:

```
=== LLM Deployment Planner — Hardware Profile ===
Timestamp: 2026-08-15 14:32:07

--- System Memory ---
Total RAM:      32.00 GB
Available RAM:  24.31 GB
Page File:      40.00 GB total, 31.22 GB available

--- GPU ---
Model:          NVIDIA GeForce RTX 3080
VRAM Total:     10.00 GB
VRAM Free:       8.73 GB
Memory Clock:   9501 MHz
Bus Width:      320-bit
Bandwidth:      760.1 GB/s (derived)
Temperature:    42°C
Compute:        sm_86

--- Storage (C:\dev\models\) ---
Sequential Read:  4,217 MB/s  (2.0s benchmark, 1MB blocks)
Random 4K Read:     127 MB/s  (2.0s benchmark, 4KB blocks, 62,012 IOPS)

=================================================
```

### Important Details in the Report
- Include the **benchmark duration and block size** in the storage section — this makes the numbers reproducible and comparable
- Include the **path** being benchmarked — different drives will give different numbers
- Include a **timestamp** — hardware state changes over time, and the calibration log in Step 7 will reference these
- The "derived" note on bandwidth is honest — it tells the user this is calculated, not directly measured

---

## Phase F — Validation Against Ground Truth

This is the most important part of Step 1. Run these checks before declaring the step complete.

### RAM Validation
| Your Profiler Says | Check Against | Acceptable Delta |
|---|---|---|
| Total RAM | Task Manager → Performance → Memory → Total | Within 1% |
| Available RAM | Task Manager → Performance → Memory → Available | Within 500MB (it changes every second) |

### GPU Validation
| Your Profiler Says | Check Against | Acceptable Delta |
|---|---|---|
| GPU Name | `nvidia-smi` output, first line | Exact match |
| VRAM Total | `nvidia-smi` → "Memory-Usage" total | Exact match |
| VRAM Free | `nvidia-smi` → "Memory-Usage" free | Within 200MB |
| Temperature | `nvidia-smi` → temperature field | Within 3°C |

### Disk Validation
| Your Profiler Says | Check Against | Acceptable Delta |
|---|---|---|
| Sequential Read | CrystalDiskMark (if installed) → SEQ1M Q8T1 | Within 20% (different tools measure differently) |
| Random 4K Read | CrystalDiskMark → RND4K Q1T1 | Within 30% (high variance is normal for random reads) |
| Sequential Read | Drive's advertised spec | Your number should be 50-80% of spec |
| Random vs Sequential | Common sense | Random should be at least 10× lower than sequential |

---

## Step 1 — Done Checklist

Before moving to Step 2, confirm every item:

- [ ] CMake project links against `nvml.lib` and builds without errors
- [ ] Binary runs and prints a complete hardware report
- [ ] RAM total matches Task Manager within 1%
- [ ] RAM available is in the right ballpark of Task Manager
- [ ] GPU name matches `nvidia-smi` exactly
- [ ] VRAM total matches `nvidia-smi` exactly
- [ ] VRAM free is within 200MB of `nvidia-smi`
- [ ] Memory bandwidth is calculated and displayed (even if from a lookup table)
- [ ] Sequential disk read is in a plausible range for your drive
- [ ] Random 4K disk read is dramatically lower than sequential
- [ ] Running the benchmark twice gives consistent results (within ~10%)
- [ ] Report includes timestamp, benchmark parameters, and storage path
- [ ] Windows Defender exclusion is active for the models folder

---

## Common Failure Points at Step 1

| Problem | Likely Cause | Fix |
|---|---|---|
| `nvmlInit()` returns `NVML_ERROR_DRIVER_NOT_LOADED` | NVIDIA driver not properly installed or GPU not detected | Reinstall driver, confirm `nvidia-smi` works |
| `nvml.lib` not found during linking | CUDA Toolkit include/lib paths not set in CMake | Add explicit paths to `target_include_directories` and `target_link_directories` |
| `ReadFile()` fails with error 87 (invalid parameter) | Buffer or read size not sector-aligned when using `FILE_FLAG_NO_BUFFERING` | Ensure buffer is allocated with `_aligned_malloc(size, 4096)` and read size is a multiple of 4096 |
| Disk benchmark shows impossibly high numbers (>10 GB/s) | OS cache is serving reads, not the disk | Confirm `FILE_FLAG_NO_BUFFERING` is set; confirm Defender exclusion is active |
| Disk benchmark shows impossibly low numbers (<10 MB/s sequential) | Defender scanning every read, or file is on a network/HDD drive | Add Defender exclusion; confirm file is on local NVMe |
| Random and sequential numbers are nearly identical | Cache is absorbing random reads | Reduce benchmark file to something the OS can't fully cache, or increase file size |
| `GlobalMemoryStatusEx()` returns 0 for all fields | `dwLength` not set before calling | Set `memStatus.dwLength = sizeof(MEMORYSTATUSEX)` before the call |
| GPU temperature reads 0 or unreasonable value | GPU is in a low-power state and sensor isn't active | Run a GPU workload briefly, then re-check |

---

## Time Estimate for Step 1
- Phase A (Understanding + architecture): **30 minutes**
- Phase B (RAM profiler): **1–2 hours** (simplest subsystem)
- Phase C (GPU profiler via NVML): **3–5 hours** (NVML setup + bandwidth derivation takes the most debugging)
- Phase D (Disk benchmark): **4–6 hours** (alignment requirements and cache-bypass debugging are the time sinks)
- Phase E (Report formatting): **1 hour**
- Phase F (Validation): **1–2 hours**

**Total: 1–2 days, as originally estimated. The disk benchmark is where most of the debugging time will go.**