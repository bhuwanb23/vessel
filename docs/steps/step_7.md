# Step 7 — Calibration Log: Full Detailed Plan

---

## Goal of Step 7
Close the feedback loop. After every real execution run, record what the predictor said would happen alongside what actually happened, keyed by the specific hardware and model combination. Over time, aggregate these records to derive per-hardware-class calibration constants that replace the initial guesses from Step 3. The tool gets smarter every time it runs — not because the formulas change, but because the constants inside them converge toward reality.

---

## What You Need Before Starting

### From Steps 1–6 (already done)
- The executor produces an `ExecutionResult` struct with actual measured metrics
- The predictor produces a `Prediction` struct with predicted metrics
- Both structs share the same `HardwareSpec`, `ModelMetadata`, and `StrategyConfig` inputs
- The hardware profiler generates a consistent hardware fingerprint (or you need to build one)

### From the Spec (§7)
- The JSON log format is already locked
- The log is local-only (a plain JSON file on disk)
- Cross-user aggregation is explicitly out of scope for MVP

---

## Phase A — The Hardware Fingerprint

### Why It Matters
The calibration log is only useful if you can match future runs to past runs on the **same hardware**. A prediction that was wrong on an RTX 3060 tells you nothing about an RTX 4090. The fingerprint is the key that groups calibration records by hardware class.

### The Fingerprint Format (From §7)
```
"<cpu_model>|<gpu_model>|<ram_total>|<nvme_model>"
```

### How to Generate Each Component

| Component | Source | How to Extract | Example |
|---|---|---|---|
| CPU model | `hwloc` or Windows Registry | Read `HKEY_LOCAL_MACHINE\HARDWARE\DESCRIPTION\System\CentralProcessor\0\ProcessorNameString` | `Intel Core i7-12700K` |
| GPU model | NVML `nvmlDeviceGetName()` | You already have this from Step 1 | `NVIDIA GeForce RTX 3080` |
| RAM total | `GlobalMemoryStatusEx()` | Round to nearest GB to avoid false mismatches from reserved memory | `32GB` |
| NVMe model | Windows Device Manager API or WMI | Query `Win32_DiskDrive` for the drive containing the models folder | `Samsung 980 PRO 1TB` |

### Normalization Rules
The fingerprint must be **stable across reboots and driver updates**. Apply these rules:

- **CPU:** Strip the clock speed and "CPU" suffix. `"Intel(R) Core(TM) i7-12700K @ 3.60GHz"` becomes `"i7-12700K"`. The parenthetical trademark symbols change between Windows versions — strip them.
- **GPU:** Use the exact NVML string. It's already stable. `"NVIDIA GeForce RTX 3080"` stays as-is.
- **RAM:** Round to the nearest power-of-two GB. `31.8 GB` becomes `32GB`. `15.7 GB` becomes `16GB`. This prevents false mismatches when the OS reserves slightly different amounts of memory on different boots.
- **NVMe:** Use the model string without the serial number or firmware version. `"Samsung SSD 980 PRO 1TB"` becomes `"Samsung 980 PRO"`. The serial number changes per drive, but the model is what determines performance characteristics.

### Implementation
A single function `std::string generateHardwareFingerprint(const HardwareSpec& hw)` that concatenates the normalized components with `|` separators. Call it once during the hardware profile step and store the result in `HardwareSpec`.

### Edge Case: Hardware Changes
If the user upgrades their GPU or adds RAM, the fingerprint changes, and old calibration records won't match. **This is correct behavior.** The old records were for different hardware. The tool will start fresh with default constants and re-calibrate for the new configuration.

---

## Phase B — The Log Entry Format

### The JSON Schema (From §7, Locked)

```json
{
  "hardware_fingerprint": "i7-12700K|NVIDIA GeForce RTX 3080|32GB|Samsung 980 PRO",
  "model_id": "bartowski/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
  "strategy": {
    "backend": "llama.cpp",
    "quant": "Q4_K_M",
    "placement": "FULL_GPU",
    "gpu_layers": 28,
    "context": 4096,
    "kv_quant_bits": 16
  },
  "predicted": {
    "tokens_per_sec": 385.0,
    "ttft_ms": 45.0,
    "vram_bytes": 2576980378,
    "ram_bytes": 536870912,
    "confidence": "HIGH"
  },
  "actual": {
    "tokens_per_sec": 371.2,
    "ttft_ms": 52.3,
    "peak_vram_bytes": 2791728742,
    "peak_ram_bytes": 751619276,
    "throttled": false,
    "tokens_generated": 100,
    "duration_sec": 0.27
  },
  "timestamp": "2026-08-15T14:32:07Z",
  "tool_version": "0.1.0"
}
```

### Field-by-Field Notes

| Field | Source | Notes |
|---|---|---|
| `hardware_fingerprint` | Step 1 + Phase A | Stable across reboots |
| `model_id` | Step 2 | Use the Hugging Face repo path + filename, not a local file path (local paths vary between machines) |
| `strategy.backend` | Hardcoded | `"llama.cpp"` for MVP. Future: `"ik_llama.cpp"`, `"ktransformers"` |
| `strategy.quant` | ModelMetadata | The quantization label |
| `strategy.placement` | StrategyConfig | `"FULL_GPU"`, `"GPU_CPU_SPLIT"`, or `"CPU_ONLY"` |
| `strategy.gpu_layers` | StrategyConfig | Exact number, not a fraction |
| `strategy.context` | StrategyConfig | The context length used for this run |
| `strategy.kv_quant_bits` | StrategyConfig | 16, 8, or 4 |
| `predicted.*` | Prediction struct | The numbers your formulas produced before the run |
| `actual.*` | ExecutionResult struct | The numbers measured during the run |
| `actual.throttled` | Live sampler | From the thermal throttle detection in Step 6 |
| `actual.tokens_generated` | Executor | How many tokens were actually generated (may be less than requested if stop token hit) |
| `actual.duration_sec` | Executor | Total decode time in seconds |
| `timestamp` | System clock | ISO 8601 format, UTC |
| `tool_version` | Hardcoded | Increment this when you change formulas or constants, so you can filter out old records |

### What NOT to Include
- **Generated text:** It's large, private, and irrelevant to calibration
- **Full hardware specs:** The fingerprint is sufficient. Storing the entire `HardwareSpec` would bloat the log and create privacy concerns
- **Prompt text:** Same privacy concern
- **Error details:** If the run failed, don't log it as a calibration entry. Failed runs don't produce valid actual metrics

---

## Phase C — Writing the Log

### When to Write
Write one entry **immediately after** a successful execution run completes. "Successful" means:
- The model loaded without errors
- At least 10 tokens were generated (fewer than 10 gives unreliable tok/s measurements)
- No CUDA errors or OOM events occurred during generation

**Do NOT write entries for:**
- Failed model loads (no actual metrics to record)
- Runs that generated fewer than 10 tokens (statistically meaningless)
- Runs where the user pressed Ctrl+C before 10 tokens (incomplete data)
- Advisor-only runs (no execution happened, no actual metrics)

### Where to Write
Store the log file in a predictable, user-accessible location:

**Windows path:** `%APPDATA%\llm-planner\calibration.jsonl`

This resolves to something like `C:\Users\YourName\AppData\Roaming\llm-planner\calibration.jsonl`.

**Why `%APPDATA%`:**
- It's the standard Windows location for application data
- It persists across updates
- It doesn't require admin permissions
- It's hidden from casual browsing but accessible if the user wants to inspect it

**Why `.jsonl` (JSON Lines) instead of `.json`:**
- Each entry is one line of JSON, terminated by a newline
- You can **append** to the file without reading and rewriting the entire thing
- The file can grow to thousands of entries without performance issues
- It's trivially parseable: read line by line, parse each line as JSON
- A corrupt line doesn't invalidate the entire file (just skip that line)

### The Write Sequence

```
1. Construct the log entry as a JSON object (using nlohmann::json)
2. Serialize to a single line (no pretty-printing — compact JSON)
3. Open the log file in append mode (std::ofstream with std::ios::app)
4. If the file doesn't exist, it will be created automatically
5. Write the line + newline
6. Flush and close
```

**Thread safety:** The write happens on the main thread after the executor and sampler have both finished. No concurrent writes are possible in MVP (single-threaded execution). Don't add mutex complexity for a case that doesn't exist yet.

### Error Handling
If the log file can't be written (disk full, permissions issue, path doesn't exist):
- Print a warning: `"⚠️ Could not write calibration log: <error>"`
- **Do not fail the run.** The execution results are still valid and displayed to the user. The log is a background feature, not a critical path.
- Try to create the directory if it doesn't exist (`std::filesystem::create_directories()`)

---

## Phase D — Reading and Aggregating the Log

### When to Read
Read the log at the **start of each prediction run** (Step 3's predictor function). The aggregated calibration data adjusts the predictor's constants before generating predictions.

### The Aggregation Logic

**Step D1: Load all entries**
Read the `.jsonl` file line by line. Parse each line as JSON. Skip lines that fail to parse (corrupt entries). Skip entries with a different `tool_version` than the current version (stale data from before a formula change).

**Step D2: Filter by hardware fingerprint**
Keep only entries where `hardware_fingerprint` matches the current machine's fingerprint. This gives you the calibration history for this specific hardware.

**Step D3: Filter by relevance**
For each predictor constant you want to calibrate, filter entries to those that are most relevant:

| Constant to Calibrate | Filter Criteria | Why |
|---|---|---|
| `runtime_overhead_bytes` (CUDA) | `placement == "FULL_GPU"`, not throttled | Overhead is placement-specific and distorted by throttling |
| `runtime_overhead_bytes` (CPU) | `placement == "CPU_ONLY"`, not throttled | Different overhead for CPU backend |
| `efficiency_factor` (GPU) | `placement == "FULL_GPU"`, `tokens_generated >= 50`, not throttled | Efficiency is most stable with full GPU offload and long runs |
| `effective_gpu_bandwidth` | `placement == "FULL_GPU"`, not throttled | Derive from actual tok/s and model size |
| `effective_ram_bandwidth` | `placement == "CPU_ONLY"`, not throttled | Derive from actual tok/s and model size |

**Step D4: Calculate adjusted constants**

For each constant, compute the adjustment factor from the filtered entries:

**Runtime Overhead Adjustment:**
```
predicted_memory = weight_bytes + kv_cache_bytes + current_overhead
actual_memory = entry.actual.peak_vram (or peak_ram for CPU)
overhead_error = actual_memory - (weight_bytes + kv_cache_bytes)

// Average the error across all matching entries
adjusted_overhead = average(overhead_error for all entries)
```

**Bandwidth Adjustment:**
```
// From the decode speed formula: tok/s = bandwidth / bytes_per_token
// Rearranged: bandwidth = tok/s × bytes_per_token
implied_bandwidth = entry.actual.tokens_per_sec × bytes_per_token

// Average across all matching entries
adjusted_bandwidth = average(implied_bandwidth for all entries)
```

**Efficiency Factor Adjustment:**
```
// From the TTFT formula: ttft = (2 × params × prompt_tokens) / (tflops × efficiency × 1e12)
// Rearranged: efficiency = (2 × params × prompt_tokens) / (ttft × tflops × 1e12)
implied_efficiency = (2 × params × prompt_tokens) / (entry.actual.ttft_ms × 1e-3 × tflops × 1e12)

// Average across all matching entries
adjusted_efficiency = average(implied_efficiency for all entries)
```

**Step D5: Apply confidence weighting**
Don't blindly trust the aggregated data. Weight it against the initial defaults based on sample size:

```
if (num_matching_entries >= 10):
    final_constant = adjusted_constant  // trust the data
elif (num_matching_entries >= 5):
    final_constant = 0.7 × adjusted + 0.3 × default  // blend
elif (num_matching_entries >= 2):
    final_constant = 0.4 × adjusted + 0.6 × default  // lean toward default
else:
    final_constant = default  // not enough data, use initial guess
```

This prevents a single outlier run from dramatically shifting the constants. As more data accumulates, the calibrated values gradually take over.

### Where to Store Calibrated Constants
For MVP, compute the adjusted constants in memory at the start of each run. Don't write them back to a config file — that adds complexity and creates stale-state bugs. The log file is the single source of truth; the constants are derived from it on demand.

**Performance note:** Reading and aggregating a few hundred JSON lines takes <10ms. This is negligible compared to the 2-3 second disk benchmark in Step 1. Don't optimize this until it becomes a problem.

---

## Phase E — Integration with the Predictor

### The Modified Predictor Signature
In Step 3, the predictor was a pure function with no I/O. Now it accepts an optional calibration data parameter:

**Before (Step 3):**
```
Prediction predict(HardwareSpec hw, ModelMetadata model, StrategyConfig strategy)
```

**After (Step 7):**
```
Prediction predict(HardwareSpec hw, ModelMetadata model, StrategyConfig strategy, 
                   const CalibrationData& cal = CalibrationData::defaults())
```

The `CalibrationData` struct holds the adjusted constants:

```
struct CalibrationData {
    double runtime_overhead_cuda_bytes = 512 * 1024 * 1024  // 512 MB default
    double runtime_overhead_cpu_bytes = 128 * 1024 * 1024   // 128 MB default
    double gpu_bandwidth_gbs = 0.0    // 0 = use hardware profiler's number
    double ram_bandwidth_gbs = 0.0    // 0 = use hardware profiler's number
    double efficiency_factor = 0.3     // TTFT efficiency default
    int sample_count = 0               // how many records informed these values
    
    static CalibrationData defaults() { return CalibrationData(); }
}
```

### How the Predictor Uses Calibration Data
Inside the predictor functions, replace the hardcoded constants with the calibration values:

**Memory formula:**
```
// Before:
total_memory = weight_bytes + kv_cache_bytes + 512_MB

// After:
total_memory = weight_bytes + kv_cache_bytes + cal.runtime_overhead_cuda_bytes
```

**Speed formula:**
```
// Before:
effective_bandwidth = hw.gpu_bandwidth_gbs

// After:
effective_bandwidth = (cal.gpu_bandwidth_gbs > 0) ? cal.gpu_bandwidth_gbs : hw.gpu_bandwidth_gbs
```

**TTFT formula:**
```
// Before:
device_throughput = hw.gpu_tflops × 0.3

// After:
device_throughput = hw.gpu_tflops × cal.efficiency_factor
```

### The Confidence Band Update
Now that you have real calibration data, the confidence band logic from Step 3 Phase F becomes meaningful:

**High Confidence:** GGUF header available + `cal.sample_count >= 5` for this hardware fingerprint
**Medium Confidence:** GGUF header available + `cal.sample_count < 5`
**Low Confidence:** config.json fallback, or MoE, or unknown quant

The `sample_count` field in `CalibrationData` is what drives this. Before Step 7, it was always 0, so everything was "medium" at best. Now it reflects real usage history.

---

## Phase F — Log File Management

### Size Limits
The log file will grow over time. A single entry is roughly 500 bytes of JSON. After 1,000 runs, the file is ~500KB. After 10,000 runs, ~5MB. This is negligible for modern storage.

**For MVP:** No size limit. Let it grow. If it becomes a problem in the future, add log rotation.

### Future Log Rotation (Not MVP)
When you eventually add rotation:
- Keep the most recent 1,000 entries
- Archive older entries to `calibration.archive.jsonl`
- Or: keep entries per hardware fingerprint and prune fingerprints that haven't been seen in 90 days

### User Access
The user should be able to inspect and delete the log file. Add a CLI command:

```
llm-planner --calibration-info
```

This prints:
```
=== Calibration Log ===
Location: C:\Users\You\AppData\Roaming\llm-planner\calibration.jsonl
Total entries: 47
Entries for this hardware: 31
Hardware fingerprint: i7-12700K|NVIDIA GeForce RTX 3080|32GB|Samsung 980 PRO
Calibrated constants:
  CUDA overhead:  587 MB (default: 512 MB, from 12 samples)
  GPU bandwidth:  724 GB/s (profiled: 760 GB/s, from 8 samples)
  Efficiency:     0.28 (default: 0.30, from 6 samples)
```

And a reset command:
```
llm-planner --calibration-reset
```

This deletes the log file (with confirmation). Useful if the user changes hardware or suspects the calibration has drifted.

---

## Phase G — Manual Calibration Workflow (The First Few Runs)

### Why Manual First
From the original plan: "Compare them by hand at first — you're looking for how far off your formulas are, so you know what to recalibrate."

The automatic aggregation in Phase D will eventually handle this, but for the first few runs, you should manually inspect the log to understand the error patterns. This builds intuition about which constants need the most adjustment.

### The Manual Workflow

**Run 1: Baseline**
- Run a model with full GPU offload at 4K context
- Check the log entry
- Compare predicted vs actual memory, tok/s, TTFT
- Note which predictions were off and by how much

**Run 2: Different Context**
- Run the same model at max-safe context
- Check if the KV cache prediction was accurate
- If memory is off proportionally to context, the KV cache formula is correct but the overhead constant is wrong
- If memory is off disproportionately, the KV cache formula itself has a bug

**Run 3: Different Placement**
- Run with a GPU+CPU split
- Check if the speed prediction matches (this tests the sequential time composition formula)
- If split speed is much lower than predicted, the RAM bandwidth measurement from Step 1 might be too optimistic

**Run 4: Different Model Size**
- Run a larger or smaller model
- Check if the errors scale with model size (indicating a bpw or param count issue) or stay constant (indicating an overhead issue)

### What to Look For

| Pattern | Likely Cause | What to Adjust |
|---|---|---|
| Memory consistently 200-400 MB too low | CUDA overhead underestimated | Increase `runtime_overhead_cuda_bytes` |
| Memory error scales with context | KV cache formula bug | Check `kv_heads` vs `attention_heads` |
| Memory error scales with model size | Bits-per-weight wrong | Re-derive from file size |
| tok/s consistently 10-20% too high | Bandwidth overestimated | Reduce `effective_gpu_bandwidth` or check for thermal throttling |
| tok/s way off for split only | RAM bandwidth wrong | Re-run the Step 1 memcpy benchmark |
| TTFT consistently 2× too low | Efficiency factor too high | Reduce `efficiency_factor` |
| TTFT wildly inconsistent | Prompt length varies | Ensure you're using the same benchmark prompt |
| Throttled = true on most runs | Cooling issue | Not a software problem — the user needs better cooling |

---

## Phase H — Testing the Calibration Log

### Test Scenarios

**Test 1: First Run (Empty Log)**
- Delete the log file (or run on a fresh install)
- Execute a model
- Verify: Log file is created, contains exactly one entry
- Verify: The entry's JSON is valid (parse it with a JSON validator)
- Verify: All fields are populated with correct values
- Verify: Predictions used default constants (no calibration data available)

**Test 2: Accumulation**
- Run 5 different models/strategies
- Verify: Log file contains 5 entries, one per line
- Verify: Each entry has a unique timestamp
- Verify: The file is valid JSONL (each line parses independently)

**Test 3: Calibration Takes Effect**
- Run the same model+strategy 10 times
- Manually inspect the log — the actual values should be consistent across runs (within ~5%)
- Run the advisor mode (no execution) for the same model
- Verify: The predicted values have shifted slightly from the defaults toward the actual values
- Verify: The confidence band shows "HIGH" (because sample_count >= 5)

**Test 4: Hardware Fingerprint Isolation**
- Manually edit the log file to change the hardware fingerprint on some entries
- Run the advisor
- Verify: Only entries with the matching fingerprint are used for calibration
- Verify: Entries with different fingerprints are ignored

**Test 5: Corrupt Entry Handling**
- Manually add a line of garbage text to the middle of the log file
- Run the advisor
- Verify: The tool doesn't crash
- Verify: The corrupt line is skipped
- Verify: Valid entries before and after the corrupt line are still used

**Test 6: Version Filtering**
- Manually edit some entries to have `tool_version: "0.0.1"` (old version)
- Run the advisor with current version `"0.1.0"`
- Verify: Old-version entries are excluded from calibration

**Test 7: Log Write Failure**
- Make the log directory read-only (or set the file to read-only)
- Execute a model
- Verify: A warning is printed about the log write failure
- Verify: The execution results are still displayed correctly
- Verify: The tool doesn't crash

**Test 8: Calibration Reset**
- Run `--calibration-reset`
- Verify: Log file is deleted (or emptied)
- Verify: Next run uses default constants
- Verify: Confidence bands revert to "MEDIUM"

---

## Step 7 — Done Checklist

Before declaring the MVP complete, confirm every item:

- [ ] Hardware fingerprint is generated correctly and is stable across reboots
- [ ] Log entry JSON matches the §7 schema exactly
- [ ] Log entries are written in JSONL format (one JSON object per line)
- [ ] Log file is stored in `%APPDATA%\llm-planner\calibration.jsonl`
- [ ] Log directory is created automatically if it doesn't exist
- [ ] Entries are only written after successful runs (≥10 tokens, no errors)
- [ ] Failed runs do not produce log entries
- [ ] Log write failure prints a warning but doesn't crash the tool
- [ ] Log is read and parsed at the start of each prediction run
- [ ] Corrupt log lines are skipped without crashing
- [ ] Entries are filtered by hardware fingerprint
- [ ] Entries are filtered by tool version
- [ ] Calibration constants are derived from aggregated entries
- [ ] Confidence weighting blends calibrated values with defaults based on sample count
- [ ] Predictor uses calibrated constants when available
- [ ] Confidence bands reflect sample count (HIGH when ≥5 matching entries)
- [ ] `--calibration-info` prints log statistics and current calibrated constants
- [ ] `--calibration-reset` deletes the log file with confirmation
- [ ] Manual inspection of first 3-5 log entries confirms predicted-vs-actual deltas are reasonable
- [ ] After 10+ runs, predictions are measurably closer to actuals than the initial defaults

---

## Common Failure Points at Step 7

| Problem | Likely Cause | Fix |
|---|---|---|
| Log file grows but predictions don't improve | Aggregation logic not wired into the predictor | Confirm the `CalibrationData` struct is actually passed to the `predict()` function, not just constructed and discarded |
| Calibrated constants are wildly wrong | A single outlier run (e.g., thermal throttle) skewing the average | Filter out throttled runs from calibration. Consider using median instead of mean for small sample sizes. |
| Hardware fingerprint changes between reboots | Normalization rules not stripping variable components (clock speed, serial number) | Re-check the normalization in Phase A. The fingerprint should be identical across reboots. |
| JSONL file becomes unreadable | A log entry contains a newline character inside a string field | Ensure all string fields are properly escaped by nlohmann::json (it should handle this automatically, but verify) |
| Calibration makes predictions worse | Not enough data — 2-3 entries can shift constants in the wrong direction | The confidence weighting in Phase D Step 5 should prevent this. If it doesn't, increase the threshold (require ≥5 entries before using calibrated values). |
| Log write is slow | Flushing to disk after every entry on a slow drive | For MVP, this is fine (one write per run). If it becomes an issue, buffer writes and flush periodically. |
| Old entries from before a formula change corrupt calibration | `tool_version` not incremented when formulas change | Every time you modify the predictor formulas or default constants, bump the version string. This invalidates old entries automatically. |
| Multiple users on the same machine share a log | `%APPDATA%` is per-user on Windows | This is actually correct behavior — each user has their own `%APPDATA%` directory. No issue. |
| Disk full prevents log write | Unlikely but possible | The error handling in Phase C should catch this. The tool continues without logging. |

---

## Time Estimate for Step 7
- Phase A (Hardware fingerprint): **1–2 hours** (Windows Registry + WMI queries for CPU and NVMe model names)
- Phase B (Log entry format): **1 hour** (constructing the JSON with nlohmann::json)
- Phase C (Writing the log): **1–2 hours** (file I/O, directory creation, error handling)
- Phase D (Reading and aggregation): **3–4 hours** (the filtering and constant derivation logic is the most math-heavy part)
- Phase E (Predictor integration): **1–2 hours** (modifying the predictor signature and replacing hardcoded constants)
- Phase F (Log management + CLI commands): **1–2 hours** (`--calibration-info` and `--calibration-reset`)
- Phase G (Manual calibration workflow): **1–2 hours** (running models, inspecting log entries, adjusting defaults)
- Phase H (Testing): **2–3 hours** (all test scenarios)

**Total: Half a day, as originally estimated. The aggregation logic in Phase D is the most complex part, but it's straightforward math once you understand which entries to filter and which constants to derive. The rest is file I/O and JSON serialization.**

---

## MVP Complete — What You Have Now

After finishing all 7 steps, you have:

1. **A hardware profiler** that measures real RAM, GPU, and disk performance
2. **A metadata fetcher** that reads model specs without downloading weights
3. **A predictor** that calculates memory, speed, and latency for any deployment strategy
4. **A method matrix generator** that enumerates all viable ways to run a model
5. **A ranker** that sorts strategies by the user's priority (speed/quality/safety)
6. **An executor** that runs the chosen strategy and measures actual performance
7. **A calibration log** that makes predictions more accurate over time

**This is a deployment planner + execution engine, not a calculator.** It answers "how will this model perform?" not just "will it fit?" It is the tool described in the spec doc, built incrementally, validated at every step, and ready for real users.