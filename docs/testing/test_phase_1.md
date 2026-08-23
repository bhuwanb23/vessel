# End-to-End Test Plan + Project Status Report

---

## PART 1: PROJECT STATUS REPORT

---

### Current Stage: MVP Design-Locked, Implementation Plan Complete

The project has completed the **full design and implementation planning phase** across all 7 build steps. Every component has been specified down to the function signature, formula, data structure, and validation criteria. No code has been written yet — what exists is a comprehensive engineering blueprint.

### What Has Been Completed (Planning Phase)

| Deliverable | Status | Detail |
|---|---|---|
| Problem Statement | ✅ Locked | Gap analysis against 6+ existing tools confirmed |
| Product Vision | ✅ Locked | Deployment planner + execution engine, not a calculator |
| Design Stress-Tests | ✅ Locked | 11 flaws identified and resolved in architecture |
| MVP Scope | ✅ Locked | Linux→Windows pivot, NVIDIA-only, dense models |
| System Architecture | ✅ Locked | 7 components, interface-driven, extensible |
| Predictor Formulas | ✅ Locked | Memory, decode speed, TTFT, confidence bands |
| Tech Stack | ✅ Locked | C++, llama.cpp, NVML, libcurl, nlohmann::json |
| Calibration Log Schema | ✅ Locked | JSONL format, hardware-fingerprinted |
| Platform Decision | ✅ Locked | Native Windows (MSVC), not WSL |
| Step 0 Plan | ✅ Detailed | Environment setup, toolchain, first model run |
| Step 1 Plan | ✅ Detailed | Hardware profiler (RAM, GPU, NVMe) |
| Step 2 Plan | ✅ Detailed | Metadata fetcher (GGUF header, config.json fallback) |
| Step 3 Plan | ✅ Detailed | Predictor math (memory, tok/s, TTFT) |
| Step 4 Plan | ✅ Detailed | Pipeline integration + method matrix |
| Step 5 Plan | ✅ Detailed | Ranker (speed/quality/safety priorities) |
| Step 6 Plan | ✅ Detailed | Executor (llama.cpp library linking, live sampler) |
| Step 7 Plan | ✅ Detailed | Calibration log (write, read, aggregate, recalibrate) |
| E2E Test Plan | ✅ This document | Comprehensive validation across all scenarios |

### What Has NOT Been Done Yet

| Item | Status | Notes |
|---|---|---|
| Actual C++ code | ❌ Not started | All 7 steps are planned but unimplemented |
| CMake project skeleton | ❌ Not started | Step 0 Phase E |
| Any compiled binary | ❌ Not started | |
| Real validation data | ❌ Not started | Requires running actual models |

### What Is Explicitly Deferred (Phase 2+)

| Feature | Phase | Reason for Deferral |
|---|---|---|
| AMD/ROCm support | Phase 2 | Separate code path behind `IHardwareProfiler` |
| Apple Metal support | Phase 2-3 | Separate code path |
| MoE expert-offload prediction | Phase 2 | Requires `ktransformers`/`ik_llama.cpp` integration |
| Multi-node distributed inference | Phase 3+ | Entirely different architecture |
| Fine-tuning/training workloads | Out of scope | Inference-planning only |
| Cross-user calibration aggregation | Phase 2+ | Requires backend server |
| Model download manager | Phase 2 | MVP assumes pre-downloaded models |
| GUI/Web interface | Phase 2+ | CLI-first for MVP |
| Quality-delta scoring with perplexity | Phase 2 | Requires benchmark database |

### Project Maturity Assessment

| Dimension | Rating | Notes |
|---|---|---|
| **Design completeness** | 🟢 High | Every component specified to implementation level |
| **Risk identification** | 🟢 High | 11 stress-test flaws resolved, common failure points documented per step |
| **Formula validation plan** | 🟢 High | Clear methodology for comparing predictions vs llama.cpp actuals |
| **Scope discipline** | 🟢 High | Strict MVP boundaries, explicit deferrals |
| **Implementation progress** | 🔴 Zero | No code written yet |
| **Real-world validation** | 🔴 Zero | No models run through the tool yet |
| **Estimated implementation time** | 🟡 ~2-4 weeks | Based on per-step estimates (Step 0: 0.5d, Step 1: 1-2d, Step 2: 1-2d, Step 3: 2-3d, Step 4: 1d, Step 5: 0.5d, Step 6: 3-5d, Step 7: 0.5d) |

---

## PART 2: END-TO-END TEST PLAN

---

### Test Plan Overview

This test plan validates the complete LLM Deployment Planner as an integrated system. It is organized into five tiers, from basic smoke tests to real-world scenario validation. Every test assumes all 7 steps are implemented and the tool is a single compiled binary.

### Test Environment Requirements

| Requirement | Specification |
|---|---|
| OS | Windows 10/11 |
| GPU | NVIDIA GPU with ≥8GB VRAM |
| RAM | ≥16GB |
| Storage | NVMe SSD with ≥20GB free |
| Models | At least 3 GGUF files of different sizes (3B, 7B, 13B) |
| Network | Internet access (for metadata fetch) |
| Tool | `llm-planner.exe` built in Release mode with CUDA |

### Test Models (Acquire Before Testing)

| # | Model | Arch | Params | Quant | File Size | Purpose |
|---|---|---|---|---|---|---|
| M1 | Llama-3.2-3B-Instruct-Q4_K_M | llama | ~3B | Q4_K_M | ~2GB | Small model, full GPU fit |
| M2 | Qwen2.5-7B-Instruct-Q4_K_M | qwen2 | ~7B | Q4_K_M | ~4.5GB | Medium model, tight fit on 8GB |
| M3 | Mistral-7B-Instruct-v0.3-Q5_K_M | llama/mistral | ~7B | Q5_K_M | ~5.5GB | Different quant, same size |
| M4 | Llama-3.1-8B-Instruct-Q2_K | llama | ~8B | Q2_K | ~3.5GB | Low quant, tests quality ranking |
| M5 | A 13B+ model in Q4_K_M | varies | ~13B | Q4_K_M | ~8GB | Too large for 8GB GPU, tests NO FIT |

---

### TIER 1: Smoke Tests (Run First, Every Time)

These tests verify the tool launches and produces output without crashing. Run these after every build.

| ID | Test | Command | Expected Result | Pass/Fail |
|---|---|---|---|---|
| S1 | Help flag | `llm-planner --help` | Prints usage with all flags listed | |
| S2 | No arguments | `llm-planner` | Prints error + usage, exits cleanly | |
| S3 | Hardware profile only | `llm-planner --profile` | Prints hardware report (RAM, GPU, NVMe), exits | |
| S4 | Advisor mode, small model | `llm-planner --model <M1_URL>` | Prints model metadata + prediction table, no crash | |
| S5 | Advisor mode, medium model | `llm-planner --model <M2_URL>` | Same as S4, different numbers | |
| S6 | Invalid URL | `llm-planner --model https://example.com/fake.gguf` | Prints clear error, exits cleanly, no crash | |
| S7 | Non-GGUF URL | `llm-planner --model https://huggingface.co/some-repo/resolve/main/config.json` | Prints "not a GGUF file" error | |

---

### TIER 2: Component Validation Tests

These tests verify each subsystem independently within the integrated tool.

#### 2A: Hardware Profiler Validation

| ID | Test | Method | Expected Result | Pass/Fail |
|---|---|---|---|---|
| H1 | RAM total accuracy | Compare tool output vs Task Manager | Within 1% | |
| H2 | RAM available accuracy | Compare tool output vs Task Manager | Within 500MB | |
| H3 | GPU name accuracy | Compare tool output vs `nvidia-smi` | Exact match | |
| H4 | VRAM total accuracy | Compare tool output vs `nvidia-smi` | Exact match | |
| H5 | VRAM free accuracy | Compare tool output vs `nvidia-smi` | Within 200MB | |
| H6 | GPU bandwidth plausibility | Check derived bandwidth vs spec sheet | 50-100% of spec | |
| H7 | Sequential read plausibility | Check vs drive spec | 50-80% of spec | |
| H8 | Random read plausibility | Check vs sequential | At least 10× lower than sequential | |
| H9 | Benchmark consistency | Run profiler twice | Numbers within 10% of each other | |
| H10 | Hardware fingerprint stability | Run profiler, reboot, run again | Identical fingerprint string | |

#### 2B: Metadata Fetcher Validation

| ID | Test | Method | Expected Result | Pass/Fail |
|---|---|---|---|---|
| F1 | GGUF header fetch (M1) | Run with M1 URL | HTTP 206, metadata printed | |
| F2 | GGUF header fetch (M2) | Run with M2 URL | HTTP 206, different arch (qwen2) | |
| F3 | Architecture detection | Check M1 vs M2 output | M1 = "llama", M2 = "qwen2" | |
| F4 | Parameter count accuracy | Compare vs model card | Within 1% | |
| F5 | Layer count accuracy | Compare vs llama.cpp load output | Exact match | |
| F6 | Context length accuracy | Compare vs model card | Exact match | |
| F7 | Quant type detection | Check M1 (Q4_K_M) vs M3 (Q5_K_M) | Correct labels and file_type integers | |
| F8 | KV heads vs attention heads | Check GQA models | kv_heads < attention_heads for GQA models | |
| F9 | Fetch speed | Time the metadata fetch | Under 5 seconds | |
| F10 | config.json fallback | Use a repo URL without .gguf extension | Falls back to config.json, flags low confidence | |

#### 2C: Predictor Validation

| ID | Test | Method | Expected Result | Pass/Fail |
|---|---|---|---|---|
| P1 | Memory prediction (M1, full GPU, 4K) | Compare predicted vs llama.cpp actual | Within 10% | |
| P2 | Memory prediction (M2, full GPU, 4K) | Same | Within 10% | |
| P3 | Memory prediction (M1, CPU only, 4K) | Same | Within 15% | |
| P4 | Memory scales with context | Compare 4K vs 32K predictions | ~8× increase in KV cache component | |
| P5 | Memory scales with quant | Compare Q4_K_M vs Q5_K_M predictions | Q5 should be ~15% larger | |
| P6 | tok/s prediction (M1, full GPU) | Compare predicted vs llama.cpp actual | Within 20% | |
| P7 | tok/s prediction (M1, CPU only) | Same | Within 25% | |
| P8 | tok/s prediction (M2, split) | Same | Within 25% | |
| P9 | Split speed < full GPU speed | Compare split vs full GPU for same model | Split must be slower | |
| P10 | CPU speed < GPU speed | Compare CPU-only vs full GPU for same model | CPU must be significantly slower | |
| P11 | TTFT prediction (M1, full GPU) | Compare predicted vs llama.cpp actual | Within 40% | |
| P12 | TTFT scales with prompt length | Compare 20-token vs 200-token prompt | ~10× increase | |
| P13 | Non-viable detection (M5, full GPU) | Check if 13B model on 8GB GPU flagged | `viable = false` | |
| P14 | Max-safe context calculation | Check computed max context | Plausible given VRAM/RAM | |
| P15 | Confidence bands | Check M1 (GGUF) vs config.json fallback | GGUF = medium/high, config = low | |

---

### TIER 3: Integration Tests (Full Pipeline)

These tests exercise the complete pipeline from input to output.

| ID | Test | Command | Expected Result | Pass/Fail |
|---|---|---|---|---|
| I1 | Full pipeline, small model | `llm-planner --model <M1_URL>` | Hardware → Metadata → Matrix → Predictions → Ranked Table, all in <10 sec | |
| I2 | Full pipeline, medium model | `llm-planner --model <M2_URL>` | Same, with some strategies showing NO FIT | |
| I3 | Full pipeline, too-large model | `llm-planner --model <M5_URL>` | All GPU strategies NO FIT, CPU strategies viable | |
| I4 | Speed priority | `llm-planner --model <M1_URL> --priority speed` | Full GPU ranked #1 | |
| I5 | Safety priority | `llm-planner --model <M2_URL> --priority safety` | Strategy with most headroom ranked #1 | |
| I6 | Quality priority | `llm-planner --model <M1_URL> --priority quality` | Highest quant/KV precision ranked #1 | |
| I7 | Priority changes order | Compare I4 vs I5 output | Table order visibly different | |
| I8 | Verbose mode | `llm-planner --model <M1_URL> --verbose` | Full hardware + metadata reports printed before table | |
| I9 | Context filter | `llm-planner --model <M1_URL> --context 4k` | Only 4K strategies shown, no max-safe | |
| I10 | Invalid priority | `llm-planner --model <M1_URL> --priority fast` | Error message listing valid options | |
| I11 | Matrix size | Count strategies in output for M1 | Between 8 and 24 entries | |
| I12 | Deduplication | Check for duplicate rows | No identical strategies in the table | |
| I13 | Recommendation line | Check bottom of table | Priority-specific recommendation present | |
| I14 | Tight warning | Run M2 on 8GB GPU | Strategies using >90% VRAM show ⚠️ TIGHT | |
| I15 | Pipeline timing | Time the full pipeline | Under 10 seconds total | |

---

### TIER 4: Execution Tests (Step 6 Validation)

These tests require the `--execute` flag and actual model execution.

| ID | Test | Command | Expected Result | Pass/Fail |
|---|---|---|---|---|
| E1 | Execute full GPU (M1) | `llm-planner --model <M1_URL> --execute` | Model loads, generates 100 tokens, predicted-vs-actual report printed | |
| E2 | Execute CPU only (M1) | Select CPU-only strategy | All layers on CPU, slower generation, VRAM stays near zero | |
| E3 | Execute split (M2) | Select split strategy | Partial GPU offload, intermediate speed | |
| E4 | tok/s accuracy | Compare predicted vs actual in E1 report | Within 20% | |
| E5 | TTFT accuracy | Compare predicted vs actual in E1 report | Within 40% | |
| E6 | Memory accuracy | Compare predicted vs actual peak VRAM in E1 | Within 10% | |
| E7 | Generated text quality | Read the generated output | Coherent English text, not garbage | |
| E8 | Custom prompt | `--execute --prompt "Explain gravity"` | Generation uses the custom prompt | |
| E9 | Custom token limit | `--execute --max-tokens 50` | Generates exactly ~50 tokens | |
| E10 | Non-viable execution warning | Select a NO FIT strategy | Warning printed, user can proceed or cancel | |
| E11 | OOM handling | Force a strategy that exceeds VRAM | Graceful error, no crash, GPU memory freed | |
| E12 | Ctrl+C abort | Press Ctrl+C during generation | Clean stop, partial results reported, GPU memory freed | |
| E13 | Memory cleanup | Run `nvidia-smi` after E1 | VRAM returns to pre-run baseline | |
| E14 | Multiple consecutive runs | Execute 3 different strategies back-to-back | All succeed, no memory leaks, consistent results | |
| E15 | Thermal throttle detection | Run long generation on laptop | Sampler detects throttle if it occurs, report flags it | |
| E16 | Live sampler accuracy | Compare sampler's peak VRAM vs `nvidia-smi` during run | Within 100MB | |

---

### TIER 5: Calibration Tests (Step 7 Validation)

These tests verify the feedback loop works correctly.

| ID | Test | Method | Expected Result | Pass/Fail |
|---|---|---|---|---|
| C1 | First run creates log | Delete log, run E1, check file | Log file exists with 1 entry | |
| C2 | Log entry validity | Open log, parse JSON | Valid JSON, all fields populated | |
| C3 | Log accumulation | Run 5 executions | Log contains 5 entries | |
| C4 | Failed runs not logged | Trigger an OOM error | Log count doesn't increase | |
| C5 | Short runs not logged | Execute with `--max-tokens 3` | Log count doesn't increase (<10 tokens) | |
| C6 | Calibration info | `llm-planner --calibration-info` | Prints entry count, fingerprint, calibrated constants | |
| C7 | Calibration affects predictions | Run 10 executions, then run advisor | Predictions shifted toward actual values | |
| C8 | Confidence band upgrade | After 5+ runs on same hardware | Confidence changes from MEDIUM to HIGH | |
| C9 | Calibration reset | `llm-planner --calibration-reset` | Log deleted, next run uses defaults | |
| C10 | Corrupt log handling | Add garbage line to log file | Tool doesn't crash, skips corrupt line | |
| C11 | Hardware fingerprint isolation | Edit log to change fingerprint on some entries | Only matching entries used for calibration | |
| C12 | Version filtering | Edit log to change tool_version | Old-version entries excluded | |
| C13 | Log write failure | Make log directory read-only | Warning printed, execution still succeeds | |
| C14 | Calibration improves accuracy | Compare prediction deltas before and after 10+ runs | Average delta decreases | |

---

### TIER 6: Edge Case and Stress Tests

| ID | Test | Method | Expected Result | Pass/Fail |
|---|---|---|---|---|
| X1 | Very low free VRAM | Open a game or GPU app to consume VRAM, then run tool | Tool detects low VRAM, most GPU strategies NO FIT | |
| X2 | Very low free RAM | Open many browser tabs, then run tool | Tool detects low RAM, CPU strategies may NO FIT | |
| X3 | No internet | Disconnect network, run tool | Clear error on metadata fetch, hardware profile still works | |
| X4 | Slow network | Throttle connection, run tool | Fetch takes longer but completes within timeout | |
| X5 | Hugging Face redirect | Use a URL that redirects to CDN | libcurl follows redirect, fetch succeeds | |
| X6 | Very large context | Request 128K context on a model that supports it | KV cache dominates memory, some strategies NO FIT | |
| X7 | Minimum context | Request 512 context | Works, KV cache negligible | |
| X8 | Unknown quant type | Use a GGUF with a rare/new file_type | Tool prints raw integer + "unknown", doesn't crash | |
| X9 | Old GGUF version | Use a GGUF v1 file (if you can find one) | Tool rejects with clear version error | |
| X10 | Corrupted GGUF file | Truncate a GGUF file to 1KB | Tool detects invalid magic or truncated header | |
| X11 | Path with spaces | Move models to `C:\My Models\` | Tool handles correctly (or fails with clear error) | |
| X12 | Very long model name | Use a model with a 200+ character filename | Table formatting doesn't break | |
| X13 | Multiple GPUs (if available) | Run on a multi-GPU system | Tool detects primary GPU, uses it | |
| X14 | Laptop on battery | Run on a power-limited laptop | Thermal throttle more likely, tool detects it | |
| X15 | Windows Defender active | Remove models folder exclusion | Disk benchmark numbers drop, tool warns | |

---

### Test Execution Order

```
1. Run TIER 1 (Smoke) — if any fail, fix before proceeding
2. Run TIER 2A (Hardware) — validate against nvidia-smi and Task Manager
3. Run TIER 2B (Fetcher) — validate against model cards
4. Run TIER 2C (Predictor) — validate against manual llama.cpp runs
5. Run TIER 3 (Integration) — full pipeline, all priorities
6. Run TIER 4 (Execution) — actual model runs, predicted-vs-actual
7. Run TIER 5 (Calibration) — feedback loop, log management
8. Run TIER 6 (Edge Cases) — break it on purpose
```

### Test Reporting Template

For each test run, record:

| Field | Value |
|---|---|
| Date | |
| Tool version | |
| Git commit | |
| Hardware | |
| Driver version | |
| CUDA version | |
| Tests run | / total |
| Tests passed | |
| Tests failed | |
| Tests skipped | |
| Critical failures | (list IDs) |
| Notes | |

---

### Summary

**Project Status:** The LLM Deployment Planner has a complete, implementation-ready design spanning all 7 build steps. The architecture, formulas, data structures, error handling, and validation methodology are fully specified. The next action is to begin Step 0 (environment setup) and start writing code.

**Test Coverage:** The E2E test plan covers 80+ test cases across 6 tiers, from basic smoke tests to edge-case stress tests. Every component, integration point, and failure mode identified in the design stress-tests has at least one corresponding test case.

**Estimated Timeline to Testable MVP:** 2-4 weeks of focused implementation, with the advisor (Steps 0-5) testable after ~1-2 weeks and the full executor+calibration (Steps 6-7) testable after ~3-4 weeks.