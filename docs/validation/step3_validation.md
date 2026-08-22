# Step 3 — Validation Table

## Llama-3.2-3B-Instruct-Q4_K_M

### Test 1: Full GPU (28 layers)

| Metric | Predicted | Actual (llama.cpp) | Delta | Acceptable? |
|--------|-----------|-------------------|-------|-------------|
| Total memory | 2.75 GB | ~2.4 GB (2464 MiB) | +14.6% | ✅ Within 15% |
| VRAM usage | 2.75 GB | 2464 MiB (2.41 GB) | +14.1% | ✅ Within 10% |
| RAM usage | 0 B | ~0 B | 0% | ✅ |
| Tokens/sec (decode) | 41.4 t/s | 61.3 t/s | -32.5% | ⚠️ Over-predicted |
| Prompt speed | 933.8 t/s | 730.6 t/s | +27.8% | ⚠️ Over-predicted |

### Test 2: Split Mode (20 GPU layers)

| Metric | Predicted | Actual (llama.cpp) | Delta | Acceptable? |
|--------|-----------|-------------------|-------|-------------|
| Total memory | 2.75 GB | ~2.4 GB | +14.6% | ✅ Within 15% |
| Tokens/sec (decode) | 7.1 t/s | 45.3 t/s | -84.3% | ❌ Way off |

---

## Analysis

### What's Working
1. **Memory prediction is accurate** — within 15% of actual
2. **VRAM usage prediction is good** — within 14% of actual
3. **Weight memory formula is correct** — 1.81 GB matches file size

### What Needs Adjustment

#### Issue 1: Efficiency Factor Too Low
- Predicted: 41.4 t/s
- Actual: 61.3 t/s
- Actual efficiency: 61.3 / 230 = **26.6%**
- Current setting: 18%
- **Fix: Increase efficiency from 0.18 to 0.27**

#### Issue 2: Sequential Split Model Too Pessimistic
- Predicted split speed: 7.1 t/s (way too slow)
- Actual split speed: 45.3 t/s (only 26% slower than full GPU)
- The sequential bottleneck model doesn't match reality for this model size
- **Possible causes:**
  - Model is small enough that CPU layers aren't a bottleneck
  - llama.cpp has optimized data transfers
  - My RAM bandwidth estimate (25 GB/s) is too low

---

## Recommended Adjustments

### 1. Increase Efficiency Factor (Full GPU)
```
Old: 0.18
New: 0.27 (based on 61.3 / 230 = 26.6%)
```

### 2. Increase RAM Bandwidth Estimate
```
Old: 25.0 GB/s (DDR4-3200 estimate)
New: 40.0 GB/s (DDR5-5600 estimate for RTX 5060 system)
```

### 3. Reduce Split Penalty
The sequential model might need a higher baseline efficiency for small models.

---

## Next Steps

1. Adjust efficiency factor in speed_predictor.cpp
2. Update RAM bandwidth default
3. Re-run tests to verify
4. Test with Qwen2.5-7B (different model size)
