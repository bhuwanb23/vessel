# Step 5 — Ranker: Full Detailed Plan

---

## Goal of Step 5
Take the prediction table from Step 4 and sort it according to what the user actually cares about. "Best" is not a single answer — the fastest strategy, the highest-quality strategy, and the safest strategy are usually three different rows in the table. The ranker makes this explicit by letting the user declare their priority and re-sorting accordingly. This is the smallest step in the entire project.

---

## What You Need Before Starting

### From Steps 1–4 (already done)
- The integrated advisor binary that produces a `vector<Prediction>` for a given model + hardware combination
- Each `Prediction` contains: memory usage, tokens/sec, TTFT, confidence band, viable flag
- The CLI already accepts a `--priority` flag (added in Step 4)
- The output table already prints strategies in some order (probably the order the matrix generator produced them)

### What Changes in This Step
- The prediction table output order becomes **deterministic and meaningful** based on user priority
- A scoring function is added that converts multi-dimensional predictions into a single sortable score
- The recommendation line at the bottom of the table becomes priority-aware

---

## Phase A — Understand the Three Priority Axes

### Why Three Axes Exist
From §3 item 9: "Fastest, lowest-memory-pressure, and highest-quality are usually three different strategies; picking one winner imposes an opinion."

**Concrete example** — 7B Q4_K_M model on RTX 3080 (10GB VRAM), 32GB RAM:

| Strategy | tok/s | Quality Proxy | VRAM Headroom | Best For |
|---|---|---|---|---|
| Full GPU, 4K, FP16 KV | 180 | Medium | 4.2 GB free | Speed |
| Full GPU, 4K, Q8 KV | 182 | Medium | 4.8 GB free | Speed + slight safety |
| Split 24/32, 32K, Q8 KV | 95 | Medium | 2.1 GB free | Long documents |
| CPU Only, 4K, FP16 KV | 21 | Medium | 10 GB free (VRAM untouched) | Safety (GPU free for other tasks) |
| Full GPU, 4K, Q4 KV | 185 | Lower | 5.5 GB free | Maximum speed at quality cost |

No single row wins on all three axes. The ranker's job is to let the user pick which axis matters most.

### The Three Priorities Defined

**1. Speed (`--priority speed`)**
- Primary sort: **tokens/sec descending** (higher is better)
- Secondary sort: **TTFT ascending** (lower is better, breaks ties between similar tok/s)
- Tertiary sort: **confidence descending** (prefer high-confidence predictions when speeds are close)
- Use case: Interactive chat, real-time generation, coding assistants

**2. Quality (`--priority quality`)**
- Primary sort: **bits-per-weight descending** (higher quant = less quality loss)
- Secondary sort: **KV cache precision descending** (FP16 > Q8 > Q4)
- Tertiary sort: **GPU layers descending** (more layers on GPU = fewer CPU quantization artifacts)
- Use case: Creative writing, research, tasks where output accuracy matters more than speed
- **MVP caveat:** From §3 item 5, true quality scoring requires published perplexity data that doesn't exist for every model×quant pair. For MVP, use the proxy metrics above. When real benchmark data is available (e.g., from the model card), incorporate it. Otherwise, be honest: "quality ranking is based on quantization level, not measured perplexity."

**3. Safety (`--priority safety`)**
- Primary sort: **memory headroom descending** (more free memory after loading = safer)
- Memory headroom is calculated as: `min(vram_free - vram_predicted, ram_free - ram_predicted) / max(vram_free, ram_free)` — the percentage of the most constrained resource that remains free
- Secondary sort: **confidence descending** (prefer strategies you're more sure about)
- Tertiary sort: **tokens/sec descending** (among equally safe strategies, prefer faster)
- Use case: Production deployments, long-running sessions, machines with background processes that might allocate memory, laptops that might thermal-throttle

### The "Balanced" Option (Optional for MVP)
A fourth option `--priority balanced` that weights all three axes equally. This is nice to have but not required for MVP. If you add it, use a normalized weighted sum of the three scores.

---

## Phase B — The Scoring Function

### Why a Scoring Function Instead of Direct Sorting
You could sort directly by tokens/sec for speed priority, but that breaks down when you need to handle:
- Non-viable strategies (they should always be at the bottom regardless of priority)
- Confidence bands (a high-speed prediction with low confidence is less trustworthy)
- Multi-dimensional tiebreaking

A scoring function converts each prediction into a single `double` score, then you just sort by that score descending. Clean, extensible, and easy to test.

### The Score Calculation

**Step 1: Non-viable strategies get score = -1.0**
They always sort to the bottom. No exceptions. A strategy that doesn't fit in memory is never "best" regardless of how fast it would theoretically be.

**Step 2: Calculate the primary metric score (0.0 to 1.0)**

Normalize the primary metric across all viable strategies so that the best gets 1.0 and the worst gets 0.0:

```
For speed priority:
    metric_value = prediction.tokens_per_sec
    normalized = (metric_value - min_tok_s) / (max_tok_s - min_tok_s)

For quality priority:
    metric_value = model.bits_per_weight + (kv_bits == 16 ? 1.0 : kv_bits == 8 ? 0.5 : 0.0)
    normalized = (metric_value - min_quality) / (max_quality - min_quality)

For safety priority:
    vram_headroom = (hardware.vram_free - prediction.vram_usage) / hardware.vram_free
    ram_headroom = (hardware.ram_free - prediction.ram_usage) / hardware.ram_free
    metric_value = min(vram_headroom, ram_headroom)
    normalized = (metric_value - min_headroom) / (max_headroom - min_headroom)
```

**Edge case:** If all viable strategies have the same primary metric value (e.g., all have the same tokens/sec because they're all full-GPU with different KV quants), the denominator is zero. In that case, set `normalized = 0.5` for all and let the secondary metric decide.

**Step 3: Apply confidence modifier**

```
confidence_multiplier:
    HIGH   → 1.0
    MEDIUM → 0.9
    LOW    → 0.7
```

This slightly penalizes low-confidence predictions. A medium-confidence strategy that predicts 200 tok/s scores lower than a high-confidence strategy that predicts 195 tok/s. This prevents the tool from confidently recommending a strategy it's actually unsure about.

**Step 4: Combine into final score**

```
score = normalized_primary × 0.7 + normalized_secondary × 0.2 + normalized_tertiary × 0.1
score ×= confidence_multiplier
```

The 70/20/10 weighting ensures the primary priority dominates while secondary and tertiary metrics break ties meaningfully.

### Why These Weights
- 70% primary: The user explicitly told you what they care about. Respect that.
- 20% secondary: Among strategies that are roughly equal on the primary axis, the secondary axis should be the tiebreaker.
- 10% tertiary: A gentle nudge, not a strong signal.
- These are starting values. The calibration log in Step 7 could eventually inform better weights, but for MVP, fixed weights are fine.

---

## Phase C — The Sorting Logic

### The Algorithm

```
function rankStrategies(predictions, priority):
    for each prediction in predictions:
        prediction.score = calculateScore(prediction, priority, all_predictions)
    
    sort predictions by:
        1. viable descending (viable before non-viable)
        2. score descending (higher is better)
        3. confidence descending (tiebreaker)
    
    return sorted predictions
```

### Stable Sort
Use `std::stable_sort` instead of `std::sort`. The reason: when two strategies have identical scores, stable sort preserves their original order from the matrix generator. This means the output is deterministic and reproducible — running the tool twice with the same inputs gives the same ordering.

### Handling Ties
Two strategies might have genuinely identical scores (e.g., Full GPU with FP16 KV vs Full GPU with Q8 KV when the memory difference is negligible and speed is the same). In this case, the stable sort preserves the matrix generator's order, which is fine. Don't add arbitrary tiebreakers — they create false precision.

---

## Phase D — The Recommendation Line

### Priority-Aware Recommendations
The recommendation at the bottom of the table should change based on the priority. Don't just say "Strategy #1 is best" — explain **why** it's best for the user's stated priority.

**Speed priority:**
```
💡 Fastest option: Strategy #1 (Full GPU, 4K) at ~385 tok/s.
   If you need longer context, Strategy #3 (Full GPU, 128K, Q8 KV) 
   stays on GPU at ~290 tok/s.
```

**Quality priority:**
```
💡 Highest quality: Strategy #1 (Full GPU, 4K, FP16 KV) — highest 
   quantization level that fits on GPU.
   Note: quality ranking is based on quantization level, not measured 
   perplexity. No published benchmark data available for this model×quant.
```

**Safety priority:**
```
💡 Safest option: Strategy #2 (Full GPU, 4K, Q8 KV) — uses only 21% 
   of VRAM, leaving 7.9 GB free for OS and background apps.
   Strategy #1 is 2% faster but uses 30% more VRAM.
```

### The "Tradeoff Callout"
When the top two strategies differ significantly on a secondary axis, mention it. This is the kind of insight that makes the tool genuinely useful:

```
⚠️  Tradeoff: Strategy #1 is 4× faster than #4 but only fits at 4K context.
   For documents >4K tokens, Strategy #4 (Split, 128K) is your best option.
```

### The "Nothing Fits" Message
If all strategies are non-viable:

```
❌ This model does not fit on your hardware in any configuration.
   Options:
   • Try a smaller quantization (e.g., Q3_K_M instead of Q4_K_M)
   • Try a smaller model (e.g., 3B instead of 7B)
   • Close GPU-intensive applications to free VRAM (currently 1.3 GB free)
```

This is actionable advice, not just a failure message.

---

## Phase E — Integration with Step 4

### Where the Ranker Fits in the Pipeline

```
Step 4 pipeline (before Step 5):
    Profile → Fetch → Generate Matrix → Predict → Print Table (unsorted)

Step 4 pipeline (after Step 5):
    Profile → Fetch → Generate Matrix → Predict → RANK → Print Table (sorted)
```

The ranker is a single function call inserted between prediction and output. It takes the `vector<Prediction>` and returns a sorted copy. It does not modify the predictions themselves (except adding the score field).

### The `--priority` Flag
This was already added to the CLI in Step 4. Now it actually does something.

**Default behavior:** If the user doesn't specify `--priority`, default to `speed`. This matches what most users care about when running local LLMs — they want to know "how fast will it be?"

**Validation:** If the user passes an invalid priority string (e.g., `--priority fast`), print an error listing the valid options and exit. Don't silently fall back to the default — that's confusing.

---

## Phase F — Testing the Ranker

### Test Scenarios

**Test 1: Speed Priority**
- Run with a model that has multiple viable strategies
- Pass `--priority speed`
- Verify: Full GPU strategy is ranked #1 (it should be fastest)
- Verify: CPU-only is near the bottom
- Verify: Non-viable strategies are at the very bottom

**Test 2: Quality Priority**
- Run with a model available in multiple KV cache precisions
- Pass `--priority quality`
- Verify: FP16 KV strategies rank above Q8 KV strategies (when both are viable)
- Verify: Higher bpw quants rank above lower bpw (if testing across quant types)

**Test 3: Safety Priority**
- Run with a model that has both tight-fit and comfortable-fit strategies
- Pass `--priority safety`
- Verify: Strategies with the most memory headroom rank highest
- Verify: A strategy using 95% VRAM ranks below one using 50% VRAM, even if the 95% one is faster

**Test 4: Priority Changes Order**
- Run the same model three times with three different priorities
- Verify: The table order actually changes between runs
- If all three priorities produce the same order, the model might be too small to show meaningful tradeoffs — try a larger model

**Test 5: All Non-Viable**
- Run with a model that's too large for the hardware
- Verify: All strategies show `❌ NO FIT`
- Verify: The recommendation provides actionable alternatives

**Test 6: Single Viable Strategy**
- Run with a model that only fits in one configuration
- Verify: That strategy is ranked #1
- Verify: The recommendation doesn't misleadingly suggest alternatives that don't exist

---

## Step 5 — Done Checklist

Before moving to Step 6, confirm every item:

- [ ] Scoring function produces values in the 0.0–1.0 range for viable strategies
- [ ] Non-viable strategies always score -1.0 and sort to the bottom
- [ ] Speed priority sorts by tokens/sec descending
- [ ] Quality priority sorts by quantization level descending
- [ ] Safety priority sorts by memory headroom descending
- [ ] Confidence modifier correctly penalizes low-confidence predictions
- [ ] `std::stable_sort` is used for deterministic output
- [ ] `--priority speed` produces a different order than `--priority safety` for at least one test model
- [ ] Default priority is `speed` when no flag is provided
- [ ] Invalid priority string produces a clear error message
- [ ] Recommendation line changes based on priority
- [ ] Recommendation includes tradeoff callouts when top strategies differ significantly
- [ ] "Nothing fits" scenario produces actionable advice
- [ ] Single-viable-strategy scenario doesn't produce misleading recommendations
- [ ] Division-by-zero is handled when all strategies have the same primary metric
- [ ] Ranker is a pure function (no side effects, no I/O)

---

## Common Failure Points at Step 5

| Problem | Likely Cause | Fix |
|---|---|---|
| All priorities produce the same order | Model is too small — all strategies fit easily with similar speeds | Test with a larger model that creates real tradeoffs (e.g., 7B on 8GB VRAM) |
| Non-viable strategies appear above viable ones | Sort comparator doesn't check `viable` flag first | Ensure `viable` is the primary sort key, before `score` |
| Score is NaN | Division by zero in normalization (all strategies have the same metric value) | Add a guard: if `max == min`, set `normalized = 0.5` |
| Quality ranking seems wrong | Using bits-per-weight of the model quant, not the KV cache precision | Quality score should combine both model quant AND KV cache precision |
| Safety ranking favors CPU-only over GPU | CPU-only leaves all VRAM free, so headroom looks huge | The headroom formula should consider the **most constrained** resource, not the least. A strategy that uses 0% VRAM but 90% RAM is less safe than one using 50% VRAM and 20% RAM. |
| Recommendation contradicts the table | Recommendation logic uses a different ranking than the table sort | The recommendation should simply reference `sorted_predictions[0]` — don't recalculate |
| Score overflow with extreme values | Very large tokens/sec or very small headroom | Normalize all metrics to 0.0–1.0 before combining. Never use raw values in the weighted sum. |

---

## Time Estimate for Step 5
- Phase A (Understanding priority axes): **30 minutes**
- Phase B (Scoring function): **1–2 hours** (normalization logic and edge cases)
- Phase C (Sorting logic): **30 minutes** (it's literally one `std::stable_sort` call with a custom comparator)
- Phase D (Recommendation line): **1–2 hours** (writing the priority-aware text templates)
- Phase E (Integration with Step 4): **30 minutes** (one function call insertion)
- Phase F (Testing): **1–2 hours** (running all test scenarios)

**Total: Half a day, as originally estimated. This is genuinely the easiest step. The hard work was done in Steps 1–4. The ranker is just sorting a list that already exists.**