# Step 12 — Auto-Recommendation Engine: Full Detailed Plan

---

## Goal of Step 12
Transform the tool from "tell me how to run this specific model" to "tell me what model I should run." Given only the user's hardware (auto-profiled), the tool queries a curated model catalog, predicts performance for every model×strategy combination, and presents the top recommendations ranked by the user's priority. The user no longer needs to know model names, quantization levels, or Hugging Face URLs. They just run the tool and get an answer.

---

## Why This Step Exists

The current tool requires the user to provide a Hugging Face GGUF URL. This assumes the user already knows:
- Which model family to use (Llama vs Qwen vs Mistral vs Phi)
- Which size fits their hardware (3B vs 7B vs 13B vs 70B)
- Which quantization level to pick (Q4_K_M vs Q5_K_M vs Q8_0)
- Where to find the GGUF file on Hugging Face

Most users don't know any of this. They know they have a GPU and they want to run a local LLM. The tool should bridge that gap.

**The product shift:** Before Step 12, the tool is a **deployment planner** (given a model, find the best strategy). After Step 12, it is also a **model advisor** (given hardware, find the best model AND strategy).

---

## What You Need Before Starting

### From Steps 1–11 (must be solid)
- **Step 1:** Hardware profiler works on all supported platforms (NVIDIA, AMD, Apple)
- **Step 2:** Metadata fetcher can extract model dimensions from GGUF headers
- **Step 3:** Predictor produces accurate memory, tok/s, and TTFT estimates
- **Step 4:** Method matrix generator enumerates strategies per model
- **Step 5:** Ranker sorts by speed/quality/safety
- **Steps 8–10:** Download manager, MoE offload, hot/cold split all functional
- **Step 11:** Platform expansion complete (the recommendation engine must work on all platforms)

### New Dependencies
- **A model catalog file** — a JSON file shipped with the tool containing curated model metadata. No external server required for MVP.
- **No new libraries.** Everything uses existing components (predictor, profiler, ranker).

---

## Phase A — The Model Catalog

### A1. What the Catalog Contains

The catalog is a local JSON file (`models_catalog.json`) shipped alongside the binary. It contains the most popular and well-tested models with their GGUF variants, pre-computed metadata, and quality scores.

**Why a local file instead of querying Hugging Face:**
- Hugging Face doesn't have a "list all GGUF models" API
- Querying individual repos is slow (one HTTP request per model)
- The catalog can include quality scores and human curation that raw HF data doesn't provide
- Works offline (the user might not have internet when first running the tool)

### A2. The Catalog Schema

```json
{
  "version": "2026.08.15",
  "models": [
    {
      "id": "llama-3.2-3b-instruct",
      "name": "Llama 3.2 3B Instruct",
      "family": "llama",
      "use_case": ["chat", "instruction", "coding"],
      "description": "Fast, capable small model. Best for real-time chat and simple tasks.",
      "params_billions": 3.2,
      "architecture": "llama",
      "max_context": 131072,
      "quality_score": 7.2,
      "quality_source": "Open LLM Leaderboard v2, Aug 2026",
      "gguf_variants": [
        {
          "quant": "Q2_K",
          "file_size_gb": 1.4,
          "bpw": 2.96,
          "hf_repo": "bartowski/Llama-3.2-3B-Instruct-GGUF",
          "hf_file": "Llama-3.2-3B-Instruct-Q2_K.gguf",
          "hf_url": "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q2_K.gguf"
        },
        {
          "quant": "Q4_K_M",
          "file_size_gb": 2.0,
          "bpw": 4.85,
          "hf_repo": "bartowski/Llama-3.2-3B-Instruct-GGUF",
          "hf_file": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
          "hf_url": "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
        },
        {
          "quant": "Q8_0",
          "file_size_gb": 3.4,
          "bpw": 8.5,
          "hf_repo": "bartowski/Llama-3.2-3B-Instruct-GGUF",
          "hf_file": "Llama-3.2-3B-Instruct-Q8_0.gguf",
          "hf_url": "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q8_0.gguf"
        }
      ],
      "dimensions": {
        "layers": 28,
        "embedding_dim": 3072,
        "attention_heads": 24,
        "kv_heads": 8,
        "head_dim": 128,
        "ffn_dim": 8192
      },
      "is_moe": false
    }
  ]
}
```

### A3. Fields Explained

| Field | Purpose | Source |
|---|---|---|
| `id` | Unique identifier for deduplication | Manual |
| `name` | Human-readable display name | Model card |
| `family` | Model family for grouping | Model card |
| `use_case` | Tags for filtering by user intent | Manual curation |
| `description` | One-line summary for the recommendation output | Manual |
| `params_billions` | Parameter count in billions | Model card |
| `architecture` | GGUF architecture string | Step 2 metadata |
| `max_context` | Maximum context length | Model card |
| `quality_score` | 0–10 quality rating | Benchmark data (see Phase A5) |
| `quality_source` | Where the quality score came from | Citation |
| `gguf_variants[]` | Available GGUF quants with download info | Hugging Face |
| `dimensions` | Pre-extracted model dimensions | Step 2 metadata (cached) |
| `is_moe` | Whether the model uses MoE | Model card |

### A4. The Initial Catalog (MVP)

For MVP, include **15–25 models** covering the most popular families and sizes. This is enough to give useful recommendations without overwhelming the user or the prediction pipeline.

**Recommended initial catalog:**

| Family | Sizes | Quants per Size | Total Variants |
|---|---|---|---|
| Llama 3.2 | 1B, 3B | Q4_K_M, Q8_0 | 4 |
| Llama 3.1 | 8B | Q3_K_M, Q4_K_M, Q5_K_M, Q8_0 | 4 |
| Qwen 2.5 | 3B, 7B, 14B | Q4_K_M, Q5_K_M | 6 |
| Mistral | 7B (v0.3) | Q4_K_M, Q5_K_M | 2 |
| Phi 3.5 | Mini (3.8B) | Q4_K_M, Q8_0 | 2 |
| Gemma 2 | 2B, 9B | Q4_K_M | 2 |
| DeepSeek V2 | Lite (16B) | Q4_K_M | 1 (MoE) |
| Mixtral | 8×7B | Q3_K_M, Q4_K_M | 2 (MoE) |

**Total: ~23 variants across 12 base models.** This covers the range from "fits on any GPU" (1B) to "needs serious hardware" (Mixtral 8×7B).

### A5. Quality Scores

**Where to get them:**
- **Open LLM Leaderboard v2** (Hugging Face) — the most widely cited benchmark for open models
- **LMSYS Chatbot Arena** — Elo ratings based on human preference
- **Model card benchmarks** — individual model cards often report MMLU, HumanEval, etc.

**How to normalize:**
Different benchmarks use different scales. Normalize everything to a 0–10 scale:
- Open LLM Leaderboard average score (typically 5–25 for small models, 25–50 for large) → map to 0–10
- Chatbot Arena Elo (typically 1000–1300) → map to 0–10
- If multiple sources exist, average them

**Honesty note:** Quality scores are approximate and benchmark-dependent. Display them as "Quality: ★★★★☆" (star rating) rather than "Quality: 7.2/10" to avoid false precision. The star rating maps:
- ★☆☆☆☆ (0–2): Very limited capability
- ★★☆☆☆ (2–4): Basic tasks only
- ★★★☆☆ (4–6): Competent for most tasks
- ★★★★☆ (6–8): Strong performer
- ★★★★★ (8–10): State-of-the-art

### A6. Catalog Maintenance

**For MVP:** The catalog is a static JSON file shipped with the binary. Users get updates when they update the tool.

**For Phase 3+:** Add a `--update-catalog` flag that downloads the latest catalog from a hosted URL (e.g., GitHub raw URL or a simple CDN). This allows weekly catalog updates without requiring a full tool rebuild.

**For Phase 4:** Community-contributed catalog entries via pull requests to a GitHub repository.

---

## Phase B — The Recommendation Pipeline

### B1. The End-to-End Flow

```
User runs: llm-planner --recommend [--priority speed|quality|balanced] [--use-case chat|coding|all]

Step 1: Profile hardware
    → HardwareSpec (from Step 1)
    → Available VRAM/RAM, bandwidth, platform

Step 2: Load catalog
    → Parse models_catalog.json
    → Filter by use_case if specified
    → Get list of model variants

Step 3: Pre-filter by viability
    → For each variant, quick-check if the smallest viable strategy fits
    → Discard variants where even CPU-only at 4K context exceeds RAM
    → This eliminates models that are obviously too large

Step 4: Predict for each surviving variant
    → For each variant, generate a mini method matrix (2-3 strategies, not the full 24)
    → Run the predictor for each strategy
    → Keep only the best strategy per variant (the one that would rank #1 for the user's priority)

Step 5: Rank across all variants
    → Sort by the user's priority (speed, quality, or balanced)
    → Select top 5-10 recommendations

Step 6: Display
    → Print the recommendation table
    → Include download URLs for the top picks
```

### B2. The Pre-Filter (Critical for Performance)

Without pre-filtering, the predictor would run for every variant × every strategy. With 23 variants × 12 strategies each, that's 276 predictions. Each prediction is pure math (microseconds), so this is actually fast enough. But the pre-filter reduces noise:

**Quick viability check:**
```
smallest_memory = weight_bytes(variant, smallest_quant) 
                + kv_cache_bytes(4096, FP16) 
                + runtime_overhead

if smallest_memory > max(vram_free, ram_free):
    discard variant  // doesn't fit in any configuration
```

This eliminates variants like Mixtral 8×7B Q4_K_M on a machine with 8GB RAM. No need to predict 12 strategies for a model that can't run at all.

**Expected survivors:** On an 8GB VRAM / 16GB RAM machine, roughly 12–18 of 23 variants survive the pre-filter. On a 24GB VRAM / 64GB RAM machine, all 23 survive.

### B3. The Mini Method Matrix

For recommendations, you don't need the full 24-strategy matrix per model. You need the **best** strategy per model for the user's priority. Generate a reduced matrix:

| Strategy | When to Include |
|---|---|
| Full GPU at 4K | Always (if viable) |
| Full GPU at max-safe | If VRAM allows |
| Best split at 4K | If full GPU doesn't fit |
| CPU-only at 4K | If nothing else fits |
| MoE offload | Only if model is MoE |

This gives 2–4 strategies per variant instead of 12–24. The predictor runs in <1ms total for all variants.

### B4. The Cross-Model Ranking

After predicting the best strategy per variant, rank all variants by the user's priority:

**Speed priority:**
- Primary: `tokens_per_sec` of the best strategy (descending)
- Secondary: `ttft_ms` (ascending)
- Tertiary: `quality_score` (descending, so among equally fast models, prefer better quality)

**Quality priority:**
- Primary: `quality_score` (descending)
- Secondary: `tokens_per_sec` (descending, so among equally good models, prefer faster)
- Tertiary: `file_size_gb` (ascending, prefer smaller downloads)

**Balanced priority (new, specific to recommendations):**
- Score = `0.4 × normalized_speed + 0.4 × normalized_quality + 0.2 × normalized_safety`
- This gives a "best overall" recommendation that balances speed and quality

### B5. The "Best For" Labels

Each recommendation gets a label explaining why it's recommended:

| Label | Condition |
|---|---|
| "🏆 Best Overall" | Highest balanced score |
| "⚡ Fastest" | Highest tok/s |
| "🧠 Highest Quality" | Highest quality score among viable models |
| "💾 Smallest Download" | Smallest file size among viable models |
| "📄 Best for Long Documents" | Largest max-safe context |
| "💻 Best for Coding" | Highest quality score among models tagged "coding" |

---

## Phase C — The Output Format

### C1. The Recommendation Table

```
=== LLM Deployment Planner — Model Recommendations ===
Hardware: RTX 3080 (10GB VRAM, 8.7GB free) | 32GB RAM | NVMe 4.2 GB/s
Priority: balanced (use --priority to change)
Use case: all (use --use-case to filter)

 #  Model                    Quant    Strategy      VRAM    tok/s   Quality  Download
─── ──────────────────────── ──────── ──────────── ─────── ─────── ──────── ────────
 1  🏆 Qwen 2.5 7B Instruct  Q4_K_M   Full GPU, 4K  5.2 GB  ~45     ★★★★☆   4.5 GB
 2  ⚡ Llama 3.2 3B Instruct Q8_0     Full GPU, 4K  3.4 GB  ~95     ★★★☆☆   3.4 GB
 3  🧠 Llama 3.1 8B Instruct Q5_K_M   Full GPU, 4K  6.1 GB  ~32     ★★★★☆   5.7 GB
 4     Mistral 7B v0.3        Q4_K_M   Full GPU, 4K  5.0 GB  ~42     ★★★★☆   4.4 GB
 5     Phi 3.5 Mini           Q4_K_M   Full GPU, 4K  2.5 GB  ~65     ★★★☆☆   2.4 GB
 6     Gemma 2 9B             Q4_K_M   Split 28/42   6.8 GB  ~22     ★★★★☆   5.5 GB
 7  📄 Qwen 2.5 14B Instruct  Q3_K_M   Split 20/48   8.9 GB  ~14     ★★★★☆   7.2 GB
 8     Llama 3.2 1B Instruct  Q8_0     Full GPU, 4K  1.3 GB  ~150    ★★☆☆☆   1.3 GB

💡 Top pick: #1 Qwen 2.5 7B — best balance of speed (45 tok/s) and quality.
   For maximum speed: #2 Llama 3.2 3B Q8 at 95 tok/s.
   For maximum quality: #3 Llama 3.1 8B Q5_K_M (slower but smarter).
   For long documents: #7 Qwen 2.5 14B supports 32K context on your hardware.

To download and run the top pick:
  llm-planner --model https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf --execute
```

### C2. Key Design Decisions in the Output

**Show the quant, not just the model name.** "Qwen 2.5 7B" is meaningless without knowing the quant. "Qwen 2.5 7B Q4_K_M" tells the user exactly what they're getting.

**Show the download size.** Users care about how much they need to download. A 4.5GB download is very different from a 40GB download.

**Show the strategy.** The user should see that "Qwen 2.5 14B" requires a split strategy, not full GPU. This sets expectations.

**Show the quality as stars, not numbers.** "★★★★☆" is immediately understandable. "7.2/10" invites questions about methodology.

**Include the run command.** The user should be able to copy-paste the command to immediately download and run the recommended model. This closes the loop from recommendation to execution.

### C3. The "Nothing Fits" Output

If the hardware is extremely constrained (e.g., 2GB VRAM, 4GB RAM):

```
=== Model Recommendations ===
Hardware: GTX 1050 (2GB VRAM, 1.1GB free) | 8GB RAM

Very few models fit on this hardware. Here are your options:

 #  Model                    Quant    Strategy    VRAM   tok/s   Quality  Download
 1  Llama 3.2 1B Instruct   Q4_K_M   Full GPU    0.8 GB ~35     ★★☆☆☆   0.7 GB
 2  Phi 3.5 Mini            Q2_K     CPU Only    1.6 GB ~3      ★★☆☆☆   1.5 GB

💡 Your hardware is very constrained. Consider:
   • Closing GPU-intensive applications to free VRAM
   • Upgrading RAM (16GB+ recommended for local LLMs)
   • Using a cloud API for larger models
```

---

## Phase D — CLI Interface

### D1. New Flags

| Flag | Type | Default | Description |
|---|---|---|---|
| `--recommend` | bool | false | Activate recommendation mode |
| `--priority` | string | "balanced" | Ranking priority: speed, quality, balanced |
| `--use-case` | string | "all" | Filter by use case: chat, coding, reasoning, all |
| `--max-download` | float | none | Maximum download size in GB (e.g., `--max-download 5` excludes models >5GB) |
| `--top` | int | 8 | Number of recommendations to show |
| `--catalog` | string | built-in | Path to a custom catalog JSON file |

### D2. Example Invocations

```bash
# Basic recommendation
llm-planner --recommend

# Fastest model for coding
llm-planner --recommend --priority speed --use-case coding

# Best quality model under 5GB download
llm-planner --recommend --priority quality --max-download 5

# Show top 3 only
llm-planner --recommend --top 3

# Use a custom catalog
llm-planner --recommend --catalog my_models.json
```

### D3. Interaction with Existing Modes

The `--recommend` flag is mutually exclusive with `--model`. If both are provided, print an error:

```
❌ Cannot use --recommend and --model together.
   --recommend suggests models for your hardware.
   --model predicts strategies for a specific model.
```

---

## Phase E — The Catalog Data Structure in Code

### E1. The Catalog Structs

```cpp
struct GgufVariant {
    std::string quant;
    double file_size_gb;
    double bpw;
    std::string hf_repo;
    std::string hf_file;
    std::string hf_url;
};

struct ModelDimensions {
    uint32_t layers;
    uint32_t embedding_dim;
    uint32_t attention_heads;
    uint32_t kv_heads;
    uint32_t head_dim;
    uint32_t ffn_dim;
};

struct CatalogModel {
    std::string id;
    std::string name;
    std::string family;
    std::vector<std::string> use_cases;
    std::string description;
    double params_billions;
    std::string architecture;
    uint32_t max_context;
    double quality_score;
    std::string quality_source;
    std::vector<GgufVariant> variants;
    ModelDimensions dimensions;
    bool is_moe;
};

struct ModelCatalog {
    std::string version;
    std::vector<CatalogModel> models;
};
```

### E2. Loading the Catalog

```cpp
ModelCatalog loadCatalog(const std::string& path) {
    // Read JSON file
    // Parse into ModelCatalog struct using nlohmann::json
    // Validate required fields
    // Return catalog
}
```

**Built-in catalog:** Embed the default catalog as a string constant in the binary (using CMake's `configure_file` or a raw string literal). This ensures the tool works out of the box without requiring an external file. If `--catalog` is provided, load from that path instead.

---

## Phase F — The Recommendation Engine Logic

### F1. The Core Algorithm

```cpp
struct Recommendation {
    CatalogModel model;
    GgufVariant variant;
    StrategyConfig best_strategy;
    Prediction best_prediction;
    std::string label;  // "🏆 Best Overall", "⚡ Fastest", etc.
    double rank_score;
};

std::vector<Recommendation> generateRecommendations(
    const HardwareSpec& hw,
    const ModelCatalog& catalog,
    const std::string& priority,
    const std::string& use_case,
    double max_download_gb,
    int top_n
) {
    std::vector<Recommendation> candidates;
    
    for (const auto& model : catalog.models) {
        // Filter by use case
        if (use_case != "all" && !contains(model.use_cases, use_case))
            continue;
        
        for (const auto& variant : model.variants) {
            // Filter by download size
            if (max_download_gb > 0 && variant.file_size_gb > max_download_gb)
                continue;
            
            // Build ModelMetadata from catalog dimensions
            ModelMetadata meta = catalogToMetadata(model, variant);
            
            // Quick viability check
            double min_memory = estimateMinMemory(meta, hw);
            if (min_memory > std::max(hw.vram_free_bytes, hw.ram_free_bytes))
                continue;  // doesn't fit at all
            
            // Generate mini matrix and predict
            auto strategies = generateMiniMatrix(hw, meta);
            Prediction best_pred;
            StrategyConfig best_config;
            double best_score = -1;
            
            for (const auto& strategy : strategies) {
                auto pred = predict(hw, meta, strategy);
                if (!pred.viable) continue;
                
                double score = scoreForPriority(pred, model.quality_score, priority);
                if (score > best_score) {
                    best_score = score;
                    best_pred = pred;
                    best_config = strategy;
                }
            }
            
            if (best_score > 0) {
                candidates.push_back({model, variant, best_config, best_pred, "", best_score});
            }
        }
    }
    
    // Sort by rank_score descending
    std::sort(candidates.begin(), candidates.end(),
        [](const auto& a, const auto& b) { return a.rank_score > b.rank_score; });
    
    // Assign labels
    assignLabels(candidates);
    
    // Return top N
    if (candidates.size() > top_n)
        candidates.resize(top_n);
    
    return candidates;
}
```

### F2. The Scoring Function for Recommendations

This is different from the Step 5 ranker scoring because it compares **across models**, not just across strategies for the same model.

```cpp
double scoreForPriority(const Prediction& pred, double quality, const std::string& priority) {
    if (priority == "speed") {
        return pred.tokens_per_sec * 0.7 
             + pred.tokens_per_sec / std::max(pred.ttft_ms, 1.0) * 0.2  // speed + responsiveness
             + quality * 0.1;
    } else if (priority == "quality") {
        return quality * 0.6 
             + std::min(pred.tokens_per_sec / 20.0, 1.0) * 0.3  // "fast enough" threshold
             + (pred.vram_headroom > 0.2 ? 0.1 : 0.0);  // slight safety bonus
    } else {  // balanced
        double speed_norm = std::min(pred.tokens_per_sec / 50.0, 1.0);  // normalize to ~50 tok/s ceiling
        double quality_norm = quality / 10.0;
        double safety_norm = std::min(pred.vram_headroom / 0.5, 1.0);
        return speed_norm * 0.4 + quality_norm * 0.4 + safety_norm * 0.2;
    }
}
```

**Key design choice:** The "balanced" score normalizes speed to a ceiling of ~50 tok/s. This prevents a 1B model at 200 tok/s from dominating a 7B model at 40 tok/s. Above 50 tok/s, additional speed has diminishing returns for most users. The quality dimension differentiates models that are "fast enough."

### F3. Label Assignment

```cpp
void assignLabels(std::vector<Recommendation>& recs) {
    if (recs.empty()) return;
    
    // Find extremes
    auto fastest = std::max_element(recs.begin(), recs.end(),
        [](const auto& a, const auto& b) { return a.best_prediction.tokens_per_sec < b.best_prediction.tokens_per_sec; });
    
    auto best_quality = std::max_element(recs.begin(), recs.end(),
        [](const auto& a, const auto& b) { return a.model.quality_score < b.model.quality_score; });
    
    auto smallest = std::min_element(recs.begin(), recs.end(),
        [](const auto& a, const auto& b) { return a.variant.file_size_gb < b.variant.file_size_gb; });
    
    auto longest_ctx = std::max_element(recs.begin(), recs.end(),
        [](const auto& a, const auto& b) { return a.best_prediction.max_safe_context < b.best_prediction.max_safe_context; });
    
    recs[0].label = "🏆 Best Overall";
    fastest->label += (fastest->label.empty() ? "" : " ") + std::string("⚡ Fastest");
    best_quality->label += (best_quality->label.empty() ? "" : " ") + std::string("🧠 Highest Quality");
    smallest->label += (smallest->label.empty() ? "" : " ") + std::string("💾 Smallest");
    longest_ctx->label += (longest_ctx->label.empty() ? "" : " ") + std::string("📄 Longest Context");
}
```

---

## Phase G — Integration with Existing Pipeline

### G1. The Main Function Branch

```cpp
int main(int argc, char* argv[]) {
    auto args = parseArgs(argc, argv);
    
    if (args.recommend) {
        // Recommendation mode
        auto hw = profileHardware();
        auto catalog = loadCatalog(args.catalog_path);
        auto recs = generateRecommendations(hw, catalog, args.priority, 
                                            args.use_case, args.max_download, args.top);
        printRecommendations(hw, recs);
    } else if (!args.model_url.empty()) {
        // Existing advisor/executor mode (Steps 1-10)
        auto hw = profileHardware();
        auto meta = fetchMetadata(args.model_url);
        auto matrix = generateMatrix(hw, meta);
        auto predictions = predictAll(hw, meta, matrix);
        auto ranked = rankStrategies(predictions, args.priority);
        printStrategyTable(hw, meta, ranked);
        
        if (args.execute) {
            auto selected = userSelect(ranked);
            auto result = execute(selected, hw, meta);
            printPredictedVsActual(result);
            logCalibration(result);
        }
    } else {
        printUsage();
    }
}
```

### G2. Shared Components

The recommendation engine reuses these existing components without modification:
- `profileHardware()` — Step 1
- `predict()` — Step 3
- `generateMatrix()` — Step 4 (mini version)
- The `Prediction` and `StrategyConfig` structs — Step 3

New components:
- `loadCatalog()` — catalog parsing
- `generateRecommendations()` — cross-model ranking
- `printRecommendations()` — output formatting
- `assignLabels()` — label logic

---

## Phase H — Testing

### H1. Test Scenarios

**Test 1: Basic Recommendation**
- Run `--recommend` on a machine with 10GB VRAM, 32GB RAM
- Verify: 5–8 models recommended, all viable on the hardware
- Verify: Top pick is a 7B-class model (best balance for this hardware)
- Verify: No 70B models in the list (they don't fit)

**Test 2: Speed Priority**
- Run `--recommend --priority speed`
- Verify: Smallest models (1B, 3B) rank highest
- Verify: tok/s values are in descending order

**Test 3: Quality Priority**
- Run `--recommend --priority quality`
- Verify: Largest viable models (7B, 14B) rank highest
- Verify: Quality stars are in descending order (or tied)

**Test 4: Use Case Filter**
- Run `--recommend --use-case coding`
- Verify: Only models tagged "coding" appear
- Verify: Models not tagged "coding" are excluded

**Test 5: Download Size Limit**
- Run `--recommend --max-download 3`
- Verify: No models with file_size > 3GB appear
- Verify: Smaller quants of the same model may appear (Q4 instead of Q8)

**Test 6: Constrained Hardware**
- Simulate 2GB VRAM, 4GB RAM (hardcode in profiler for testing)
- Verify: Only 1B–3B models appear
- Verify: "Hardware is very constrained" message is shown

**Test 7: Powerful Hardware**
- Simulate 24GB VRAM, 64GB RAM
- Verify: Larger models (14B, Mixtral) appear in the list
- Verify: Full GPU strategies dominate

**Test 8: Empty Catalog**
- Run with an empty catalog file
- Verify: Clear error message, no crash

**Test 9: Custom Catalog**
- Run with `--catalog custom.json` containing 2 models
- Verify: Only those 2 models appear in recommendations

**Test 10: Recommendation → Execution Flow**
- Run `--recommend`, note the top pick's URL
- Run `--model <url> --execute`
- Verify: The execution matches the recommendation's predicted strategy and performance

---

## Step 12 — Done Checklist

- [ ] Catalog JSON schema defined and validated
- [ ] Default catalog embedded in binary with 15–25 model variants
- [ ] Catalog includes models from at least 5 different families
- [ ] Catalog includes MoE models (Mixtral, DeepSeek)
- [ ] Quality scores sourced from real benchmarks and normalized to 0–10
- [ ] `--recommend` flag activates recommendation mode
- [ ] `--priority` flag works with speed, quality, and balanced
- [ ] `--use-case` flag filters by model tags
- [ ] `--max-download` flag excludes large models
- [ ] `--top` flag controls output count
- [ ] `--catalog` flag loads custom catalog
- [ ] Pre-filter eliminates models that don't fit on the hardware
- [ ] Mini method matrix generated per variant (2–4 strategies)
- [ ] Predictor runs for all surviving variants in <100ms total
- [ ] Cross-model ranking produces sensible order for all three priorities
- [ ] Labels assigned correctly (Best Overall, Fastest, Highest Quality, etc.)
- [ ] Output table includes model name, quant, strategy, VRAM, tok/s, quality, download size
- [ ] Recommendation includes copy-pasteable run command
- [ ] "Nothing fits" scenario handled gracefully with suggestions
- [ ] Constrained hardware shows appropriate warning
- [ ] Powerful hardware shows larger models
- [ ] Recommendation → execution flow works end-to-end
- [ ] `--recommend` and `--model` are mutually exclusive (error if both)
- [ ] Tested on at least 2 different hardware configurations

---

## Common Failure Points at Step 12

| Problem | Likely Cause | Fix |
|---|---|---|
| 1B model always ranks #1 | Speed score dominates because 1B is 5× faster than 7B | The balanced scoring must normalize speed with a ceiling. Above ~50 tok/s, additional speed has diminishing value. |
| 70B model appears on 8GB machine | Pre-filter viability check is wrong | Check that the pre-filter uses the smallest quant's memory footprint, not the largest. |
| All models show the same strategy | Mini matrix generator isn't considering different placements | Ensure the mini matrix includes full-GPU, split, and CPU-only options. |
| Quality scores seem arbitrary | No real benchmark data, just guesses | Source from Open LLM Leaderboard or Chatbot Arena. Cite the source. |
| Catalog is outdated | Shipped catalog doesn't include newest models | Add `--update-catalog` flag in a future step. For MVP, document the catalog version date. |
| Recommendation takes too long | Predicting all 23 variants × all strategies | Use the mini matrix (2–4 strategies per variant). Total predictions should be <100. |
| MoE models show wrong memory | Catalog dimensions don't include expert counts | Ensure MoE entries in the catalog have `is_moe: true` and the predictor uses the MoE memory formula. |
| Download URLs are broken | Hugging Face repos moved or renamed | Test all URLs periodically. Use the `/resolve/main/` path which follows redirects. |
| Star ratings don't match user expectations | Quality scores from benchmarks don't reflect real-world quality | Add a disclaimer: "Quality ratings are based on automated benchmarks and may not reflect your specific use case." |
| Balanced score ignores small models | Normalization ceiling too low | Tune the speed ceiling (~50 tok/s) based on user feedback. This is a subjective parameter. |

---

## Time Estimate for Step 12

| Phase | Work | Time |
|---|---|---|
| A | Catalog schema + initial catalog data (researching models, sourcing quality scores) | 1–1.5 days |
| B | Recommendation pipeline (pre-filter, mini matrix, cross-model ranking) | 1 day |
| C | Output formatting (table, labels, run command, edge cases) | 0.5–1 day |
| D | CLI interface (new flags, mutual exclusion with `--model`) | 0.5 day |
| E | Catalog data structures + loading logic | 0.5 day |
| F | Scoring function + label assignment | 0.5 day |
| G | Integration with main pipeline | 0.5 day |
| H | Testing (all 10 scenarios) | 1 day |

**Total: 3–5 days, as estimated. The most time-consuming part is Phase A — researching models, sourcing quality scores, and building the initial catalog. The code itself is straightforward because it reuses the predictor and ranker from Steps 3–5.**