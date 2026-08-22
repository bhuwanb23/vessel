// =============================================================================
// LLM Deployment Planner — Step 4+5: CLI & Orchestration
// =============================================================================
// This file is intentionally small. All logic lives in:
//   profiler.cpp  — hardware profiling (Step 1)
//   fetcher.cpp   — metadata fetching (Step 2)
//   matrix.cpp    — strategy generation (Step 4)
//   ranker.cpp    — priority sorting (Step 5)
//   output.cpp    — table formatting & printing
//   predictor.cpp — math formulas (Step 3)
// =============================================================================

#include "types.h"
#include "profiler.h"
#include "fetcher.h"
#include "matrix.h"
#include "ranker.h"
#include "output.h"
#include "../predictor/predictor.h"

#include <cstdio>
#include <string>
#include <chrono>

// Timer utility
class Timer {
    std::chrono::high_resolution_clock::time_point t0;
public:
    Timer() : t0(std::chrono::high_resolution_clock::now()) {}
    double elapsed_ms() {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(now - t0).count();
    }
};

// =============================================================================
// Main
// =============================================================================

int main(int argc, char* argv[]) {
    // --- Parse CLI arguments ---
    std::string model_url;       // URL or path from --model
    std::string model_local;     // Local path from --model-path
    PriorityMode priority    = PriorityMode::SPEED;
    ContextMode ctx_mode     = ContextMode::BOTH;
    bool verbose             = false;
    bool model_specified     = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage();
            return 0;
        } else if (arg == "--model" && i + 1 < argc) {
            model_url = argv[++i];
            model_specified = true;
        } else if (arg == "--model-path" && i + 1 < argc) {
            model_local = argv[++i];
        } else if (arg == "--priority" && i + 1 < argc) {
            std::string val = argv[++i];
            if (!is_valid_priority(val)) {
                fprintf(stderr, "Error: Invalid priority '%s'\n", val.c_str());
                fprintf(stderr, "  Valid options: speed, quality, safety\n");
                return 1;
            }
            priority = parse_priority(val);
        } else if (arg == "--context" && i + 1 < argc) {
            ctx_mode = parse_context(argv[++i]);
        } else if (arg == "--verbose") {
            verbose = true;
        } else if (arg[0] != '-' && !model_specified) {
            model_url = arg;
            model_specified = true;
        }
    }

    if (!model_specified) {
        fprintf(stderr, "Error: --model <url_or_path> is required\n\n");
        print_usage();
        return 1;
    }

    // --- Resolve model path ---
    std::string model_path = resolve_model_path(model_url, model_local);
    if (model_path.empty()) {
        fprintf(stderr, "\nError: Model file not found locally.\n");
        if (!model_local.empty()) {
            fprintf(stderr, "  Tried: %s\n", model_local.c_str());
        } else {
            // Extract filename from URL
            std::string filename = model_url;
            auto pos = model_url.find_last_of('/');
            if (pos != std::string::npos) filename = model_url.substr(pos + 1);
            fprintf(stderr, "  Expected: models/%s\n", filename.c_str());
        }
        fprintf(stderr, "\nDownload it first:\n");
        fprintf(stderr, "  huggingface-cli download <repo> <filename> --local-dir models/\n");
        fprintf(stderr, "\nOr specify the local path:\n");
        fprintf(stderr, "  llm-planner --model <url> --model-path ./models/<file>.gguf\n");
        return 1;
    }

    printf("\n");

    // --- Step 1: Profile Hardware ---
    Timer t_hw;
    HardwareSpec hw = profile_hardware(model_path);

    const ProfileErrors& pe = get_profile_errors();
    if (pe.gpu_failed && pe.ram_failed) {
        fprintf(stderr, "\nError: Could not detect any hardware subsystems.\n");
        fprintf(stderr, "Check: NVIDIA driver, CUDA Toolkit, system RAM.\n");
        return 1;
    }

    if (verbose) print_hardware_full(hw);
    else         print_hardware_brief(hw);
    print_warnings(hw);

    // --- Step 2: Fetch Model Metadata ---
    Timer t_meta;
    ModelSpec model = fetch_metadata(model_path);

    if (model.layers == 0) {
        const std::string& err = get_fetch_error();
        int http = get_fetch_http_status();

        fprintf(stderr, "\nError: Failed to fetch model metadata.\n");
        if (http > 0)        fprintf(stderr, "  HTTP Status: %d\n", http);
        if (!err.empty())    fprintf(stderr, "  Details: %s\n", err.c_str());
        fprintf(stderr, "  Source: %s\n", model_path.c_str());

        if (model_path.find("huggingface.co") != std::string::npos) {
            if (model_path.find(".gguf") == std::string::npos) {
                fprintf(stderr, "\nTip: Append the GGUF filename to the URL:\n");
                fprintf(stderr, "  .../resolve/main/Model-Q4_K_M.gguf\n");
            } else if (http == 404) {
                fprintf(stderr, "\nTip: File not found. Check for typos.\n");
            } else if (http == 403) {
                fprintf(stderr, "\nTip: Model may be gated. Accept the license first.\n");
            }
        }
        return 1;
    }

    if (model.source == MetadataSource::CONFIG_JSON) {
        fprintf(stderr, "\n⚠️  Using config.json fallback (lower confidence).\n");
        fprintf(stderr, "   For best results, use a pre-quantized GGUF file.\n");
    }

    if (verbose) print_model_full(model);
    else         print_model_brief(model);

    // --- Step 3: Generate Strategy Matrix ---
    Timer t_matrix;
    std::vector<StrategyResult> results = generate_matrix(hw, model);
    results = filter_by_context(results, ctx_mode, model.context_length);

    // --- Step 4: Rank by Priority ---
    sort_by_priority(results, priority, hw);

    // --- Step 5: Print Table ---
    print_prediction_table(results, hw, priority);
    print_post_table_warnings(results, hw);

    printf("\n=================================================\n");

    // --- Timing summary (verbose) ---
    if (verbose) {
        printf("\nTiming: HW=%.0fms  Meta=%.0fms  Matrix=%.1fms  Total=%.0fms\n",
               t_hw.elapsed_ms(), t_meta.elapsed_ms(),
               t_matrix.elapsed_ms(),
               t_hw.elapsed_ms() + t_meta.elapsed_ms() + t_matrix.elapsed_ms());
    }

    return 0;
}
