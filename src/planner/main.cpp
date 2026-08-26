// =============================================================================
// LLM Deployment Planner — Step 4+5+6: CLI & Orchestration
// =============================================================================
// This file is intentionally small. All logic lives in:
//   profiler.cpp  — hardware profiling (Step 1)
//   fetcher.cpp   — metadata fetching (Step 2)
//   matrix.cpp    — strategy generation (Step 4)
//   ranker.cpp    — priority sorting (Step 5)
//   output.cpp    — table formatting & printing
//   predictor.cpp — math formulas (Step 3)
//   executor.cpp  — llama.cpp inference (Step 6)
//   comparison_report.cpp — predicted vs actual (Step 6)
// =============================================================================

#include "types.h"
#include "profiler.h"
#include "fetcher.h"
#include "matrix.h"
#include "ranker.h"
#include "output.h"
#include "executor.h"
#include "comparison_report.h"
#include "calibration_log.h"
#include "calibration_aggregator.h"
#include "download_manager.h"
#include "../predictor/predictor.h"
#include "../hotcold/neuron_profiler.h"
#include "../hotcold/mask_file.h"
#include "platform/platform_factory.h"
#include "recommend/catalog_loader.h"
#include "recommend/recommendation_engine.h"

#include <cstdio>
#include <string>
#include <chrono>
#include <cstdlib>
#include <fstream>

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
// Default benchmark prompt
// =============================================================================

static const char* DEFAULT_PROMPT =
    "The following is a detailed explanation of how quantum computing works, "
    "step by step:";

// =============================================================================
// Main
// =============================================================================

int main(int argc, char* argv[]) {
    // Phase H: Register Ctrl+C handler for graceful abort
    register_abort_handler();

    // --- Parse CLI arguments ---
    std::string model_url;       // URL or path from --model
    std::string model_local;     // Local path from --model-path
    PriorityMode priority    = PriorityMode::SPEED;
    ContextMode ctx_mode     = ContextMode::BOTH;
    bool verbose             = false;
    bool model_specified     = false;
    bool execute_mode        = false;   // --execute: run inference after planning
    bool profile_neurons     = false;   // --profile-neurons: run neuron activation profiling
    std::string prompt       = DEFAULT_PROMPT;
    std::string download_dir;           // --download-dir override
    bool skip_verify         = false;   // --skip-verify: skip SHA256 verification
    int max_tokens           = 100;
    double hot_ratio         = 0.15;   // --hot-ratio: target hot neuron ratio
    uint64_t vram_budget     = 0;      // --vram-budget: VRAM for hot neurons (bytes)
    std::string platform_override;     // --platform: force specific platform (cuda, hip, metal, cpu)
    int gpu_index            = -1;     // --gpu: select specific GPU by index
    std::string gpu_name_pattern;      // --gpu-name: select GPU by name pattern
    bool recommend_mode      = false;  // --recommend: show model recommendations
    std::string use_case     = "all";  // --use-case: filter by use case
    int top_n                = 8;      // --top: number of recommendations to show

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage();
            return 0;
        } else if (arg == "--calibration-info") {
            // Print calibration log info and exit
            std::string log_path = get_log_path();
            auto records = read_all_records(log_path);
            printf("\n=== Calibration Log ===\n");
            printf("Location: %s\n", log_path.c_str());
            printf("Total entries: %zu\n", records.size());
            if (records.empty()) {
                printf("\nNo calibration data yet. Run with --execute to generate data.\n");
            } else {
                // Find a model file for NVMe detection in fingerprint
                std::string model_for_fingerprint;
                {
                    // Try common model directories for any .gguf file
                    std::ifstream test;
                    const char* candidates[] = {
                        "models/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
                        "../models/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
                        "./models/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
                        "C:/dev/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
                    };
                    for (const char* c : candidates) {
                        test.open(c);
                        if (test.is_open()) { model_for_fingerprint = c; test.close(); break; }
                    }
                }
                HardwareSpec hw_tmp = profile_hardware(model_for_fingerprint);
                int match_count = 0;
                for (const auto& r : records) {
                    if (r.hardware_fingerprint == hw_tmp.hardware_fingerprint) match_count++;
                }
                printf("Entries for this hardware: %d\n", match_count);
                printf("Hardware fingerprint: %s\n", hw_tmp.hardware_fingerprint.c_str());
                if (match_count > 0) {
                    CalibrationAggregator agg(hw_tmp.hardware_fingerprint);
                    CalibrationData cal = agg.get_calibration_data();
                    printf("\nCalibrated constants:\n");
                    printf("  GPU overhead:     %llu MB (default: 512 MB, from %d samples)\n",
                           (unsigned long long)(cal.adjusted_gpu_overhead_bytes / 1024 / 1024),
                           cal.overhead_gpu_records);
                    printf("  GPU decode eff:   %.3f (default: 0.270, from %d samples)\n",
                           cal.adjusted_gpu_decode_efficiency > 0 ? cal.adjusted_gpu_decode_efficiency : 0.27,
                           cal.decode_gpu_records);
                    printf("  CPU decode eff:   %.3f (default: 0.800, from %d samples)\n",
                           cal.adjusted_cpu_decode_efficiency > 0 ? cal.adjusted_cpu_decode_efficiency : 0.80,
                           cal.decode_cpu_records);
                    printf("  GPU prefill eff:  %.3f (default: 0.230, from %d samples)\n",
                           cal.adjusted_gpu_prefill_efficiency > 0 ? cal.adjusted_gpu_prefill_efficiency : 0.23,
                           cal.prefill_records);
                } else {
                    printf("\nNo entries match this hardware.\n");
                }
            }
            printf("\n");
            return 0;
        } else if (arg == "--calibration-reset") {
            // Delete calibration log with confirmation
            std::string log_path = get_log_path();
            auto records = read_all_records(log_path);
            if (records.empty()) {
                printf("No calibration log found at: %s\n", log_path.c_str());
                return 0;
            }
            printf("This will delete %zu calibration records.\n", records.size());
            printf("Location: %s\n", log_path.c_str());
            printf("Delete? (y/n): ");
            char confirm[8];
            if (!fgets(confirm, sizeof(confirm), stdin)) {
                printf("Cancelled.\n");
                return 0;
            }
            if (confirm[0] != 'y' && confirm[0] != 'Y') {
                printf("Cancelled.\n");
                return 0;
            }
            if (remove(log_path.c_str()) == 0) {
                printf("Calibration log deleted.\n");
            } else {
                fprintf(stderr, "Error: Could not delete %s\n", log_path.c_str());
            }
            return 0;
        } else if (arg == "--profile-neurons") {
            profile_neurons = true;
        } else if (arg == "--hot-ratio" && i + 1 < argc) {
            hot_ratio = atof(argv[++i]);
            if (hot_ratio <= 0.0 || hot_ratio > 1.0) hot_ratio = 0.15;
        } else if (arg == "--vram-budget" && i + 1 < argc) {
            // Accept as GB string (e.g., "4GB" or "4")
            std::string val = argv[++i];
            double gb = atof(val.c_str());
            vram_budget = static_cast<uint64_t>(gb * 1024.0 * 1024.0 * 1024.0);
        } else if (arg == "--model" && i + 1 < argc) {
            model_url = argv[++i];
            model_specified = true;
        } else if (arg == "--model-path" && i + 1 < argc) {
            model_local = argv[++i];
        } else if (arg == "--download-dir" && i + 1 < argc) {
            download_dir = argv[++i];
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
        } else if (arg == "--prompt" && i + 1 < argc) {
            prompt = argv[++i];
        } else if (arg == "--max-tokens" && i + 1 < argc) {
            max_tokens = atoi(argv[++i]);
            if (max_tokens <= 0) max_tokens = 100;
        } else if (arg == "--execute") {
            execute_mode = true;
        } else if (arg == "--skip-verify") {
            skip_verify = true;
        } else if (arg == "--platform" && i + 1 < argc) {
            platform_override = argv[++i];
        } else if (arg == "--gpu" && i + 1 < argc) {
            gpu_index = atoi(argv[++i]);
            if (gpu_index < 0) gpu_index = -1;
        } else if (arg == "--gpu-name" && i + 1 < argc) {
            gpu_name_pattern = argv[++i];
        } else if (arg == "--recommend") {
            recommend_mode = true;
        } else if (arg == "--use-case" && i + 1 < argc) {
            use_case = argv[++i];
        } else if (arg == "--top" && i + 1 < argc) {
            top_n = atoi(argv[++i]);
            if (top_n <= 0) top_n = 8;
            if (top_n > 20) top_n = 20;
        } else if (arg == "--verbose") {
            verbose = true;
        } else if (arg[0] != '-' && !model_specified) {
            model_url = arg;
            model_specified = true;
        }
    }

    if (!model_specified && !recommend_mode) {
        fprintf(stderr, "Error: --model <url_or_path> is required (or use --recommend)\n\n");
        print_usage();
        return 1;
    }

    // --- Skip URL resolution in recommend mode ---
    bool is_url = false;
    std::string model_path;  // local file path (empty if URL)

    if (!recommend_mode) {
        is_url = (model_url.find("http://") == 0 || model_url.find("https://") == 0);

        if (is_url) {
            // URL mode: use --model-path for execution, URL for metadata
            if (!model_local.empty()) {
                model_path = model_local;
                { std::ifstream test(model_path);
                  if (!test.is_open()) {
                    fprintf(stderr, "\nError: Model file not found: %s\n", model_path.c_str());
                    return 1;
                  }
                }
            }
            // model_path may be empty — that's fine for advisor mode
        } else {
            // Local path mode
            model_path = resolve_model_path(model_url, model_local);
            if (model_path.empty()) {
                fprintf(stderr, "\nError: Model file not found locally.\n");
                fprintf(stderr, "  Tried: %s\n", model_url.c_str());
                fprintf(stderr, "\nDownload it first:\n");
                fprintf(stderr, "  huggingface-cli download <repo> <filename> --local-dir models/\n");
                return 1;
            }
        }
    }

    printf("\n");

    // =========================================================================
    // --recommend: Model Recommendation Mode
    // =========================================================================
    if (recommend_mode) {
        // Step 1: Profile hardware
        std::string model_for_fingerprint;
        {
            std::ifstream test;
            const char* candidates[] = {
                "models/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
                "../models/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
                "./models/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
                "C:/dev/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
            };
            for (const char* c : candidates) {
                test.open(c);
                if (test.is_open()) { model_for_fingerprint = c; test.close(); break; }
            }
        }
        
        Timer t_hw;
        HardwareSpec hw = profile_hardware(model_for_fingerprint);
        double hw_time = t_hw.elapsed_ms();
        
        if (verbose) {
            print_hardware_full(hw);
        } else {
            print_hardware_brief(hw);
        }
        
        // Step 2: Load catalog
        ModelCatalog catalog = load_builtin_catalog();
        if (catalog.models.empty()) {
            fprintf(stderr, "Error: Model catalog is empty.\n");
            return 1;
        }
        
        // Step 3: Generate recommendations
        RecommendationRequest req;
        req.priority = (priority == PriorityMode::SPEED) ? "speed" :
                       (priority == PriorityMode::QUALITY) ? "quality" : "balanced";
        req.use_case = use_case;
        req.top_n = top_n;
        
        Timer t_rec;
        auto recs = generate_recommendations(hw, catalog, req);
        double rec_time = t_rec.elapsed_ms();
        
        // Step 4: Print recommendation table
        print_recommendation_table(recs, hw, req.priority, req.use_case);
        
        printf("\n(Profiling: %.0f ms, Recommendations: %.0f ms)\n",
               hw_time, rec_time);
        
        return 0;
    }

    // =========================================================================
    // --profile-neurons: Neuron Activation Profiling Mode
    // =========================================================================
    if (profile_neurons) {
        printf("=== Neuron Activation Profiling ===\n");
        printf("Model:     %s\n", model_url.c_str());
        printf("Hot ratio: %.0f%%\n", hot_ratio * 100.0);
        if (vram_budget > 0) {
            printf("VRAM budget: %.1f GB\n", vram_budget / 1e9);
        }
        printf("\n");

        // Determine model path
        std::string profile_model_path = model_path;
        if (profile_model_path.empty() && is_url) {
            // Try to find the model locally
            profile_model_path = model_url;
            // Check if file exists
            std::ifstream test(profile_model_path);
            if (!test.is_open()) {
                fprintf(stderr, "Error: --profile-neurons requires a local model file.\n");
                fprintf(stderr, "  Download the model first, then use --model-path.\n");
                return 1;
            }
        }

        if (profile_model_path.empty()) {
            fprintf(stderr, "Error: --profile-neurons requires a model file path.\n");
            return 1;
        }

        // Run profiling
        ProfilingConfig config;
        config.model_path = profile_model_path;
        config.prompts_path = "";  // Use bundled prompts
        config.max_prompts = 500;
        config.hot_ratio = hot_ratio;
        config.activation = ActivationType::SILU;
        config.activation_threshold = 0.01f;
        config.n_gpu_layers = 0;  // CPU-only for profiling
        config.vram_budget_bytes = vram_budget;
        config.validate_stability = true;

        // Progress callback
        auto progress = [](const ProfilingProgress& p) {
            if (p.current_prompt > 0 && p.current_prompt % 10 == 0) {
                fprintf(stderr, "\r[Profiling] Prompt %u/%u | Tokens: %llu | ETA: %.0fs ",
                        p.current_prompt, p.total_prompts,
                        (unsigned long long)p.tokens_processed,
                        p.estimated_remaining_seconds);
            }
        };

        printf("Starting neuron profiling...\n");
        printf("This may take 1-4 hours depending on model size.\n\n");

        ProfilingResult result = run_neuron_profiling(config, progress);
        fprintf(stderr, "\n");

        if (!result.success) {
            fprintf(stderr, "Error: Profiling failed: %s\n", result.error_message.c_str());
            return 1;
        }

        // Print results
        printf("=== Profiling Complete ===\n");
        printf("Time:      %.1f seconds\n", result.total_seconds);
        printf("Tokens:    %llu\n", (unsigned long long)result.total_tokens);
        printf("Layers:    %u\n", result.profile.num_layers);
        printf("FFN dim:   %u\n", result.profile.ffn_dim);
        printf("Hot ratio: %.0f%%\n", result.profile.hot_ratio * 100.0);
        
        if (!result.profile.layers.empty()) {
            printf("Hot/layer: %u / %u (%.0f%%)\n",
                   result.profile.layers[0].n_hot,
                   result.profile.ffn_dim,
                   100.0 * result.profile.layers[0].n_hot / result.profile.ffn_dim);
        }

        printf("\nMask file: %s\n", get_mask_file_path(profile_model_path).c_str());

        if (config.validate_stability) {
            printf("Stability: %.1f%% overlap (%s)\n",
                   result.stability_overlap * 100.0,
                   result.stability_passed ? "PASS" : "FAIL");
        }

        printf("\n");
        return 0;
    }

    // --- Step 1: Profile Hardware ---
    Timer t_hw;
    
    // Platform auto-detection (Phase F)
    std::unique_ptr<IHardwareProfiler> platform_profiler;
    if (!platform_override.empty()) {
        // Manual platform override (--platform flag)
        platform_profiler = create_platform_profiler_for_platform(platform_override);
        if (!platform_profiler) {
            fprintf(stderr, "Error: Platform '%s' is not available on this system.\n",
                    platform_override.c_str());
            fprintf(stderr, "  Run without --platform to auto-detect.\n");
            return 1;
        }
    } else {
        // Auto-detect best available platform
        platform_profiler = create_platform_profiler_auto();
    }
    
    // Profile hardware using the detected/specified platform
    HardwareSpec hw;
    if (platform_profiler && platform_profiler->isAvailable()) {
        hw = platform_profiler->profile(model_path);
        
        // Multi-GPU selection (--gpu flag)
        if (gpu_index >= 0 && platform_profiler->supportsMultiGPU()) {
            uint32_t gpu_count = platform_profiler->getGPUCount();
            if (static_cast<uint32_t>(gpu_index) >= gpu_count) {
                fprintf(stderr, "Error: GPU index %d out of range (0-%d).\n",
                        gpu_index, gpu_count - 1);
                return 1;
            }
            // TODO: Re-profile with specific GPU index
            fprintf(stderr, "[PlatformFactory] Selected GPU %d of %d\n",
                    gpu_index, gpu_count);
        }
        
        // GPU name pattern matching (--gpu-name flag)
        if (!gpu_name_pattern.empty()) {
            // Find GPU matching the pattern
            auto gpus = enumerate_gpus();
            bool found = false;
            for (const auto& gpu : gpus) {
                if (gpu.name.find(gpu_name_pattern) != std::string::npos) {
                    fprintf(stderr, "[PlatformFactory] Matched GPU: %s\n", gpu.name.c_str());
                    found = true;
                    break;
                }
            }
            if (!found) {
                fprintf(stderr, "Warning: No GPU matching '%s' found.\n",
                        gpu_name_pattern.c_str());
            }
        }
    } else {
        // Fallback to legacy profiler if platform factory fails
        hw = profile_hardware(model_path);
    }

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
    std::string metadata_source = is_url ? model_url : model_path;
    ModelSpec model = fetch_metadata(metadata_source);

    if (model.layers == 0) {
        const std::string& err = get_fetch_error();
        int http = get_fetch_http_status();

        fprintf(stderr, "\nError: Failed to fetch model metadata.\n");
        if (http > 0)        fprintf(stderr, "  HTTP Status: %d\n", http);
        if (!err.empty())    fprintf(stderr, "  Details: %s\n", err.c_str());
        fprintf(stderr, "  Source: %s\n", metadata_source.c_str());

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
        fprintf(stderr, "\nWarning: Using config.json fallback (lower confidence).\n");
        fprintf(stderr, "   For best results, use a pre-quantized GGUF file.\n");
    }

    if (verbose) print_model_full(model);
    else         print_model_brief(model);

    // --- Step 7: Load Calibration Data ---
    CalibrationAggregator cal_agg(hw.hardware_fingerprint);
    CalibrationData cal = cal_agg.get_calibration_data();
    if (cal.has_calibration_data && verbose) {
        printf("\n--- Calibration Data ---\n");
        printf("Records:        %d matching / %d total\n",
               cal.matching_record_count, cal.total_record_count);
        printf("GPU overhead:   %.0f MB (default: 512 MB)\n",
               cal.adjusted_gpu_overhead_bytes / 1e6);
        printf("GPU decode eff: %.3f (default: 0.270)\n",
               cal.adjusted_gpu_decode_efficiency);
        printf("CPU decode eff: %.3f (default: 0.800)\n",
               cal.adjusted_cpu_decode_efficiency);
        printf("GPU prefill eff: %.3f (default: 0.230)\n",
               cal.adjusted_gpu_prefill_efficiency);
    } else if (verbose) {
        printf("\nNo calibration data for this hardware. Using default constants.\n");
    }

    // --- Step 3: Generate Strategy Matrix (with calibration) ---
    Timer t_matrix;
    std::vector<StrategyResult> results = generate_matrix(hw, model, cal);
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

    // =========================================================================
    // Step 6: Interactive Strategy Selection & Execution
    // =========================================================================
    if (!execute_mode) {
        // Plan-only mode: just print the table and exit
        return 0;
    }

    // --- Step 8: Download Manager (URL mode only) ---
    if (is_url && model_path.empty()) {
        // Resolve download directory
        std::string target_dir = download_dir.empty() ? get_default_download_dir() : download_dir;
        if (!ensure_download_dir(target_dir)) {
            fprintf(stderr, "Error: Could not create download directory: %s\n", target_dir.c_str());
            return 1;
        }

        std::string filename = extract_filename_from_url(model_url);
        printf("Checking for model file...\n");

        // --- Scenario 1: Model already on disk ---
        std::string local_candidate = target_dir + "\\" + filename;
        { std::ifstream test(local_candidate);
          if (test.is_open()) {
            model_path = local_candidate;
            printf("  Found: %s\n", model_path.c_str());
          }
        }

        if (model_path.empty()) {
            // --- Phase D: Detect shards ---
            ModelShards shards = detect_model_shards(model_url);
            if (!shards.error_message.empty()) {
                fprintf(stderr, "  Error detecting shards: %s\n", shards.error_message.c_str());
                return 1;
            }

            if (shards.is_sharded) {
                // Multi-shard: check if all shards exist
                if (all_shards_present(shards, target_dir)) {
                    model_path = get_first_shard_path(shards, target_dir);
                    printf("  Found all %d shards: %s\n", shards.total_shards, model_path.c_str());
                } else {
                    // Need to download shards
                    printf("  Not found. Model has %d shards (%.1f GB total).\n",
                           shards.total_shards, shards.total_size_bytes / 1e9);

                    // Disk space check
                    uint64_t required = (uint64_t)(shards.total_size_bytes * 1.15);
                    uint64_t available = get_disk_free_bytes(target_dir);
                    if (available < required) {
                        fprintf(stderr, "\nInsufficient disk space.\n");
                        fprintf(stderr, "   Required: %.1f GB | Available: %.1f GB | Shortfall: %.1f GB\n",
                                required / 1e9, available / 1e9, (required - available) / 1e9);
                        return 1;
                    }

                    printf("  Downloading %d shards...\n", shards.total_shards);
                    bool ok = download_all_shards(shards, target_dir, abort_requested, skip_verify);
                    if (!ok) { fprintf(stderr, "\nDownload failed or paused.\n"); return 1; }
                    model_path = get_first_shard_path(shards, target_dir);
                    printf("  Model ready: %s (all %d shards)\n", model_path.c_str(), shards.total_shards);
                }
            } else {
                // --- Scenario 2/3: Single-file model ---
                // Check for partial file (resume scenario)
                std::string partial = get_partial_path(model_url, target_dir);
                uint64_t partial_size = get_partial_file_size(partial);

                // Get expected file size
                uint64_t file_size = get_file_size_via_head(model_url);
                if (file_size == 0) {
                    file_size = estimate_file_size_from_metadata(model);
                }

                if (partial_size > 0 && file_size > 0 && partial_size < file_size) {
                    // Scenario 3: Partial download exists
                    printf("  Found partial download (%.2f / %.2f GB, %.0f%% complete).\n",
                           partial_size / 1e9, file_size / 1e9, 100.0 * partial_size / file_size);
                } else {
                    printf("  Not found locally.\n");
                }

                // Disk space check
                uint64_t required = (uint64_t)(file_size * 1.15);
                uint64_t available = get_disk_free_bytes(target_dir);
                if (available < required) {
                    fprintf(stderr, "\nInsufficient disk space.\n");
                    fprintf(stderr, "   Required: %.1f GB | Available: %.1f GB | Shortfall: %.1f GB\n",
                            required / 1e9, available / 1e9, (required - available) / 1e9);
                    // Suggest smaller quant
                    std::string smaller = find_smaller_quant_that_fits(available, model);
                    if (!smaller.empty()) {
                        double smaller_bpw = get_bits_per_weight(smaller);
                        uint64_t smaller_size = (smaller_bpw > 0)
                            ? (uint64_t)((double)model.param_count * smaller_bpw / 8.0 * 1.05) : 0;
                        fprintf(stderr, "   Suggestion: %s quantization (~%.1f GB) would fit.\n",
                                smaller.c_str(), smaller_size / 1e9);
                    }
                    return 1;
                }

                // Download
                printf("  Downloading %s...\n", filename.c_str());
                DownloadResult dl = download_model_file(
                    model_url, target_dir, file_size, abort_requested, skip_verify);

                if (dl.paused) { printf("\nDownload paused. Re-run to resume.\n"); return 0; }
                if (!dl.success) { fprintf(stderr, "  Download failed: %s\n", dl.error_message.c_str()); return 1; }

                model_path = dl.final_path;
                printf("  Model ready: %s\n", model_path.c_str());
            }
        }
    }

    // For URL mode, require local model after pre-download check
    if (is_url && model_path.empty()) {
        fprintf(stderr, "\nError: Model file not found locally.\n");
        fprintf(stderr, "  Download the model first, then use --model-path.\n");
        return 1;
    }

    // Find viable strategies for selection
    int viable_count = 0;
    int total_count = 0;
    for (size_t i = 0; i < results.size(); i++) {
        StrategyStatus st = determine_status(hw, results[i].prediction, results[i].strategy);
        total_count++;
        if (st == StrategyStatus::VIABLE || st == StrategyStatus::TIGHT)
            viable_count++;
    }

    if (viable_count == 0) {
        fprintf(stderr, "\nNo viable strategies. Cannot execute.\n");
        return 1;
    }

    // Prompt user to select
    printf("\nSelect a strategy to execute (1-%d), or 'q' to quit: ",
           total_count);

    char input[32];
    if (!fgets(input, sizeof(input), stdin)) {
        printf("\nNo input. Exiting.\n");
        return 0;
    }

    // Parse input
    std::string input_str(input);
    // Strip trailing newline
    while (!input_str.empty() && (input_str.back() == '\n' || input_str.back() == '\r'))
        input_str.pop_back();

    if (input_str == "q" || input_str == "Q" || input_str.empty()) {
        printf("Exiting.\n");
        return 0;
    }

    int selection = 0;
    try {
        selection = std::stoi(input_str);
    } catch (...) {
        fprintf(stderr, "Invalid input. Exiting.\n");
        return 1;
    }

    if (selection < 1 || selection > total_count) {
        fprintf(stderr, "Error: Strategy #%d is out of range (1-%d).\n",
                selection, total_count);
        return 1;
    }

    // Get the selected strategy
    const StrategyResult& selected = results[selection - 1];
    StrategyStatus sel_status = determine_status(hw, selected.prediction, selected.strategy);

    // Warn if non-viable
    if (sel_status == StrategyStatus::NO_FIT) {
        uint64_t vram_needed = selected.prediction.memory_vram_bytes;
        uint64_t vram_have = hw.vram_free_bytes;
        double vram_exceed_gb = (vram_needed > vram_have)
            ? (vram_needed - vram_have) / 1e9 : 0.0;

        fprintf(stderr, "\nWarning: Strategy #%d is not viable", selection);
        if (vram_exceed_gb > 0)
            fprintf(stderr, " (exceeds VRAM by %.1f GB)", vram_exceed_gb);
        fprintf(stderr, ".\n");
        fprintf(stderr, "   The model will likely fail to load. Execute anyway? (y/n): ");

        char confirm[8];
        if (!fgets(confirm, sizeof(confirm), stdin)) {
            printf("No input. Exiting.\n");
            return 0;
        }
        if (confirm[0] != 'y' && confirm[0] != 'Y') {
            printf("Cancelled.\n");
            return 0;
        }
    }

    // Print what we're about to do
    const auto& strat = selected.strategy;
    const char* placement_name;
    switch (strat.placement) {
        case PlacementStrategy::FULL_GPU:       placement_name = "Full GPU"; break;
        case PlacementStrategy::GPU_CPU_SPLIT:  placement_name = "GPU+CPU Split"; break;
        case PlacementStrategy::CPU_ONLY:       placement_name = "CPU Only"; break;
        case PlacementStrategy::HOT_COLD_SPLIT: placement_name = "Hot/Cold Split"; break;
        case PlacementStrategy::LAYER_STREAM:   placement_name = "Layer-Stream"; break;
        default:                               placement_name = "Unknown"; break;
    }

    printf("\n--- Executing Strategy #%d ---\n", selection);
    printf("Placement:  %s (%u/%u layers)\n", placement_name, strat.gpu_layers, model.layers);
    printf("Context:    %uK\n", strat.context_length / 1024);
    printf("KV Cache:   %s\n", strat.kv_quant_bits == 16 ? "FP16" : "Q8");
    printf("Prompt:     \"%s\"\n", prompt.c_str());
    printf("Max tokens: %d\n\n", max_tokens);

    // Initialize executor
    printf("Initializing executor...\n");
    if (!executor_init()) {
        fprintf(stderr, "Failed to initialize executor.\n");
        return 1;
    }

    // Run inference
    printf("Running inference...\n\n");
    ExecutionResult result = execute(
        model_path,
        strat,
        prompt,
        max_tokens,
        nullptr
    );

    if (!result.success) {
        fprintf(stderr, "\nExecution failed: %s\n", result.error_message.c_str());
        executor_shutdown();
        return 1;
    }

    // Print raw results
    printf("--- Raw Results ---\n");
    printf("Prompt eval:  %.1f ms (%.1f tok/s)\n",
           result.prompt_eval_ms, result.prompt_eval_tokens_per_sec);
    printf("Decode:       %.1f ms (%.1f tok/s)\n",
           result.decode_ms, result.decode_tokens_per_sec);
    printf("Tokens:       %d generated\n", result.tokens_generated);
    printf("Peak VRAM:    %.2f GB\n", result.peak_vram_used_bytes / 1e9);
    printf("Peak RAM:     %.2f GB\n", result.peak_ram_used_bytes / 1e9);
    printf("Peak Temp:    %.0f C\n", result.peak_gpu_temp_c);
    printf("Throttled:    %s\n", result.throttled ? "YES" : "No");
    printf("Output:       \"%s\"\n", result.generated_text.c_str());

    // Print comparison report (use calibrated predictions)
    Prediction prediction = predict(hw, model, strat, cal);
    print_comparison_report(prediction, result, strat);

    // Write calibration log (Phase C)
    // Model ID: extract from URL or use filename
    std::string model_id;
    {
        auto pos = model_url.find_last_of('/');
        if (pos != std::string::npos) {
            model_id = model_url.substr(pos + 1);  // filename only
        } else {
            model_id = model_url;
        }
    }
    write_calibration_entry(hw, model, strat, prediction, result, model_id);

    // Shutdown
    executor_shutdown();
    printf("\nDone.\n");

    return 0;
}
