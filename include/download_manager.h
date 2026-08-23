#pragma once

#include "types.h"
#include <string>
#include <vector>
#include <atomic>

// =============================================================================
// Download Manager — Step 8: Pre-Download Check + Resumable Download
// =============================================================================

// ---------------------------------------------------------------------------
// Phase A: Pre-Download Safety Check
// ---------------------------------------------------------------------------

// Where did we get the file size from?
enum class SizeSource {
    HEAD_REQUEST,       // HTTP HEAD → Content-Length (most reliable)
    CONTENT_RANGE,      // Content-Range header from a range request
    METADATA_ESTIMATE,  // param_count × bpw / 8 × 1.05 (fallback)
    UNKNOWN             // Could not determine
};

// Result of the pre-download safety check
struct PreDownloadCheck {
    bool pass = false;              // true if safe to download
    uint64_t file_size_bytes = 0;   // estimated file size
    uint64_t required_bytes = 0;    // file_size × 1.15 (with safety margin)
    uint64_t available_bytes = 0;   // free space on target drive
    std::string target_dir;         // resolved download directory
    SizeSource size_source = SizeSource::UNKNOWN;
    std::string error_message;      // human-readable error if !pass
    std::string suggestion;         // alternative quant that fits (if any)
};

// ---------------------------------------------------------------------------
// Phase B: Resumable Download
// ---------------------------------------------------------------------------

// Result of a download attempt
struct DownloadResult {
    bool success = false;           // true if download completed + verified
    bool paused = false;            // true if user pressed Ctrl+C (resume possible)
    std::string final_path;         // path to the verified .gguf file
    std::string partial_path;       // path to the .partial file (if paused)
    uint64_t bytes_downloaded = 0;  // total bytes written to disk
    uint64_t file_size = 0;         // expected total size
    std::string error_message;      // human-readable error if !success
};

// =============================================================================
// Public API — Phase A
// =============================================================================

// Perform a pre-download safety check.
PreDownloadCheck pre_download_check(const std::string& url,
                                     const ModelSpec& model,
                                     const std::string& download_dir);

// Get file size via HTTP HEAD request (Content-Length).
uint64_t get_file_size_via_head(const std::string& url);

// Estimate file size from model metadata (fallback).
uint64_t estimate_file_size_from_metadata(const ModelSpec& model);

// Check available disk space on a directory's drive.
uint64_t get_disk_free_bytes(const std::string& dir);

// Ensure the download directory exists (create if missing).
bool ensure_download_dir(const std::string& dir);

// Find a smaller quantization that would fit in the available space.
std::string find_smaller_quant_that_fits(uint64_t available_bytes,
                                          const ModelSpec& model);

// Extract filename from a URL.
std::string extract_filename_from_url(const std::string& url);

// Default download directory
std::string get_default_download_dir();

// =============================================================================
// Public API — Phase B
// =============================================================================

// Download a GGUF file with resumable support.
// Uses .partial file strategy: writes to <name>.gguf.partial during download,
// renames to <name>.gguf after successful completion.
// Supports resume: if .partial exists, continues from where it left off.
//
// Parameters:
//   url          — HuggingFace GGUF URL
//   download_dir — Target directory
//   file_size    — Expected file size (from HEAD request, 0 = auto-detect)
//   abort_flag   — Atomic flag: set to true to cancel download gracefully
//
// Returns DownloadResult with success/fail and file paths.
DownloadResult download_model_file(const std::string& url,
                                    const std::string& download_dir,
                                    uint64_t file_size,
                                    std::atomic<bool>& abort_flag);

// Get the .partial file path for a given URL and directory.
std::string get_partial_path(const std::string& url, const std::string& dir);

// Get the final .gguf file path for a given URL and directory.
std::string get_final_path(const std::string& url, const std::string& dir);

// Check if a .partial file exists and return its size.
// Returns 0 if no .partial file exists.
uint64_t get_partial_file_size(const std::string& partial_path);
