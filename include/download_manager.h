#pragma once

#include "types.h"
#include <string>
#include <vector>

// =============================================================================
// Download Manager — Step 8 Phase A: Pre-Download Safety Check
// =============================================================================
// Before downloading a GGUF file, verify:
//   1. We know the file size (HEAD request, Content-Range, or metadata estimate)
//   2. The target drive has enough space (with 1.15x safety margin)
//   3. If not, suggest a smaller quant that fits
// =============================================================================

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

// =============================================================================
// Public API
// =============================================================================

// Perform a pre-download safety check.
// Uses HTTP HEAD to get exact file size, checks disk space, suggests alternatives.
//
// Parameters:
//   url          — HuggingFace GGUF URL (for HEAD request)
//   model        — ModelSpec for fallback size estimation
//   download_dir — Target directory for the download
//
// Returns a PreDownloadCheck with pass/fail and details.
PreDownloadCheck pre_download_check(const std::string& url,
                                     const ModelSpec& model,
                                     const std::string& download_dir);

// Get file size via HTTP HEAD request (Content-Length).
// Returns 0 on failure.
uint64_t get_file_size_via_head(const std::string& url);

// Estimate file size from model metadata (fallback).
// formula: param_count × bits_per_weight / 8 × 1.05
uint64_t estimate_file_size_from_metadata(const ModelSpec& model);

// Check available disk space on a directory's drive.
// Uses GetDiskFreeSpaceExW. Returns 0 on failure.
uint64_t get_disk_free_bytes(const std::string& dir);

// Ensure the download directory exists (create if missing).
// Returns true if directory exists or was created successfully.
bool ensure_download_dir(const std::string& dir);

// Find a smaller quantization that would fit in the available space.
// Returns the quant type name (e.g., "Q3_K_M") or empty string if nothing fits.
std::string find_smaller_quant_that_fits(uint64_t available_bytes,
                                          const ModelSpec& model);

// Extract filename from a URL.
std::string extract_filename_from_url(const std::string& url);

// Default download directory
std::string get_default_download_dir();
