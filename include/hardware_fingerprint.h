#pragma once

#include "types.h"
#include <string>

// =============================================================================
// Hardware Fingerprint — Step 7 Calibration Log
// =============================================================================
// Generates a stable string key that identifies a hardware configuration.
// Used to group calibration records by hardware class.
//
// Format: "<cpu_model>|<gpu_model>|<ram_total>|<nvme_model>"
//
// The fingerprint is stable across reboots and driver updates because it
// strips volatile info (clock speeds, temperatures, exact RAM amounts).
// =============================================================================

// Generate a hardware fingerprint from a HardwareSpec.
// The fingerprint is a pipe-separated string that uniquely identifies
// the hardware class (not the specific run).
//
// Example: "i7-12700K|NVIDIA GeForce RTX 3080|32GB|Samsung 980 PRO"
std::string generateHardwareFingerprint(const HardwareSpec& hw);

// Overload that accepts a file path for NVMe model detection.
// The file path is used to determine which physical drive to query.
std::string generateHardwareFingerprint(const HardwareSpec& hw, const std::string& model_file_path);

// Normalize CPU model string by stripping trademarks, clock speed, "CPU" suffix.
// "Intel(R) Core(TM) i7-12700K @ 3.60GHz" -> "i7-12700K"
std::string normalizeCpuModel(const std::string& raw);

// Normalize GPU model string (already stable from NVML, but strip "NVIDIA" prefix).
// "NVIDIA GeForce RTX 3080" -> "GeForce RTX 3080"
std::string normalizeGpuModel(const std::string& raw);

// Round RAM total bytes to nearest power-of-two GB.
// 31.8 GB -> "32GB", 15.7 GB -> "16GB"
std::string normalizeRamTotal(uint64_t ram_bytes);

// Extract NVMe model name from the drive containing a given file path.
// Uses Windows DeviceIoControl to query STORAGE_DEVICE_DESCRIPTOR.
// Returns empty string on failure.
std::string getNvmeModel(const std::string& file_path);
