#pragma once

#include "types.h"

// =============================================================================
// Hardware Profiler (Step 1)
// =============================================================================
// Profiles the system hardware and returns a HardwareSpec struct.
// No model loading, no inference, no networking.
// =============================================================================

// Profile all hardware subsystems and return complete HardwareSpec
HardwareSpec profile_hardware(const std::string& model_path_for_disk_bench = "");

// Profile RAM only
uint64_t profile_ram_total();
uint64_t profile_ram_free();

// Profile GPU only (first GPU)
std::string profile_gpu_name();
uint64_t profile_vram_total();
uint64_t profile_vram_free();
double profile_gpu_bandwidth();

// Profile disk only
double profile_disk_sequential(const std::string& file_path);
double profile_disk_random_4k(const std::string& file_path);
