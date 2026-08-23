#pragma once

#include "hotcold_types.h"
#include <string>
#include <cstdint>

// =============================================================================
// Step 10 Phase B — Hot Neuron Mask File Format
// =============================================================================
// Binary format for storing per-layer hot neuron masks.
// File extension: .hot_neurons.bin
//
// Format:
//   Header (16 bytes):
//     magic:     "HOTM" (4 bytes)
//     version:   uint32 (4 bytes) — currently 1
//     n_layers:  uint32 (4 bytes)
//     ffn_dim:   uint32 (4 bytes)
//
//   Metadata:
//     model_name_len: uint32
//     model_name:     char[n]
//     activation:     uint32 (ActivationType enum)
//     hot_ratio:      double
//     n_prompts:      uint32
//     avg_act_rate:   double
//
//   Per layer (n_layers entries):
//     n_hot:      uint32
//     hot_mask:   uint8[ceil(ffn_dim / 8)]  — bit vector, 1=hot, 0=cold
//     hot_count:  uint32 (redundant, for quick validation)
// =============================================================================

// Save a HotNeuronProfile to the binary mask file format
bool save_mask_file(const HotNeuronProfile& profile, const std::string& path);

// Load a HotNeuronProfile from the binary mask file format
HotNeuronProfile load_mask_file(const std::string& path);

// Validate a mask file (check magic, version, consistency)
// Returns empty string on success, error message on failure
std::string validate_mask_file(const std::string& path);

// Generate the default mask file path from a model path
// e.g., "models/Llama-7B.gguf" → "models/Llama-7B.hot_neurons.bin"
std::string get_mask_file_path(const std::string& model_path);

// Check if a mask file exists for a given model
bool mask_file_exists(const std::string& model_path);

// Get mask file info (print metadata without loading full data)
// Returns true if file is valid
bool print_mask_file_info(const std::string& path);
