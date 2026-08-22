#pragma once

#include "types.h"
#include <vector>

// =============================================================================
// Step 5 — Ranker: Three Priority Axes
// =============================================================================
// From spec: "Fastest, lowest-memory-pressure, and highest-quality are usually
//            three different strategies; picking one winner imposes an opinion."
//
// Priority modes:
//   SPEED   — tokens/sec desc, TTFT asc, confidence desc
//   QUALITY — bpw desc, kv_bits desc, gpu_layers desc
//   SAFETY  — memory headroom desc, confidence desc, tokens/sec desc
// =============================================================================

// Priority modes (reused by CLI and output)
enum class PriorityMode { SPEED, QUALITY, SAFETY };

// Parse priority string from CLI
PriorityMode parse_priority(const std::string& str);

// Get human-readable priority name
const char* get_priority_name(PriorityMode mode);

// Calculate memory headroom (0.0 to 1.0)
// From spec: min(vram_free - vram_used, ram_free - ram_used) / max(vram_free, ram_free)
// Higher = safer (more margin for background processes)
double calculate_memory_headroom(const HardwareSpec& hw, const Prediction& pred);

// Sort strategies by priority (main entry point)
void sort_by_priority(std::vector<StrategyResult>& results, PriorityMode priority,
                      const HardwareSpec& hw);
