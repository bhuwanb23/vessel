#pragma once

#include <cstdint>
#include <string>

// Result of RAM profiling
struct RamProfile {
    double total_ram_gb;        // Total physical RAM in GB
    double available_ram_gb;    // Available RAM in GB (includes reclaimable standby cache)
    double total_pagefile_gb;   // Total commit charge limit in GB
    double available_pagefile_gb; // Available commit charge in GB
    uint64_t total_ram_bytes;   // Raw value for calculations
    uint64_t available_ram_bytes;
};

// Profile system RAM using Windows GlobalMemoryStatusEx()
// Returns a populated RamProfile struct.
RamProfile profile_ram();

// Print RAM profile to stdout in a readable format
void print_ram_profile(const RamProfile& profile);
