#include "ram_profiler.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <cstdio>

static constexpr double BYTES_PER_GB = 1024.0 * 1024.0 * 1024.0;

RamProfile profile_ram() {
    RamProfile profile = {};

    MEMORYSTATUSEX mem_status;
    mem_status.dwLength = sizeof(MEMORYSTATUSEX);

    if (!GlobalMemoryStatusEx(&mem_status)) {
        fprintf(stderr, "Error: GlobalMemoryStatusEx() failed (error %lu)\n", GetLastError());
        return profile;
    }

    profile.total_ram_bytes = mem_status.ullTotalPhys;
    profile.available_ram_bytes = mem_status.ullAvailPhys;
    profile.total_ram_gb = static_cast<double>(mem_status.ullTotalPhys) / BYTES_PER_GB;
    profile.available_ram_gb = static_cast<double>(mem_status.ullAvailPhys) / BYTES_PER_GB;
    profile.total_pagefile_gb = static_cast<double>(mem_status.ullTotalPageFile) / BYTES_PER_GB;
    profile.available_pagefile_gb = static_cast<double>(mem_status.ullAvailPageFile) / BYTES_PER_GB;

    return profile;
}

void print_ram_profile(const RamProfile& profile) {
    printf("=== RAM Profile ===\n");
    printf("  Total RAM:        %.2f GB\n", profile.total_ram_gb);
    printf("  Available RAM:    %.2f GB (includes reclaimable standby cache)\n", profile.available_ram_gb);
    printf("  Total Pagefile:   %.2f GB\n", profile.total_pagefile_gb);
    printf("  Available Pagefile: %.2f GB\n", profile.available_pagefile_gb);
    printf("  Memory Load:      %.1f%%\n", 100.0 - (profile.available_ram_gb / profile.total_ram_gb * 100.0));
    printf("\n");
}
