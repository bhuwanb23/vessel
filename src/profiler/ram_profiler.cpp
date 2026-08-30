#include "ram_profiler.h"

#include <cstdio>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#include <mach/mach.h>
#else
#include <fstream>
#endif

static constexpr double BYTES_PER_GB = 1024.0 * 1024.0 * 1024.0;

RamProfile profile_ram() {
    RamProfile profile = {};

#if defined(_WIN32)
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

#elif defined(__APPLE__)
    // macOS: use sysctl for total, mach for available
    uint64_t total_mem = 0;
    size_t size = sizeof(total_mem);
    sysctlbyname("hw.memsize", &total_mem, &size, NULL, 0);

    mach_port_t host = mach_host_self();
    vm_statistics64_data_t vm_stats;
    mach_msg_type_number_t info_count = HOST_VM_INFO64_COUNT;
    host_statistics64(host, HOST_VM_INFO64, (host_info64_t)&vm_stats, &info_count);

    uint64_t page_size = 0;
    size = sizeof(page_size);
    sysctlbyname("hw.pagesize", &page_size, &size, NULL, 0);

    uint64_t free_mem = (uint64_t)(vm_stats.free_count + vm_stats.inactive_count) * page_size;

    profile.total_ram_bytes = total_mem;
    profile.available_ram_bytes = free_mem;
    profile.total_ram_gb = static_cast<double>(total_mem) / BYTES_PER_GB;
    profile.available_ram_gb = static_cast<double>(free_mem) / BYTES_PER_GB;
    profile.total_pagefile_gb = 0.0;  // macOS doesn't have pagefile
    profile.available_pagefile_gb = 0.0;

#else
    // Linux: read /proc/meminfo
    std::ifstream meminfo("/proc/meminfo");
    uint64_t total_kb = 0, available_kb = 0;
    std::string line;
    while (std::getline(meminfo, line)) {
        if (sscanf(line.c_str(), "MemTotal: %lu kB", &total_kb) == 1) continue;
        if (sscanf(line.c_str(), "MemAvailable: %lu kB", &available_kb) == 1) continue;
    }
    uint64_t total_bytes = total_kb * 1024ULL;
    uint64_t avail_bytes = available_kb * 1024ULL;

    profile.total_ram_bytes = total_bytes;
    profile.available_ram_bytes = avail_bytes;
    profile.total_ram_gb = static_cast<double>(total_bytes) / BYTES_PER_GB;
    profile.available_ram_gb = static_cast<double>(avail_bytes) / BYTES_PER_GB;
    profile.total_pagefile_gb = 0.0;
    profile.available_pagefile_gb = 0.0;
#endif

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
