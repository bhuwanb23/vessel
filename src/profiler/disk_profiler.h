#pragma once

#include <cstdint>
#include <string>

// Result of disk profiling
struct DiskProfile {
    std::string file_path;
    uint64_t file_size_bytes;
    double file_size_gb;

    // Sequential read
    double sequential_read_mb_per_sec;
    uint64_t sequential_bytes_read;
    double sequential_elapsed_sec;

    // Random read (4K blocks)
    double random_read_mb_per_sec;
    double random_read_iops;
    uint64_t random_bytes_read;
    uint64_t random_num_reads;
    double random_elapsed_sec;
};

// Benchmark disk read performance using the given file path.
// The file should be large (2GB+) and on the target drive.
// Uses FILE_FLAG_NO_BUFFERING to bypass OS cache.
DiskProfile profile_disk(const std::string& file_path);

// Print disk profile to stdout in a readable format
void print_disk_profile(const DiskProfile& profile);
