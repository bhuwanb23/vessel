#include "disk_profiler.h"

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#endif

#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <algorithm>

// Constants
static constexpr int SECTOR_SIZE = 4096;
static constexpr int SEQ_BUFFER_SIZE = 4 * 1024 * 1024;  // 4MB for sequential reads
static constexpr int RAND_BUFFER_SIZE = 4096;             // 4KB for random reads
static constexpr double BENCHMARK_DURATION_SEC = 2.5;     // Target benchmark duration
static constexpr double WARMUP_DURATION_SEC = 1.0;        // Warm-up duration (untimed)

// High-resolution timer helpers (cross-platform)
static double get_time_sec() {
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(now.time_since_epoch()).count();
}

#if defined(_WIN32)

// =============================================================================
// Windows Implementation
// =============================================================================

// Sequential read benchmark
static void benchmark_sequential(HANDLE file_handle, DiskProfile& profile) {
    char* buffer = static_cast<char*>(_aligned_malloc(SEQ_BUFFER_SIZE, SECTOR_SIZE));
    if (!buffer) {
        fprintf(stderr, "Error: _aligned_malloc failed for sequential buffer\n");
        return;
    }

    // Warm-up phase (untimed)
    double warmup_start = get_time_sec();
    while (get_time_sec() - warmup_start < WARMUP_DURATION_SEC) {
        DWORD bytes_read = 0;
        if (!ReadFile(file_handle, buffer, SEQ_BUFFER_SIZE, &bytes_read, NULL) || bytes_read == 0) {
            break;
        }
    }

    // Reset file pointer to beginning
    LARGE_INTEGER zero_offset;
    zero_offset.QuadPart = 0;
    SetFilePointerEx(file_handle, zero_offset, NULL, FILE_BEGIN);

    // Timed benchmark
    uint64_t total_bytes_read = 0;
    double start_time = get_time_sec();
    double elapsed = 0;

    while (elapsed < BENCHMARK_DURATION_SEC) {
        DWORD bytes_read = 0;
        if (!ReadFile(file_handle, buffer, SEQ_BUFFER_SIZE, &bytes_read, NULL) || bytes_read == 0) {
            break;
        }
        total_bytes_read += bytes_read;
        elapsed = get_time_sec() - start_time;
    }

    profile.sequential_bytes_read = total_bytes_read;
    profile.sequential_elapsed_sec = elapsed;
    profile.sequential_read_mb_per_sec = (elapsed > 0) ?
        (static_cast<double>(total_bytes_read) / elapsed / (1024.0 * 1024.0)) : 0;

    _aligned_free(buffer);
}

// Random read benchmark
static void benchmark_random(HANDLE file_handle, uint64_t file_size, DiskProfile& profile) {
    char* buffer = static_cast<char*>(_aligned_malloc(RAND_BUFFER_SIZE, SECTOR_SIZE));
    if (!buffer) {
        fprintf(stderr, "Error: _aligned_malloc failed for random buffer\n");
        return;
    }

    uint64_t max_offset = file_size - RAND_BUFFER_SIZE;
    max_offset = (max_offset / SECTOR_SIZE) * SECTOR_SIZE;

    std::mt19937_64 rng(42);
    std::uniform_int_distribution<uint64_t> dist(0, max_offset / SECTOR_SIZE);

    // Warm-up
    double warmup_start = get_time_sec();
    while (get_time_sec() - warmup_start < WARMUP_DURATION_SEC) {
        uint64_t offset = dist(rng) * SECTOR_SIZE;
        LARGE_INTEGER li_offset;
        li_offset.QuadPart = static_cast<LONGLONG>(offset);
        SetFilePointerEx(file_handle, li_offset, NULL, FILE_BEGIN);
        DWORD bytes_read = 0;
        ReadFile(file_handle, buffer, RAND_BUFFER_SIZE, &bytes_read, NULL);
    }

    // Timed benchmark
    uint64_t num_reads = 0;
    double start_time = get_time_sec();
    double elapsed = 0;

    while (elapsed < BENCHMARK_DURATION_SEC) {
        uint64_t offset = dist(rng) * SECTOR_SIZE;
        LARGE_INTEGER li_offset;
        li_offset.QuadPart = static_cast<LONGLONG>(offset);
        SetFilePointerEx(file_handle, li_offset, NULL, FILE_BEGIN);

        DWORD bytes_read = 0;
        if (!ReadFile(file_handle, buffer, RAND_BUFFER_SIZE, &bytes_read, NULL)) {
            break;
        }
        num_reads++;
        elapsed = get_time_sec() - start_time;
    }

    profile.random_bytes_read = num_reads * RAND_BUFFER_SIZE;
    profile.random_num_reads = num_reads;
    profile.random_elapsed_sec = elapsed;
    profile.random_read_mb_per_sec = (elapsed > 0) ?
        (static_cast<double>(num_reads * RAND_BUFFER_SIZE) / elapsed / (1024.0 * 1024.0)) : 0;
    profile.random_read_iops = (elapsed > 0) ?
        (static_cast<double>(num_reads) / elapsed) : 0;

    _aligned_free(buffer);
}

DiskProfile profile_disk(const std::string& file_path) {
    DiskProfile profile = {};
    profile.file_path = file_path;

    HANDLE file_handle = CreateFileA(
        file_path.c_str(),
        GENERIC_READ,
        FILE_SHARE_READ,
        NULL,
        OPEN_EXISTING,
        FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN,
        NULL
    );

    if (file_handle == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "Error: Cannot open file '%s' (error %lu)\n", file_path.c_str(), GetLastError());
        return profile;
    }

    LARGE_INTEGER file_size;
    if (!GetFileSizeEx(file_handle, &file_size)) {
        fprintf(stderr, "Error: GetFileSizeEx() failed\n");
        CloseHandle(file_handle);
        return profile;
    }
    profile.file_size_bytes = static_cast<uint64_t>(file_size.QuadPart);
    profile.file_size_gb = static_cast<double>(file_size.QuadPart) / (1024.0 * 1024.0 * 1024.0);

    printf("  File: %s (%.2f GB)\n", file_path.c_str(), profile.file_size_gb);
    printf("  Running sequential read benchmark...\n");

    benchmark_sequential(file_handle, profile);

    LARGE_INTEGER zero_offset;
    zero_offset.QuadPart = 0;
    SetFilePointerEx(file_handle, zero_offset, NULL, FILE_BEGIN);

    printf("  Running random read benchmark (4K blocks)...\n");
    benchmark_random(file_handle, profile.file_size_bytes, profile);

    CloseHandle(file_handle);
    return profile;
}

#else

// =============================================================================
// Unix (Linux/macOS) Implementation
// =============================================================================

// Sequential read benchmark
static void benchmark_sequential(int fd, DiskProfile& profile) {
    std::vector<char> buffer(SEQ_BUFFER_SIZE);

    // Warm-up
    double warmup_start = get_time_sec();
    while (get_time_sec() - warmup_start < WARMUP_DURATION_SEC) {
        ssize_t bytes_read = read(fd, buffer.data(), SEQ_BUFFER_SIZE);
        if (bytes_read <= 0) break;
    }

    // Reset to beginning
    lseek(fd, 0, SEEK_SET);

    // Timed benchmark
    uint64_t total_bytes_read = 0;
    double start_time = get_time_sec();
    double elapsed = 0;

    while (elapsed < BENCHMARK_DURATION_SEC) {
        ssize_t bytes_read = read(fd, buffer.data(), SEQ_BUFFER_SIZE);
        if (bytes_read <= 0) break;
        total_bytes_read += bytes_read;
        elapsed = get_time_sec() - start_time;
    }

    profile.sequential_bytes_read = total_bytes_read;
    profile.sequential_elapsed_sec = elapsed;
    profile.sequential_read_mb_per_sec = (elapsed > 0) ?
        (static_cast<double>(total_bytes_read) / elapsed / (1024.0 * 1024.0)) : 0;
}

// Random read benchmark
static void benchmark_random(int fd, uint64_t file_size, DiskProfile& profile) {
    std::vector<char> buffer(RAND_BUFFER_SIZE);

    uint64_t max_offset = file_size - RAND_BUFFER_SIZE;
    max_offset = (max_offset / SECTOR_SIZE) * SECTOR_SIZE;

    std::mt19937_64 rng(42);
    std::uniform_int_distribution<uint64_t> dist(0, max_offset / SECTOR_SIZE);

    // Warm-up
    double warmup_start = get_time_sec();
    while (get_time_sec() - warmup_start < WARMUP_DURATION_SEC) {
        uint64_t offset = dist(rng) * SECTOR_SIZE;
        pread(fd, buffer.data(), RAND_BUFFER_SIZE, offset);
    }

    // Timed benchmark
    uint64_t num_reads = 0;
    double start_time = get_time_sec();
    double elapsed = 0;

    while (elapsed < BENCHMARK_DURATION_SEC) {
        uint64_t offset = dist(rng) * SECTOR_SIZE;
        ssize_t bytes_read = pread(fd, buffer.data(), RAND_BUFFER_SIZE, offset);
        if (bytes_read <= 0) break;
        num_reads++;
        elapsed = get_time_sec() - start_time;
    }

    profile.random_bytes_read = num_reads * RAND_BUFFER_SIZE;
    profile.random_num_reads = num_reads;
    profile.random_elapsed_sec = elapsed;
    profile.random_read_mb_per_sec = (elapsed > 0) ?
        (static_cast<double>(num_reads * RAND_BUFFER_SIZE) / elapsed / (1024.0 * 1024.0)) : 0;
    profile.random_read_iops = (elapsed > 0) ?
        (static_cast<double>(num_reads) / elapsed) : 0;
}

DiskProfile profile_disk(const std::string& file_path) {
    DiskProfile profile = {};
    profile.file_path = file_path;

    int fd = open(file_path.c_str(), O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "Error: Cannot open file '%s'\n", file_path.c_str());
        return profile;
    }

    // Get file size
    struct stat st;
    if (fstat(fd, &st) < 0) {
        fprintf(stderr, "Error: fstat() failed\n");
        close(fd);
        return profile;
    }
    profile.file_size_bytes = static_cast<uint64_t>(st.st_size);
    profile.file_size_gb = static_cast<double>(st.st_size) / (1024.0 * 1024.0 * 1024.0);

    printf("  File: %s (%.2f GB)\n", file_path.c_str(), profile.file_size_gb);
    printf("  Running sequential read benchmark...\n");

    benchmark_sequential(fd, profile);

    // Reset to beginning
    lseek(fd, 0, SEEK_SET);

    printf("  Running random read benchmark (4K blocks)...\n");
    benchmark_random(fd, profile.file_size_bytes, profile);

    close(fd);
    return profile;
}

#endif

void print_disk_profile(const DiskProfile& profile) {
    printf("=== Disk Profile ===\n");
    printf("  File: %s (%.2f GB)\n", profile.file_path.c_str(), profile.file_size_gb);
    printf("\n");
    printf("  Sequential Read:\n");
    printf("    Throughput:     %.2f MB/s\n", profile.sequential_read_mb_per_sec);
    printf("    Data read:      %.2f MB in %.2f sec\n",
           profile.sequential_bytes_read / (1024.0 * 1024.0), profile.sequential_elapsed_sec);
    printf("\n");
    printf("  Random Read (4K blocks):\n");
    printf("    Throughput:     %.2f MB/s\n", profile.random_read_mb_per_sec);
    printf("    IOPS:           %.0f\n", profile.random_read_iops);
    printf("    Data read:      %.2f MB in %.2f sec (%llu reads)\n",
           profile.random_bytes_read / (1024.0 * 1024.0), profile.random_elapsed_sec,
           (unsigned long long)profile.random_num_reads);
    printf("\n");

    // Sanity check
    if (profile.random_read_mb_per_sec > 0 && profile.sequential_read_mb_per_sec > 0) {
        double ratio = profile.sequential_read_mb_per_sec / profile.random_read_mb_per_sec;
        printf("  Sequential/Random ratio: %.1fx\n", ratio);
        if (ratio < 5.0) {
            printf("  WARNING: Ratio seems low. Sequential and random should differ by 10-50x.\n");
            printf("  This may indicate caching is not fully bypassed.\n");
        }
    }
    printf("\n");
}
