#include "hardware_fingerprint.h"
#include <cstring>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <winioctl.h>
#include <ntddstor.h>
#endif

#include <cstdio>
#include <cmath>
#include <algorithm>
#include <sstream>

// =============================================================================
// CPU Model — from Windows Registry (Windows), /proc/cpuinfo (Linux), or sysctl (macOS)
// =============================================================================

std::string normalizeCpuModel(const std::string& raw) {
    std::string s = raw;

    // Strip common prefixes
    const char* prefixes[] = {
        "Intel(R) Core(TM) ", "Intel(R) Xeon(R) ", "Intel(R) Pentium(R) ",
        "Intel(R) Celeron(R) ", "AMD Ryzen(TM) ", "AMD Athlon(TM) ",
        "AMD FX(TM) ", "AMD EPYC(TM) ", "Apple ",
    };
    for (const char* prefix : prefixes) {
        size_t pos = s.find(prefix);
        if (pos != std::string::npos) { s = s.substr(pos + strlen(prefix)); break; }
    }

    // Strip suffixes
    auto strip = [](std::string& str, const char* suffix) {
        size_t pos = str.find(suffix);
        if (pos != std::string::npos) str = str.substr(0, pos);
    };
    strip(s, " CPU"); strip(s, " Processor");
    strip(s, " 6-Core"); strip(s, " 8-Core"); strip(s, " 10-Core"); strip(s, " 12-Core");
    size_t at = s.find(" @ ");
    if (at != std::string::npos) s = s.substr(0, at);
    return s;
}

std::string normalizeGpuModel(const std::string& raw) {
    std::string s = raw;
    const char* prefixes[] = { "NVIDIA ", "AMD ", "Intel(R) " };
    for (const char* prefix : prefixes) {
        if (s.find(prefix) == 0) { s = s.substr(strlen(prefix)); break; }
    }
    return s;
}

std::string normalizeRamTotal(uint64_t ram_bytes) {
    int gb = (int)((double)ram_bytes / (1024.0 * 1024.0 * 1024.0) + 0.5);
    if (gb <= 4) return "4GB"; if (gb <= 8) return "8GB";
    if (gb <= 16) return "16GB"; if (gb <= 32) return "32GB";
    if (gb <= 64) return "64GB"; if (gb <= 128) return "128GB";
    return std::to_string(gb) + "GB";
}

// =============================================================================
// NVMe Model — Windows only via DeviceIoControl
// =============================================================================

#if defined(_WIN32)
std::string getNvmeModel(const std::string& file_path) {
    if (file_path.size() < 2 || file_path[1] != ':') return "";
    char device_path[] = "\\\\.\\C:";
    device_path[4] = (char)toupper((unsigned char)file_path[0]);

    HANDLE h = CreateFileA(device_path, FILE_READ_ATTRIBUTES,
        FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_EXISTING, 0, NULL);
    if (h == INVALID_HANDLE_VALUE) return "";

    STORAGE_PROPERTY_QUERY query = {};
    query.PropertyId = StorageDeviceProperty;
    query.QueryType = PropertyStandardQuery;
    BYTE buf[4096] = {};
    DWORD ret = 0;
    BOOL ok = DeviceIoControl(h, IOCTL_STORAGE_QUERY_PROPERTY, &query, sizeof(query), buf, sizeof(buf), &ret, NULL);
    CloseHandle(h);
    if (!ok || ret < sizeof(STORAGE_DEVICE_DESCRIPTOR)) return "";

    auto* desc = (STORAGE_DEVICE_DESCRIPTOR*)buf;
    if (desc->ProductIdOffset > 0 && desc->ProductIdOffset < ret) {
        return std::string((const char*)(buf + desc->ProductIdOffset));
    }
    return "";
}
#else
std::string getNvmeModel(const std::string&) { return ""; }
#endif

// =============================================================================
// Get CPU model from OS (cross-platform)
// =============================================================================

#if defined(_WIN32)
static std::string getCpuModelFromOS() {
    char buf[256] = {};
    DWORD size = sizeof(buf);
    // Try registry first
    HKEY key;
    if (RegOpenKeyExA(HKEY_LOCAL_MACHINE,
        "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0", 0, KEY_READ, &key) == ERROR_SUCCESS) {
        RegQueryValueExA(key, "ProcessorNameString", NULL, NULL, (LPBYTE)buf, &size);
        RegCloseKey(key);
    }
    return buf;
}
#elif defined(__APPLE__)
#include <sys/sysctl.h>
static std::string getCpuModelFromOS() {
    char buf[256] = {};
    size_t size = sizeof(buf);
    sysctlbyname("machdep.cpu.brand_string", buf, &size, NULL, 0);
    return buf;
}
#else
static std::string getCpuModelFromOS() {
    FILE* f = fopen("/proc/cpuinfo", "r");
    if (!f) return "Unknown CPU";
    char line[512] = {};
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "model name", 10) == 0) {
            char* colon = strchr(line, ':');
            if (colon) {
                fclose(f);
                colon += 2; // skip ": "
                // trim trailing newline
                char* nl = strchr(colon, '\n'); if (nl) *nl = '\0';
                return colon;
            }
        }
    }
    fclose(f);
    return "Unknown CPU";
}
#endif

// =============================================================================
// Generate Hardware Fingerprint
// =============================================================================

std::string generateHardwareFingerprint(const HardwareSpec& hw) {
    std::ostringstream oss;
    oss << normalizeCpuModel(getCpuModelFromOS())
        << "|"
        << normalizeGpuModel(hw.gpu_name.empty() ? "No GPU" : hw.gpu_name)
        << "|"
        << normalizeRamTotal(hw.ram_total_bytes)
        << "|"
        << "Unknown SSD";  // NVMe model needs a file path to query
    return oss.str();
}

std::string generateHardwareFingerprint(const HardwareSpec& hw, const std::string& model_file_path) {
    std::string nvme = getNvmeModel(model_file_path);
    std::ostringstream oss;
    oss << normalizeCpuModel(getCpuModelFromOS())
        << "|"
        << normalizeGpuModel(hw.gpu_name.empty() ? "No GPU" : hw.gpu_name)
        << "|"
        << normalizeRamTotal(hw.ram_total_bytes)
        << "|"
        << (nvme.empty() ? "Unknown SSD" : nvme);
    return oss.str();
}
