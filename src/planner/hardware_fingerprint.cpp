#include "hardware_fingerprint.h"

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
// CPU Model — from Windows Registry (Windows) or /proc/cpuinfo (Linux) or sysctl (macOS)
// =============================================================================

std::string normalizeCpuModel(const std::string& raw) {
    std::string s = raw;

    // Strip "Intel(R) Core(TM) " prefix
    const char* prefixes[] = {
        "Intel(R) Core(TM) ",
        "Intel(R) Xeon(R) ",
        "Intel(R) Pentium(R) ",
        "Intel(R) Celeron(R) ",
        "AMD Ryzen(TM) ",
        "AMD Athlon(TM) ",
        "AMD FX(TM) ",
        "AMD EPYC(TM) ",
        "Apple ",
    };
    for (const char* prefix : prefixes) {
        size_t pos = s.find(prefix);
        if (pos != std::string::npos) {
            s = s.substr(pos + strlen(prefix));
            break;
        }
    }

    // Strip " CPU" or " Processor" suffix
    auto strip_suffix = [](std::string& str, const char* suffix) {
        size_t pos = str.find(suffix);
        if (pos != std::string::npos) {
            str = str.substr(0, pos);
        }
    };
    strip_suffix(s, " CPU");
    strip_suffix(s, " Processor");
    strip_suffix(s, " 6-Core");
    strip_suffix(s, " 8-Core");
    strip_suffix(s, " 10-Core");
    strip_suffix(s, " 12-Core");

    // Strip " @ 3.60GHz" clock speed
    size_t at_pos = s.find(" @ ");
    if (at_pos != std::string::npos) {
        s = s.substr(0, at_pos);
    }

    return s;
}

// =============================================================================
// GPU Model — strip "NVIDIA" prefix
// =============================================================================

std::string normalizeGpuModel(const std::string& raw) {
    std::string s = raw;
    const char* prefixes[] = { "NVIDIA ", "AMD ", "Intel(R) " };
    for (const char* prefix : prefixes) {
        size_t pos = s.find(prefix);
        if (pos == 0) {
            s = s.substr(strlen(prefix));
            break;
        }
    }
    return s;
}

// =============================================================================
// RAM Total — round to nearest power-of-two GB
// =============================================================================

std::string normalizeRamTotal(uint64_t ram_bytes) {
    double gb = (double)ram_bytes / (1024.0 * 1024.0 * 1024.0);
    // Round to nearest integer
    int rounded = (int)(gb + 0.5);
    // Round to nearest power of two for clean display
    if (rounded <= 4) return "4GB";
    if (rounded <= 8) return "8GB";
    if (rounded <= 16) return "16GB";
    if (rounded <= 32) return "32GB";
    if (rounded <= 64) return "64GB";
    if (rounded <= 128) return "128GB";
    if (rounded <= 256) return "256GB";
    return std::to_string(rounded) + "GB";
}

// =============================================================================
// NVMe Model — Windows DeviceIoControl (Windows only)
// =============================================================================

#if defined(_WIN32)
std::string getNvmeModel(const std::string& file_path) {
    // Extract drive letter from file path
    if (file_path.size() < 2 || file_path[1] != ':') return "";
    char drive_letter = (char)toupper((unsigned char)file_path[0]);

    // Open the physical drive
    char device_path[] = "\\\\.\\C:";
    device_path[4] = drive_letter;

    HANDLE hDevice = CreateFileA(
        device_path,
        FILE_READ_ATTRIBUTES,
        FILE_SHARE_READ | FILE_SHARE_WRITE,
        NULL,
        OPEN_EXISTING,
        0,
        NULL
    );
    if (hDevice == INVALID_HANDLE_VALUE) return "";

    // Query storage device descriptor
    STORAGE_PROPERTY_QUERY query = {};
    query.PropertyId = StorageDeviceProperty;
    query.QueryType = PropertyStandardQuery;

    BYTE buffer[4096] = {};
    DWORD bytes_returned = 0;

    BOOL success = DeviceIoControl(
        hDevice,
        IOCTL_STORAGE_QUERY_PROPERTY,
        &query, sizeof(query),
        buffer, sizeof(buffer),
        &bytes_returned,
        NULL
    );

    CloseHandle(hDevice);

    if (!success || bytes_returned < sizeof(STORAGE_DEVICE_DESCRIPTOR)) return "";

    STORAGE_DEVICE_DESCRIPTOR* descriptor = (STORAGE_DEVICE_DESCRIPTOR*)buffer;
    if (descriptor->VendorIdOffset > 0 && descriptor->VendorIdOffset < bytes_returned) {
        const char* vendor = (const char*)(buffer + descriptor->VendorIdOffset);
        if (descriptor->ProductIdOffset > 0 && descriptor->ProductIdOffset < bytes_returned) {
            const char* product = (const char*)(buffer + descriptor->ProductIdOffset);
            std::string model = std::string(vendor) + " " + std::string(product);
            // Trim whitespace
            size_t start = model.find_first_not_of(" \t");
            size_t end = model.find_last_not_of(" \t");
            if (start != std::string::npos) return model.substr(start, end - start + 1);
        }
    }
    return "";
}
#else
// Linux/macOS: NVMe model detection not implemented (returns empty)
std::string getNvmeModel(const std::string&) { return ""; }
#endif

// =============================================================================
// Generate Hardware Fingerprint
// =============================================================================

std::string generateHardwareFingerprint(const HardwareSpec& hw) {
    std::ostringstream oss;
    oss << normalizeCpuModel(hw.cpu_name.empty() ? "Unknown CPU" : hw.cpu_name)
        << "|"
        << normalizeGpuModel(hw.gpu_name.empty() ? "No GPU" : hw.gpu_name)
        << "|"
        << normalizeRamTotal(hw.ram_total_bytes)
        << "|"
        << (hw.nvme_model.empty() ? "Unknown SSD" : hw.nvme_model);
    return oss.str();
}

std::string generateHardwareFingerprint(const HardwareSpec& hw, const std::string& model_file_path) {
    HardwareSpec hw_copy = hw;
    if (hw_copy.nvme_model.empty()) {
        hw_copy.nvme_model = getNvmeModel(model_file_path);
    }
    return generateHardwareFingerprint(hw_copy);
}
