#include "hardware_fingerprint.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <winioctl.h>
#include <ntddstor.h>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <sstream>

// =============================================================================
// CPU Model — from Windows Registry
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
    strip_suffix(s, " 16-Core");

    // Strip clock speed: " @ 3.60GHz" or " @ 3600MHz"
    size_t at_pos = s.find(" @ ");
    if (at_pos != std::string::npos) {
        s = s.substr(0, at_pos);
    }

    // Strip trailing whitespace
    while (!s.empty() && s.back() == ' ') s.pop_back();

    return s;
}

// =============================================================================
// GPU Model — normalize NVML string
// =============================================================================

std::string normalizeGpuModel(const std::string& raw) {
    std::string s = raw;

    // Strip "NVIDIA " prefix (keep the rest)
    const char* nvidia_prefix = "NVIDIA ";
    size_t pos = s.find(nvidia_prefix);
    if (pos == 0) {
        s = s.substr(strlen(nvidia_prefix));
    }

    return s;
}

// =============================================================================
// RAM Total — round to nearest power-of-two GB
// =============================================================================

std::string normalizeRamTotal(uint64_t ram_bytes) {
    double ram_gb = static_cast<double>(ram_bytes) / (1024.0 * 1024.0 * 1024.0);

    // Round to nearest power of 2
    // Powers of 2: 4, 8, 16, 32, 64, 128, 256
    int powers[] = {4, 8, 16, 32, 64, 128, 256};
    int closest = 4;
    int min_dist = 999;
    for (int p : powers) {
        int dist = abs(static_cast<int>(ram_gb) - p);
        if (dist < min_dist) {
            min_dist = dist;
            closest = p;
        }
    }

    char buf[32];
    snprintf(buf, sizeof(buf), "%dGB", closest);
    return std::string(buf);
}

// =============================================================================
// NVMe Model — from Windows DeviceIoControl
// =============================================================================

// Extract drive letter (e.g., "D:") from a file path
static std::string extractDriveLetter(const std::string& file_path) {
    if (file_path.size() >= 2 && file_path[1] == ':') {
        std::string drive = file_path.substr(0, 2);
        // Uppercase
        if (drive[0] >= 'a' && drive[0] <= 'z') drive[0] -= 32;
        return drive;
    }
    return "";
}

// Get physical disk number for a drive letter using DeviceIoControl
static int getDiskNumber(const std::string& drive_letter) {
    // Open the volume handle (e.g., "\\.\D:")
    std::string volume_path = "\\\\.\\" + drive_letter;
    HANDLE hDevice = CreateFileA(
        volume_path.c_str(),
        0,
        FILE_SHARE_READ | FILE_SHARE_WRITE,
        NULL,
        OPEN_EXISTING,
        0,
        NULL
    );

    if (hDevice == INVALID_HANDLE_VALUE) return -1;

    STORAGE_DEVICE_NUMBER sdn = {};
    DWORD bytes_returned = 0;
    BOOL success = DeviceIoControl(
        hDevice,
        IOCTL_STORAGE_GET_DEVICE_NUMBER,
        NULL, 0,
        &sdn, sizeof(sdn),
        &bytes_returned,
        NULL
    );

    CloseHandle(hDevice);

    return success ? static_cast<int>(sdn.DeviceNumber) : -1;
}

// Query storage device descriptor for model name
static std::string queryStorageModel(int disk_number) {
    char device_path[64];
    snprintf(device_path, sizeof(device_path), "\\\\.\\PhysicalDrive%d", disk_number);

    HANDLE hDevice = CreateFileA(
        device_path,
        0,
        FILE_SHARE_READ | FILE_SHARE_WRITE,
        NULL,
        OPEN_EXISTING,
        0,
        NULL
    );

    if (hDevice == INVALID_HANDLE_VALUE) return "";

    // Allocate enough space for the descriptor + vendor + product strings
    const int buf_size = 4096;
    char* buf = new char[buf_size];
    memset(buf, 0, buf_size);

    STORAGE_PROPERTY_QUERY query = {};
    query.PropertyId = StorageDeviceProperty;
    query.QueryType = PropertyStandardQuery;

    DWORD bytes_returned = 0;
    BOOL success = DeviceIoControl(
        hDevice,
        IOCTL_STORAGE_QUERY_PROPERTY,
        &query, sizeof(query),
        buf, buf_size,
        &bytes_returned,
        NULL
    );

    CloseHandle(hDevice);

    std::string model;
    if (success && bytes_returned > sizeof(STORAGE_DEVICE_DESCRIPTOR)) {
        STORAGE_DEVICE_DESCRIPTOR* descriptor = reinterpret_cast<STORAGE_DEVICE_DESCRIPTOR*>(buf);

        // Product ID string (the model name)
        if (descriptor->ProductIdOffset > 0 && descriptor->ProductIdOffset < bytes_returned) {
            const char* product = buf + descriptor->ProductIdOffset;
            // Trim leading/trailing whitespace
            while (*product == ' ') product++;
            model = product;
            while (!model.empty() && model.back() == ' ') model.pop_back();
        }

        // Vendor ID string (manufacturer)
        if (descriptor->VendorIdOffset > 0 && descriptor->VendorIdOffset < bytes_returned) {
            const char* vendor = buf + descriptor->VendorIdOffset;
            while (*vendor == ' ') vendor++;
            std::string vendor_str = vendor;
            while (!vendor_str.empty() && vendor_str.back() == ' ') vendor_str.pop_back();
            if (!vendor_str.empty() && !model.empty()) {
                model = vendor_str + " " + model;
            }
        }
    }

    delete[] buf;
    return model;
}

// Normalize NVMe model: strip serial number and firmware version
// "Samsung SSD 980 PRO 1TB S5JYNS0T123456" -> "Samsung 980 PRO"
static std::string normalizeNvmeModel(const std::string& raw) {
    if (raw.empty()) return "Unknown";

    std::string s = raw;

    // Strip common prefixes
    auto strip = [](std::string& str, const char* token) {
        size_t pos = str.find(token);
        if (pos != std::string::npos) {
            str.erase(pos, strlen(token));
        }
    };
    strip(s, "NVMe ");
    strip(s, "SSD ");
    strip(s, "Solid State Drive ");
    strip(s, "SATA ");
    strip(s, "HD ");

    // Remove trailing serial-like tokens (long alphanumeric strings)
    // e.g., "Samsung 980 PRO 1TB S5JYNS0T123456" -> "Samsung 980 PRO 1TB"
    // The serial is typically 8+ chars of mixed letters/digits at the end
    size_t last_space = s.find_last_of(' ');
    if (last_space != std::string::npos) {
        std::string last_token = s.substr(last_space + 1);
        // If last token looks like a serial number (mostly alphanumeric, >6 chars)
        bool is_serial = last_token.size() > 6;
        for (char c : last_token) {
            if (!isalnum(c)) { is_serial = false; break; }
        }
        if (is_serial) {
            s = s.substr(0, last_space);
        }
    }

    // Trim
    while (!s.empty() && s.back() == ' ') s.pop_back();
    while (!s.empty() && s.front() == ' ') s.erase(0, 1);

    return s.empty() ? "Unknown" : s;
}

std::string getNvmeModel(const std::string& file_path) {
    std::string drive_letter = extractDriveLetter(file_path);
    if (drive_letter.empty()) return "Unknown";

    int disk_number = getDiskNumber(drive_letter);
    if (disk_number < 0) return "Unknown";

    std::string model = queryStorageModel(disk_number);
    if (model.empty()) return "Unknown";

    return normalizeNvmeModel(model);
}

// =============================================================================
// Main Fingerprint Generator
// =============================================================================

std::string generateHardwareFingerprint(const HardwareSpec& hw) {
    // CPU model — read from registry
    HKEY hKey;
    LONG result = RegOpenKeyExA(
        HKEY_LOCAL_MACHINE,
        "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
        0,
        KEY_READ,
        &hKey
    );

    std::string cpu_model = "Unknown CPU";
    if (result == ERROR_SUCCESS) {
        char value[256] = {};
        DWORD value_size = sizeof(value);
        DWORD type = 0;
        result = RegQueryValueExA(hKey, "ProcessorNameString", NULL, &type,
                                  reinterpret_cast<LPBYTE>(value), &value_size);
        if (result == ERROR_SUCCESS && type == REG_SZ) {
            cpu_model = normalizeCpuModel(std::string(value));
        }
        RegCloseKey(hKey);
    }

    // GPU model — from HardwareSpec
    std::string gpu_model = normalizeGpuModel(hw.gpu_name);

    // RAM total — rounded to power-of-2 GB
    std::string ram_total = normalizeRamTotal(hw.ram_total_bytes);

    // NVMe model — read from storage device
    // Note: we don't have a file path here, so we try common drive letters
    // The caller should set this separately if needed
    std::string nvme_model = "Unknown";

    // Build the fingerprint
    std::string fingerprint = cpu_model + "|" + gpu_model + "|" + ram_total + "|" + nvme_model;

    return fingerprint;
}

// Overload that accepts a file path for NVMe detection
std::string generateHardwareFingerprint(const HardwareSpec& hw, const std::string& model_file_path) {
    // CPU model — read from registry
    HKEY hKey;
    LONG result = RegOpenKeyExA(
        HKEY_LOCAL_MACHINE,
        "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
        0,
        KEY_READ,
        &hKey
    );

    std::string cpu_model = "Unknown CPU";
    if (result == ERROR_SUCCESS) {
        char value[256] = {};
        DWORD value_size = sizeof(value);
        DWORD type = 0;
        result = RegQueryValueExA(hKey, "ProcessorNameString", NULL, &type,
                                  reinterpret_cast<LPBYTE>(value), &value_size);
        if (result == ERROR_SUCCESS && type == REG_SZ) {
            cpu_model = normalizeCpuModel(std::string(value));
        }
        RegCloseKey(hKey);
    }

    // GPU model — from HardwareSpec
    std::string gpu_model = normalizeGpuModel(hw.gpu_name);

    // RAM total — rounded to power-of-2 GB
    std::string ram_total = normalizeRamTotal(hw.ram_total_bytes);

    // NVMe model — from the drive containing the model file
    std::string nvme_model = "Unknown";
    if (!model_file_path.empty()) {
        nvme_model = getNvmeModel(model_file_path);
    }

    // Build the fingerprint
    std::string fingerprint = cpu_model + "|" + gpu_model + "|" + ram_total + "|" + nvme_model;

    return fingerprint;
}
