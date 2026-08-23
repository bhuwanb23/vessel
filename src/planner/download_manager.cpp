#include "download_manager.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <winhttp.h>
#include <shlobj.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <filesystem>
#include <algorithm>

#pragma comment(lib, "winhttp.lib")

// =============================================================================
// Helper: Convert string to wide string (for WinHTTP)
// =============================================================================
static std::wstring utf8_to_wide(const std::string& str) {
    int len = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), -1, NULL, 0);
    std::wstring wide(len, 0);
    MultiByteToWideChar(CP_UTF8, 0, str.c_str(), -1, &wide[0], len);
    return wide;
}

// =============================================================================
// Helper: Parse URL components for WinHTTP
// =============================================================================
struct ParsedUrl {
    std::wstring host;
    std::wstring path;
    INTERNET_PORT port = 443;
    bool use_ssl = true;
};

static bool parse_url(const std::string& url, ParsedUrl& out) {
    std::wstring wide = utf8_to_wide(url);

    URL_COMPONENTS comp = {};
    comp.dwStructSize = sizeof(comp);
    comp.dwSchemeLength = 1;
    comp.dwHostNameLength = 1;
    comp.dwUrlPathLength = 1;
    comp.dwExtraInfoLength = 1;

    if (!WinHttpCrackUrl(wide.c_str(), 0, 0, &comp)) {
        return false;
    }

    out.host = std::wstring(comp.lpszHostName, comp.dwHostNameLength);
    out.path = std::wstring(comp.lpszUrlPath, comp.dwUrlPathLength);
    if (comp.dwExtraInfoLength > 0) {
        out.path += std::wstring(comp.lpszExtraInfo, comp.dwExtraInfoLength);
    }
    out.port = comp.nPort;
    out.use_ssl = (comp.nScheme == INTERNET_SCHEME_HTTPS);

    return true;
}

// =============================================================================
// HTTP HEAD request to get file size (Content-Length)
// =============================================================================
uint64_t get_file_size_via_head(const std::string& url) {
    ParsedUrl parsed;
    if (!parse_url(url, parsed)) {
        fprintf(stderr, "[download_manager] Failed to parse URL: %s\n", url.c_str());
        return 0;
    }

    HINTERNET hSession = WinHttpOpen(L"LLMPlanner/1.0",
                                      WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,
                                      WINHTTP_NO_PROXY_NAME,
                                      WINHTTP_NO_PROXY_BYPASS, 0);
    if (!hSession) return 0;

    WinHttpSetTimeouts(hSession, 10000, 10000, 30000, 30000);

    // Enable redirect following (HuggingFace redirects to CDN)
    DWORD redirect_policy = WINHTTP_OPTION_REDIRECT_POLICY_ALWAYS;
    WinHttpSetOption(hSession, WINHTTP_OPTION_REDIRECT_POLICY,
                     &redirect_policy, sizeof(redirect_policy));

    HINTERNET hConnect = WinHttpConnect(hSession, parsed.host.c_str(),
                                         parsed.port, 0);
    if (!hConnect) {
        WinHttpCloseHandle(hSession);
        return 0;
    }

    DWORD flags = parsed.use_ssl ? WINHTTP_FLAG_SECURE : 0;
    HINTERNET hRequest = WinHttpOpenRequest(hConnect, L"HEAD", parsed.path.c_str(),
                                             NULL, WINHTTP_NO_REFERER,
                                             WINHTTP_DEFAULT_ACCEPT_TYPES, flags);
    if (!hRequest) {
        WinHttpCloseHandle(hConnect);
        WinHttpCloseHandle(hSession);
        return 0;
    }

    if (!WinHttpSendRequest(hRequest, WINHTTP_NO_ADDITIONAL_HEADERS, 0,
                            WINHTTP_NO_REQUEST_DATA, 0, 0, 0)) {
        WinHttpCloseHandle(hRequest);
        WinHttpCloseHandle(hConnect);
        WinHttpCloseHandle(hSession);
        return 0;
    }

    if (!WinHttpReceiveResponse(hRequest, NULL)) {
        WinHttpCloseHandle(hRequest);
        WinHttpCloseHandle(hConnect);
        WinHttpCloseHandle(hSession);
        return 0;
    }

    // Check HTTP status (200 OK for HEAD)
    DWORD status_code = 0;
    DWORD status_size = sizeof(status_code);
    WinHttpQueryHeaders(hRequest,
                         WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
                         WINHTTP_HEADER_NAME_BY_INDEX,
                         &status_code, &status_size, WINHTTP_NO_HEADER_INDEX);

    if (status_code != 200) {
        fprintf(stderr, "[download_manager] HEAD request returned HTTP %lu\n", status_code);
        WinHttpCloseHandle(hRequest);
        WinHttpCloseHandle(hConnect);
        WinHttpCloseHandle(hSession);
        return 0;
    }

    // Query Content-Length
    wchar_t content_length_str[64] = {};
    DWORD cl_size = sizeof(content_length_str);
    BOOL ok = WinHttpQueryHeaders(hRequest,
                                   WINHTTP_QUERY_CONTENT_LENGTH,
                                   WINHTTP_HEADER_NAME_BY_INDEX,
                                   content_length_str, &cl_size,
                                   WINHTTP_NO_HEADER_INDEX);

    WinHttpCloseHandle(hRequest);
    WinHttpCloseHandle(hConnect);
    WinHttpCloseHandle(hSession);

    if (!ok) return 0;

    // Parse the wide string to uint64
    uint64_t size = 0;
    for (int i = 0; content_length_str[i] >= L'0' && content_length_str[i] <= L'9'; i++) {
        size = size * 10 + (content_length_str[i] - L'0');
    }

    return size;
}

// =============================================================================
// Estimate file size from model metadata (fallback)
// =============================================================================
uint64_t estimate_file_size_from_metadata(const ModelSpec& model) {
    if (model.param_count == 0 || model.bits_per_weight <= 0) return 0;

    // param_count × bpw / 8 × 1.05 (header overhead)
    double estimated = (double)model.param_count * model.bits_per_weight / 8.0 * 1.05;
    return (uint64_t)estimated;
}

// =============================================================================
// Check available disk space on a directory's drive
// =============================================================================
uint64_t get_disk_free_bytes(const std::string& dir) {
    std::wstring wide_dir = utf8_to_wide(dir);

    ULARGE_INTEGER free_bytes_available = {};
    ULARGE_INTEGER total_bytes = {};
    ULARGE_INTEGER total_free_bytes = {};

    if (!GetDiskFreeSpaceExW(wide_dir.c_str(),
                              &free_bytes_available,
                              &total_bytes,
                              &total_free_bytes)) {
        fprintf(stderr, "[download_manager] GetDiskFreeSpaceExW failed for: %s\n",
                dir.c_str());
        return 0;
    }

    // Use free_bytes_available (respects per-user quotas)
    return free_bytes_available.QuadPart;
}

// =============================================================================
// Ensure download directory exists
// =============================================================================
bool ensure_download_dir(const std::string& dir) {
    std::error_code ec;
    if (std::filesystem::exists(dir, ec)) return true;

    if (std::filesystem::create_directories(dir, ec)) {
        return true;
    }

    fprintf(stderr, "[download_manager] Could not create directory: %s (%s)\n",
            dir.c_str(), ec.message().c_str());
    return false;
}

// =============================================================================
// Default download directory
// =============================================================================
std::string get_default_download_dir() {
    // Try %USERPROFILE%\models first
    char user_profile[MAX_PATH] = {};
    if (GetEnvironmentVariableA("USERPROFILE", user_profile, MAX_PATH)) {
        return std::string(user_profile) + "\\models";
    }
    // Fallback
    return "C:\\dev\\models";
}

// =============================================================================
// Extract filename from URL
// =============================================================================
std::string extract_filename_from_url(const std::string& url) {
    auto pos = url.find_last_of('/');
    if (pos != std::string::npos && pos + 1 < url.size()) {
        std::string filename = url.substr(pos + 1);
        // Strip query parameters
        auto qpos = filename.find('?');
        if (qpos != std::string::npos) {
            filename = filename.substr(0, qpos);
        }
        return filename;
    }
    return "model.gguf";
}

// =============================================================================
// Find a smaller quantization that fits in available space
// =============================================================================
std::string find_smaller_quant_that_fits(uint64_t available_bytes,
                                          const ModelSpec& model) {
    if (model.param_count == 0) return "";

    // Candidate quants in descending bpw order (best quality first)
    struct QuantCandidate {
        const char* name;
        double bpw;
    };

    static const QuantCandidate candidates[] = {
        {"Q5_K_M",  5.69},
        {"Q5_K_S",  5.54},
        {"Q4_K_M",  4.85},
        {"Q4_K_S",  4.58},
        {"Q4_0",    4.5},
        {"Q3_K_L",  3.91},
        {"Q3_K_M",  3.69},
        {"Q3_K_S",  3.44},
        {"Q2_K",    2.96},
        {"IQ3_M",   3.3},
        {"IQ2_XS",  2.31},
    };

    // Find the current quant's position
    int current_idx = -1;
    for (int i = 0; i < (int)(sizeof(candidates) / sizeof(candidates[0])); i++) {
        if (candidates[i].name == model.quant_type) {
            current_idx = i;
            break;
        }
    }

    // Start from one step below the current quant
    int start = (current_idx >= 0) ? current_idx + 1 : 0;

    // Safety margin: require available >= size × 1.15
    uint64_t safe_available = (uint64_t)(available_bytes / 1.15);

    for (int i = start; i < (int)(sizeof(candidates) / sizeof(candidates[0])); i++) {
        uint64_t est_size = (uint64_t)((double)model.param_count * candidates[i].bpw / 8.0 * 1.05);
        if (est_size <= safe_available) {
            return candidates[i].name;
        }
    }

    return "";
}

// =============================================================================
// Main: Pre-Download Safety Check
// =============================================================================
PreDownloadCheck pre_download_check(const std::string& url,
                                     const ModelSpec& model,
                                     const std::string& download_dir) {
    PreDownloadCheck result;
    result.target_dir = download_dir;

    // --- Step 1: Determine file size ---

    // Source 1: HTTP HEAD request
    uint64_t head_size = get_file_size_via_head(url);
    if (head_size > 0) {
        result.file_size_bytes = head_size;
        result.size_source = SizeSource::HEAD_REQUEST;
    }

    // Source 2: Metadata estimate (fallback)
    if (result.file_size_bytes == 0) {
        uint64_t estimated = estimate_file_size_from_metadata(model);
        if (estimated > 0) {
            result.file_size_bytes = estimated;
            result.size_source = SizeSource::METADATA_ESTIMATE;
        }
    }

    if (result.file_size_bytes == 0) {
        result.pass = false;
        result.error_message = "Could not determine file size (HEAD failed, metadata incomplete)";
        return result;
    }

    // --- Step 2: Calculate required space with safety margin ---
    // 1.15x safety margin for filesystem overhead + fragmentation
    result.required_bytes = (uint64_t)(result.file_size_bytes * 1.15);

    // --- Step 3: Check disk space ---
    result.available_bytes = get_disk_free_bytes(download_dir);

    if (result.available_bytes == 0) {
        result.pass = false;
        result.error_message = "Could not check disk space for: " + download_dir;
        return result;
    }

    if (result.available_bytes >= result.required_bytes) {
        result.pass = true;
        return result;
    }

    // --- Step 4: Not enough space — build error with suggestions ---
    result.pass = false;

    char buf[512];
    snprintf(buf, sizeof(buf),
             "Insufficient disk space.\n"
             "   Model size:    %.1f GB\n"
             "   Required:      %.1f GB (with 15%% safety margin)\n"
             "   Available:     %.1f GB on %s\n"
             "   Shortfall:     %.1f GB",
             result.file_size_bytes / 1e9,
             result.required_bytes / 1e9,
             result.available_bytes / 1e9,
             download_dir.c_str(),
             (result.required_bytes - result.available_bytes) / 1e9);
    result.error_message = buf;

    // Suggest a smaller quant
    std::string smaller = find_smaller_quant_that_fits(result.available_bytes, model);
    if (!smaller.empty()) {
        uint64_t smaller_size = estimate_file_size_from_metadata(model);
        // Re-estimate with the smaller quant's bpw
        double smaller_bpw = get_bits_per_weight(smaller);
        if (smaller_bpw > 0) {
            smaller_size = (uint64_t)((double)model.param_count * smaller_bpw / 8.0 * 1.05);
        }
        char sug[256];
        snprintf(sug, sizeof(sug),
                 "   \n   Suggestions:\n"
                 "   • Free up disk space and try again\n"
                 "   • Try a smaller quantization: %s (~%.1f GB) would fit\n"
                 "   • Download to a different drive: --download-dir D:\\models\\",
                 smaller.c_str(), smaller_size / 1e9);
        result.suggestion = sug;
    } else {
        result.suggestion =
            "   \n   Suggestions:\n"
            "   • Free up disk space and try again\n"
            "   • Try a smaller model (e.g., 3B instead of 7B+)\n"
            "   • Download to a different drive: --download-dir D:\\models\\";
    }

    return result;
}
