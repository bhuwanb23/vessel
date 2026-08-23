#include "download_manager.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <winhttp.h>
#include <shlobj.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <filesystem>
#include <algorithm>
#include <thread>
#include <chrono>
#include <atomic>
#include <mutex>

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
// Phase B: Resumable Download
// =============================================================================

// Get the .partial file path for a given URL and directory
std::string get_partial_path(const std::string& url, const std::string& dir) {
    std::string filename = extract_filename_from_url(url);
    return dir + "\\" + filename + ".partial";
}

// Get the final .gguf file path
std::string get_final_path(const std::string& url, const std::string& dir) {
    std::string filename = extract_filename_from_url(url);
    return dir + "\\" + filename;
}

// Check if a .partial file exists and return its size
uint64_t get_partial_file_size(const std::string& partial_path) {
    std::error_code ec;
    if (!std::filesystem::exists(partial_path, ec)) return 0;
    uint64_t size = std::filesystem::file_size(partial_path, ec);
    return ec ? 0 : size;
}

// ---------------------------------------------------------------------------
// Progress Display
// ---------------------------------------------------------------------------

struct ProgressState {
    uint64_t file_size = 0;
    uint64_t bytes_downloaded = 0;
    std::chrono::steady_clock::time_point start_time;
    // Sliding window for speed calculation (last 5 seconds)
    static constexpr int WINDOW_SIZE = 10;  // 5s / 0.5s = 10 samples
    uint64_t window_bytes[WINDOW_SIZE] = {};
    int window_idx = 0;
    std::chrono::steady_clock::time_point last_update;
    double current_speed_mbs = 0.0;
};

static void update_progress(ProgressState& state, uint64_t current_bytes) {
    auto now = std::chrono::steady_clock::now();
    state.bytes_downloaded = current_bytes;

    // Update sliding window every 500ms
    double elapsed_since_update = std::chrono::duration<double>(now - state.last_update).count();
    if (elapsed_since_update < 0.5) return;

    state.window_bytes[state.window_idx % ProgressState::WINDOW_SIZE] = current_bytes;
    state.window_idx++;
    state.last_update = now;

    // Calculate speed from window
    int count = std::min(state.window_idx, ProgressState::WINDOW_SIZE);
    if (count < 2) {
        // Not enough samples — use total average
        double total_elapsed = std::chrono::duration<double>(now - state.start_time).count();
        if (total_elapsed > 0.1) {
            state.current_speed_mbs = (current_bytes / 1e6) / total_elapsed;
        }
        return;
    }

    uint64_t oldest = state.window_bytes[(state.window_idx - count) % ProgressState::WINDOW_SIZE];
    uint64_t newest = state.window_bytes[(state.window_idx - 1) % ProgressState::WINDOW_SIZE];
    double window_time = (count - 1) * 0.5;  // approximate time between samples
    if (window_time > 0.1) {
        state.current_speed_mbs = ((newest - oldest) / 1e6) / window_time;
    }
}

static void print_progress_bar(const ProgressState& state) {
    if (state.file_size == 0) return;

    double percent = 100.0 * state.bytes_downloaded / state.file_size;
    double downloaded_gb = state.bytes_downloaded / 1e9;
    double total_gb = state.file_size / 1e9;
    double speed = state.current_speed_mbs;

    // ETA
    char eta_str[32] = "";
    if (speed > 0.1) {
        double remaining_sec = ((state.file_size - state.bytes_downloaded) / 1e6) / speed;
        if (remaining_sec > 3600) {
            snprintf(eta_str, sizeof(eta_str), ">1h");
        } else if (remaining_sec > 60) {
            snprintf(eta_str, sizeof(eta_str), "%dm %02ds", (int)(remaining_sec / 60), (int)remaining_sec % 60);
        } else {
            snprintf(eta_str, sizeof(eta_str), "%ds", (int)remaining_sec);
        }
    }

    // Progress bar (40 chars wide)
    int bar_width = 40;
    int filled = (int)(percent / 100.0 * bar_width);
    filled = std::min(filled, bar_width);

    printf("\r  [");
    for (int i = 0; i < bar_width; i++) {
        printf(i < filled ? "=" : " ");
    }
    printf("] %5.1f%% | %.2f / %.2f GB | %.1f MB/s | ETA: %s  ",
           percent, downloaded_gb, total_gb, speed, eta_str);
    fflush(stdout);
}

// ---------------------------------------------------------------------------
// WinHTTP Download with Range Request
// ---------------------------------------------------------------------------

// Callback for WinHTTP async read — stores data and tracks bytes
struct DownloadCallbackData {
    FILE* file;
    uint64_t bytes_written;
    ProgressState* progress;
    std::atomic<bool>* abort_flag;
};

// WinHTTP status callback for progress tracking
static void CALLBACK winhttp_status_callback(
    HINTERNET hInternet, DWORD_PTR dwContext,
    DWORD dwInternetStatus, LPVOID lpvStatusInformation,
    DWORD dwStatusInformationLength)
{
    // We don't use this for progress — we poll file size instead
    // But we need it registered for WinHTTP to work with async I/O
    (void)hInternet; (void)dwContext; (void)dwInternetStatus;
    (void)lpvStatusInformation; (void)dwStatusInformationLength;
}

static DownloadResult do_download(
    const std::string& url,
    const std::string& partial_path,
    const std::string& final_path,
    uint64_t file_size,
    uint64_t resume_offset,
    std::atomic<bool>& abort_flag)
{
    DownloadResult result;
    result.partial_path = partial_path;
    result.file_size = file_size;

    // Parse URL
    ParsedUrl parsed;
    if (!parse_url(url, parsed)) {
        result.error_message = "Failed to parse URL: " + url;
        return result;
    }

    // Open WinHTTP session
    HINTERNET hSession = WinHttpOpen(L"LLMPlanner/1.0",
                                      WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,
                                      WINHTTP_NO_PROXY_NAME,
                                      WINHTTP_NO_PROXY_BYPASS, 0);
    if (!hSession) {
        result.error_message = "WinHttpOpen failed (error " + std::to_string(GetLastError()) + ")";
        return result;
    }

    // No total timeout for large files, but kill slow connections
    WinHttpSetTimeouts(hSession, 10000, 10000, 0, 0);

    // Enable redirect following
    DWORD redirect_policy = WINHTTP_OPTION_REDIRECT_POLICY_ALWAYS;
    WinHttpSetOption(hSession, WINHTTP_OPTION_REDIRECT_POLICY,
                     &redirect_policy, sizeof(redirect_policy));

    // Connect
    HINTERNET hConnect = WinHttpConnect(hSession, parsed.host.c_str(),
                                         parsed.port, 0);
    if (!hConnect) {
        result.error_message = "WinHttpConnect failed (error " + std::to_string(GetLastError()) + ")";
        WinHttpCloseHandle(hSession);
        return result;
    }

    // Open GET request
    DWORD flags = parsed.use_ssl ? WINHTTP_FLAG_SECURE : 0;
    HINTERNET hRequest = WinHttpOpenRequest(hConnect, L"GET", parsed.path.c_str(),
                                             NULL, WINHTTP_NO_REFERER,
                                             WINHTTP_DEFAULT_ACCEPT_TYPES, flags);
    if (!hRequest) {
        result.error_message = "WinHttpOpenRequest failed (error " + std::to_string(GetLastError()) + ")";
        WinHttpCloseHandle(hConnect);
        WinHttpCloseHandle(hSession);
        return result;
    }

    // Add Range header for resume
    if (resume_offset > 0) {
        wchar_t range_header[64];
        swprintf(range_header, 64, L"Range: bytes=%llu-", resume_offset);
        WinHttpAddRequestHeaders(hRequest, range_header, (DWORD)-1L, WINHTTP_ADDREQ_FLAG_ADD);
    }

    // Send request
    if (!WinHttpSendRequest(hRequest, WINHTTP_NO_ADDITIONAL_HEADERS, 0,
                            WINHTTP_NO_REQUEST_DATA, 0, 0, 0)) {
        result.error_message = "WinHttpSendRequest failed (error " + std::to_string(GetLastError()) + ")";
        WinHttpCloseHandle(hRequest);
        WinHttpCloseHandle(hConnect);
        WinHttpCloseHandle(hSession);
        return result;
    }

    if (!WinHttpReceiveResponse(hRequest, NULL)) {
        result.error_message = "WinHttpReceiveResponse failed (error " + std::to_string(GetLastError()) + ")";
        WinHttpCloseHandle(hRequest);
        WinHttpCloseHandle(hConnect);
        WinHttpCloseHandle(hSession);
        return result;
    }

    // Check HTTP status
    DWORD status_code = 0;
    DWORD status_size = sizeof(status_code);
    WinHttpQueryHeaders(hRequest,
                         WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
                         WINHTTP_HEADER_NAME_BY_INDEX,
                         &status_code, &status_size, WINHTTP_NO_HEADER_INDEX);

    // Accept 200 (fresh) or 206 (partial content / resume)
    bool is_resume = (status_code == 206);
    if (status_code != 200 && status_code != 206) {
        result.error_message = "HTTP error " + std::to_string(status_code) + " (expected 200 or 206)";
        WinHttpCloseHandle(hRequest);
        WinHttpCloseHandle(hConnect);
        WinHttpCloseHandle(hSession);
        return result;
    }

    // If server sent 200 instead of 206 for a resume, restart from beginning
    if (resume_offset > 0 && status_code == 200) {
        resume_offset = 0;  // Server doesn't support resume — restart fresh
    }

    // Open file for writing
    FILE* file = nullptr;
    if (resume_offset > 0 && is_resume) {
        file = fopen(partial_path.c_str(), "ab");  // append mode
    } else {
        file = fopen(partial_path.c_str(), "wb");  // truncate and start fresh
        resume_offset = 0;
    }

    if (!file) {
        result.error_message = "Could not open file for writing: " + partial_path;
        WinHttpCloseHandle(hRequest);
        WinHttpCloseHandle(hConnect);
        WinHttpCloseHandle(hSession);
        return result;
    }

    // Initialize progress
    ProgressState progress;
    progress.file_size = file_size;
    progress.start_time = std::chrono::steady_clock::now();
    progress.last_update = progress.start_time;
    progress.bytes_downloaded = resume_offset;

    printf("\nDownloading %s\n", extract_filename_from_url(url).c_str());
    if (resume_offset > 0) {
        printf("  Resuming from %.2f GB\n", resume_offset / 1e9);
    }

    // Read loop
    DWORD bytes_available = 0;
    uint64_t total_written = resume_offset;
    const size_t BUF_SIZE = 64 * 1024;  // 64KB chunks
    char buffer[65536];

    while (WinHttpQueryDataAvailable(hRequest, &bytes_available) && bytes_available > 0) {
        // Check abort
        if (abort_flag.load()) {
            result.paused = true;
            result.bytes_downloaded = total_written;
            result.error_message = "Download cancelled by user";
            fclose(file);
            WinHttpCloseHandle(hRequest);
            WinHttpCloseHandle(hConnect);
            WinHttpCloseHandle(hSession);
            return result;
        }

        // Read in chunks
        DWORD remaining = bytes_available;
        while (remaining > 0) {
            DWORD to_read = std::min(remaining, (DWORD)BUF_SIZE);
            DWORD bytes_read = 0;

            if (!WinHttpReadData(hRequest, buffer, to_read, &bytes_read) || bytes_read == 0) {
                break;
            }

            size_t written = fwrite(buffer, 1, bytes_read, file);
            if (written != bytes_read) {
                // Disk full or write error
                result.error_message = "Write error — disk may be full";
                fclose(file);
                WinHttpCloseHandle(hRequest);
                WinHttpCloseHandle(hConnect);
                WinHttpCloseHandle(hSession);
                return result;
            }

            total_written += bytes_read;
            remaining -= bytes_read;

            // Update progress display
            update_progress(progress, total_written);
            print_progress_bar(progress);
        }
    }

    fclose(file);
    WinHttpCloseHandle(hRequest);
    WinHttpCloseHandle(hConnect);
    WinHttpCloseHandle(hSession);

    // Final progress line
    printf("\n");

    result.bytes_downloaded = total_written;
    result.success = true;
    result.final_path = final_path;
    return result;
}

// ---------------------------------------------------------------------------
// Main Download Entry Point
// ---------------------------------------------------------------------------

DownloadResult download_model_file(const std::string& url,
                                    const std::string& download_dir,
                                    uint64_t file_size,
                                    std::atomic<bool>& abort_flag) {
    DownloadResult result;

    std::string partial_path = get_partial_path(url, download_dir);
    std::string final_path_str = get_final_path(url, download_dir);

    // Check if final file already exists
    {
        std::error_code ec;
        if (std::filesystem::exists(final_path_str, ec)) {
            result.success = true;
            result.final_path = final_path_str;
            result.bytes_downloaded = file_size;  // assume complete
            result.file_size = file_size;
            printf("Model already downloaded: %s\n", final_path_str.c_str());
            return result;
        }
    }

    // Auto-detect file size if not provided
    if (file_size == 0) {
        file_size = get_file_size_via_head(url);
        if (file_size == 0) {
            result.error_message = "Could not determine file size";
            return result;
        }
    }
    result.file_size = file_size;

    // Check for existing .partial file (resume)
    uint64_t resume_offset = get_partial_file_size(partial_path);

    // Validate .partial file
    if (resume_offset > 0) {
        if (resume_offset >= file_size) {
            // .partial is complete or larger than expected — delete and restart
            printf("Partial file is complete or oversized — restarting download.\n");
            std::remove(partial_path.c_str());
            resume_offset = 0;
        } else {
            printf("Found partial download: %.2f / %.2f GB (%.1f%% complete)\n",
                   resume_offset / 1e9, file_size / 1e9,
                   100.0 * resume_offset / file_size);
        }
    }

    // Perform the download
    result = do_download(url, partial_path, final_path_str, file_size,
                         resume_offset, abort_flag);

    if (result.paused) {
        // User cancelled — .partial file preserved for resume
        printf("\nDownload paused. Run again to resume from %.2f GB.\n",
               result.bytes_downloaded / 1e9);
        return result;
    }

    if (!result.success) {
        // Error — .partial file preserved for resume
        fprintf(stderr, "\nDownload failed: %s\n", result.error_message.c_str());
        fprintf(stderr, "Run again to resume from %.2f GB.\n",
                result.bytes_downloaded / 1e9);
        return result;
    }

    // Download complete — verify size
    if (result.bytes_downloaded != file_size) {
        result.error_message = "Size mismatch: expected " + std::to_string(file_size)
                             + " bytes, got " + std::to_string(result.bytes_downloaded) + " bytes";
        fprintf(stderr, "\n%s\n", result.error_message.c_str());
        // Don't delete .partial — user may want to resume
        return result;
    }

    // Rename .partial to final filename
    std::error_code ec;
    std::filesystem::rename(partial_path, final_path_str, ec);
    if (ec) {
        result.error_message = "Could not rename .partial to final: " + ec.message();
        return result;
    }

    result.success = true;
    result.final_path = final_path_str;
    printf("Download complete: %s\n", final_path_str.c_str());
    return result;
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
