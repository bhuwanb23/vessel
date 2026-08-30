#include "http_fetcher.h"

// Safety limit: abort if we receive more than this (prevents downloading full file)
static constexpr uint64_t MAX_RANGE_RESPONSE = 256 * 1024;  // 256KB

// =============================================================================
// Windows implementation (WinHTTP)
// =============================================================================
#if defined(_WIN32)

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <winhttp.h>
#include <cstdio>
#include <cstring>

#pragma comment(lib, "winhttp.lib")

// Helper: convert std::string URL to wide string
static std::string wide_to_utf8(const wchar_t* wstr, int len) {
    if (len <= 0) return "";
    int utf8_len = WideCharToMultiByte(CP_UTF8, 0, wstr, len, NULL, 0, NULL, NULL);
    std::string result(utf8_len, 0);
    WideCharToMultiByte(CP_UTF8, 0, wstr, len, &result[0], utf8_len, NULL, NULL);
    return result;
}

bool fetch_range(const std::string& url, uint64_t range_end, std::vector<uint8_t>& output_buffer) {
    output_buffer.clear();

    // Convert URL to wide string
    int wide_len = MultiByteToWideChar(CP_UTF8, 0, url.c_str(), -1, NULL, 0);
    std::wstring wide_url(wide_len, 0);
    MultiByteToWideChar(CP_UTF8, 0, url.c_str(), -1, &wide_url[0], wide_len);

    // Parse URL components
    URL_COMPONENTS url_comp = {};
    url_comp.dwStructSize = sizeof(url_comp);
    url_comp.dwSchemeLength = 1;
    url_comp.dwHostNameLength = 1;
    url_comp.dwUrlPathLength = 1;
    url_comp.dwExtraInfoLength = 1;

    if (!WinHttpCrackUrl(wide_url.c_str(), 0, 0, &url_comp)) {
        fprintf(stderr, "Error: WinHttpCrackUrl() failed (error %lu)\n", GetLastError());
        return false;
    }

    std::wstring host_name(url_comp.lpszHostName, url_comp.dwHostNameLength);
    std::wstring url_path(url_comp.lpszUrlPath, url_comp.dwUrlPathLength);
    if (url_comp.dwExtraInfoLength > 0) {
        url_path += std::wstring(url_comp.lpszExtraInfo, url_comp.dwExtraInfoLength);
    }

    // Open session
    HINTERNET hSession = WinHttpOpen(L"Vessel/1.0",
                                      WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,
                                      WINHTTP_NO_PROXY_NAME,
                                      WINHTTP_NO_PROXY_BYPASS, 0);
    if (!hSession) {
        fprintf(stderr, "Error: WinHttpOpen() failed (error %lu)\n", GetLastError());
        return false;
    }

    // Set timeouts (10s connect, 30s total)
    WinHttpSetTimeouts(hSession, 10000, 10000, 30000, 30000);

    // Enable automatic redirect following (HuggingFace redirects to CDN)
    DWORD redirect_policy = WINHTTP_OPTION_REDIRECT_POLICY_ALWAYS;
    WinHttpSetOption(hSession, WINHTTP_OPTION_REDIRECT_POLICY, &redirect_policy, sizeof(redirect_policy));

    // Connect
    HINTERNET hConnect = WinHttpConnect(hSession, host_name.c_str(),
                                         url_comp.nPort, 0);
    if (!hConnect) {
        fprintf(stderr, "Error: WinHttpConnect() failed (error %lu)\n", GetLastError());
        WinHttpCloseHandle(hSession);
        return false;
    }

    // Open request
    HINTERNET hRequest = WinHttpOpenRequest(hConnect, L"GET", url_path.c_str(),
                                             NULL, WINHTTP_NO_REFERER,
                                             WINHTTP_DEFAULT_ACCEPT_TYPES,
                                             WINHTTP_FLAG_SECURE);
    if (!hRequest) {
        fprintf(stderr, "Error: WinHttpOpenRequest() failed (error %lu)\n", GetLastError());
        WinHttpCloseHandle(hConnect);
        WinHttpCloseHandle(hSession);
        return false;
    }

    // Add Range header
    wchar_t range_header[64];
    swprintf(range_header, 64, L"Range: bytes=0-%llu", range_end);
    WinHttpAddRequestHeaders(hRequest, range_header, (DWORD)-1L, WINHTTP_ADDREQ_FLAG_ADD);

    // Send request
    if (!WinHttpSendRequest(hRequest, WINHTTP_NO_ADDITIONAL_HEADERS, 0,
                            WINHTTP_NO_REQUEST_DATA, 0, 0, 0)) {
        fprintf(stderr, "Error: WinHttpSendRequest() failed (error %lu)\n", GetLastError());
        WinHttpCloseHandle(hRequest);
        WinHttpCloseHandle(hConnect);
        WinHttpCloseHandle(hSession);
        return false;
    }

    // Receive response
    if (!WinHttpReceiveResponse(hRequest, NULL)) {
        fprintf(stderr, "Error: WinHttpReceiveResponse() failed (error %lu)\n", GetLastError());
        WinHttpCloseHandle(hRequest);
        WinHttpCloseHandle(hConnect);
        WinHttpCloseHandle(hSession);
        return false;
    }

    // Check HTTP status code (must be 206 for range request)
    DWORD status_code = 0;
    DWORD status_size = sizeof(status_code);
    WinHttpQueryHeaders(hRequest,
                         WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
                         WINHTTP_HEADER_NAME_BY_INDEX,
                         &status_code, &status_size, WINHTTP_NO_HEADER_INDEX);

    if (status_code != 206) {
        fprintf(stderr, "Error: Expected HTTP 206 (Partial Content), got HTTP %lu\n", status_code);
        if (status_code == 200) {
            fprintf(stderr, "Server sent full file! Aborting to prevent downloading entire file.\n");
        }
        WinHttpCloseHandle(hRequest);
        WinHttpCloseHandle(hConnect);
        WinHttpCloseHandle(hSession);
        return false;
    }

    // Read response body with safety check
    uint64_t total_received = 0;
    DWORD bytes_available = 0;

    while (WinHttpQueryDataAvailable(hRequest, &bytes_available) && bytes_available > 0) {
        // Safety check: abort if receiving too much data
        if (total_received + bytes_available > MAX_RANGE_RESPONSE) {
            fprintf(stderr, "Error: Received more data than expected (%llu bytes). Aborting.\n",
                    total_received + bytes_available);
            WinHttpCloseHandle(hRequest);
            WinHttpCloseHandle(hConnect);
            WinHttpCloseHandle(hSession);
            return false;
        }

        // Grow buffer
        size_t old_size = output_buffer.size();
        output_buffer.resize(old_size + bytes_available);

        // Read into buffer
        DWORD bytes_read = 0;
        if (!WinHttpReadData(hRequest, output_buffer.data() + old_size, bytes_available, &bytes_read)) {
            fprintf(stderr, "Error: WinHttpReadData() failed (error %lu)\n", GetLastError());
            WinHttpCloseHandle(hRequest);
            WinHttpCloseHandle(hConnect);
            WinHttpCloseHandle(hSession);
            return false;
        }

        total_received += bytes_read;
    }

    WinHttpCloseHandle(hRequest);
    WinHttpCloseHandle(hConnect);
    WinHttpCloseHandle(hSession);

    return true;
}

bool fetch_gguf_header(const std::string& url, std::vector<uint8_t>& output_buffer) {
    // Fetch first 64KB — enough for all GGUF metadata
    return fetch_range(url, 65535, output_buffer);
}

bool fetch_full(const std::string& url, std::vector<uint8_t>& output_buffer) {
    output_buffer.clear();

    // Convert URL to wide string
    int wide_len = MultiByteToWideChar(CP_UTF8, 0, url.c_str(), -1, NULL, 0);
    std::wstring wide_url(wide_len, 0);
    MultiByteToWideChar(CP_UTF8, 0, url.c_str(), -1, &wide_url[0], wide_len);

    // Parse URL components
    URL_COMPONENTS url_comp = {};
    url_comp.dwStructSize = sizeof(url_comp);
    url_comp.dwSchemeLength = 1;
    url_comp.dwHostNameLength = 1;
    url_comp.dwUrlPathLength = 1;
    url_comp.dwExtraInfoLength = 1;

    if (!WinHttpCrackUrl(wide_url.c_str(), 0, 0, &url_comp)) {
        fprintf(stderr, "Error: WinHttpCrackUrl() failed (error %lu)\n", GetLastError());
        return false;
    }

    std::wstring host_name(url_comp.lpszHostName, url_comp.dwHostNameLength);
    std::wstring url_path(url_comp.lpszUrlPath, url_comp.dwUrlPathLength);
    if (url_comp.dwExtraInfoLength > 0) {
        url_path += std::wstring(url_comp.lpszExtraInfo, url_comp.dwExtraInfoLength);
    }

    HINTERNET hSession = WinHttpOpen(L"Vessel/1.0",
                                      WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,
                                      WINHTTP_NO_PROXY_NAME,
                                      WINHTTP_NO_PROXY_BYPASS, 0);
    if (!hSession) { fprintf(stderr, "[fetch_full] WinHttpOpen failed: %lu\n", GetLastError()); return false; }

    WinHttpSetTimeouts(hSession, 10000, 10000, 30000, 30000);

    HINTERNET hConnect = WinHttpConnect(hSession, host_name.c_str(), url_comp.nPort, 0);
    if (!hConnect) { fprintf(stderr, "[fetch_full] WinHttpConnect failed: %lu\n", GetLastError()); WinHttpCloseHandle(hSession); return false; }

    HINTERNET hRequest = WinHttpOpenRequest(hConnect, L"GET", url_path.c_str(),
                                             NULL, WINHTTP_NO_REFERER,
                                             WINHTTP_DEFAULT_ACCEPT_TYPES, WINHTTP_FLAG_SECURE);
    if (!hRequest) { fprintf(stderr, "[fetch_full] WinHttpOpenRequest failed: %lu\n", GetLastError()); WinHttpCloseHandle(hConnect); WinHttpCloseHandle(hSession); return false; }

    if (!WinHttpSendRequest(hRequest, WINHTTP_NO_ADDITIONAL_HEADERS, 0,
                            WINHTTP_NO_REQUEST_DATA, 0, 0, 0)) {
        fprintf(stderr, "[fetch_full] WinHttpSendRequest failed: %lu\n", GetLastError());
        WinHttpCloseHandle(hRequest); WinHttpCloseHandle(hConnect); WinHttpCloseHandle(hSession);
        return false;
    }

    if (!WinHttpReceiveResponse(hRequest, NULL)) {
        fprintf(stderr, "[fetch_full] WinHttpReceiveResponse failed: %lu\n", GetLastError());
        WinHttpCloseHandle(hRequest); WinHttpCloseHandle(hConnect); WinHttpCloseHandle(hSession);
        return false;
    }

    DWORD status_code = 0;
    DWORD status_size = sizeof(status_code);
    WinHttpQueryHeaders(hRequest, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
                         WINHTTP_HEADER_NAME_BY_INDEX, &status_code, &status_size, WINHTTP_NO_HEADER_INDEX);

    if (status_code != 200) {
        fprintf(stderr, "Error: HTTP %lu (expected 200)\n", status_code);
        WinHttpCloseHandle(hRequest); WinHttpCloseHandle(hConnect); WinHttpCloseHandle(hSession);
        return false;
    }

    DWORD bytes_available = 0;
    while (WinHttpQueryDataAvailable(hRequest, &bytes_available) && bytes_available > 0) {
        size_t old_size = output_buffer.size();
        output_buffer.resize(old_size + bytes_available);
        DWORD bytes_read = 0;
        WinHttpReadData(hRequest, output_buffer.data() + old_size, bytes_available, &bytes_read);
    }

    WinHttpCloseHandle(hRequest); WinHttpCloseHandle(hConnect); WinHttpCloseHandle(hSession);
    return true;
}

// =============================================================================
// Linux/macOS implementation (libcurl)
// =============================================================================
#else

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <curl/curl.h>

// libcurl write callback: appends data to a std::vector<uint8_t>
static size_t curl_write_data(void* contents, size_t size, size_t nmemb, void* userp) {
    size_t total = size * nmemb;
    auto* buffer = static_cast<std::vector<uint8_t>*>(userp);
    size_t old_size = buffer->size();
    buffer->resize(old_size + total);
    memcpy(buffer->data() + old_size, contents, total);
    return total;
}

bool fetch_range(const std::string& url, uint64_t range_end, std::vector<uint8_t>& output_buffer) {
    output_buffer.clear();

    CURL* curl = curl_easy_init();
    if (!curl) {
        fprintf(stderr, "Error: curl_easy_init() failed\n");
        return false;
    }

    // Build Range header
    char range_str[64];
    snprintf(range_str, sizeof(range_str), "0-%llu", (unsigned long long)range_end);

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_RANGE, range_str);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_write_data);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &output_buffer);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);  // Follow redirects
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 10L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 1L);

    CURLcode res = curl_easy_perform(curl);

    if (res != CURLE_OK) {
        fprintf(stderr, "Error: curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
        curl_easy_cleanup(curl);
        return false;
    }

    // Check HTTP status code (must be 206 for range request)
    long status_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &status_code);
    curl_easy_cleanup(curl);

    if (status_code != 206) {
        fprintf(stderr, "Error: Expected HTTP 206 (Partial Content), got HTTP %ld\n", status_code);
        if (status_code == 200) {
            fprintf(stderr, "Server sent full file! Aborting to prevent downloading entire file.\n");
        }
        output_buffer.clear();
        return false;
    }

    return true;
}

bool fetch_gguf_header(const std::string& url, std::vector<uint8_t>& output_buffer) {
    // Fetch first 64KB — enough for all GGUF metadata
    return fetch_range(url, 65535, output_buffer);
}

bool fetch_full(const std::string& url, std::vector<uint8_t>& output_buffer) {
    output_buffer.clear();

    CURL* curl = curl_easy_init();
    if (!curl) {
        fprintf(stderr, "Error: curl_easy_init() failed\n");
        return false;
    }

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_write_data);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &output_buffer);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 10L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 60L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 1L);

    CURLcode res = curl_easy_perform(curl);

    if (res != CURLE_OK) {
        fprintf(stderr, "[fetch_full] curl_easy_perform failed: %s\n", curl_easy_strerror(res));
        curl_easy_cleanup(curl);
        return false;
    }

    long status_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &status_code);
    curl_easy_cleanup(curl);

    if (status_code != 200) {
        fprintf(stderr, "Error: HTTP %ld (expected 200)\n", status_code);
        output_buffer.clear();
        return false;
    }

    return true;
}

#endif // _WIN32
