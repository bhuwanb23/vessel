#include "http_fetcher.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <winhttp.h>
#include <cstdio>
#include <cstring>

#pragma comment(lib, "winhttp.lib")

// Safety limit: abort if we receive more than this (prevents downloading full file)
static constexpr uint64_t MAX_RANGE_RESPONSE = 256 * 1024;  // 256KB

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
    HINTERNET hSession = WinHttpOpen(L"LLMPlanner/1.0",
                                      WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,
                                      WINHTTP_NO_PROXY_NAME,
                                      WINHTTP_NO_PROXY_BYPASS, 0);
    if (!hSession) {
        fprintf(stderr, "Error: WinHttpOpen() failed (error %lu)\n", GetLastError());
        return false;
    }

    // Set timeouts (10s connect, 30s total)
    WinHttpSetTimeouts(hSession, 10000, 10000, 30000, 30000);

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

    HINTERNET hSession = WinHttpOpen(L"LLMPlanner/1.0",
                                      WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,
                                      WINHTTP_NO_PROXY_NAME,
                                      WINHTTP_NO_PROXY_BYPASS, 0);
    if (!hSession) return false;

    WinHttpSetTimeouts(hSession, 10000, 10000, 30000, 30000);

    HINTERNET hConnect = WinHttpConnect(hSession, host_name.c_str(), url_comp.nPort, 0);
    if (!hConnect) { WinHttpCloseHandle(hSession); return false; }

    HINTERNET hRequest = WinHttpOpenRequest(hConnect, L"GET", url_path.c_str(),
                                             NULL, WINHTTP_NO_REFERER,
                                             WINHTTP_DEFAULT_ACCEPT_TYPES, 0);
    if (!hRequest) { WinHttpCloseHandle(hConnect); WinHttpCloseHandle(hSession); return false; }

    if (!WinHttpSendRequest(hRequest, WINHTTP_NO_ADDITIONAL_HEADERS, 0,
                            WINHTTP_NO_REQUEST_DATA, 0, 0, 0)) {
        WinHttpCloseHandle(hRequest); WinHttpCloseHandle(hConnect); WinHttpCloseHandle(hSession);
        return false;
    }

    if (!WinHttpReceiveResponse(hRequest, NULL)) {
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
