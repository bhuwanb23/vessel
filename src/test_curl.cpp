#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <winhttp.h>
#include <cstdio>
#include <string>

#pragma comment(lib, "winhttp.lib")

// Simple HTTP GET using WinHTTP
// Returns HTTP status code and response body
static int http_get(const std::wstring& url, std::string& response_body) {
    // Parse URL components
    URL_COMPONENTS url_comp = {};
    url_comp.dwStructSize = sizeof(url_comp);
    url_comp.dwSchemeLength = 1;
    url_comp.dwHostNameLength = 1;
    url_comp.dwUrlPathLength = 1;
    url_comp.dwExtraInfoLength = 1;

    // Convert char URL to wchar_t for WinHTTP
    int wide_len = MultiByteToWideChar(CP_UTF8, 0, url.c_str(), -1, NULL, 0);
    std::wstring wide_url(wide_len, 0);
    MultiByteToWideChar(CP_UTF8, 0, url.c_str(), -1, &wide_url[0], wide_len);

    if (!WinHttpCrackUrl(wide_url.c_str(), 0, 0, &url_comp)) {
        fprintf(stderr, "Error: WinHttpCrackUrl() failed (error %lu)\n", GetLastError());
        return 0;
    }

    // Extract components
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
        return 0;
    }

    // Connect
    HINTERNET hConnect = WinHttpConnect(hSession, host_name.c_str(),
                                         url_comp.nPort, 0);
    if (!hConnect) {
        fprintf(stderr, "Error: WinHttpConnect() failed (error %lu)\n", GetLastError());
        WinHttpCloseHandle(hSession);
        return 0;
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
        return 0;
    }

    // Send request
    if (!WinHttpSendRequest(hRequest, WINHTTP_NO_ADDITIONAL_HEADERS, 0,
                            WINHTTP_NO_REQUEST_DATA, 0, 0, 0)) {
        fprintf(stderr, "Error: WinHttpSendRequest() failed (error %lu)\n", GetLastError());
        WinHttpCloseHandle(hRequest);
        WinHttpCloseHandle(hConnect);
        WinHttpCloseHandle(hSession);
        return 0;
    }

    // Receive response
    if (!WinHttpReceiveResponse(hRequest, NULL)) {
        fprintf(stderr, "Error: WinHttpReceiveResponse() failed (error %lu)\n", GetLastError());
        WinHttpCloseHandle(hRequest);
        WinHttpCloseHandle(hConnect);
        WinHttpCloseHandle(hSession);
        return 0;
    }

    // Get status code
    DWORD status_code = 0;
    DWORD status_size = sizeof(status_code);
    WinHttpQueryHeaders(hRequest,
                         WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
                         WINHTTP_HEADER_NAME_BY_INDEX,
                         &status_code, &status_size, WINHTTP_NO_HEADER_INDEX);

    // Read response body
    DWORD bytes_available = 0;
    while (WinHttpQueryDataAvailable(hRequest, &bytes_available) && bytes_available > 0) {
        char buffer[4096];
        DWORD bytes_read = 0;
        WinHttpReadData(hRequest, buffer, min(bytes_available, sizeof(buffer)), &bytes_read);
        response_body.append(buffer, bytes_read);
    }

    WinHttpCloseHandle(hRequest);
    WinHttpCloseHandle(hConnect);
    WinHttpCloseHandle(hSession);

    return static_cast<int>(status_code);
}

int main() {
    printf("=== HTTP Client Smoke Test (WinHTTP) ===\n\n");

    std::string response;
    int status = http_get("https://huggingface.co", response);

    printf("URL:    https://huggingface.co\n");
    printf("Status: %d\n", status);
    printf("Expected: 200\n");

    if (status == 200) {
        printf("Result: PASS - HTTP client is working!\n");
        printf("Response size: %zu bytes\n", response.size());
    } else if (status == 301 || status == 302) {
        printf("Result: PASS - Redirect (HTTP %d) is normal for HuggingFace\n", status);
    } else {
        printf("Result: FAIL - unexpected status code\n");
    }

    return (status >= 200 && status < 400) ? 0 : 1;
}
