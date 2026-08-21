#include <cstdio>
#include <curl/curl.h>

// Callback to discard response body
static size_t write_callback(void* contents, size_t size, size_t nmemb, void* userp) {
    return size * nmemb;
}

int main() {
    printf("=== libcurl Smoke Test ===\n\n");

    curl_global_init(CURL_GLOBAL_DEFAULT);
    CURL* curl = curl_easy_init();

    if (!curl) {
        fprintf(stderr, "Error: curl_easy_init() failed\n");
        return 1;
    }

    // Test: fetch Hugging Face homepage
    curl_easy_setopt(curl, CURLOPT_URL, "https://huggingface.co");
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 10L);

    CURLcode res = curl_easy_perform(curl);

    if (res != CURLE_OK) {
        fprintf(stderr, "Error: curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
        curl_easy_cleanup(curl);
        curl_global_cleanup();
        return 1;
    }

    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

    printf("HTTP Status: %ld\n", http_code);
    printf("Expected:    200\n");

    if (http_code == 200) {
        printf("Result:      PASS - libcurl is working!\n");
    } else {
        printf("Result:      FAIL - unexpected status code\n");
    }

    curl_easy_cleanup(curl);
    curl_global_cleanup();

    return (http_code == 200) ? 0 : 1;
}
