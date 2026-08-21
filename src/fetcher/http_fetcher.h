#pragma once

#include <cstdint>
#include <string>
#include <vector>

// Fetch the first N bytes of a URL via HTTP range request
// Returns the downloaded bytes in output_buffer
// Returns true on success, false on failure
// Verifies HTTP 206 (Partial Content) response
bool fetch_range(const std::string& url, uint64_t range_end, std::vector<uint8_t>& output_buffer);

// Convenience: fetch first 64KB (standard GGUF header size)
bool fetch_gguf_header(const std::string& url, std::vector<uint8_t>& output_buffer);

// Fetch a full file via HTTP GET (no range request)
// Returns the downloaded bytes in output_buffer
bool fetch_full(const std::string& url, std::vector<uint8_t>& output_buffer);
