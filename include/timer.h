#pragma once

#include <chrono>

// Simple wall-clock timer used throughout the codebase.
class Timer {
    std::chrono::high_resolution_clock::time_point t0;
public:
    Timer() : t0(std::chrono::high_resolution_clock::now()) {}
    double elapsed_ms() {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(now - t0).count();
    }
};
