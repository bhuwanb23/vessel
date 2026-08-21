cmake_minimum_required(VERSION 3.14)
project(local-llm-planner LANGUAGES C CXX)

# ---- C++ standard ----
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---- Build type default ----
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---- Output directories ----
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# ---- Compiler warnings ----
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ============================================
# External dependencies paths
# ============================================
set(CUDA_PATH "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9" CACHE PATH "CUDA Toolkit path")
set(CURL_PATH "${CMAKE_SOURCE_DIR}/external/curl/curl-8.21.0_7-win64-mingw" CACHE PATH "libcurl path")

# ============================================
# Hardware Profiler (Step 1)
# ============================================
add_executable(planner
    src/main.cpp
    src/profiler/ram_profiler.cpp
    src/profiler/gpu_profiler.cpp
    src/profiler/disk_profiler.cpp
)

target_include_directories(planner PRIVATE
    src
    ${CUDA_PATH}/include
)

target_link_directories(planner PRIVATE
    ${CUDA_PATH}/lib/x64
)

target_link_libraries(planner PRIVATE nvml)

# ============================================
# libcurl test (Step 2 smoke test)
# ============================================
add_executable(test_curl src/test_curl.cpp)

target_include_directories(test_curl PRIVATE
    ${CURL_PATH}/include
)

target_link_directories(test_curl PRIVATE
    ${CURL_PATH}/lib
)

# Static libcurl + Windows system libraries
target_link_libraries(test_curl PRIVATE
    libcurl.a
    ws2_32
    crypt32
    wldap32
    normaliz
)

# ============================================
# Installation
# ============================================
install(TARGETS planner test_curl DESTINATION bin)
