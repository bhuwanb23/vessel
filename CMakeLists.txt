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
# Metadata Fetcher (Step 2)
# ============================================
add_executable(metadata_fetcher
    src/fetcher/metadata_fetcher.cpp
    src/fetcher/http_fetcher.cpp
    src/fetcher/gguf_parser.cpp
    src/fetcher/config_fetcher.cpp
)

target_include_directories(metadata_fetcher PRIVATE src/fetcher)
target_link_libraries(metadata_fetcher PRIVATE winhttp)

# ============================================
# HTTP Client test (Step 2 smoke test)
# ============================================
add_executable(test_http src/test_curl.cpp)
target_link_libraries(test_http PRIVATE winhttp)

# ============================================
# Predictor Math (Step 3)
# ============================================
add_executable(predictor_test
    src/predictor/predictor_test.cpp
    src/predictor/predictor.cpp
    src/predictor/memory_predictor.cpp
    src/predictor/speed_predictor.cpp
    src/predictor/context_analyzer.cpp
    src/predictor/predictor_validation.cpp
    src/predictor/confidence_calculator.cpp
)

target_include_directories(predictor_test PRIVATE
    src/predictor
)

# ============================================
# Step 4: Wire It Together (Unified Binary)
# ============================================
add_executable(llm-planner
    src/step4/main.cpp
    src/step4/profiler.cpp
    src/step4/fetcher.cpp
    src/step4/matrix.cpp
    src/profiler/ram_profiler.cpp
    src/profiler/gpu_profiler.cpp
    src/profiler/disk_profiler.cpp
    src/fetcher/http_fetcher.cpp
    src/fetcher/gguf_parser.cpp
    src/fetcher/config_fetcher.cpp
    src/predictor/predictor.cpp
    src/predictor/memory_predictor.cpp
    src/predictor/speed_predictor.cpp
    src/predictor/context_analyzer.cpp
    src/predictor/predictor_validation.cpp
    src/predictor/confidence_calculator.cpp
)

target_include_directories(llm-planner PRIVATE
    include
    src/profiler
    src/fetcher
    src/predictor
    ${CUDA_PATH}/include
)

target_link_directories(llm-planner PRIVATE
    ${CUDA_PATH}/lib/x64
)

target_link_libraries(llm-planner PRIVATE nvml winhttp)

# ============================================
# Installation
# ============================================
install(TARGETS planner metadata_fetcher test_http predictor_test llm-planner DESTINATION bin)
