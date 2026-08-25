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
# External dependencies
# ============================================
set(CUDA_PATH "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9"
    CACHE PATH "CUDA Toolkit path")

# ============================================
# llama.cpp — Pre-built library (Option B)
# ============================================
# Why Option B instead of add_subdirectory (Option A)?
# CUDA 12.9 does not support VS 2026 as a host compiler, so building
# llama.cpp from source via add_subdirectory fails at CUDA compilation.
# llama.cpp is built separately in Step 0 and we link against the result.
#
# To rebuild llama.cpp:
#   cd llama.cpp && mkdir build && cd build
#   cmake .. -G "Visual Studio 18 2026" -A x64 ^
#       -DGGML_CUDA=ON -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF
#   cmake --build . --config Release
# ============================================
set(LLAMA_PATH "${CMAKE_SOURCE_DIR}/llama.cpp"
    CACHE PATH "llama.cpp source path")
set(LLAMA_BUILD "${LLAMA_PATH}/build"
    CACHE PATH "llama.cpp build path")

# ============================================
# Shared source files (reused by multiple targets)
# ============================================
set(PROFILER_SOURCES
    src/profiler/ram_profiler.cpp
    src/profiler/gpu_profiler.cpp
    src/profiler/disk_profiler.cpp
)

set(FETCHER_SOURCES
    src/fetcher/http_fetcher.cpp
    src/fetcher/gguf_parser.cpp
    src/fetcher/config_fetcher.cpp
)

set(PREDICTOR_SOURCES
    src/predictor/predictor.cpp
    src/predictor/memory_predictor.cpp
    src/predictor/speed_predictor.cpp
    src/predictor/context_analyzer.cpp
    src/predictor/predictor_validation.cpp
    src/predictor/confidence_calculator.cpp
    src/predictor/moe_predictor.cpp
)

set(PLANNER_SOURCES
    src/planner/profiler.cpp
    src/planner/fetcher.cpp
    src/planner/matrix.cpp
    src/planner/ranker.cpp
    src/planner/output.cpp
    src/planner/executor.cpp
    src/planner/hardware_sampler.cpp
    src/planner/comparison_report.cpp
    src/planner/hardware_fingerprint.cpp
    src/planner/calibration_log.cpp
    src/planner/calibration_aggregator.cpp
    src/planner/download_manager.cpp
    src/planner/moe_placer.cpp
    src/planner/moe_tensor_overrides.cpp
)

# Step 10: Hot/Cold CPU-GPU Offload
set(HOTCOLD_SOURCES
    src/hotcold/profiler.cpp
    src/hotcold/weight_splitter.cpp
    src/hotcold/sparse_ffn.cpp
    src/hotcold/layer_streamer.cpp
    src/hotcold/hotcold_predictor.cpp
    src/hotcold/neuron_profiler.cpp
    src/hotcold/mask_file.cpp
    src/hotcold/hotcold_executor.cpp
)

# Step 11: Platform Abstraction
set(PLATFORM_SOURCES
    src/platform/nvidia_profiler.cpp
    # Future: src/platform/amd_profiler.cpp
    # Future: src/platform/apple_profiler.cpp
    # Future: src/platform/cpu_profiler.cpp
)

set(SHARED_INCLUDE_DIRS
    include
    include/hotcold
    include/platform
    src/profiler
    src/fetcher
    src/predictor
    ${CUDA_PATH}/include
    ${LLAMA_PATH}/include
    ${LLAMA_PATH}/ggml/include
)

set(SHARED_LINK_DIRS
    ${CUDA_PATH}/lib/x64
    ${LLAMA_BUILD}/src
    ${LLAMA_BUILD}/ggml/src
    ${LLAMA_BUILD}/ggml/src/ggml-cuda
    ${LLAMA_BUILD}/bin
)

# llama.cpp libraries (link in dependency order)
set(LLAMA_LIBS llama ggml ggml-base ggml-cpu ggml-cuda)

# ============================================
# llm-planner — Main unified binary
# ============================================
add_executable(llm-planner
    src/planner/main.cpp
    ${PLANNER_SOURCES}
    ${PROFILER_SOURCES}
    ${FETCHER_SOURCES}
    ${PREDICTOR_SOURCES}
    ${HOTCOLD_SOURCES}
    ${PLATFORM_SOURCES}
)

target_include_directories(llm-planner PRIVATE ${SHARED_INCLUDE_DIRS})
target_link_directories(llm-planner PRIVATE ${SHARED_LINK_DIRS})
target_link_libraries(llm-planner PRIVATE ${LLAMA_LIBS} nvml winhttp)

# ============================================
# e2e_test — End-to-end test suite
# ============================================
add_executable(e2e_test
    src/planner/e2e_test.cpp
    ${PLANNER_SOURCES}
    ${PROFILER_SOURCES}
    ${FETCHER_SOURCES}
    ${PREDICTOR_SOURCES}
    ${HOTCOLD_SOURCES}
    ${PLATFORM_SOURCES}
)

target_include_directories(e2e_test PRIVATE ${SHARED_INCLUDE_DIRS})
target_link_directories(e2e_test PRIVATE ${SHARED_LINK_DIRS})
target_link_libraries(e2e_test PRIVATE ${LLAMA_LIBS} nvml winhttp)

# ============================================
# executor_test — Step 6 smoke test
# ============================================
add_executable(executor_test
    src/planner/executor_test.cpp
    src/planner/executor.cpp
    src/planner/hardware_sampler.cpp
    src/planner/comparison_report.cpp
    src/planner/profiler.cpp
    src/planner/output.cpp
    src/planner/ranker.cpp
    src/planner/moe_placer.cpp
    src/planner/moe_tensor_overrides.cpp
    src/planner/hardware_fingerprint.cpp
    src/hotcold/hotcold_executor.cpp
    src/hotcold/mask_file.cpp
    ${PROFILER_SOURCES}
    ${PREDICTOR_SOURCES}
)

target_include_directories(executor_test PRIVATE ${SHARED_INCLUDE_DIRS})
target_link_directories(executor_test PRIVATE ${SHARED_LINK_DIRS})
target_link_libraries(executor_test PRIVATE ${LLAMA_LIBS} nvml winhttp)

# ============================================
# executor_phase_j_test — Phase J test suite
# ============================================
add_executable(executor_phase_j_test
    src/planner/executor_phase_j_test.cpp
    src/planner/executor.cpp
    src/planner/hardware_sampler.cpp
    src/planner/moe_placer.cpp
    src/planner/moe_tensor_overrides.cpp
    src/planner/hardware_fingerprint.cpp
    src/hotcold/hotcold_executor.cpp
    src/hotcold/mask_file.cpp
    ${PROFILER_SOURCES}
    ${PREDICTOR_SOURCES}
)

target_include_directories(executor_phase_j_test PRIVATE ${SHARED_INCLUDE_DIRS})
target_link_directories(executor_phase_j_test PRIVATE ${SHARED_LINK_DIRS})
target_link_libraries(executor_phase_j_test PRIVATE ${LLAMA_LIBS} nvml winhttp)

# ============================================
# calibration_log_test — Step 7 round-trip test
# ============================================
add_executable(calibration_log_test
    src/planner/calibration_log_test.cpp
    src/planner/calibration_log.cpp
    src/planner/hardware_fingerprint.cpp
)

target_include_directories(calibration_log_test PRIVATE
    include
)

target_link_libraries(calibration_log_test PRIVATE winhttp)

# ============================================
# predictor_test — Step 3 standalone test
# ============================================
add_executable(predictor_test
    src/predictor/predictor_test.cpp
    src/planner/moe_placer.cpp
    ${PREDICTOR_SOURCES}
)

# ============================================
# sparse_ffn_test — Step 10 Phase D validation
# ============================================
add_executable(sparse_ffn_test
    src/hotcold/sparse_ffn_test.cpp
    src/hotcold/sparse_ffn.cpp
    src/hotcold/mask_file.cpp
)

target_include_directories(sparse_ffn_test PRIVATE
    include
    include/hotcold
    ${CUDA_PATH}/include
    ${LLAMA_PATH}/include
    ${LLAMA_PATH}/ggml/include
)

target_link_directories(sparse_ffn_test PRIVATE
    ${CUDA_PATH}/lib/x64
    ${LLAMA_BUILD}/src
    ${LLAMA_BUILD}/ggml/src
    ${LLAMA_BUILD}/ggml/src/ggml-cpu
    ${LLAMA_BUILD}/ggml/src/ggml-cuda
)

target_link_libraries(sparse_ffn_test PRIVATE llama ggml ggml-base ggml-cpu ggml-cuda)

# ============================================
# Layer-Streaming Prediction Test
# ============================================
add_executable(layer_stream_test
    src/hotcold/layer_stream_test.cpp
    src/hotcold/hotcold_predictor.cpp
    src/hotcold/weight_splitter.cpp
    src/predictor/predictor.cpp
    src/predictor/speed_predictor.cpp
    src/predictor/memory_predictor.cpp
    src/predictor/confidence_calculator.cpp
    src/predictor/moe_predictor.cpp
    src/predictor/context_analyzer.cpp
    src/planner/moe_placer.cpp
)

target_include_directories(layer_stream_test PRIVATE
    include
    src/predictor
)

target_link_libraries(layer_stream_test PRIVATE ws2_32)

# ============================================
# Phase I Validation Test
# ============================================
add_executable(phase_i_test
    src/hotcold/phase_i_validation.cpp
    src/hotcold/sparse_ffn.cpp
    src/hotcold/mask_file.cpp
    src/hotcold/hotcold_predictor.cpp
    src/hotcold/weight_splitter.cpp
    src/predictor/predictor.cpp
    src/predictor/speed_predictor.cpp
    src/predictor/memory_predictor.cpp
    src/predictor/confidence_calculator.cpp
    src/predictor/moe_predictor.cpp
    src/predictor/context_analyzer.cpp
    src/planner/moe_placer.cpp
)

target_include_directories(phase_i_test PRIVATE
    include
    include/hotcold
    src/predictor
)

target_link_libraries(phase_i_test PRIVATE ws2_32)

target_include_directories(predictor_test PRIVATE
    include
    src/predictor
)

# ============================================
# Installation
# ============================================
install(TARGETS llm-planner e2e_test executor_test predictor_test DESTINATION bin)
