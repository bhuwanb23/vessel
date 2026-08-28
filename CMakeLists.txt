cmake_minimum_required(VERSION 3.14)
project(vessel LANGUAGES C CXX)

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
# Platform detection
# ============================================
if(WIN32)
    set(PLATFORM_WINDOWS TRUE)
elseif(APPLE)
    set(PLATFORM_MACOS TRUE)
elseif(UNIX)
    set(PLATFORM_LINUX TRUE)
endif()

# ============================================
# External dependencies
# ============================================
# CUDA (optional)
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    set(HAS_CUDA TRUE)
    message(STATUS "CUDA found: ${CUDAToolkit_VERSION}")
else()
    set(HAS_CUDA FALSE)
    message(STATUS "CUDA not found — building CPU-only")
endif()

# NVML (Windows only, for GPU monitoring)
if(PLATFORM_WINDOWS)
    set(HAS_NVML TRUE)
else()
    set(HAS_NVML FALSE)
endif()

# ============================================
# llama.cpp — Pre-built library
# ============================================
set(LLAMA_PATH "${CMAKE_SOURCE_DIR}/llama.cpp" CACHE PATH "llama.cpp source path")
set(LLAMA_BUILD "${LLAMA_PATH}/build" CACHE PATH "llama.cpp build path")

# ============================================
# Platform-specific source files
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

set(RECOMMEND_SOURCES
    src/recommend/catalog_loader.cpp
    src/recommend/recommendation_engine.cpp
)

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

# Platform-specific backends
set(PLATFORM_SOURCES
    src/platform/cpu_profiler.cpp
    src/platform/platform_factory.cpp
)

if(PLATFORM_WINDOWS)
    list(APPEND PLATFORM_SOURCES
        src/platform/nvidia_profiler.cpp
        src/platform/nvidia_executor.cpp
    )
endif()

# ============================================
# Include directories
# ============================================
set(SHARED_INCLUDE_DIRS
    include
    include/hotcold
    include/platform
    include/recommend
    src/profiler
    src/fetcher
    src/predictor
    src/recommend
    ${LLAMA_PATH}/include
    ${LLAMA_PATH}/ggml/include
)

if(HAS_CUDA)
    list(APPEND SHARED_INCLUDE_DIRS ${CUDAToolkit_INCLUDE_DIRS})
endif()

# ============================================
# Link directories
# ============================================
set(SHARED_LINK_DIRS
    ${LLAMA_BUILD}/src
    ${LLAMA_BUILD}/ggml/src
    ${LLAMA_BUILD}/bin
)

if(HAS_CUDA)
    list(APPEND SHARED_LINK_DIRS ${LLAMA_BUILD}/ggml/src/ggml-cuda)
endif()

# ============================================
# llama.cpp libraries
# ============================================
set(LLAMA_LIBS llama ggml ggml-base ggml-cpu)
if(HAS_CUDA)
    list(APPEND LLAMA_LIBS ggml-cuda)
endif()

# ============================================
# Platform-specific link libraries
# ============================================
set(PLATFORM_LIBS "")
if(PLATFORM_WINDOWS)
    list(APPEND PLATFORM_LIBS nvml winhttp ws2_32)
elseif(PLATFORM_LINUX)
    list(APPEND PLATFORM_LIBS curl pthread dl)
elseif(PLATFORM_MACOS)
    list(APPEND PLATFORM_LIBS curl pthread)
endif()

# ============================================
# vessel — Main unified binary
# ============================================
add_executable(vessel
    src/planner/main.cpp
    ${PLANNER_SOURCES}
    ${PROFILER_SOURCES}
    ${FETCHER_SOURCES}
    ${PREDICTOR_SOURCES}
    ${HOTCOLD_SOURCES}
    ${PLATFORM_SOURCES}
    ${RECOMMEND_SOURCES}
)

target_include_directories(vessel PRIVATE ${SHARED_INCLUDE_DIRS})
target_link_directories(vessel PRIVATE ${SHARED_LINK_DIRS})
target_link_libraries(vessel PRIVATE ${LLAMA_LIBS} ${PLATFORM_LIBS})

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
target_link_libraries(e2e_test PRIVATE ${LLAMA_LIBS} ${PLATFORM_LIBS})

# ============================================
# predictor_test — Step 3 standalone test
# ============================================
add_executable(predictor_test
    src/predictor/predictor_test.cpp
    src/planner/moe_placer.cpp
    ${PREDICTOR_SOURCES}
)

target_include_directories(predictor_test PRIVATE
    include
    src/predictor
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
    ${LLAMA_PATH}/include
    ${LLAMA_PATH}/ggml/include
)

target_link_directories(sparse_ffn_test PRIVATE
    ${LLAMA_BUILD}/src
    ${LLAMA_BUILD}/ggml/src
    ${LLAMA_BUILD}/ggml/src/ggml-cpu
)

if(HAS_CUDA)
    target_link_directories(sparse_ffn_test PRIVATE ${LLAMA_BUILD}/ggml/src/ggml-cuda)
    target_link_libraries(sparse_ffn_test PRIVATE llama ggml ggml-base ggml-cpu ggml-cuda)
else()
    target_link_libraries(sparse_ffn_test PRIVATE llama ggml ggml-base ggml-cpu)
endif()

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

if(PLATFORM_WINDOWS)
    target_link_libraries(layer_stream_test PRIVATE ws2_32)
endif()

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

if(PLATFORM_WINDOWS)
    target_link_libraries(phase_i_test PRIVATE ws2_32)
endif()

# ============================================
# Step 11 Phase I Validation Test
# ============================================
add_executable(step11_test
    src/platform/step11_validation.cpp
    src/platform/platform_factory.cpp
    src/platform/cpu_profiler.cpp
    src/predictor/predictor.cpp
    src/predictor/speed_predictor.cpp
    src/predictor/memory_predictor.cpp
    src/predictor/confidence_calculator.cpp
    src/predictor/moe_predictor.cpp
    src/predictor/context_analyzer.cpp
    src/hotcold/hotcold_predictor.cpp
    src/hotcold/weight_splitter.cpp
    src/planner/moe_placer.cpp
)

if(PLATFORM_WINDOWS)
    list(APPEND step11_test_SOURCES
        src/platform/nvidia_profiler.cpp
        src/platform/nvidia_executor.cpp
    )
    target_sources(step11_test PRIVATE ${step11_test_SOURCES})
endif()

target_include_directories(step11_test PRIVATE
    include
    include/hotcold
    include/platform
    src/predictor
    src/planner
    ${LLAMA_PATH}/include
    ${LLAMA_PATH}/ggml/include
)

target_link_directories(step11_test PRIVATE
    ${LLAMA_BUILD}/src
    ${LLAMA_BUILD}/ggml/src
    ${LLAMA_BUILD}/bin
)

target_link_libraries(step11_test PRIVATE ${PLATFORM_LIBS})

# ============================================
# Step 12 Phase A — Catalog Test
# ============================================
add_executable(catalog_test
    src/recommend/catalog_test.cpp
    src/recommend/catalog_loader.cpp
)

target_include_directories(catalog_test PRIVATE
    include
    include/recommend
)

# ============================================
# Step 12 Phase B — Recommendation Engine Test
# ============================================
add_executable(recommend_test
    src/recommend/recommendation_test.cpp
    src/recommend/recommendation_engine.cpp
    src/recommend/catalog_loader.cpp
    src/predictor/predictor.cpp
    src/predictor/speed_predictor.cpp
    src/predictor/memory_predictor.cpp
    src/predictor/confidence_calculator.cpp
    src/predictor/moe_predictor.cpp
    src/predictor/context_analyzer.cpp
    src/planner/moe_placer.cpp
)

target_include_directories(recommend_test PRIVATE
    include
    include/recommend
    include/hotcold
    src/predictor
    src/planner
    ${LLAMA_PATH}/include
    ${LLAMA_PATH}/ggml/include
)

target_link_directories(recommend_test PRIVATE
    ${LLAMA_BUILD}/src
    ${LLAMA_BUILD}/ggml/src
    ${LLAMA_BUILD}/bin
)

if(HAS_CUDA)
    target_link_directories(recommend_test PRIVATE ${LLAMA_BUILD}/ggml/src/ggml-cuda)
    target_link_libraries(recommend_test PRIVATE llama ggml ggml-base ggml-cpu ggml-cuda ${PLATFORM_LIBS})
else()
    target_link_libraries(recommend_test PRIVATE llama ggml ggml-base ggml-cpu ${PLATFORM_LIBS})
endif()

# ============================================
# Installation
# ============================================
install(TARGETS vessel e2e_test predictor_test DESTINATION bin)
