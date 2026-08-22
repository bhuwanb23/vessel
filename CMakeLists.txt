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

# llama.cpp (built in Step 0)
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
)

set(PLANNER_SOURCES
    src/planner/profiler.cpp
    src/planner/fetcher.cpp
    src/planner/matrix.cpp
    src/planner/ranker.cpp
    src/planner/output.cpp
    src/planner/executor.cpp
)

set(SHARED_INCLUDE_DIRS
    include
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
)

target_include_directories(executor_test PRIVATE
    include
    ${LLAMA_PATH}/include
    ${LLAMA_PATH}/ggml/include
)
target_link_directories(executor_test PRIVATE
    ${LLAMA_BUILD}/src
    ${LLAMA_BUILD}/ggml/src
    ${LLAMA_BUILD}/ggml/src/ggml-cuda
    ${LLAMA_BUILD}/bin
)
target_link_libraries(executor_test PRIVATE ${LLAMA_LIBS})

# ============================================
# predictor_test — Step 3 standalone test
# ============================================
add_executable(predictor_test
    src/predictor/predictor_test.cpp
    ${PREDICTOR_SOURCES}
)

target_include_directories(predictor_test PRIVATE
    include
    src/predictor
)

# ============================================
# Installation
# ============================================
install(TARGETS llm-planner e2e_test executor_test predictor_test DESTINATION bin)
