cmake_minimum_required(VERSION 3.14)
project(local-llm-planner LANGUAGES CXX)

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
# llama.cpp integration
# ============================================
# Option A: Link against a pre-built llama.cpp (recommended for development)
# Option B: Add llama.cpp as a subdirectory (builds together)
#
# For now, we build llama.cpp separately via build_llama.ps1,
# then reference it via find_package or manual linking.
# This keeps build times short during development.

# TODO: Uncomment when ready to link llama.cpp
# add_subdirectory(external/llama.cpp)

# ============================================
# Main executable
# ============================================
add_executable(planner src/main.cpp)

# TODO: Link against llama.cpp when ready
# target_link_libraries(planner PRIVATE llama llama-common)

# ============================================
# Installation
# ============================================
install(TARGETS planner DESTINATION bin)
