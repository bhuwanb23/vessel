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
# Main executable
# ============================================
add_executable(planner
    src/main.cpp
    src/profiler/ram_profiler.cpp
)

target_include_directories(planner PRIVATE src)

# ============================================
# Installation
# ============================================
install(TARGETS planner DESTINATION bin)
