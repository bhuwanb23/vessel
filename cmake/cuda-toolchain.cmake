# Custom CUDA toolchain for building llama.cpp without VS CUDA integration
# This tells CMake to use nvcc directly as the CUDA compiler

set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/bin/nvcc.exe")
set(CMAKE_C_COMPILER "C:/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/MSVC/14.50.35717/bin/Hostx64/x64/cl.exe")
set(CMAKE_CXX_COMPILER "C:/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/MSVC/14.50.35717/bin/Hostx64/x64/cl.exe")
set(CMAKE_CUDA_HOST_COMPILER "C:/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/MSVC/14.50.35717/bin/Hostx64/x64/cl.exe")

# Use Ninja generator which doesn't need VS CUDA toolset
set(CMAKE_GENERATOR "Ninja" CACHE STRING "Build generator" FORCE)

# CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES "120" CACHE STRING "CUDA architecture" FORCE)
