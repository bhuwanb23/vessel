@echo off
echo === Setting up MSVC environment ===
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" amd64 >nul 2>&1

echo === Configuring CMake with Ninja (CUDA enabled, Release mode) ===
cd /d D:\projects\software\local_llm\llama.cpp

REM Clean previous build artifacts
if exist build\CMakeCache.txt del build\CMakeCache.txt
if exist build\CMakeFiles rmdir /s /q build\CMakeFiles

REM Set CUDA path
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9

REM Configure with Ninja generator
REM Note: -allow-unsupported-compiler is needed because CUDA 12.9 doesn't officially support VS 2026
cmake -G Ninja -B build -S . ^
    -DGGML_CUDA=ON ^
    -DCMAKE_CUDA_ARCHITECTURES=120 ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_CUDA_COMPILER="%CUDA_PATH%\bin\nvcc.exe" ^
    -DCMAKE_CUDA_FLAGS="-allow-unsupported-compiler"

if %ERRORLEVEL% neq 0 (
    echo CMake configure failed!
    exit /b 1
)

echo === Building llama.cpp (this takes 5-15 min first time) ===
cmake --build build --config Release -j

if %ERRORLEVEL% neq 0 (
    echo Build failed!
    exit /b 1
)

REM Verify
set LLAMA_CLI=build\bin\llama-cli.exe
if exist %LLAMA_CLI% (
    echo === Build successful! ===
    echo Binary: %CD%\%LLAMA_CLI%
    echo.
    echo === Testing --help ===
    %LLAMA_CLI% --help
) else (
    echo Build completed but binary not found at expected path.
    echo Searching for llama-cli.exe...
    dir /s /b build\*llama-cli.exe 2>nul
    dir /s /b build\*main.exe 2>nul
)
