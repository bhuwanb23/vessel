# build_llama.ps1 — Build llama.cpp with CUDA support
# Run from: D:\projects\software\local_llm
#
# This script configures and builds llama.cpp with CUDA (NVIDIA GPU) support.
# It uses the Ninja generator with MSVC compiler and explicit CUDA paths.
#
# Prerequisites:
#   - Visual Studio 2022/2026 with "Desktop development with C++" workload
#   - CUDA Toolkit 12.x installed
#   - pip install ninja (for Ninja build system)

$CMAKE = "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
$LLAMA_DIR = "D:\projects\software\local_llm\llama.cpp"
$CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"

Write-Host "=== Configuring CMake (CUDA enabled, Release mode) ===" -ForegroundColor Cyan

# Clean previous build
Remove-Item "$LLAMA_DIR\build\CMakeCache.txt" -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force "$LLAMA_DIR\build\CMakeFiles" -ErrorAction SilentlyContinue

# Configure with Ninja, CUDA compiler, and -allow-unsupported-compiler flag
# (needed because CUDA 12.9 doesn't officially support VS 2026)
& $CMAKE -G Ninja -B "$LLAMA_DIR\build" -S "$LLAMA_DIR" `
    -DGGML_CUDA=ON `
    -DCMAKE_CUDA_ARCHITECTURES=120 `
    -DCMAKE_BUILD_TYPE=Release `
    -DCMAKE_CUDA_COMPILER="$CUDA_PATH\bin\nvcc.exe" `
    -DCMAKE_CUDA_FLAGS="-allow-unsupported-compiler"

if ($LASTEXITCODE -ne 0) {
    Write-Host "CMake configure failed!" -ForegroundColor Red
    exit 1
}

Write-Host "=== Building llama.cpp (this takes 5-15 min first time) ===" -ForegroundColor Cyan

& $CMAKE --build "$LLAMA_DIR\build" --config Release -j

if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed!" -ForegroundColor Red
    exit 1
}

# Verify
$llamaCli = "$LLAMA_DIR\build\bin\llama-cli.exe"
if (Test-Path $llamaCli) {
    Write-Host "=== Build successful! ===" -ForegroundColor Green
    Write-Host "Binary: $llamaCli"
    Write-Host ""
    Write-Host "=== Testing --help ===" -ForegroundColor Cyan
    & $llamaCli --help
} else {
    Write-Host "Build completed but binary not found at expected path." -ForegroundColor Yellow
    Write-Host "Searching for llama-cli.exe..."
    Get-ChildItem "$LLAMA_DIR\build" -Recurse -Filter "llama-cli.exe" | Select-Object FullName
}
