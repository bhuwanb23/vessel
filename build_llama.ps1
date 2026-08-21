# build_llama.ps1 — Build llama.cpp with CUDA support
# Run from: D:\projects\software\local_llm

$CMAKE = "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
$VSVARS = "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat"
$LLAMA_DIR = "D:\projects\software\local_llm\llama.cpp"

# Step 1: Configure with CMake (run inside VS developer environment)
Write-Host "=== Configuring CMake (CUDA enabled, Release mode) ===" -ForegroundColor Cyan

# Use cmd.exe to set up VS environment and run cmake in one shell
$configureCmd = @"
call "$VSVARS" amd64 >nul 2>&1
"$CMAKE" -B "$LLAMA_DIR\build" -S "$LLAMA_DIR" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release
"@

cmd /c $configureCmd

if ($LASTEXITCODE -ne 0) {
    Write-Host "CMake configure failed!" -ForegroundColor Red
    exit 1
}

# Step 2: Build (also inside VS developer environment)
Write-Host "=== Building llama.cpp (this takes 5-15 min first time) ===" -ForegroundColor Cyan

$buildCmd = @"
call "$VSVARS" amd64 >nul 2>&1
"$CMAKE" --build "$LLAMA_DIR\build" --config Release -j
"@

cmd /c $buildCmd

if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed!" -ForegroundColor Red
    exit 1
}

# Step 3: Verify
$llamaCli = "$LLAMA_DIR\build\bin\Release\llama-cli.exe"
if (Test-Path $llamaCli) {
    Write-Host "=== Build successful! ===" -ForegroundColor Green
    Write-Host "Binary: $llamaCli"
    Write-Host ""
    Write-Host "=== Testing --help ===" -ForegroundColor Cyan
    & $llamaCli --help
} else {
    # Try alternate output path
    $llamaCli2 = "$LLAMA_DIR\build\bin\Release\main.exe"
    if (Test-Path $llamaCli2) {
        Write-Host "=== Build successful! ===" -ForegroundColor Green
        Write-Host "Binary: $llamaCli2"
        & $llamaCli2 --help
    } else {
        Write-Host "Build completed but binary not found at expected path." -ForegroundColor Yellow
        Write-Host "Searching for llama-cli.exe..."
        Get-ChildItem "$LLAMA_DIR\build" -Recurse -Filter "llama-cli.exe" | Select-Object FullName
        Get-ChildItem "$LLAMA_DIR\build" -Recurse -Filter "main.exe" | Select-Object FullName
    }
}
