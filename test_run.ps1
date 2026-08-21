# test_run.ps1 — Run llama.cpp with test model (Phase D)
# Run this after downloading the GGUF model file.

$LLAMA_CLI = "D:\projects\software\local_llm\llama.cpp\build\bin\llama-cli.exe"
$MODEL = "D:\projects\software\local_llm\models\Llama-3.2-3B-Instruct-Q4_K_M.gguf"

# Check model exists
if (-not (Test-Path $MODEL)) {
    Write-Host "Model not found at: $MODEL" -ForegroundColor Red
    Write-Host "Download it first:" -ForegroundColor Yellow
    Write-Host 'curl -L -o "D:\projects\software\local_llm\models\Llama-3.2-3B-Instruct-Q4_K_M.gguf" "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf"'
    exit 1
}

Write-Host "=== Running llama-cli with Llama-3.2-3B-Instruct-Q4_K_M ===" -ForegroundColor Cyan
Write-Host "GPU layers: 20 (fits in 8GB VRAM for 3B model)" -ForegroundColor Cyan
Write-Host "Prompt: The capital of France is" -ForegroundColor Cyan
Write-Host "Token limit: 50" -ForegroundColor Cyan
Write-Host ""

& $LLAMA_CLI `
    -m $MODEL `
    -ngl 20 `
    -p "The capital of France is" `
    -n 50 `
    --verbose

Write-Host ""
Write-Host "=== Done! Record numbers in baseline.txt ===" -ForegroundColor Green
