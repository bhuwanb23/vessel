# =============================================================================
# Vessel — Windows Installer
# Usage: powershell -c "irm https://raw.githubusercontent.com/bhuwanb23/vessel/main/install.ps1 | iex"
# =============================================================================
param(
    [string]$InstallDir = "$env:USERPROFILE\.vessel\bin"
)

$ErrorActionPreference = "Stop"
$Repo = "bhuwanb23/vessel"

# --- Helper functions ---
function Info($msg)  { Write-Host "  >> $msg" -ForegroundColor Cyan }
function Ok($msg)    { Write-Host "  OK $msg" -ForegroundColor Green }
function Warn($msg)  { Write-Host "  !! $msg" -ForegroundColor Yellow }
function Err($msg)   { Write-Host "  XX $msg" -ForegroundColor Red; exit 1 }

# --- Banner ---
Write-Host ""
Write-Host "  ================================"
Write-Host "    Vessel - Windows Installer"
Write-Host "  ================================"
Write-Host ""

# --- Get latest version ---
Info "Checking latest version..."
try {
    $release = Invoke-RestMethod -Uri "https://api.github.com/repos/$Repo/releases/latest" -UseBasicParsing
    $version = $release.tag_name
} catch {
    try {
        $tags = Invoke-RestMethod -Uri "https://api.github.com/repos/$Repo/tags" -UseBasicParsing
        $version = $tags[0].name
    } catch {
        Err "Could not determine latest version. Check https://github.com/$Repo/releases"
    }
}
Info "Latest version: $version"

# --- Download zip ---
$url = "https://github.com/$Repo/releases/download/$version/vessel-windows-amd64.zip"
$tmpZip = Join-Path $env:TEMP "vessel-install.zip"
$tmpDir = Join-Path $env:TEMP "vessel-install"

Info "Downloading vessel-windows-amd64.zip..."
try {
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    Invoke-WebRequest -Uri $url -OutFile $tmpZip -UseBasicParsing
} catch {
    # Fallback: try single exe (older releases)
    $urlExe = "https://github.com/$Repo/releases/download/$version/vessel-windows-amd64.exe"
    Info "Zip not found, trying single exe download..."
    try {
        $target = Join-Path $InstallDir "vessel.exe"
        if (!(Test-Path $InstallDir)) { New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null }
        Invoke-WebRequest -Uri $urlExe -OutFile $target -UseBasicParsing
        Ok "Installed vessel.exe to $target"
        # Add to PATH
        $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
        if ($userPath -notlike "*$InstallDir*") {
            Info "Adding $InstallDir to user PATH..."
            [Environment]::SetEnvironmentVariable("Path", "$userPath;$InstallDir", "User")
            $env:Path = "$env:Path;$InstallDir"
            Warn "Restart your terminal for PATH changes to take effect."
        }
        Write-Host ""
        Ok "Quick start:"
        Write-Host "  vessel --recommend"
        Write-Host "  vessel --serve --models-dir ./models"
        Write-Host ""
        exit 0
    } catch {
        Err "Download failed. URL: $url"
    }
}

# --- Extract ---
Info "Extracting..."
if (Test-Path $tmpDir) { Remove-Item $tmpDir -Recurse -Force }
Expand-Archive -Path $tmpZip -DestinationPath $tmpDir -Force

# --- Install ---
Info "Installing to $InstallDir..."
if (!(Test-Path $InstallDir)) { New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null }

# Copy all files from extracted dir
Get-ChildItem -Path $tmpDir -Recurse -File | ForEach-Object {
    $dest = Join-Path $InstallDir $_.Name
    Copy-Item $_.FullName $dest -Force
}

# Cleanup
Remove-Item $tmpZip -Force -ErrorAction SilentlyContinue
Remove-Item $tmpDir -Recurse -Force -ErrorAction SilentlyContinue

# --- Add to PATH ---
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($userPath -notlike "*$InstallDir*") {
    Info "Adding $InstallDir to user PATH..."
    [Environment]::SetEnvironmentVariable("Path", "$userPath;$InstallDir", "User")
    $env:Path = "$env:Path;$InstallDir"
    Warn "Restart your terminal for PATH changes to take effect."
}

# --- Verify ---
$exe = Join-Path $InstallDir "vessel.exe"
if (Test-Path $exe) {
    Ok "Installed vessel to $exe"
    try {
        $ver = & $exe --version 2>$null
        Ok "Version: $ver"
    } catch {
        Ok "Binary installed successfully"
    }
} else {
    Err "Installation failed. Binary not found at $exe"
}

# --- Done ---
Write-Host ""
Ok "Quick start:"
Write-Host ""
Write-Host "  vessel --recommend              # Find models for your hardware"
Write-Host "  vessel --model <url>            # Predict deployment strategy"
Write-Host "  vessel --serve                  # Start OpenAI-compatible API"
Write-Host ""
