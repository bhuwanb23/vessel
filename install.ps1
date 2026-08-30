# =============================================================================
# Vessel — Windows Installer
# Usage: powershell -c "irm https://raw.githubusercontent.com/bhuwanb23/vessel/main/install.ps1 | iex"
# =============================================================================
param(
    [string]$InstallDir = "$env:USERPROFILE\.vessel\bin"
)

$ErrorActionPreference = "Stop"
$Repo = "bhuwanb23/vessel"
$BinaryName = "vessel.exe"

# --- Helper functions ---
function Info($msg)  { Write-Host "  `u{25B8} $msg" -ForegroundColor Cyan }
function Ok($msg)    { Write-Host "  `u{2714} $msg" -ForegroundColor Green }
function Warn($msg)  { Write-Host "  `u{26A0} $msg" -ForegroundColor Yellow }
function Err($msg)   { Write-Host "  `u{2716} $msg" -ForegroundColor Red; exit 1 }

# --- Banner ---
Write-Host ""
Write-Host "  `u{2554}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2557}" -ForegroundColor Cyan
Write-Host "  `u{2551}     Vessel `u{2014} Windows Installer     `u{2551}" -ForegroundColor Cyan
Write-Host "  `u{255A}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{2550}`u{255D}" -ForegroundColor Cyan
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

# --- Download ---
$assetName = "vessel-windows-amd64.exe"
$url = "https://github.com/$Repo/releases/download/$version/$assetName"
$tmpFile = Join-Path $env:TEMP "vessel-install.exe"

Info "Downloading $assetName..."
try {
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    Invoke-WebRequest -Uri $url -OutFile $tmpFile -UseBasicParsing
} catch {
    Err "Download failed. URL: $url"
}

# --- Install ---
Info "Installing to $InstallDir..."
if (!(Test-Path $InstallDir)) {
    New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
}

$target = Join-Path $InstallDir $BinaryName
Copy-Item $tmpFile $target -Force
Remove-Item $tmpFile -Force

# --- Add to PATH if needed ---
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($userPath -notlike "*$InstallDir*") {
    Info "Adding $InstallDir to user PATH..."
    [Environment]::SetEnvironmentVariable("Path", "$userPath;$InstallDir", "User")
    $env:Path = "$env:Path;$InstallDir"
    Warn "You may need to restart your terminal for PATH changes to take effect."
}

# --- Verify ---
if (Test-Path $target) {
    Ok "Installed vessel to $target"
    try {
        $ver = & $target --version 2>$null
        Ok "Version: $ver"
    } catch {
        Ok "Binary installed successfully"
    }
} else {
    Err "Installation failed. Binary not found at $target"
}

# --- Done ---
Write-Host ""
Ok "Quick start:"
Write-Host ""
Write-Host "  vessel --recommend              # Find models for your hardware"
Write-Host "  vessel --model <url>            # Predict deployment strategy"
Write-Host "  vessel --serve                  # Start OpenAI-compatible API"
Write-Host ""
