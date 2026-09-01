# =============================================================================
# Vessel — Enterprise Installer (No raw.githubusercontent.com dependency)
# =============================================================================
# This installer uses ONLY GitHub Releases API and assets.
# It works in corporate environments (Zscaler, etc.) that block
# raw.githubusercontent.com but allow github.com/releases.
#
# Usage (download from GitHub Releases, then run locally):
#
#   # Step 1: Download the installer from GitHub Releases
#   Invoke-WebRequest -Uri "https://github.com/bhuwanb23/vessel/releases/latest/download/install-enterprise.ps1" -OutFile "install-enterprise.ps1"
#
#   # Step 2: Run the installer
#   powershell -ExecutionPolicy Bypass -File .\install-enterprise.ps1
#
# Or one-liner (if your environment allows it):
#   powershell -ExecutionPolicy Bypass -Command "& { Invoke-WebRequest -Uri 'https://github.com/bhuwanb23/vessel/releases/latest/download/install-enterprise.ps1' -OutFile '$env:TEMP\install-enterprise.ps1'; & $env:TEMP\install-enterprise.ps1 }"
#
# =============================================================================

param(
    [string]$InstallDir = "$env:USERPROFILE\.vessel\bin",
    [string]$Version = "",
    [switch]$Force,
    [switch]$NoVerify,
    [switch]$Help
)

$ErrorActionPreference = "Stop"

# =============================================================================
# Configuration
# =============================================================================
$Repo = "bhuwanb23/vessel"
$GitHubApiBase = "https://api.github.com/repos/$Repo"
$GitHubReleasesBase = "https://github.com/$Repo/releases"
$MaxRetries = 3
$RetryDelaySeconds = 5

# =============================================================================
# Helper Functions
# =============================================================================
function Write-Status {
    param([string]$Message, [string]$Level = "info")
    switch ($Level) {
        "info"    { Write-Host "  >> $Message" -ForegroundColor Cyan }
        "success" { Write-Host "  OK $Message" -ForegroundColor Green }
        "warn"    { Write-Host "  !! $Message" -ForegroundColor Yellow }
        "error"   { Write-Host "  XX $Message" -ForegroundColor Red }
        "step"    { Write-Host "`n--- $Message ---" -ForegroundColor White }
    }
}

function Write-Banner {
    Write-Host ""
    Write-Host "  ============================================" -ForegroundColor Cyan
    Write-Host "    Vessel — Enterprise Installer" -ForegroundColor Cyan
    Write-Host "    (GitHub Releases only, no raw.githubusercontent.com)" -ForegroundColor DarkGray
    Write-Host "  ============================================" -ForegroundColor Cyan
    Write-Host ""
}

function Show-Help {
    Write-Banner
    Write-Host "Usage:" -ForegroundColor White
    Write-Host "  .\install-enterprise.ps1 [options]"
    Write-Host ""
    Write-Host "Options:" -ForegroundColor White
    Write-Host "  -InstallDir <path>   Install directory (default: %USERPROFILE%\.vessel\bin)"
    Write-Host "  -Version <tag>       Install a specific version (default: latest)"
    Write-Host "  -Force               Overwrite existing installation"
    Write-Host "  -NoVerify            Skip post-install verification"
    Write-Host "  -Help                Show this help message"
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor White
    Write-Host "  .\install-enterprise.ps1                          # Install latest"
    Write-Host "  .\install-enterprise.ps1 -Version v0.2.0         # Install specific version"
    Write-Host "  .\install-enterprise.ps1 -Force                  # Force reinstall"
    Write-Host ""
    Write-Host "Installation directory: $InstallDir" -ForegroundColor DarkGray
    Write-Host ""
    exit 0
}

function Invoke-WithRetry {
    param(
        [scriptblock]$ScriptBlock,
        [string]$Description,
        [int]$Retries = $MaxRetries
    )

    $attempt = 0
    while ($true) {
        $attempt++
        try {
            return & $ScriptBlock
        }
        catch {
            if ($attempt -ge $Retries) {
                throw "$Description failed after $attempt attempts: $($_.Exception.Message)"
            }
            Write-Status "Attempt $attempt/$Retries failed for $Description. Retrying in ${RetryDelaySeconds}s..." "warn"
            Start-Sleep -Seconds $RetryDelaySeconds
        }
    }
}

# =============================================================================
# Step 1: Detect System Information
# =============================================================================
function Get-SystemInfo {
    Write-Status "Detecting system information" "step"

    # Detect OS
    $os = "windows"  # This is a PowerShell script, so it's Windows
    Write-Status "Detected OS: Windows"

    # Detect architecture
    $arch = $env:PROCESSOR_ARCHITECTURE
    switch ($arch) {
        "AMD64"   { $archName = "amd64"; Write-Status "Detected Architecture: x86_64 (amd64)" }
        "ARM64"   { $archName = "arm64"; Write-Status "Detected Architecture: ARM64" }
        default   {
            Write-Status "Unknown architecture: $arch. Defaulting to amd64." "warn"
            $archName = "amd64"
        }
    }

    return @{
        OS       = $os
        Arch     = $archName
        ArchRaw  = $arch
    }
}

# =============================================================================
# Step 2: Detect Latest Release
# =============================================================================
function Get-LatestRelease {
    param([string]$TargetVersion = "")

    Write-Status "Checking latest release" "step"

    if ($TargetVersion) {
        Write-Status "Using specified version: $TargetVersion"
        return $TargetVersion
    }

    Write-Status "Querying GitHub API for latest release..."

    try {
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

        $release = Invoke-RestMethod -Uri "$GitHubApiBase/releases/latest" -UseBasicParsing -TimeoutSec 30
        $version = $release.tag_name

        if (-not $version) {
            throw "No tag_name in response"
        }

        Write-Status "Latest version: $version" "success"
        Write-Status "Release: $($release.name)" "info"

        return $version
    }
    catch {
        Write-Status "Failed to fetch latest release from API: $($_.Exception.Message)" "warn"
        Write-Status "Trying tags endpoint as fallback..." "warn"

        try {
            $tags = Invoke-RestMethod -Uri "$GitHubApiBase/tags" -UseBasicParsing -TimeoutSec 30
            if ($tags -and $tags.Count -gt 0) {
                $version = $tags[0].name
                Write-Status "Latest tag: $version" "success"
                return $version
            }
        }
        catch {
            Write-Status "Tags endpoint also failed: $($_.Exception.Message)" "warn"
        }

        throw "Could not determine latest version. Check https://github.com/$Repo/releases"
    }
}

# =============================================================================
# Step 3: Select Correct Asset
# =============================================================================
function Select-Asset {
    param(
        [string]$Version,
        [hashtable]$SystemInfo
    )

    Write-Status "Selecting correct release asset" "step"

    $assetName = "vessel-$($SystemInfo.OS)-$($SystemInfo.Arch)"
    Write-Status "Target asset: $assetName.zip"

    # Check if asset exists in the release
    try {
        $release = Invoke-RestMethod -Uri "$GitHubApiBase/releases/tags/$Version" -UseBasicParsing -TimeoutSec 30
        $found = $false

        foreach ($asset in $release.assets) {
            if ($asset.name -eq "$assetName.zip") {
                $found = $true
                $downloadUrl = $asset.browser_download_url
                $fileSize = $asset.size
                Write-Status "Found asset: $($asset.name) ($([math]::Round($fileSize / 1MB, 1)) MB)" "success"
                break
            }
        }

        if (-not $found) {
            # Try .exe fallback (older releases)
            $exeName = "vessel-$($SystemInfo.OS)-$($SystemInfo.Arch).exe"
            foreach ($asset in $release.assets) {
                if ($asset.name -eq $exeName) {
                    $found = $true
                    $downloadUrl = $asset.browser_download_url
                    $fileSize = $asset.size
                    Write-Status "Found single executable: $($asset.name) ($([math]::Round($fileSize / 1MB, 1)) MB)" "success"
                    return @{
                        Url         = $downloadUrl
                        FileName    = $exeName
                        IsExe       = $true
                        FileSize    = $fileSize
                    }
                }
            }

            if (-not $found) {
                Write-Status "Available assets in this release:" "warn"
                foreach ($asset in $release.assets) {
                    Write-Status "  - $($asset.name)" "warn"
                }
                throw "No matching asset found for $assetName. See available assets above."
            }
        }

        return @{
            Url         = $downloadUrl
            FileName    = "$assetName.zip"
            IsExe       = $false
            FileSize    = $fileSize
        }
    }
    catch {
        Write-Status "Could not query release assets: $($_.Exception.Message)" "warn"
        Write-Status "Attempting direct download..." "warn"

        # Fallback: try direct download URL
        $downloadUrl = "$GitHubReleasesBase/download/$Version/$assetName.zip"
        return @{
            Url         = $downloadUrl
            FileName    = "$assetName.zip"
            IsExe       = $false
            FileSize    = 0
        }
    }
}

# =============================================================================
# Step 4: Download with Progress
# =============================================================================
function Download-Asset {
    param(
        [string]$Url,
        [string]$OutputPath,
        [string]$Description,
        [long]$ExpectedSize = 0
    )

    Write-Status "Downloading $Description" "step"

    $tempFile = Join-Path $env:TEMP "vessel-install-$([System.IO.Path]::GetRandomFileName())"

    try {
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

        # Use WebClient for progress reporting
        $webClient = New-Object System.Net.WebClient
        $webClient.Headers.Add("User-Agent", "Vessel-Installer/1.0")

        $downloadStarted = Get-Date

        # Register progress handler
        $progressActivity = "Downloading $Description"
        Register-ObjectEvent -InputObject $webClient -EventName DownloadProgressChanged -Action {
            $percent = $Event.SourceEventArgs.ProgressPercentage
            $bytesReceived = $Event.SourceEventArgs.BytesReceived
            $totalBytes = $Event.SourceEventArgs.TotalBytesToReceive

            if ($totalBytes -gt 0) {
                $mbReceived = [math]::Round($bytesReceived / 1MB, 1)
                $mbTotal = [math]::Round($totalBytes / 1MB, 1)
                Write-Progress -Activity $progressActivity -Status "$mbReceived / $mbTotal MB ($percent%)" -PercentComplete $percent
            }
        } | Out-Null

        try {
            $webClient.DownloadFile($Url, $tempFile)
        }
        finally {
            Write-Progress -Activity $progressActivity -Completed
            Get-EventSubscriber | Where-Object { $_.SourceObject -eq $webClient } | Unregister-Event -ErrorAction SilentlyContinue
        }

        $downloadTime = ((Get-Date) - $downloadStarted).TotalSeconds
        $fileSize = (Get-Item $tempFile).Length
        $speed = if ($downloadTime -gt 0) { [math]::Round($fileSize / $downloadTime / 1MB, 1) } else { 0 }

        Write-Status "Downloaded $([math]::Round($fileSize / 1MB, 1)) MB in $([math]::Round($downloadTime, 1))s ($speed MB/s)" "success"

        return $tempFile
    }
    catch {
        # Cleanup on failure
        if (Test-Path $tempFile) { Remove-Item $tempFile -Force -ErrorAction SilentlyContinue }
        throw "Download failed: $($_.Exception.Message)"
    }
}

# =============================================================================
# Step 5: Extract Archive
# =============================================================================
function Expand-ArchiveAsset {
    param(
        [string]$ArchivePath,
        [string]$IsExe
    )

    Write-Status "Extracting archive" "step"

    if ($IsExe) {
        # Single executable, no extraction needed
        Write-Status "Single executable file, no extraction needed" "info"
        return $ArchivePath
    }

    $extractDir = Join-Path $env:TEMP "vessel-extract-$([System.IO.Path]::GetRandomFileName())"

    try {
        if (Test-Path $extractDir) { Remove-Item $extractDir -Recurse -Force }
        New-Item -ItemType Directory -Path $extractDir -Force | Out-Null

        Expand-Archive -Path $ArchivePath -DestinationPath $extractDir -Force

        # Find the vessel executable
        $exe = Get-ChildItem -Path $extractDir -Recurse -Filter "vessel.exe" | Select-Object -First 1
        if (-not $exe) {
            # Try any executable
            $exe = Get-ChildItem -Path $extractDir -Recurse -Filter "*.exe" | Select-Object -First 1
        }

        if (-not $exe) {
            # Maybe it's a directory with the binary directly
            $files = Get-ChildItem -Path $extractDir -Recurse -File
            Write-Status "Extracted files:" "info"
            foreach ($f in $files) {
                Write-Status "  - $($f.Name)" "info"
            }
            throw "No executable found in archive"
        }

        Write-Status "Found executable: $($exe.Name)" "success"
        return $exe.FullName
    }
    catch {
        if (Test-Path $extractDir) { Remove-Item $extractDir -Recurse -Force -ErrorAction SilentlyContinue }
        throw "Extraction failed: $($_.Exception.Message)"
    }
}

# =============================================================================
# Step 6: Install Binary
# =============================================================================
function Install-Binary {
    param(
        [string]$SourcePath,
        [string]$InstallDirectory,
        [bool]$ForceInstall
    )

    Write-Status "Installing to $InstallDirectory" "step"

    # Check if already installed
    $targetExe = Join-Path $InstallDirectory "vessel.exe"
    if ((Test-Path $targetExe) -and -not $ForceInstall) {
        Write-Status "Vessel is already installed at $targetExe" "warn"
        Write-Status "Use -Force to overwrite" "warn"

        $response = Read-Host "Overwrite? (y/N)"
        if ($response -ne "y" -and $response -ne "Y") {
            Write-Status "Installation cancelled" "warn"
            exit 0
        }
    }

    # Create install directory
    if (-not (Test-Path $InstallDirectory)) {
        Write-Status "Creating directory: $InstallDirectory"
        New-Item -ItemType Directory -Path $InstallDirectory -Force | Out-Null
    }

    # Copy binary
    if ($SourcePath -ne $targetExe) {
        Copy-Item -Path $SourcePath -Destination $targetExe -Force
    }

    # Verify it was copied
    if (-not (Test-Path $targetExe)) {
        throw "Binary not found at $targetExe after installation"
    }

    $fileSize = (Get-Item $targetExe).Length
    Write-Status "Installed vessel.exe to $targetExe ($([math]::Round($fileSize / 1MB, 1)) MB)" "success"

    return $targetExe
}

# =============================================================================
# Step 7: Configure PATH
# =============================================================================
function Update-Path {
    param([string]$InstallDirectory)

    Write-Status "Configuring PATH" "step"

    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    if (-not $userPath) { $userPath = "" }

    # Check if already in PATH
    $pathEntries = $userPath -split ";" | Where-Object { $_.Trim() -ne "" }
    $alreadyInPath = $pathEntries | Where-Object { $_.Trim() -eq $InstallDirectory }

    if ($alreadyInPath) {
        Write-Status "Install directory already in user PATH" "success"
        return
    }

    # Add to PATH
    $newPath = if ($userPath) { "$userPath;$InstallDirectory" } else { $InstallDirectory }
    [Environment]::SetEnvironmentVariable("Path", $newPath, "User")

    # Update current session
    $env:Path = "$env:Path;$InstallDirectory"

    Write-Status "Added $InstallDirectory to user PATH" "success"
    Write-Status "Restart your terminal for PATH changes to take effect in other sessions" "warn"
}

# =============================================================================
# Step 8: Verify Installation
# =============================================================================
function Confirm-Installation {
    param(
        [string]$InstallDirectory,
        [bool]$SkipVerify
    )

    Write-Status "Verifying installation" "step"

    $exe = Join-Path $InstallDirectory "vessel.exe"

    if (-not (Test-Path $exe)) {
        throw "Binary not found at $exe"
    }

    if ($SkipVerify) {
        Write-Status "Skipping verification (-NoVerify flag)" "warn"
        Write-Status "Binary exists at $exe" "success"
        return
    }

    try {
        $ver = & $exe --version 2>&1
        if ($ver -match "vessel\s+(\S+)") {
            Write-Status "Version: $ver" "success"
        } else {
            Write-Status "Binary executes successfully" "success"
        }
    }
    catch {
        Write-Status "Warning: Could not run --version, but binary exists" "warn"
        Write-Status "This may be expected if GPU drivers are not installed" "info"
    }
}

# =============================================================================
# Step 9: Cleanup
# =============================================================================
function Remove-TempFiles {
    param([string[]]$Files)

    Write-Status "Cleaning up" "step"

    foreach ($file in $Files) {
        if ($file -and (Test-Path $file)) {
            Remove-Item $file -Force -ErrorAction SilentlyContinue
        }
    }

    Write-Status "Temporary files cleaned up" "success"
}

# =============================================================================
# Main
# =============================================================================
function Main {
    # Handle help flag
    if ($Help) { Show-Help }

    Write-Banner

    # Track temp files for cleanup
    $tempFiles = @()

    try {
        # Step 1: Detect system
        $systemInfo = Get-SystemInfo

        # Step 2: Get version
        $version = Get-LatestRelease -TargetVersion $Version

        # Step 3: Select asset
        $asset = Select-Asset -Version $version -SystemInfo $systemInfo

        # Step 4: Download
        $downloadPath = Download-Asset `
            -Url $asset.Url `
            -OutputPath (Join-Path $env:TEMP "vessel-download.zip") `
            -Description $asset.FileName `
            -ExpectedSize $asset.FileSize
        $tempFiles += $downloadPath

        # Step 5: Extract
        if ($asset.IsExe) {
            $binaryPath = $downloadPath
        } else {
            $binaryPath = Expand-ArchiveAsset -ArchivePath $downloadPath -IsExe $false
        }

        # Step 6: Install
        $installedPath = Install-Binary `
            -SourcePath $binaryPath `
            -InstallDirectory $InstallDir `
            -ForceInstall $Force

        # Step 7: Update PATH
        Update-Path -InstallDirectory $InstallDir

        # Step 8: Verify
        Confirm-Installation -InstallDirectory $InstallDir -SkipVerify $NoVerify

        # Step 9: Cleanup
        Remove-TempFiles -Files $tempFiles

        # Done!
        Write-Host ""
        Write-Host "  ============================================" -ForegroundColor Green
        Write-Host "    Installation Complete!" -ForegroundColor Green
        Write-Host "  ============================================" -ForegroundColor Green
        Write-Host ""
        Write-Host "  Installed to: $InstallDir\vessel.exe" -ForegroundColor White
        Write-Host "  Version:      $version" -ForegroundColor White
        Write-Host ""
        Write-Host "  Quick start:" -ForegroundColor Cyan
        Write-Host "    vessel --recommend                    # Find models for your hardware"
        Write-Host "    vessel --model <url>                  # Predict deployment strategy"
        Write-Host "    vessel --model <url> --execute        # Download and run inference"
        Write-Host "    vessel --serve                        # Start OpenAI-compatible API"
        Write-Host ""
        Write-Host "  Note: You may need to restart your terminal for PATH changes." -ForegroundColor Yellow
        Write-Host ""

    }
    catch {
        # Cleanup on error
        Remove-TempFiles -Files $tempFiles

        Write-Host ""
        Write-Status "Installation failed!" "error"
        Write-Host ""
        Write-Host "  Error: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host ""

        # Provide actionable guidance
        $errMsg = $_.Exception.Message.ToLower()
        if ($errMsg -match "network|connection|timeout|dns") {
            Write-Host "  Possible causes:" -ForegroundColor Yellow
            Write-Host "    - No internet connection" -ForegroundColor Yellow
            Write-Host "    - GitHub API is blocked by your firewall" -ForegroundColor Yellow
            Write-Host "    - Proxy configuration required" -ForegroundColor Yellow
            Write-Host ""
            Write-Host "  Try:" -ForegroundColor Yellow
            Write-Host "    - Check your internet connection" -ForegroundColor Yellow
            Write-Host "    - Verify github.com is accessible" -ForegroundColor Yellow
            Write-Host "    - Configure proxy: `$env:HTTP_PROXY = 'http://proxy:port'" -ForegroundColor Yellow
        }
        elseif ($errMsg -match "no matching asset") {
            Write-Host "  Possible causes:" -ForegroundColor Yellow
            Write-Host "    - No release available for your architecture ($($systemInfo.Arch))" -ForegroundColor Yellow
            Write-Host "    - Release naming convention changed" -ForegroundColor Yellow
            Write-Host ""
            Write-Host "  Try:" -ForegroundColor Yellow
            Write-Host "    - Visit https://github.com/$Repo/releases manually" -ForegroundColor Yellow
            Write-Host "    - Download the correct binary for your platform" -ForegroundColor Yellow
        }
        elseif ($errMsg -match "permission|access") {
            Write-Host "  Possible causes:" -ForegroundColor Yellow
            Write-Host "    - Insufficient permissions to write to $InstallDir" -ForegroundColor Yellow
            Write-Host ""
            Write-Host "  Try:" -ForegroundColor Yellow
            Write-Host "    - Run as Administrator" -ForegroundColor Yellow
            Write-Host "    - Choose a different install directory: -InstallDir 'C:\tools\vessel'" -ForegroundColor Yellow
        }
        else {
            Write-Host "  For help, visit: https://github.com/$Repo/issues" -ForegroundColor Yellow
        }

        Write-Host ""
        exit 1
    }
}

# Run main
Main
