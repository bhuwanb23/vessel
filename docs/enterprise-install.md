# Enterprise Installation Guide

This guide covers installing Vessel in corporate environments where `raw.githubusercontent.com` is blocked by firewalls or proxies (e.g., Zscaler, Cisco Umbrella, etc.).

## Why the Standard Installer Fails

The standard one-line install commands use `raw.githubusercontent.com`:

```powershell
# Standard installer (may be blocked)
powershell -c "irm https://raw.githubusercontent.com/bhuwanb23/vessel/main/install.ps1 | iex"
```

Many corporate environments block `raw.githubusercontent.com` while allowing `github.com` and its Releases CDN. The enterprise installer solves this by using **only** GitHub Releases API and assets.

## Enterprise Installers

| Platform | Script | Download |
|----------|--------|----------|
| Windows | `install-enterprise.ps1` | [GitHub Releases](https://github.com/bhuwanb23/vessel/releases/latest/download/install-enterprise.ps1) |
| Linux/macOS | `install-enterprise.sh` | [GitHub Releases](https://github.com/bhuwanb23/vessel/releases/latest/download/install-enterprise.sh) |

## Quick Start

### Windows (PowerShell)

```powershell
# Step 1: Download the installer
Invoke-WebRequest -Uri "https://github.com/bhuwanb23/vessel/releases/latest/download/install-enterprise.ps1" -OutFile "install-enterprise.ps1"

# Step 2: Run the installer
powershell -ExecutionPolicy Bypass -File .\install-enterprise.ps1
```

### Linux / macOS

```bash
# Step 1: Download the installer
curl -fsSL -o install-enterprise.sh https://github.com/bhuwanb23/vessel/releases/latest/download/install-enterprise.sh

# Step 2: Make executable and run
chmod +x install-enterprise.sh
./install-enterprise.sh
```

## Installer Options

### PowerShell (`install-enterprise.ps1`)

```powershell
.\install-enterprise.ps1 [options]

Options:
  -InstallDir <path>   Install directory (default: %USERPROFILE%\.vessel\bin)
  -Version <tag>       Install a specific version (e.g., v0.2.0)
  -Force               Overwrite existing installation without prompting
  -NoVerify            Skip post-install verification
  -Help                Show help message
```

### Bash (`install-enterprise.sh`)

```bash
./install-enterprise.sh [options]

Options:
  --install-dir <path>   Install directory (default: ~/.vessel/bin)
  --version <tag>        Install a specific version (e.g., v0.2.0)
  --force                Overwrite existing installation
  --no-verify            Skip post-install verification
  --help                 Show help message
```

## What the Installer Does

1. **Detects system** — OS and architecture (x86_64, ARM64)
2. **Queries GitHub API** — Fetches latest release info from `api.github.com`
3. **Selects correct asset** — Matches your OS/arch to the right binary
4. **Downloads with retry** — Up to 3 attempts with progress reporting
5. **Extracts and installs** — Places binary in `~/.vessel/bin`
6. **Configures PATH** — Adds install directory to user PATH
7. **Verifies** — Runs `vessel --version` to confirm installation

## Network Requirements

The enterprise installer only requires access to:

| Domain | Purpose | Port |
|--------|---------|------|
| `api.github.com` | Release metadata | 443 |
| `github.com` | Release assets download | 443 |

It does **NOT** require:
- `raw.githubusercontent.com`
- `objects.githubusercontent.com`
- Any other GitHub raw content domains

## Offline / Air-Gapped Installation

For environments with no internet access:

1. **On a connected machine**, download the correct binary from [GitHub Releases](https://github.com/bhuwanb23/vessel/releases)
2. **Transfer** the binary to the target machine (USB, internal file share, etc.)
3. **Install manually**:

```powershell
# Windows
mkdir "%USERPROFILE%\.vessel\bin" -Force
copy vessel.exe "%USERPROFILE%\.vessel\bin\vessel.exe"

# Add to PATH (current user)
$currentPath = [Environment]::GetEnvironmentVariable("Path", "User")
[Environment]::SetEnvironmentVariable("Path", "$currentPath;%USERPROFILE%\.vessel\bin", "User")
```

```bash
# Linux/macOS
mkdir -p ~/.vessel/bin
cp vessel ~/.vessel/bin/vessel
chmod +x ~/.vessel/bin/vessel

# Add to PATH
echo 'export PATH="$HOME/.vessel/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

## Proxy Configuration

If your environment requires a proxy for HTTPS traffic:

### PowerShell

```powershell
# Set proxy for current session
$env:HTTP_PROXY = "http://proxy.yourcompany.com:8080"
$env:HTTPS_PROXY = "http://proxy.yourcompany.com:8080"

# Run installer
.\install-enterprise.ps1
```

### Bash

```bash
# Set proxy for current session
export HTTP_PROXY="http://proxy.yourcompany.com:8080"
export HTTPS_PROXY="http://proxy.yourcompany.com:8080"

# Run installer
./install-enterprise.sh
```

## Troubleshooting

### "Could not determine latest version"

- Verify `api.github.com` is accessible: `curl -I https://api.github.com/repos/bhuwanb23/vessel/releases/latest`
- Check if GitHub API is rate-limited (unauthenticated: 60 requests/hour)
- Try specifying a version: `--version v0.2.0`

### "No matching asset found"

- Verify `github.com` releases page is accessible
- Check available assets at: https://github.com/bhuwanb23/vessel/releases
- Try downloading the binary manually

### "Download failed"

- Check internet connectivity
- Verify proxy settings if applicable
- Try increasing timeout or using a different network

### "Permission denied"

- Run PowerShell as Administrator
- Or choose a user-writable install directory: `-InstallDir "C:\Users\$env:USERNAME\tools\vessel"`

### Binary exists but `--version` fails

This is normal if GPU drivers (NVIDIA CUDA) are not installed on the machine. The binary will work for CPU-only inference. Install NVIDIA drivers for GPU support.

## Group Policy / SCCM Deployment

For mass deployment across an organization:

1. Download the binary from GitHub Releases
2. Place it on an internal file share
3. Deploy via Group Policy or SCCM:

```powershell
# Silent install script for SCCM
$installDir = "$env:ProgramFiles\vessel"
$sourceBinary = "\\fileshare\software\vessel\vessel.exe"

New-Item -ItemType Directory -Path $installDir -Force -ErrorAction SilentlyContinue
Copy-Item $sourceBinary "$installDir\vessel.exe" -Force

# Add to system PATH
$machinePath = [Environment]::GetEnvironmentVariable("Path", "Machine")
if ($machinePath -notlike "*$installDir*") {
    [Environment]::SetEnvironmentVariable("Path", "$machinePath;$installDir", "Machine")
}
```

## Release Asset Naming Convention

Release assets follow this naming pattern:

```
vessel-{os}-{arch}.{ext}
```

| OS | Architecture | Asset |
|----|-------------|-------|
| Windows | x86_64 | `vessel-windows-amd64.zip` |
| Windows | ARM64 | `vessel-windows-arm64.zip` |
| Linux | x86_64 | `vessel-linux-amd64.tar.gz` |
| Linux | ARM64 | `vessel-linux-arm64.tar.gz` |
| macOS | Apple Silicon | `vessel-darwin-arm64.tar.gz` |
