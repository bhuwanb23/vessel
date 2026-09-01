#!/usr/bin/env bash
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
#   curl -fsSL -o install-enterprise.sh https://github.com/bhuwanb23/vessel/releases/latest/download/install-enterprise.sh
#
#   # Step 2: Make it executable and run
#   chmod +x install-enterprise.sh
#   ./install-enterprise.sh
#
# Or one-liner:
#   curl -fsSL https://github.com/bhuwanb23/vessel/releases/latest/download/install-enterprise.sh | bash
#
# =============================================================================

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================
REPO="bhuwanb23/vessel"
GITHUB_API_BASE="https://api.github.com/repos/${REPO}"
GITHUB_RELEASES_BASE="https://github.com/${REPO}/releases"
INSTALL_DIR="${VESSEL_INSTALL_DIR:-$HOME/.vessel/bin}"
VERSION=""
FORCE=0
NO_VERIFY=0
MAX_RETRIES=3
RETRY_DELAY=5

# =============================================================================
# Colors
# =============================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
DIM='\033[2m'
NC='\033[0m'

# =============================================================================
# Helper Functions
# =============================================================================
info()  { printf "${CYAN}  >>${NC} %s\n" "$1"; }
ok()    { printf "${GREEN}  OK${NC} %s\n" "$1"; }
warn()  { printf "${YELLOW}  !!${NC} %s\n" "$1"; }
err()   { printf "${RED}  XX${NC} %s\n" "$1" >&2; exit 1; }
step()  { printf "\n${WHITE}--- %s ---${NC}\n" "$1"; }

show_help() {
    printf "\n"
    printf "${CYAN}  ============================================${NC}\n"
    printf "${CYAN}    Vessel — Enterprise Installer${NC}\n"
    printf "${CYAN}    (GitHub Releases only, no raw.githubusercontent.com)${NC}\n"
    printf "${CYAN}  ============================================${NC}\n"
    printf "\n"
    printf "${WHITE}Usage:${NC}\n"
    printf "  ./install-enterprise.sh [options]\n"
    printf "\n"
    printf "${WHITE}Options:${NC}\n"
    printf "  --install-dir <path>   Install directory (default: ~/.vessel/bin)\n"
    printf "  --version <tag>        Install a specific version (default: latest)\n"
    printf "  --force                Overwrite existing installation\n"
    printf "  --no-verify            Skip post-install verification\n"
    printf "  --help                 Show this help message\n"
    printf "\n"
    printf "${WHITE}Examples:${NC}\n"
    printf "  ./install-enterprise.sh                          # Install latest\n"
    printf "  ./install-enterprise.sh --version v0.2.0         # Install specific version\n"
    printf "  ./install-enterprise.sh --force                  # Force reinstall\n"
    printf "\n"
    printf "Installation directory: %s\n" "$INSTALL_DIR"
    printf "\n"
    exit 0
}

# =============================================================================
# Parse Arguments
# =============================================================================
parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --install-dir)
                INSTALL_DIR="$2"
                shift 2
                ;;
            --version)
                VERSION="$2"
                shift 2
                ;;
            --force)
                FORCE=1
                shift
                ;;
            --no-verify)
                NO_VERIFY=1
                shift
                ;;
            --help|-h)
                show_help
                ;;
            *)
                err "Unknown option: $1 (use --help for usage)"
                ;;
        esac
    done
}

# =============================================================================
# Step 1: Detect System Information
# =============================================================================
detect_system() {
    step "Detecting system information"

    # Detect OS
    local os
    os="$(uname -s)"
    case "$os" in
        Linux*)  OS="linux" ;;
        Darwin*) OS="darwin" ;;
        MINGW*|MSYS*|CYGWIN*)
            warn "Windows detected via bash."
            warn "For best results, use install-enterprise.ps1 instead."
            warn "Continuing with bash installer..."
            OS="windows"
            ;;
        *)       err "Unsupported OS: $os" ;;
    esac
    info "Detected OS: $OS"

    # Detect architecture
    local arch
    arch="$(uname -m)"
    case "$arch" in
        x86_64|amd64)   ARCH="amd64" ;;
        aarch64|arm64)   ARCH="arm64" ;;
        *)               err "Unsupported architecture: $arch" ;;
    esac
    info "Detected Architecture: $arch ($ARCH)"

    # macOS: only arm64 binary is released for Apple Silicon
    if [ "$OS" = "darwin" ] && [ "$ARCH" != "arm64" ]; then
        warn "macOS x86_64 detected. Only Apple Silicon (arm64) binary is available."
        warn "Falling back to arm64 (requires Rosetta 2 for Intel Macs)."
        ARCH="arm64"
    fi
}

# =============================================================================
# Step 2: Detect Latest Release
# =============================================================================
get_version() {
    step "Checking latest release"

    if [ -n "$VERSION" ]; then
        info "Using specified version: $VERSION"
        return
    fi

    info "Querying GitHub API for latest release..."

    local version
    version=$(curl -fsSL --connect-timeout 10 --max-time 30 \
        -H "Accept: application/vnd.github.v3+json" \
        "${GITHUB_API_BASE}/releases/latest" 2>/dev/null \
        | grep '"tag_name"' \
        | head -1 \
        | sed -E 's/.*"([^"]+)".*/\1/') || true

    if [ -z "$version" ]; then
        warn "API query failed. Trying tags endpoint as fallback..."
        version=$(curl -fsSL --connect-timeout 10 --max-time 30 \
            -H "Accept: application/vnd.github.v3+json" \
            "${GITHUB_API_BASE}/tags" 2>/dev/null \
            | grep '"name"' \
            | head -1 \
            | sed -E 's/.*"([^"]+)".*/\1/') || true
    fi

    if [ -z "$version" ]; then
        err "Could not determine latest version. Check https://github.com/${REPO}/releases"
    fi

    VERSION="$version"
    ok "Latest version: $VERSION"
}

# =============================================================================
# Step 3: Select Correct Asset
# =============================================================================
select_asset() {
    step "Selecting correct release asset"

    ASSET_NAME="vessel-${OS}-${ARCH}"
    info "Target asset: ${ASSET_NAME}.tar.gz"

    # Check if asset exists in the release
    local assets_json
    assets_json=$(curl -fsSL --connect-timeout 10 --max-time 30 \
        -H "Accept: application/vnd.github.v3+json" \
        "${GITHUB_API_BASE}/releases/tags/${VERSION}" 2>/dev/null) || true

    if [ -n "$assets_json" ]; then
        # Look for .tar.gz
        local download_url
        download_url=$(echo "$assets_json" \
            | grep -o "\"browser_download_url\":\s*\"[^\"]*${ASSET_NAME}\.tar\.gz\"" \
            | head -1 \
            | sed -E 's/.*"browser_download_url":\s*"([^"]+)".*/\1/') || true

        if [ -n "$download_url" ]; then
            DOWNLOAD_URL="$download_url"
            IS_TAR=1
            info "Found asset: ${ASSET_NAME}.tar.gz"
            return
        fi

        # Look for single binary
        download_url=$(echo "$assets_json" \
            | grep -o "\"browser_download_url\":\s*\"[^\"]*${ASSET_NAME}\"" \
            | head -1 \
            | sed -E 's/.*"browser_download_url":\s*"([^"]+)".*/\1/') || true

        if [ -n "$download_url" ]; then
            DOWNLOAD_URL="$download_url"
            IS_TAR=0
            info "Found single binary: ${ASSET_NAME}"
            return
        fi

        # List available assets for debugging
        warn "Available assets in this release:"
        echo "$assets_json" | grep '"name"' | sed -E 's/.*"name":\s*"([^"]+)".*/  - \1/' | head -20
    fi

    # Fallback: try direct download URL
    info "Attempting direct download..."
    DOWNLOAD_URL="${GITHUB_RELEASES_BASE}/download/${VERSION}/${ASSET_NAME}.tar.gz"
    IS_TAR=1
}

# =============================================================================
# Step 4: Download with Retry
# =============================================================================
download_asset() {
    step "Downloading ${ASSET_NAME}"

    local tmp_file
    tmp_file=$(mktemp /tmp/vessel-install.XXXXXX)

    local attempt=0
    local success=0

    while [ $attempt -lt $MAX_RETRIES ]; do
        attempt=$((attempt + 1))
        info "Download attempt $attempt/$MAX_RETRIES..."

        if curl -fSL --connect-timeout 10 --max-time 300 \
            -H "User-Agent: Vessel-Installer/1.0" \
            -o "$tmp_file" \
            "$DOWNLOAD_URL" 2>/dev/null; then

            local file_size
            file_size=$(stat -f%z "$tmp_file" 2>/dev/null || stat -c%s "$tmp_file" 2>/dev/null || echo "0")
            local file_size_mb
            file_size_mb=$(echo "scale=1; $file_size / 1048576" | bc 2>/dev/null || echo "?")

            ok "Downloaded ${file_size_mb} MB"
            success=1
            break
        fi

        if [ $attempt -lt $MAX_RETRIES ]; then
            warn "Download failed. Retrying in ${RETRY_DELAY}s..."
            sleep $RETRY_DELAY
        fi
    done

    if [ $success -eq 0 ]; then
        rm -f "$tmp_file"
        err "Download failed after $MAX_RETRIES attempts. URL: $DOWNLOAD_URL"
    fi

    DOWNLOAD_PATH="$tmp_file"
}

# =============================================================================
# Step 5: Extract / Prepare Binary
# =============================================================================
extract_binary() {
    step "Extracting binary"

    if [ "$IS_TAR" -eq 0 ]; then
        # Single binary, no extraction needed
        info "Single binary file, no extraction needed"
        BINARY_PATH="$DOWNLOAD_PATH"
        chmod +x "$BINARY_PATH"
        return
    fi

    local extract_dir
    extract_dir=$(mktemp -d /tmp/vessel-extract.XXXXXX)

    if tar -xzf "$DOWNLOAD_PATH" -C "$extract_dir" 2>/dev/null; then
        # Find the vessel binary
        local vessel_bin
        vessel_bin=$(find "$extract_dir" -name "vessel" -type f -executable 2>/dev/null | head -1)

        if [ -z "$vessel_bin" ]; then
            vessel_bin=$(find "$extract_dir" -name "vessel" -type f 2>/dev/null | head -1)
        fi

        if [ -z "$vessel_bin" ]; then
            # Try any binary
            vessel_bin=$(find "$extract_dir" -type f -executable 2>/dev/null | head -1)
        fi

        if [ -n "$vessel_bin" ]; then
            BINARY_PATH="$vessel_bin"
            chmod +x "$BINARY_PATH"
            ok "Found binary: $(basename "$vessel_bin")"
        else
            warn "Extracted contents:"
            find "$extract_dir" -type f | sed 's/^/  - /'
            rm -rf "$extract_dir"
            err "No executable found in archive"
        fi
    else
        rm -rf "$extract_dir"
        err "Extraction failed. The download may be corrupted."
    fi

    EXTRACT_DIR="$extract_dir"
}

# =============================================================================
# Step 6: Install Binary
# =============================================================================
install_binary() {
    step "Installing to $INSTALL_DIR"

    local target="${INSTALL_DIR}/vessel"

    # Check if already installed
    if [ -f "$target" ] && [ "$FORCE" -eq 0 ]; then
        warn "Vessel is already installed at $target"
        warn "Use --force to overwrite"
        read -r -p "Overwrite? (y/N) " response
        if [ "$response" != "y" ] && [ "$response" != "Y" ]; then
            info "Installation cancelled"
            exit 0
        fi
    fi

    # Create install directory
    if [ ! -d "$INSTALL_DIR" ]; then
        info "Creating directory: $INSTALL_DIR"
        mkdir -p "$INSTALL_DIR" 2>/dev/null || {
            warn "Cannot create $INSTALL_DIR without sudo. Using sudo..."
            sudo mkdir -p "$INSTALL_DIR"
        }
    fi

    # Check if directory is writable
    if [ ! -w "$INSTALL_DIR" ] 2>/dev/null; then
        info "Using sudo to install to $INSTALL_DIR..."
        sudo cp "$BINARY_PATH" "$target"
        sudo chmod +x "$target"
    else
        cp "$BINARY_PATH" "$target"
        chmod +x "$target"
    fi

    if [ ! -f "$target" ]; then
        err "Binary not found at $target after installation"
    fi

    local file_size
    file_size=$(stat -f%z "$target" 2>/dev/null || stat -c%s "$target" 2>/dev/null || echo "0")
    local file_size_mb
    file_size_mb=$(echo "scale=1; $file_size / 1048576" | bc 2>/dev/null || echo "?")

    ok "Installed vessel to $target (${file_size_mb} MB)"
    INSTALLED_PATH="$target"
}

# =============================================================================
# Step 7: Configure PATH
# =============================================================================
update_path() {
    step "Configuring PATH"

    # Check if already in PATH
    if [[ ":$PATH:" == *":${INSTALL_DIR}:"* ]]; then
        ok "Install directory already in PATH"
        return
    fi

    # Determine shell profile
    local shell_profile=""
    local shell_name
    shell_name=$(basename "${SHELL:-/bin/bash}")

    case "$shell_name" in
        zsh)  shell_profile="$HOME/.zshrc" ;;
        bash)
            if [ -f "$HOME/.bashrc" ]; then
                shell_profile="$HOME/.bashrc"
            elif [ -f "$HOME/.bash_profile" ]; then
                shell_profile="$HOME/.bash_profile"
            else
                shell_profile="$HOME/.profile"
            fi
            ;;
        fish) shell_profile="$HOME/.config/fish/config.fish" ;;
        *)    shell_profile="$HOME/.profile" ;;
    esac

    # Add to PATH in shell profile
    local path_line="export PATH=\"${INSTALL_DIR}:\$PATH\""

    if [ "$shell_name" = "fish" ]; then
        path_line="set -gx PATH ${INSTALL_DIR} \$PATH"
    fi

    if [ -f "$shell_profile" ] && grep -qF "$INSTALL_DIR" "$shell_profile" 2>/dev/null; then
        ok "Install directory already in $shell_profile"
    else
        info "Adding $INSTALL_DIR to $shell_profile"
        echo "" >> "$shell_profile"
        echo "# Vessel - added by install-enterprise.sh" >> "$shell_profile"
        echo "$path_line" >> "$shell_profile"
        ok "Added $INSTALL_DIR to $shell_profile"
    fi

    # Also add to current session
    export PATH="${INSTALL_DIR}:$PATH"

    warn "Run 'source $shell_profile' or restart your terminal for PATH changes."
}

# =============================================================================
# Step 8: Verify Installation
# =============================================================================
verify_installation() {
    step "Verifying installation"

    if [ "$NO_VERIFY" -eq 1 ]; then
        warn "Skipping verification (--no-verify flag)"
        ok "Binary exists at $INSTALLED_PATH"
        return
    fi

    if [ ! -x "$INSTALLED_PATH" ]; then
        err "Binary not executable at $INSTALLED_PATH"
    fi

    local version_output
    if version_output=$("$INSTALLED_PATH" --version 2>&1); then
        ok "Version: $version_output"
    else
        warn "Warning: Could not run --version, but binary exists"
        info "This may be expected if GPU drivers are not installed"
    fi
}

# =============================================================================
# Step 9: Cleanup
# =============================================================================
cleanup() {
    step "Cleaning up"

    if [ -n "${DOWNLOAD_PATH:-}" ] && [ -f "$DOWNLOAD_PATH" ]; then
        rm -f "$DOWNLOAD_PATH"
    fi

    if [ -n "${EXTRACT_DIR:-}" ] && [ -d "$EXTRACT_DIR" ]; then
        rm -rf "$EXTRACT_DIR"
    fi

    ok "Temporary files cleaned up"
}

# =============================================================================
# Main
# =============================================================================
main() {
    parse_args "$@"

    printf "\n"
    printf "${CYAN}  ============================================${NC}\n"
    printf "${CYAN}    Vessel — Enterprise Installer${NC}\n"
    printf "${CYAN}    (GitHub Releases only)${NC}\n"
    printf "${CYAN}  ============================================${NC}\n"
    printf "\n"

    detect_system
    get_version
    select_asset
    download_asset
    extract_binary
    install_binary
    update_path
    verify_installation
    cleanup

    printf "\n"
    printf "${GREEN}  ============================================${NC}\n"
    printf "${GREEN}    Installation Complete!${NC}\n"
    printf "${GREEN}  ============================================${NC}\n"
    printf "\n"
    printf "  Installed to: %s\n" "$INSTALLED_PATH"
    printf "  Version:      %s\n" "$VERSION"
    printf "\n"
    printf "  ${CYAN}Quick start:${NC}\n"
    printf "    vessel --recommend                    # Find models for your hardware\n"
    printf "    vessel --model <url>                  # Predict deployment strategy\n"
    printf "    vessel --model <url> --execute        # Download and run inference\n"
    printf "    vessel --serve                        # Start OpenAI-compatible API\n"
    printf "\n"
}

# Run main
main "$@"
