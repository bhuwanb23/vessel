#!/usr/bin/env bash
# =============================================================================
# Vessel — One-line installer
# Usage: curl -fsSL https://raw.githubusercontent.com/bhuwanb23/vessel/main/install.sh | bash
# =============================================================================
set -euo pipefail

REPO="bhuwanb23/vessel"
BINARY_NAME="vessel"
INSTALL_DIR="${VESSEL_INSTALL_DIR:-/usr/local/bin}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

info()  { printf "${CYAN}▸${NC} %s\n" "$1"; }
ok()    { printf "${GREEN}✔${NC} %s\n" "$1"; }
warn()  { printf "${YELLOW}⚠${NC} %s\n" "$1"; }
err()   { printf "${RED}✖${NC} %s\n" "$1" >&2; exit 1; }

# --- Detect OS ---
detect_os() {
    local os
    os="$(uname -s)"
    case "$os" in
        Linux*)  OS="linux" ;;
        Darwin*) OS="macos" ;;
        MINGW*|MSYS*|CYGWIN*) err "Windows detected. Use: powershell -c \"irm https://raw.githubusercontent.com/$REPO/main/install.ps1 | iex\"" ;;
        *)       err "Unsupported OS: $os" ;;
    esac
}

# --- Detect Architecture ---
detect_arch() {
    local arch
    arch="$(uname -m)"
    case "$arch" in
        x86_64|amd64)   ARCH="amd64" ;;
        aarch64|arm64)   ARCH="arm64" ;;
        *)               err "Unsupported architecture: $arch" ;;
    esac

    # macOS: only arm64 is released (Apple Silicon)
    if [ "$OS" = "macos" ] && [ "$ARCH" != "arm64" ]; then
        warn "macOS x86_64 detected. Only Apple Silicon (arm64) binary is available."
        warn "Falling back to arm64 (requires Rosetta 2 for Intel Macs)."
        ARCH="arm64"
    fi
}

# --- Get latest version ---
get_version() {
    local version
    version=$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" 2>/dev/null | grep '"tag_name"' | sed -E 's/.*"([^"]+)".*/\1/')
    if [ -z "$version" ]; then
        # Fallback: try to list tags
        version=$(curl -fsSL "https://api.github.com/repos/$REPO/tags" 2>/dev/null | grep '"name"' | head -1 | sed -E 's/.*"([^"]+)".*/\1/')
    fi
    if [ -z "$version" ]; then
        err "Could not determine latest version. Check https://github.com/$REPO/releases"
    fi
    echo "$version"
}

# --- Download ---
download() {
    local version="$1"
    local asset_name="vessel-${OS}-${ARCH}"
    if [ "$OS" = "windows" ]; then
        asset_name="${asset_name}.exe"
    fi

    local url="https://github.com/$REPO/releases/download/${version}/${asset_name}"
    local tmp_file
    tmp_file=$(mktemp)

    info "Downloading ${asset_name} (${version})..."
    if ! curl -fsSL -o "$tmp_file" "$url"; then
        rm -f "$tmp_file"
        err "Download failed. URL: $url"
    fi

    chmod +x "$tmp_file"
    echo "$tmp_file"
}

# --- Install ---
install_binary() {
    local tmp_file="$1"

    # Check if install dir exists and is writable
    if [ ! -d "$INSTALL_DIR" ]; then
        info "Creating install directory: $INSTALL_DIR"
        mkdir -p "$INSTALL_DIR" 2>/dev/null || sudo mkdir -p "$INSTALL_DIR"
    fi

    local target="${INSTALL_DIR}/${BINARY_NAME}"

    # Try without sudo first, fall back to sudo
    if [ -w "$INSTALL_DIR" ] 2>/dev/null; then
        mv "$tmp_file" "$target"
    else
        info "Using sudo to install to $INSTALL_DIR..."
        sudo mv "$tmp_file" "$target"
    fi

    chmod +x "$target"
}

# --- Verify ---
verify() {
    local target="${INSTALL_DIR}/${BINARY_NAME}"
    if [ -x "$target" ]; then
        local version
        version=$("$target" --version 2>/dev/null || echo "installed")
        ok "Installed ${BINARY_NAME} to ${target}"
        ok "Version: ${version}"
    else
        err "Installation failed. Binary not found at ${target}"
    fi
}

# --- PATH check ---
check_path() {
    if [[ ":$PATH:" != *":${INSTALL_DIR}:"* ]]; then
        warn "${INSTALL_DIR} is not in your PATH."
        echo ""
        echo "  Add it to your shell profile:"
        echo ""
        if [ "$OS" = "macos" ]; then
            echo "    echo 'export PATH=\"${INSTALL_DIR}:\$PATH\"' >> ~/.zshrc"
            echo "    source ~/.zshrc"
        else
            echo "    echo 'export PATH=\"${INSTALL_DIR}:\$PATH\"' >> ~/.bashrc"
            echo "    source ~/.bashrc"
        fi
        echo ""
    fi
}

# --- Main ---
main() {
    printf "\n"
    printf "${CYAN}  ╔═══════════════════════════════════╗${NC}\n"
    printf "${CYAN}  ║     Vessel — Installer             ║${NC}\n"
    printf "${CYAN}  ╚═══════════════════════════════════╝${NC}\n"
    printf "\n"

    detect_os
    detect_arch

    local version
    version=$(get_version)
    info "Latest version: ${version}"

    local tmp_file
    tmp_file=$(download "$version")

    install_binary "$tmp_file"
    verify
    check_path

    echo ""
    ok "Quick start:"
    echo ""
    echo "  vessel --recommend              # Find models for your hardware"
    echo "  vessel --model <url>            # Predict deployment strategy"
    echo "  vessel --serve                  # Start OpenAI-compatible API"
    echo ""
}

main "$@"
