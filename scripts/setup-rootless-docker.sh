#!/bin/bash
# =============================================================================
# SUJBOT Rootless Docker Setup Script
# =============================================================================
# One-time setup for rootless Docker. Run as the user who will use it (NOT root).
#
# After setup, the user can run ./scripts/dev.sh without sudo to manage
# their own isolated development environment.
#
# Usage:
#   ./scripts/setup-rootless-docker.sh
#
# What this script does:
#   1. Installs prerequisites (uidmap package)
#   2. Configures subuid/subgid ranges for user namespace isolation
#   3. Enables user lingering (so services persist after logout)
#   4. Installs and starts rootless Docker daemon
#   5. Configures shell environment
# =============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}=== SUJBOT Rootless Docker Setup ===${NC}"
echo ""

# =============================================================================
# PRE-FLIGHT CHECKS
# =============================================================================

# Check not running as root
if [[ $EUID -eq 0 ]]; then
    echo -e "${RED}ERROR: Do not run this script as root!${NC}"
    echo "Run as the user who will use rootless Docker."
    exit 1
fi

USER_UID=$(id -u)
USER_GID=$(id -g)

echo "User: $USER (UID: $USER_UID, GID: $USER_GID)"
echo ""

# Calculate port range for this user
PORT_BASE=$((10000 + (USER_UID % 1000) * 100))
echo "Your development port range will be: $PORT_BASE-$((PORT_BASE + 99))"
echo ""

# =============================================================================
# PREREQUISITES
# =============================================================================
echo -e "${YELLOW}Step 1: Checking prerequisites...${NC}"

# Check if uidmap is installed
if ! command -v newuidmap &>/dev/null; then
    echo "Installing uidmap package (requires sudo)..."
    if command -v apt-get &>/dev/null; then
        sudo apt-get update && sudo apt-get install -y uidmap
    elif command -v dnf &>/dev/null; then
        sudo dnf install -y shadow-utils
    elif command -v yum &>/dev/null; then
        sudo yum install -y shadow-utils
    else
        echo -e "${RED}ERROR: Cannot install uidmap. Please install it manually.${NC}"
        exit 1
    fi
fi
echo -e "${GREEN}✓ uidmap installed${NC}"

# Check if dbus-user-session is installed (needed for systemd --user)
if ! dpkg -l dbus-user-session &>/dev/null 2>&1; then
    echo "Installing dbus-user-session (requires sudo)..."
    sudo apt-get install -y dbus-user-session
fi
echo -e "${GREEN}✓ dbus-user-session installed${NC}"

# =============================================================================
# SUBUID/SUBGID CONFIGURATION
# =============================================================================
echo ""
echo -e "${YELLOW}Step 2: Configuring user namespace mappings...${NC}"

# Check /etc/subuid
if ! grep -q "^$USER:" /etc/subuid 2>/dev/null; then
    echo "Adding subuid range for $USER (requires sudo)..."
    sudo usermod --add-subuids 100000-165535 "$USER"
fi
echo -e "${GREEN}✓ subuid configured${NC}"

# Check /etc/subgid
if ! grep -q "^$USER:" /etc/subgid 2>/dev/null; then
    echo "Adding subgid range for $USER (requires sudo)..."
    sudo usermod --add-subgids 100000-165535 "$USER"
fi
echo -e "${GREEN}✓ subgid configured${NC}"

# Show current mappings
echo "  subuid: $(grep "^$USER:" /etc/subuid)"
echo "  subgid: $(grep "^$USER:" /etc/subgid)"

# =============================================================================
# ENABLE LINGERING
# =============================================================================
echo ""
echo -e "${YELLOW}Step 3: Enabling user lingering...${NC}"

# Enable lingering so user services run without active login
if ! loginctl show-user "$USER" 2>/dev/null | grep -q "Linger=yes"; then
    echo "Enabling lingering for $USER (requires sudo)..."
    sudo loginctl enable-linger "$USER"
fi
echo -e "${GREEN}✓ Lingering enabled${NC}"

# =============================================================================
# INSTALL ROOTLESS DOCKER
# =============================================================================
echo ""
echo -e "${YELLOW}Step 4: Installing rootless Docker...${NC}"

# Check if Docker is installed
if ! command -v docker &>/dev/null; then
    echo -e "${RED}ERROR: Docker not found. Install Docker first:${NC}"
    echo "  curl -fsSL https://get.docker.com | sudo sh"
    exit 1
fi

# Check if rootless Docker is already set up
ROOTLESS_SOCKET="/run/user/$USER_UID/docker.sock"
if [[ -S "$ROOTLESS_SOCKET" ]] && docker info &>/dev/null; then
    echo -e "${GREEN}✓ Rootless Docker already running${NC}"
else
    # Run Docker rootless setup
    if [[ -f /usr/bin/dockerd-rootless-setuptool.sh ]]; then
        echo "Running dockerd-rootless-setuptool.sh..."
        /usr/bin/dockerd-rootless-setuptool.sh install
    else
        echo -e "${RED}ERROR: dockerd-rootless-setuptool.sh not found.${NC}"
        echo "Make sure Docker 20.10+ is installed."
        echo "Try: curl -fsSL https://get.docker.com/rootless | sh"
        exit 1
    fi
fi

# =============================================================================
# CONFIGURE SHELL ENVIRONMENT
# =============================================================================
echo ""
echo -e "${YELLOW}Step 5: Configuring shell environment...${NC}"

# Detect shell profile
PROFILE_FILE="$HOME/.bashrc"
if [[ -f "$HOME/.zshrc" ]]; then
    PROFILE_FILE="$HOME/.zshrc"
fi

# Add Docker environment to profile if not already there
if ! grep -q "DOCKER_HOST.*docker.sock" "$PROFILE_FILE" 2>/dev/null; then
    echo "Adding Docker configuration to $PROFILE_FILE..."
    cat >> "$PROFILE_FILE" <<EOF

# =============================================================================
# Rootless Docker configuration (added by setup-rootless-docker.sh)
# =============================================================================
export DOCKER_HOST=unix:///run/user/$USER_UID/docker.sock
export PATH=\$HOME/bin:\$PATH
EOF
    echo -e "${GREEN}✓ Shell configuration added${NC}"
else
    echo -e "${GREEN}✓ Shell configuration already present${NC}"
fi

# =============================================================================
# START ROOTLESS DOCKER SERVICE
# =============================================================================
echo ""
echo -e "${YELLOW}Step 6: Starting rootless Docker service...${NC}"

# Export for current session
export DOCKER_HOST="unix:///run/user/$USER_UID/docker.sock"

# Enable and start the service
systemctl --user enable docker 2>/dev/null || true
systemctl --user start docker 2>/dev/null || true

# Wait for socket to appear
echo "Waiting for Docker daemon..."
for i in {1..10}; do
    if [[ -S "$ROOTLESS_SOCKET" ]]; then
        break
    fi
    sleep 1
done

# Verify Docker is working
if docker info &>/dev/null; then
    echo -e "${GREEN}✓ Rootless Docker is running${NC}"
else
    echo -e "${RED}ERROR: Docker daemon not responding.${NC}"
    echo "Try restarting with: systemctl --user restart docker"
    exit 1
fi

# =============================================================================
# SUMMARY
# =============================================================================
echo ""
echo -e "${GREEN}=== Setup Complete ===${NC}"
echo ""
echo "Rootless Docker is now configured for $USER"
echo ""
echo "Docker root directory: $HOME/.local/share/docker"
echo "Docker socket: $ROOTLESS_SOCKET"
echo ""
echo -e "${YELLOW}IMPORTANT: Log out and log back in, or run:${NC}"
echo "  source $PROFILE_FILE"
echo ""
echo "Then verify with:"
echo "  docker info | grep 'Docker Root Dir'"
echo "  docker run --rm hello-world"
echo ""
echo "Your development ports: $PORT_BASE-$((PORT_BASE + 99))"
echo ""
echo "To start your development environment:"
echo "  1. Make sure databases are running: ./scripts/db.sh up"
echo "  2. Start your dev stack: ./scripts/dev.sh up"
echo ""
