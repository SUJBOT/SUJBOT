#!/bin/bash
#
# Install SUJBOT2 Security Monitoring systemd timers
# Usage: sudo ./install.sh [username]
#
# If username is not provided, uses the user who invoked sudo (SUDO_USER)
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
SYSTEMD_DIR="/etc/systemd/system"

echo "=== SUJBOT2 Security Monitor Installation ==="
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "ERROR: This script must be run as root (sudo)"
    exit 1
fi

# Determine target user
if [ -n "${1:-}" ]; then
    TARGET_USER="$1"
elif [ -n "${SUDO_USER:-}" ]; then
    TARGET_USER="$SUDO_USER"
else
    echo "ERROR: Cannot determine target user. Please provide username as argument."
    echo "Usage: sudo ./install.sh <username>"
    exit 1
fi

# Validate user exists and get home directory safely
# Using getent instead of eval to prevent command injection
if ! id "$TARGET_USER" &>/dev/null; then
    echo "ERROR: User '$TARGET_USER' does not exist"
    exit 1
fi

TARGET_HOME=$(getent passwd "$TARGET_USER" | cut -d: -f6)
if [ -z "$TARGET_HOME" ] || [ ! -d "$TARGET_HOME" ]; then
    echo "ERROR: Cannot determine home directory for user '$TARGET_USER'"
    exit 1
fi

echo "Configuration:"
echo "  User:        $TARGET_USER"
echo "  Home:        $TARGET_HOME"
echo "  Project:     $PROJECT_DIR"
echo ""

# Generate service files from templates
echo "Generating systemd files from templates..."

for template in "$SCRIPT_DIR"/*.template; do
    if [ -f "$template" ]; then
        filename=$(basename "$template" .template)
        output="$SCRIPT_DIR/$filename"

        sed -e "s|__USER__|$TARGET_USER|g" \
            -e "s|__HOME__|$TARGET_HOME|g" \
            -e "s|__PROJECT_DIR__|$PROJECT_DIR|g" \
            "$template" > "$output"

        echo "  Generated: $filename"
    fi
done

echo ""

# Stop existing timers if running (ignore errors if not installed)
echo "Stopping existing timers (if any)..."
systemctl stop sujbot-monitor.timer 2>/dev/null || true
systemctl stop sujbot-monitor-daily.timer 2>/dev/null || true

# Copy service and timer files to systemd directory
echo "Copying systemd files to $SYSTEMD_DIR..."
cp "$SCRIPT_DIR/sujbot-monitor.service" "$SYSTEMD_DIR/"
cp "$SCRIPT_DIR/sujbot-monitor.timer" "$SYSTEMD_DIR/"
cp "$SCRIPT_DIR/sujbot-monitor-daily.service" "$SYSTEMD_DIR/"
cp "$SCRIPT_DIR/sujbot-monitor-daily.timer" "$SYSTEMD_DIR/"

# Set permissions
chmod 644 "$SYSTEMD_DIR/sujbot-monitor.service"
chmod 644 "$SYSTEMD_DIR/sujbot-monitor.timer"
chmod 644 "$SYSTEMD_DIR/sujbot-monitor-daily.service"
chmod 644 "$SYSTEMD_DIR/sujbot-monitor-daily.timer"

# Reload systemd daemon
echo "Reloading systemd daemon..."
systemctl daemon-reload

# Enable and start timers (restart with new configuration)
echo "Enabling and starting timers..."
systemctl enable --now sujbot-monitor.timer
systemctl enable --now sujbot-monitor-daily.timer

echo ""
echo "=== Installation Complete ==="
echo ""
echo "Timers status:"
systemctl list-timers | grep -E "sujbot|NEXT|PASSED" | head -5

echo ""
echo "Useful commands:"
echo "  systemctl status sujbot-monitor.timer       # Check timer status"
echo "  systemctl start sujbot-monitor.service      # Run check manually"
echo "  journalctl -u sujbot-monitor.service -f     # Watch logs"
echo "  systemctl list-timers | grep sujbot         # List scheduled runs"
