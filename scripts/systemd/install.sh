#!/bin/bash
#
# Install SUJBOT2 Security Monitoring systemd timers
# Usage: sudo ./install.sh
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYSTEMD_DIR="/etc/systemd/system"

echo "=== SUJBOT2 Security Monitor Installation ==="
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "ERROR: This script must be run as root (sudo)"
    exit 1
fi

# Copy service and timer files
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

# Enable and start timers
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
