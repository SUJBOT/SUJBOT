#!/bin/bash
# SUJBOT Automated Backup Script
# Wrapper for backup_export.sh with logging and rotation
#
# Install: Add to crontab or use systemd timer
#   crontab -e
#   0 3 * * * /home/michal/SUJBOT/scripts/backup_cron.sh
#
# Or copy systemd files:
#   sudo cp scripts/systemd/sujbot-backup.* /etc/systemd/system/
#   sudo systemctl enable --now sujbot-backup.timer

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/logs"
LOG_FILE="$LOG_DIR/backup.log"
BACKUP_DIR="$PROJECT_DIR/backups"
MAX_BACKUPS=7  # Keep last 7 days

# Ensure directories exist
mkdir -p "$LOG_DIR" "$BACKUP_DIR"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "=== Starting automated backup ==="

# Run backup
cd "$PROJECT_DIR"
if "$SCRIPT_DIR/backup_export.sh" >> "$LOG_FILE" 2>&1; then
    log "Backup completed successfully"

    # Move backup to backups directory
    LATEST_BACKUP=$(ls -t sujbot_backup_*.tar.gz 2>/dev/null | head -1)
    if [ -n "$LATEST_BACKUP" ]; then
        mv "$LATEST_BACKUP" "$BACKUP_DIR/"
        log "Moved $LATEST_BACKUP to $BACKUP_DIR/"

        # Verify archive integrity
        if tar -tzf "$BACKUP_DIR/$LATEST_BACKUP" > /dev/null 2>&1; then
            log "Archive verification: OK"
        else
            log "ERROR: Archive verification failed!"
            exit 1
        fi
    fi
else
    log "ERROR: Backup failed!"
    exit 1
fi

# Rotate old backups (keep last N)
log "Rotating old backups (keeping last $MAX_BACKUPS)..."
cd "$BACKUP_DIR"
BACKUP_COUNT=$(ls -1 sujbot_backup_*.tar.gz 2>/dev/null | wc -l)
if [ "$BACKUP_COUNT" -gt "$MAX_BACKUPS" ]; then
    REMOVE_COUNT=$((BACKUP_COUNT - MAX_BACKUPS))
    ls -t sujbot_backup_*.tar.gz | tail -n "$REMOVE_COUNT" | while read -r old_backup; do
        log "Removing old backup: $old_backup"
        rm -f "$old_backup"
    done
fi

# Report disk usage
BACKUP_SIZE=$(du -sh "$BACKUP_DIR" 2>/dev/null | cut -f1)
log "Total backup storage: $BACKUP_SIZE"

log "=== Backup complete ==="
