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
    LATEST_BACKUP=$(ls -t sujbot_backup_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9][0-9][0-9].tar.gz 2>/dev/null | head -1)
    if [ -z "$LATEST_BACKUP" ]; then
        log "ERROR: No backup archive found after export!"
        exit 1
    fi

    if ! mv "$LATEST_BACKUP" "$BACKUP_DIR/"; then
        log "ERROR: Failed to move backup to $BACKUP_DIR/"
        exit 1
    fi
    log "Moved $LATEST_BACKUP to $BACKUP_DIR/"

    # Verify archive integrity
    if tar -tzf "$BACKUP_DIR/$LATEST_BACKUP" > /dev/null 2>&1; then
        log "Archive verification: OK"
    else
        log "ERROR: Archive verification failed!"
        exit 1
    fi
else
    log "ERROR: Backup failed!"
    exit 1
fi

# Rotate old backups (keep last N)
log "Rotating old backups (keeping last $MAX_BACKUPS)..."
cd "$BACKUP_DIR"
BACKUP_COUNT=$(ls -1 sujbot_backup_[0-9]*.tar.gz 2>/dev/null | wc -l)
if [ "$BACKUP_COUNT" -gt "$MAX_BACKUPS" ]; then
    REMOVE_COUNT=$((BACKUP_COUNT - MAX_BACKUPS))
    # Use process substitution to avoid subshell issues
    while IFS= read -r old_backup; do
        log "Removing old backup: $old_backup"
        if ! rm -f "$old_backup"; then
            log "WARNING: Failed to remove $old_backup"
        fi
    done < <(ls -t sujbot_backup_[0-9]*.tar.gz | tail -n "$REMOVE_COUNT")
fi

# Report disk usage
BACKUP_SIZE=$(du -sh "$BACKUP_DIR" 2>/dev/null | cut -f1)
log "Total backup storage: $BACKUP_SIZE"

log "=== Backup complete ==="
