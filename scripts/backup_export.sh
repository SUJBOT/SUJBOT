#!/bin/bash
# SUJBOT Backup Export Script
# Creates a complete backup archive for server migration
#
# Usage: ./scripts/backup_export.sh
# Output: sujbot_backup_YYYYMMDD_HHMMSS.tar.gz

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="$PROJECT_DIR/backup_$TIMESTAMP"
ARCHIVE_NAME="sujbot_backup_$TIMESTAMP.tar.gz"

# Container names
POSTGRES_CONTAINER="sujbot_postgres"
REDIS_CONTAINER="sujbot_redis"

echo -e "${GREEN}=== SUJBOT Backup Export ===${NC}"
echo "Timestamp: $TIMESTAMP"
echo "Output: $ARCHIVE_NAME"
echo ""

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Function to check if container is running
container_running() {
    docker ps --format '{{.Names}}' | grep -q "^$1$"
}

# 1. PostgreSQL dump
echo -e "${YELLOW}[1/6] Exporting PostgreSQL database...${NC}"
if container_running "$POSTGRES_CONTAINER"; then
    if ! docker exec "$POSTGRES_CONTAINER" pg_dump -U postgres -d sujbot --no-owner --no-acl > "$BACKUP_DIR/postgres.sql"; then
        echo -e "${RED}  ERROR: PostgreSQL dump failed!${NC}"
        rm -rf "$BACKUP_DIR"
        exit 1
    fi
    # Verify dump is not empty
    if [ ! -s "$BACKUP_DIR/postgres.sql" ]; then
        echo -e "${RED}  ERROR: PostgreSQL dump is empty!${NC}"
        rm -rf "$BACKUP_DIR"
        exit 1
    fi
    echo -e "${GREEN}  PostgreSQL dump: $(du -h "$BACKUP_DIR/postgres.sql" | cut -f1)${NC}"
else
    echo -e "${RED}  ERROR: PostgreSQL container not running!${NC}"
    rm -rf "$BACKUP_DIR"
    exit 1
fi

# 2. Redis snapshot
echo -e "${YELLOW}[2/5] Exporting Redis snapshot...${NC}"
if container_running "$REDIS_CONTAINER"; then
    # Trigger BGSAVE and check result
    BGSAVE_RESULT=$(docker exec "$REDIS_CONTAINER" redis-cli BGSAVE 2>&1)
    if [[ "$BGSAVE_RESULT" == *"Background saving started"* ]] || [[ "$BGSAVE_RESULT" == *"Background saving scheduled"* ]]; then
        # Wait for BGSAVE to complete (check every second, max 30 seconds)
        for i in {1..30}; do
            LASTSAVE=$(docker exec "$REDIS_CONTAINER" redis-cli LASTSAVE 2>/dev/null)
            sleep 1
            NEWSAVE=$(docker exec "$REDIS_CONTAINER" redis-cli LASTSAVE 2>/dev/null)
            if [ "$NEWSAVE" != "$LASTSAVE" ] || [ "$i" -eq 30 ]; then
                break
            fi
        done
    fi
    # Copy dump.rdb if exists
    if docker cp "$REDIS_CONTAINER:/data/dump.rdb" "$BACKUP_DIR/" 2>/dev/null; then
        echo -e "${GREEN}  Redis snapshot: $(du -h "$BACKUP_DIR/dump.rdb" | cut -f1)${NC}"
    else
        echo -e "${YELLOW}  Redis dump.rdb not found (database may be empty)${NC}"
    fi
else
    echo -e "${YELLOW}  WARNING: Redis container not running, skipping...${NC}"
fi

# 3. Configuration files
echo -e "${YELLOW}[3/5] Copying configuration files...${NC}"
cd "$PROJECT_DIR"

# .env (contains secrets!) - REQUIRED for backup
if [ -f ".env" ]; then
    if ! cp ".env" "$BACKUP_DIR/"; then
        echo -e "${RED}  ERROR: Failed to copy .env file!${NC}"
        rm -rf "$BACKUP_DIR"
        exit 1
    fi
    echo -e "${GREEN}  .env copied (contains API keys!)${NC}"
else
    echo -e "${RED}  ERROR: .env not found - backup would be incomplete!${NC}"
    rm -rf "$BACKUP_DIR"
    exit 1
fi

# config.json
if [ -f "config.json" ]; then
    if ! cp "config.json" "$BACKUP_DIR/"; then
        echo -e "${RED}  ERROR: Failed to copy config.json!${NC}"
        rm -rf "$BACKUP_DIR"
        exit 1
    fi
    echo -e "${GREEN}  config.json copied${NC}"
fi

# docker-compose files
if [ -f "docker-compose.yml" ]; then
    if ! cp "docker-compose.yml" "$BACKUP_DIR/"; then
        echo -e "${YELLOW}  WARNING: Failed to copy docker-compose.yml${NC}"
    fi
fi
if [ -f "docker-compose.override.yml" ]; then
    if ! cp "docker-compose.override.yml" "$BACKUP_DIR/"; then
        echo -e "${YELLOW}  WARNING: Failed to copy docker-compose.override.yml${NC}"
    fi
fi

# SQL init scripts
if [ -d "docker/postgres/init" ]; then
    mkdir -p "$BACKUP_DIR/postgres_init"
    cp -r docker/postgres/init/* "$BACKUP_DIR/postgres_init/"
    echo -e "${GREEN}  SQL init scripts copied${NC}"
fi

# 4. Data directories
echo -e "${YELLOW}[4/5] Copying data directories...${NC}"

# Source documents
if [ -d "data" ]; then
    cp -r "data" "$BACKUP_DIR/"
    echo -e "${GREEN}  data/: $(du -sh data | cut -f1)${NC}"
fi

# System prompts
if [ -d "prompts" ]; then
    cp -r "prompts" "$BACKUP_DIR/"
    echo -e "${GREEN}  prompts/: $(du -sh prompts | cut -f1)${NC}"
fi

# Output files
if [ -d "output" ]; then
    cp -r "output" "$BACKUP_DIR/"
    echo -e "${GREEN}  output/: $(du -sh output | cut -f1)${NC}"
fi

# Logs
if [ -d "logs" ]; then
    cp -r "logs" "$BACKUP_DIR/"
    echo -e "${GREEN}  logs/: $(du -sh logs | cut -f1)${NC}"
fi

# Dataset files (for evaluation)
if [ -d "dataset" ]; then
    cp -r "dataset" "$BACKUP_DIR/"
    echo -e "${GREEN}  dataset/: $(du -sh dataset | cut -f1)${NC}"
fi

# 5. Create archive
echo -e "${YELLOW}[5/5] Creating archive...${NC}"
cd "$PROJECT_DIR"
tar -czf "$ARCHIVE_NAME" -C "$PROJECT_DIR" "backup_$TIMESTAMP"

# Cleanup
rm -rf "$BACKUP_DIR"

# Verify archive integrity
echo -e "${YELLOW}Verifying archive integrity...${NC}"
if tar -tzf "$ARCHIVE_NAME" > /dev/null 2>&1; then
    echo -e "${GREEN}  Archive verification: OK${NC}"
else
    echo -e "${RED}  ERROR: Archive verification failed!${NC}"
    exit 1
fi

# Summary
ARCHIVE_SIZE=$(du -h "$PROJECT_DIR/$ARCHIVE_NAME" | cut -f1)
echo ""
echo -e "${GREEN}=== Backup Complete ===${NC}"
echo -e "Archive: ${GREEN}$PROJECT_DIR/$ARCHIVE_NAME${NC}"
echo -e "Size: ${GREEN}$ARCHIVE_SIZE${NC}"
echo ""
echo -e "${YELLOW}To download to your MacBook:${NC}"
echo "  scp user@server:$PROJECT_DIR/$ARCHIVE_NAME ~/Downloads/"
echo ""
echo -e "${RED}WARNING: Archive contains .env with API keys - handle securely!${NC}"
