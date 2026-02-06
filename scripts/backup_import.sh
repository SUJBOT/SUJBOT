#!/bin/bash
# SUJBOT Backup Import Script
# Restores data from a backup archive
#
# Usage: ./scripts/backup_import.sh <backup_archive.tar.gz>
#
# Prerequisites:
#   - Docker containers must be running (docker compose up -d)
#   - Run from the SUJBOT project directory

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Container names
POSTGRES_CONTAINER="sujbot_postgres"
REDIS_CONTAINER="sujbot_redis"

# Check arguments
if [ -z "$1" ]; then
    echo -e "${RED}Usage: $0 <backup_archive.tar.gz>${NC}"
    echo ""
    echo "Example:"
    echo "  ./scripts/backup_import.sh sujbot_backup_20250106_143000.tar.gz"
    exit 1
fi

ARCHIVE="$1"

# Resolve absolute path
if [[ ! "$ARCHIVE" = /* ]]; then
    ARCHIVE="$(pwd)/$ARCHIVE"
fi

if [ ! -f "$ARCHIVE" ]; then
    echo -e "${RED}Error: Archive not found: $ARCHIVE${NC}"
    exit 1
fi

echo -e "${GREEN}=== SUJBOT Backup Import ===${NC}"
echo "Archive: $ARCHIVE"
echo "Project: $PROJECT_DIR"
echo ""

# Function to check if container is running
container_running() {
    docker ps --format '{{.Names}}' | grep -q "^$1$"
}

# Extract archive
echo -e "${YELLOW}[1/6] Extracting archive...${NC}"
cd "$PROJECT_DIR"
tar -xzf "$ARCHIVE"

# Find the backup directory
BACKUP_DIR=$(find "$PROJECT_DIR" -maxdepth 1 -type d -name "backup_*" | head -1)
if [ -z "$BACKUP_DIR" ]; then
    echo -e "${RED}Error: Could not find backup directory in archive${NC}"
    exit 1
fi
echo -e "${GREEN}  Extracted to: $BACKUP_DIR${NC}"

# 2. Restore PostgreSQL
echo -e "${YELLOW}[2/6] Restoring PostgreSQL database...${NC}"
if [ -f "$BACKUP_DIR/postgres.sql" ]; then
    if container_running "$POSTGRES_CONTAINER"; then
        # Verify connection first
        if ! docker exec "$POSTGRES_CONTAINER" psql -U postgres -d sujbot -c "SELECT 1;" >/dev/null 2>&1; then
            echo -e "${RED}  ERROR: Cannot connect to PostgreSQL!${NC}"
            exit 1
        fi

        # Drop and recreate database to ensure clean state
        echo "  Dropping existing data..."
        # Note: DROP IF EXISTS with || true is OK here - we just verified connection works
        docker exec "$POSTGRES_CONTAINER" psql -U postgres -d sujbot -c "DROP SCHEMA IF EXISTS vectors CASCADE;" 2>/dev/null || true
        docker exec "$POSTGRES_CONTAINER" psql -U postgres -d sujbot -c "DROP SCHEMA IF EXISTS checkpoints CASCADE;" 2>/dev/null || true
        docker exec "$POSTGRES_CONTAINER" psql -U postgres -d sujbot -c "DROP SCHEMA IF EXISTS metadata CASCADE;" 2>/dev/null || true
        docker exec "$POSTGRES_CONTAINER" psql -U postgres -d sujbot -c "DROP SCHEMA IF EXISTS bm25 CASCADE;" 2>/dev/null || true
        docker exec "$POSTGRES_CONTAINER" psql -U postgres -d sujbot -c "DROP SCHEMA IF EXISTS auth CASCADE;" 2>/dev/null || true

        echo "  Importing database dump..."
        if ! docker exec -i "$POSTGRES_CONTAINER" psql -U postgres -d sujbot < "$BACKUP_DIR/postgres.sql"; then
            echo -e "${RED}  ERROR: PostgreSQL import failed!${NC}"
            exit 1
        fi
        echo -e "${GREEN}  PostgreSQL restored successfully${NC}"
    else
        echo -e "${RED}  ERROR: PostgreSQL container not running!${NC}"
        echo "  Start containers first: docker compose up -d postgres"
        exit 1
    fi
else
    echo -e "${YELLOW}  No postgres.sql found, skipping...${NC}"
fi

# 3. Restore Redis
echo -e "${YELLOW}[3/5] Restoring Redis snapshot...${NC}"
if [ -f "$BACKUP_DIR/dump.rdb" ]; then
    if container_running "$REDIS_CONTAINER"; then
        if docker cp "$BACKUP_DIR/dump.rdb" "$REDIS_CONTAINER:/data/"; then
            if docker restart "$REDIS_CONTAINER"; then
                # Wait for Redis to be ready
                sleep 2
                if docker exec "$REDIS_CONTAINER" redis-cli ping >/dev/null 2>&1; then
                    echo -e "${GREEN}  Redis snapshot restored${NC}"
                else
                    echo -e "${YELLOW}  WARNING: Redis restarted but not responding${NC}"
                fi
            else
                echo -e "${RED}  ERROR: Failed to restart Redis${NC}"
            fi
        else
            echo -e "${RED}  ERROR: Failed to copy dump.rdb to Redis container${NC}"
        fi
    else
        echo -e "${YELLOW}  WARNING: Redis container not running, skipping...${NC}"
    fi
else
    echo -e "${YELLOW}  No Redis snapshot found, skipping...${NC}"
fi

# 4. Restore configuration files
echo -e "${YELLOW}[4/5] Restoring configuration files...${NC}"
cd "$PROJECT_DIR"

if [ -f "$BACKUP_DIR/.env" ]; then
    # Backup existing .env if present
    if [ -f ".env" ]; then
        cp ".env" ".env.backup_$(date +%Y%m%d_%H%M%S)"
        echo -e "${YELLOW}  Existing .env backed up${NC}"
    fi
    cp "$BACKUP_DIR/.env" "./"
    echo -e "${GREEN}  .env restored${NC}"
fi

if [ -f "$BACKUP_DIR/config.json" ]; then
    cp "$BACKUP_DIR/config.json" "./"
    echo -e "${GREEN}  config.json restored${NC}"
fi

if [ -f "$BACKUP_DIR/docker-compose.yml" ]; then
    cp "$BACKUP_DIR/docker-compose.yml" "./"
    echo -e "${GREEN}  docker-compose.yml restored${NC}"
fi

if [ -f "$BACKUP_DIR/docker-compose.override.yml" ]; then
    cp "$BACKUP_DIR/docker-compose.override.yml" "./"
    echo -e "${GREEN}  docker-compose.override.yml restored${NC}"
fi

# 5. Restore data directories
echo -e "${YELLOW}[5/5] Restoring data directories...${NC}"

# Helper function to restore a directory with proper error handling
restore_directory() {
    local src="$1"
    local dst="$2"
    local name="$3"

    if [ -d "$src" ]; then
        mkdir -p "$dst"
        # Check if source directory has files
        if [ -z "$(ls -A "$src" 2>/dev/null)" ]; then
            echo -e "${YELLOW}  $name/ was empty in backup${NC}"
        elif cp -r "$src/"* "$dst/" 2>/dev/null; then
            echo -e "${GREEN}  $name/ restored${NC}"
        else
            echo -e "${RED}  ERROR: Failed to restore $name/${NC}"
        fi
    fi
}

restore_directory "$BACKUP_DIR/data" "data" "data"
restore_directory "$BACKUP_DIR/prompts" "prompts" "prompts"
restore_directory "$BACKUP_DIR/output" "output" "output"
restore_directory "$BACKUP_DIR/logs" "logs" "logs"
restore_directory "$BACKUP_DIR/dataset" "dataset" "dataset"
restore_directory "$BACKUP_DIR/postgres_init" "docker/postgres/init" "postgres_init"

# Cleanup
echo ""
echo -e "${YELLOW}Cleaning up...${NC}"
rm -rf "$BACKUP_DIR"

# Summary
echo ""
echo -e "${GREEN}=== Import Complete ===${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "  1. Restart all services: docker compose restart"
echo "  2. Verify PostgreSQL: docker exec $POSTGRES_CONTAINER psql -U postgres -d sujbot -c 'SELECT count(*) FROM vectors.layer3;'"
echo "  3. Test the application: curl http://localhost:8000/health"
echo ""
echo -e "${GREEN}Done!${NC}"
