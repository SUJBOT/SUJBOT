#!/bin/bash
# =============================================================================
# SUJBOT Production Management Script
# =============================================================================
# This script requires password authentication to manage production containers.
# Password is stored in sudo.txt (gitignored) - NOT system sudo.
#
# Requires: docker group membership + production password
#
# Usage:
#   ./scripts/prod.sh up      - Start production (requires password)
#   ./scripts/prod.sh down    - Stop production (requires password)
#   ./scripts/prod.sh restart - Restart production (requires password)
#   ./scripts/prod.sh logs    - View production logs
#   ./scripts/prod.sh status  - Show container status
#   ./scripts/prod.sh ps      - List running containers
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# =============================================================================
# VALIDATION
# =============================================================================
validate_docker_access() {
    if ! docker info >/dev/null 2>&1; then
        echo -e "${RED}ERROR: Cannot connect to Docker daemon.${NC}"
        echo ""
        echo "Make sure you are in the 'docker' group:"
        echo "  groups | grep docker"
        echo ""
        echo "If not, ask an admin to add you:"
        echo "  sudo usermod -aG docker $USER"
        echo "  newgrp docker"
        exit 1
    fi
}

validate_compose_files() {
    if [ ! -f "$PROJECT_DIR/docker-compose.yml" ]; then
        echo -e "${RED}ERROR: docker-compose.yml not found at: $PROJECT_DIR/docker-compose.yml${NC}"
        exit 1
    fi
    if [ ! -f "$PROJECT_DIR/docker-compose.prod.yml" ]; then
        echo -e "${RED}ERROR: docker-compose.prod.yml not found at: $PROJECT_DIR/docker-compose.prod.yml${NC}"
        exit 1
    fi
}

validate_env_file() {
    if [ ! -f "$PROJECT_DIR/.env" ]; then
        echo -e "${RED}ERROR: .env file not found at: $PROJECT_DIR/.env${NC}"
        echo "Copy from .env.example and configure:"
        echo "  cp .env.example .env"
        exit 1
    fi
}

# =============================================================================
# PRODUCTION SAFETY CHECK
# =============================================================================
# Password is read from sudo.txt file (gitignored)
# SECURITY NOTE: This is basic protection against accidental production changes.
# For production servers, consider using proper access controls (SSH keys, etc.)
PROD_PASSWORD_FILE="$PROJECT_DIR/sudo.txt"

check_auth() {
    validate_docker_access
    validate_compose_files
    validate_env_file

    echo -e "${YELLOW}=== PRODUCTION ENVIRONMENT ===${NC}"
    echo -e "${RED}WARNING: You are about to modify PRODUCTION containers!${NC}"
    echo -e "${RED}This affects https://sujbot.fjfi.cvut.cz${NC}"
    echo ""

    # Check if password file exists
    if [ ! -f "$PROD_PASSWORD_FILE" ]; then
        echo -e "${RED}ERROR: Password file not found: $PROD_PASSWORD_FILE${NC}"
        echo "Create this file with a strong password (chmod 600 recommended)."
        exit 1
    fi

    # Check if password file is readable
    if [ ! -r "$PROD_PASSWORD_FILE" ]; then
        echo -e "${RED}ERROR: Cannot read password file: $PROD_PASSWORD_FILE${NC}"
        exit 1
    fi

    # Read password and strip whitespace
    PROD_PASSWORD=$(tr -d '[:space:]' < "$PROD_PASSWORD_FILE")

    # Validate password is not empty
    if [ -z "$PROD_PASSWORD" ]; then
        echo -e "${RED}ERROR: Password file is empty: $PROD_PASSWORD_FILE${NC}"
        echo "Add a strong password to this file."
        exit 1
    fi

    # Request password
    read -s -p "Enter production password: " entered_password
    echo ""

    # Strip whitespace from entered password
    entered_password=$(echo "$entered_password" | tr -d '[:space:]')

    if [ "$entered_password" != "$PROD_PASSWORD" ]; then
        # Delay to mitigate brute-force timing attacks
        sleep 2
        echo -e "${RED}ERROR: Invalid password. Aborting.${NC}"
        exit 1
    fi

    echo -e "${GREEN}Authentication successful.${NC}"
    echo ""
}

# =============================================================================
# DATABASE CHECK
# =============================================================================
ensure_databases() {
    # Check if database containers are running
    if ! docker ps --filter "name=sujbot_postgres" --format "{{.Names}}" | grep -q sujbot_postgres; then
        echo -e "${YELLOW}Database services not running. Starting them first...${NC}"
        cd "$PROJECT_DIR"
        docker compose -f docker-compose.yml -p sujbot up -d
        echo "Waiting for databases to be healthy..."
        sleep 10
    fi
}

# =============================================================================
# Commands
# =============================================================================
cmd_up() {
    check_auth
    ensure_databases

    echo -e "${GREEN}Starting PRODUCTION application stack...${NC}"
    cd "$PROJECT_DIR"

    # Start production stack (uses external database network)
    docker compose -f docker-compose.yml -f docker-compose.prod.yml -p sujbot up -d

    echo -e "${GREEN}Production started successfully!${NC}"
    echo ""
    echo "Access: https://sujbot.fjfi.cvut.cz"
}

cmd_down() {
    check_auth

    echo -e "${YELLOW}Stopping PRODUCTION application stack...${NC}"
    cd "$PROJECT_DIR"

    # Stop only production services (keep databases running)
    docker compose -f docker-compose.prod.yml -p sujbot down

    echo -e "${GREEN}Production application stopped.${NC}"
    echo -e "${YELLOW}Note: Database services are still running.${NC}"
    echo "To stop databases too: ./scripts/db.sh down"
}

cmd_restart() {
    check_auth
    ensure_databases

    echo -e "${YELLOW}Restarting PRODUCTION containers...${NC}"
    cd "$PROJECT_DIR"

    docker compose -f docker-compose.yml -f docker-compose.prod.yml -p sujbot restart

    echo -e "${GREEN}Production restarted.${NC}"
}

cmd_logs() {
    validate_docker_access
    validate_compose_files

    cd "$PROJECT_DIR"
    # Check if any containers are running
    local running
    running=$(docker compose -f docker-compose.yml -f docker-compose.prod.yml -p sujbot ps -q 2>/dev/null)
    if [ -z "$running" ]; then
        echo -e "${YELLOW}No production containers running.${NC}"
        echo "Start with: ./scripts/prod.sh up"
        exit 0
    fi

    echo -e "${GREEN}Viewing PRODUCTION logs (Ctrl+C to exit)...${NC}"
    docker compose -f docker-compose.yml -f docker-compose.prod.yml -p sujbot logs -f "${@:2}"
}

cmd_status() {
    validate_docker_access
    validate_compose_files

    echo -e "${GREEN}=== PRODUCTION STATUS ===${NC}"
    cd "$PROJECT_DIR"
    docker compose -f docker-compose.yml -f docker-compose.prod.yml -p sujbot ps
}

cmd_ps() {
    validate_docker_access

    echo -e "${GREEN}=== PRODUCTION CONTAINERS ===${NC}"
    local output
    output=$(docker ps --filter "name=sujbot_" --filter "name=!sujbot_dev" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}")
    if [ -z "$output" ] || [ "$(echo "$output" | wc -l)" -le 1 ]; then
        echo -e "${YELLOW}No production containers running.${NC}"
        echo "Start with: ./scripts/prod.sh up"
    else
        echo "$output"
    fi
}

cmd_help() {
    echo "SUJBOT Production Management"
    echo ""
    echo "Requires: docker group membership + production password"
    echo ""
    echo "Usage: $0 <command>"
    echo ""
    echo "Commands (require password):"
    echo "  up       Start production (databases + application)"
    echo "  down     Stop production application (keeps databases)"
    echo "  restart  Restart production containers"
    echo ""
    echo "Commands (no password):"
    echo "  logs     View logs (optional: service name)"
    echo "  status   Show container status"
    echo "  ps       List running production containers"
    echo "  help     Show this help message"
    echo ""
    echo "Related scripts:"
    echo "  ./scripts/db.sh     - Manage database services only"
    echo "  ./scripts/dev.sh    - Manage development environment"
}

# =============================================================================
# Main
# =============================================================================
case "${1:-help}" in
    up)      cmd_up ;;
    down)    cmd_down ;;
    restart) cmd_restart ;;
    logs)    cmd_logs "$@" ;;
    status)  cmd_status ;;
    ps)      cmd_ps ;;
    help)    cmd_help ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        cmd_help
        exit 1
        ;;
esac
