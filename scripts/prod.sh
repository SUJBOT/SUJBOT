#!/bin/bash
# =============================================================================
# SUJBOT Production Management Script
# =============================================================================
# This script requires sudo authentication to manage production containers.
# Usage:
#   ./scripts/prod.sh up      - Start production
#   ./scripts/prod.sh down    - Stop production
#   ./scripts/prod.sh restart - Restart production
#   ./scripts/prod.sh logs    - View production logs
#   ./scripts/prod.sh status  - Show container status
#   ./scripts/prod.sh ps      - List running containers
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# =============================================================================
# PRODUCTION SAFETY CHECK
# =============================================================================
# Password is read from sudo.txt file (gitignored)
PROD_PASSWORD_FILE="$PROJECT_DIR/sudo.txt"

check_auth() {
    echo -e "${YELLOW}=== PRODUCTION ENVIRONMENT ===${NC}"
    echo -e "${RED}WARNING: You are about to modify PRODUCTION containers!${NC}"
    echo -e "${RED}This affects https://sujbot.fjfi.cvut.cz${NC}"
    echo ""

    # Check if password file exists
    if [ ! -f "$PROD_PASSWORD_FILE" ]; then
        echo -e "${RED}ERROR: Password file not found: $PROD_PASSWORD_FILE${NC}"
        exit 1
    fi

    PROD_PASSWORD=$(cat "$PROD_PASSWORD_FILE")

    # Request password
    read -s -p "Enter production password: " entered_password
    echo ""

    if [ "$entered_password" != "$PROD_PASSWORD" ]; then
        echo -e "${RED}ERROR: Invalid password. Aborting.${NC}"
        exit 1
    fi

    echo -e "${GREEN}Authentication successful.${NC}"
    echo ""
}

# =============================================================================
# Commands
# =============================================================================
cmd_up() {
    check_auth
    echo -e "${GREEN}Starting PRODUCTION containers...${NC}"
    cd "$PROJECT_DIR"
    docker compose -f docker-compose.yml -p sujbot up -d
    echo -e "${GREEN}Production started successfully!${NC}"
    echo ""
    echo "Access: https://sujbot.fjfi.cvut.cz"
}

cmd_down() {
    check_auth
    echo -e "${YELLOW}Stopping PRODUCTION containers...${NC}"
    cd "$PROJECT_DIR"
    docker compose -f docker-compose.yml -p sujbot down
    echo -e "${GREEN}Production stopped.${NC}"
}

cmd_restart() {
    check_auth
    echo -e "${YELLOW}Restarting PRODUCTION containers...${NC}"
    cd "$PROJECT_DIR"
    docker compose -f docker-compose.yml -p sujbot restart
    echo -e "${GREEN}Production restarted.${NC}"
}

cmd_logs() {
    echo -e "${GREEN}Viewing PRODUCTION logs (Ctrl+C to exit)...${NC}"
    cd "$PROJECT_DIR"
    docker compose -f docker-compose.yml -p sujbot logs -f "${@:2}"
}

cmd_status() {
    echo -e "${GREEN}=== PRODUCTION STATUS ===${NC}"
    cd "$PROJECT_DIR"
    docker compose -f docker-compose.yml -p sujbot ps
}

cmd_ps() {
    echo -e "${GREEN}=== PRODUCTION CONTAINERS ===${NC}"
    docker ps --filter "name=sujbot_" --filter "name=!sujbot_dev" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
}

cmd_help() {
    echo "SUJBOT Production Management"
    echo ""
    echo "Usage: $0 <command>"
    echo ""
    echo "Commands:"
    echo "  up       Start production containers (requires sudo)"
    echo "  down     Stop production containers (requires sudo)"
    echo "  restart  Restart production containers (requires sudo)"
    echo "  logs     View logs (optional: service name)"
    echo "  status   Show container status"
    echo "  ps       List running production containers"
    echo "  help     Show this help message"
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
