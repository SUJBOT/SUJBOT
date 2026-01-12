#!/bin/bash
# =============================================================================
# SUJBOT Development Management Script
# =============================================================================
# Usage:
#   ./scripts/dev.sh up      - Start development
#   ./scripts/dev.sh down    - Stop development
#   ./scripts/dev.sh restart - Restart development
#   ./scripts/dev.sh logs    - View development logs
#   ./scripts/dev.sh status  - Show container status
#   ./scripts/dev.sh ps      - List running containers
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Validate docker-compose.yml exists
validate_compose_file() {
    if [ ! -f "$PROJECT_DIR/docker-compose.yml" ]; then
        echo -e "${RED}ERROR: docker-compose.yml not found at: $PROJECT_DIR/docker-compose.yml${NC}"
        exit 1
    fi
}

# Create combined env file (base + dev overrides)
prepare_env() {
    validate_compose_file
    # Validate .env exists
    if [ ! -f "$PROJECT_DIR/.env" ]; then
        echo -e "${RED}ERROR: Base .env file not found at: $PROJECT_DIR/.env${NC}"
        echo "Please copy .env.example to .env and configure it."
        exit 1
    fi

    # Validate .env.dev exists
    if [ ! -f "$PROJECT_DIR/.env.dev" ]; then
        echo -e "${RED}ERROR: Development override file not found at: $PROJECT_DIR/.env.dev${NC}"
        echo "Please create .env.dev with development-specific overrides."
        exit 1
    fi

    # Combine env files
    if ! cat "$PROJECT_DIR/.env" "$PROJECT_DIR/.env.dev" > "$PROJECT_DIR/.env.combined"; then
        echo -e "${RED}ERROR: Failed to create combined env file${NC}"
        exit 1
    fi
}

cleanup_env() {
    rm -f "$PROJECT_DIR/.env.combined" 2>/dev/null || true
}

# Ensure cleanup on script exit (handles Ctrl+C and errors)
trap cleanup_env EXIT

# =============================================================================
# Commands
# =============================================================================
cmd_up() {
    echo -e "${CYAN}=== DEVELOPMENT ENVIRONMENT ===${NC}"
    echo -e "${GREEN}Starting DEVELOPMENT containers...${NC}"
    cd "$PROJECT_DIR"
    prepare_env
    docker compose --env-file .env.combined -p sujbot_dev up -d
    echo -e "${GREEN}Development started successfully!${NC}"
    echo ""
    echo "Access points:"
    echo "  - Web UI:    http://localhost:8280"
    echo "  - Backend:   http://localhost:8200"
    echo "  - pgAdmin:   http://localhost:5052"
    echo "  - Neo4j:     http://localhost:7475"
}

cmd_down() {
    echo -e "${YELLOW}Stopping DEVELOPMENT containers...${NC}"
    cd "$PROJECT_DIR"
    prepare_env
    docker compose --env-file .env.combined -p sujbot_dev down
    echo -e "${GREEN}Development stopped.${NC}"
}

cmd_restart() {
    echo -e "${YELLOW}Restarting DEVELOPMENT containers...${NC}"
    cd "$PROJECT_DIR"
    prepare_env
    docker compose --env-file .env.combined -p sujbot_dev restart
    echo -e "${GREEN}Development restarted.${NC}"
}

cmd_logs() {
    cd "$PROJECT_DIR"
    prepare_env
    # Check if any containers are running
    local running
    running=$(docker compose --env-file .env.combined -p sujbot_dev ps -q 2>/dev/null)
    if [ -z "$running" ]; then
        echo -e "${YELLOW}No development containers running.${NC}"
        echo "Start with: ./scripts/dev.sh up"
        exit 0
    fi
    echo -e "${GREEN}Viewing DEVELOPMENT logs (Ctrl+C to exit)...${NC}"
    docker compose --env-file .env.combined -p sujbot_dev logs -f "${@:2}"
    # Cleanup handled by EXIT trap
}

cmd_status() {
    echo -e "${CYAN}=== DEVELOPMENT STATUS ===${NC}"
    cd "$PROJECT_DIR"
    prepare_env
    docker compose --env-file .env.combined -p sujbot_dev ps
}

cmd_ps() {
    echo -e "${CYAN}=== DEVELOPMENT CONTAINERS ===${NC}"
    local output
    output=$(docker ps --filter "name=sujbot_dev" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}")
    if [ -z "$output" ] || [ "$(echo "$output" | wc -l)" -le 1 ]; then
        echo -e "${YELLOW}No development containers running.${NC}"
        echo "Start with: ./scripts/dev.sh up"
    else
        echo "$output"
    fi
}

cmd_build() {
    echo -e "${YELLOW}Building DEVELOPMENT containers...${NC}"
    cd "$PROJECT_DIR"
    prepare_env
    docker compose --env-file .env.combined -p sujbot_dev build "$@"
    echo -e "${GREEN}Build complete.${NC}"
}

cmd_help() {
    echo "SUJBOT Development Management"
    echo ""
    echo "Usage: $0 <command>"
    echo ""
    echo "Commands:"
    echo "  up       Start development containers"
    echo "  down     Stop development containers"
    echo "  restart  Restart development containers"
    echo "  logs     View logs (optional: service name)"
    echo "  status   Show container status"
    echo "  ps       List running development containers"
    echo "  build    Build containers (optional: service name)"
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
    build)   cmd_build "${@:2}" ;;
    help)    cmd_help ;;
    *)
        echo "Unknown command: $1"
        cmd_help
        exit 1
        ;;
esac
