#!/bin/bash
# =============================================================================
# SUJBOT Database Stack Management
# =============================================================================
# Manages shared database services (PostgreSQL, Neo4j, Redis, pgAdmin).
# These run in system Docker and are shared by production and all development
# instances.
#
# Requires: docker group membership
#
# Usage:
#   ./scripts/db.sh up      - Start database services
#   ./scripts/db.sh down    - Stop database services
#   ./scripts/db.sh status  - Show container status
#   ./scripts/db.sh logs    - View database logs
#   ./scripts/db.sh ps      - List running database containers
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

validate_compose_file() {
    if [ ! -f "$PROJECT_DIR/docker-compose.yml" ]; then
        echo -e "${RED}ERROR: docker-compose.yml not found at: $PROJECT_DIR/docker-compose.yml${NC}"
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
# COMMANDS
# =============================================================================
cmd_up() {
    validate_docker_access
    validate_compose_file
    validate_env_file

    echo -e "${CYAN}=== SUJBOT DATABASE SERVICES ===${NC}"
    echo -e "${GREEN}Starting shared database services...${NC}"
    echo ""

    cd "$PROJECT_DIR"
    docker compose -f docker-compose.yml -p sujbot up -d

    echo ""
    echo -e "${GREEN}Database services started!${NC}"
    echo ""
    echo "Access points:"
    echo "  PostgreSQL: localhost:5432 (or $(hostname -I | awk '{print $1}'):5432)"
    echo "  Neo4j:      localhost:7687 (bolt), localhost:7474 (browser)"
    echo "  Redis:      localhost:6379"
    echo "  pgAdmin:    http://localhost:5050"
    echo ""
    echo "For development, start your stack with:"
    echo "  ./scripts/dev.sh up"
}

cmd_down() {
    validate_docker_access
    validate_compose_file

    echo -e "${YELLOW}Stopping database services...${NC}"
    echo -e "${RED}WARNING: This will affect ALL running development instances!${NC}"
    echo ""

    read -p "Are you sure? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi

    cd "$PROJECT_DIR"
    docker compose -f docker-compose.yml -p sujbot down

    echo -e "${GREEN}Database services stopped.${NC}"
}

cmd_status() {
    validate_docker_access
    validate_compose_file

    echo -e "${CYAN}=== DATABASE STATUS ===${NC}"
    cd "$PROJECT_DIR"
    docker compose -f docker-compose.yml -p sujbot ps
}

cmd_logs() {
    validate_docker_access
    validate_compose_file

    cd "$PROJECT_DIR"
    # Check if any containers are running
    local running
    running=$(docker compose -f docker-compose.yml -p sujbot ps -q 2>/dev/null)
    if [ -z "$running" ]; then
        echo -e "${YELLOW}No database containers running.${NC}"
        echo "Start with: ./scripts/db.sh up"
        exit 0
    fi

    echo -e "${GREEN}Viewing database logs (Ctrl+C to exit)...${NC}"
    docker compose -f docker-compose.yml -p sujbot logs -f "${@:2}"
}

cmd_ps() {
    validate_docker_access

    echo -e "${CYAN}=== DATABASE CONTAINERS ===${NC}"
    local output
    output=$(docker ps --filter "name=sujbot_postgres" --filter "name=sujbot_neo4j" --filter "name=sujbot_redis" --filter "name=sujbot_pgadmin" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}")
    if [ -z "$output" ] || [ "$(echo "$output" | wc -l)" -le 1 ]; then
        echo -e "${YELLOW}No database containers running.${NC}"
        echo "Start with: ./scripts/db.sh up"
    else
        echo "$output"
    fi
}

cmd_help() {
    echo "SUJBOT Database Stack Management"
    echo ""
    echo "Manages shared database services used by production and development."
    echo "Requires: docker group membership"
    echo ""
    echo "Usage: $0 <command>"
    echo ""
    echo "Commands:"
    echo "  up       Start database services (postgres, neo4j, redis, pgadmin)"
    echo "  down     Stop database services (WARNING: affects all dev instances)"
    echo "  status   Show container status"
    echo "  logs     View logs (optional: service name)"
    echo "  ps       List running database containers"
    echo "  help     Show this help message"
    echo ""
    echo "Access points (when running):"
    echo "  PostgreSQL: localhost:5432"
    echo "  Neo4j:      localhost:7687 (bolt), localhost:7474 (browser)"
    echo "  Redis:      localhost:6379"
    echo "  pgAdmin:    http://localhost:5050"
}

# =============================================================================
# MAIN
# =============================================================================
case "${1:-help}" in
    up)      cmd_up ;;
    down)    cmd_down ;;
    status)  cmd_status ;;
    logs)    cmd_logs "$@" ;;
    ps)      cmd_ps ;;
    help)    cmd_help ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        cmd_help
        exit 1
        ;;
esac
