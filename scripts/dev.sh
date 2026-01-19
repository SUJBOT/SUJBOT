#!/bin/bash
# =============================================================================
# SUJBOT Development Environment Management (Rootless Docker)
# =============================================================================
# Each user runs their own isolated development environment using rootless
# Docker. NO sudo required!
#
# Port allocation is based on UID to prevent conflicts:
#   PORT_BASE = 10000 + (UID % 1000) * 100
#   Example: michal (UID 1003) gets ports 10300-10399
#
# Prerequisites:
#   1. Rootless Docker installed: ./scripts/setup-rootless-docker.sh
#   2. Database services running: ./scripts/db.sh up
#
# Usage:
#   ./scripts/dev.sh up      - Start your development environment
#   ./scripts/dev.sh down    - Stop your development environment
#   ./scripts/dev.sh restart - Restart your development environment
#   ./scripts/dev.sh logs    - View development logs
#   ./scripts/dev.sh status  - Show container status
#   ./scripts/dev.sh build   - Rebuild containers
#   ./scripts/dev.sh ports   - Show your allocated ports
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
# PORT ALLOCATION (based on UID)
# =============================================================================
USER_UID=$(id -u)
PORT_BASE=$((10000 + (USER_UID % 1000) * 100))

# Validate USER variable (used in docker compose project names)
# Must contain only alphanumeric characters, underscores, and dashes
if [[ ! "$USER" =~ ^[a-zA-Z0-9_-]+$ ]]; then
    echo -e "${RED}ERROR: Invalid USER value: '$USER'${NC}"
    echo "USER must contain only alphanumeric characters, underscores, and dashes."
    exit 1
fi

export DEV_HTTP_PORT=$((PORT_BASE + 0))
export DEV_HTTPS_PORT=$((PORT_BASE + 1))
export DEV_BACKEND_PORT=$((PORT_BASE + 2))
export DEV_FRONTEND_PORT=$((PORT_BASE + 3))

# =============================================================================
# HOST IP DETECTION (for database access)
# =============================================================================
# Rootless Docker cannot use localhost to access host services.
# We need to use the actual host IP address.
get_host_ip() {
    # Try to get the primary IP
    local ip
    ip=$(ip route get 1 2>/dev/null | awk '{print $7; exit}')
    if [ -z "$ip" ]; then
        # Fallback: try hostname
        ip=$(hostname -I 2>/dev/null | awk '{print $1}')
    fi
    if [ -z "$ip" ]; then
        echo -e "${RED}ERROR: Could not determine host IP address${NC}"
        exit 1
    fi
    echo "$ip"
}

export HOST_IP=$(get_host_ip)

# =============================================================================
# VALIDATION
# =============================================================================
check_docker() {
    if ! docker info >/dev/null 2>&1; then
        echo -e "${RED}ERROR: Docker is not running or not accessible.${NC}"
        echo ""

        # Check if rootless Docker socket exists
        local rootless_socket="/run/user/$USER_UID/docker.sock"
        if [ -S "$rootless_socket" ]; then
            echo "Rootless Docker socket found but not responding."
            echo "Try restarting: systemctl --user restart docker"
        else
            echo "Rootless Docker is not set up for your user."
            echo "Run the setup script:"
            echo "  ./scripts/setup-rootless-docker.sh"
        fi
        exit 1
    fi
}

check_databases() {
    # Check if PostgreSQL is accessible via host IP
    if ! nc -z "$HOST_IP" 5432 2>/dev/null; then
        echo -e "${RED}ERROR: PostgreSQL not reachable at $HOST_IP:5432${NC}"
        echo ""
        echo "The shared database services must be running first."
        echo "Start them with (requires docker group membership):"
        echo "  ./scripts/db.sh up"
        echo ""
        echo "Or ask someone with docker group access to start them."
        exit 1
    fi
}

validate_env() {
    if [ ! -f "$PROJECT_DIR/.env" ]; then
        echo -e "${RED}ERROR: .env file not found at: $PROJECT_DIR/.env${NC}"
        echo "Copy from .env.example and configure:"
        echo "  cp .env.example .env"
        exit 1
    fi
}

validate_compose_file() {
    if [ ! -f "$PROJECT_DIR/docker-compose.dev.yml" ]; then
        echo -e "${RED}ERROR: docker-compose.dev.yml not found${NC}"
        exit 1
    fi
}

# =============================================================================
# ENVIRONMENT PREPARATION
# =============================================================================
prepare_env() {
    validate_env
    validate_compose_file

    # Create user-specific env file with secure permissions
    # Set umask to 077 to ensure files are readable only by owner
    local old_umask
    old_umask=$(umask)
    umask 077

    cat > "$PROJECT_DIR/.env.dev.user" <<EOF
# Auto-generated for user: $USER (UID: $USER_UID)
# Port allocation: $PORT_BASE-$((PORT_BASE + 99))
# Generated: $(date)

# User-specific ports
DEV_HTTP_PORT=$DEV_HTTP_PORT
DEV_HTTPS_PORT=$DEV_HTTPS_PORT
DEV_BACKEND_PORT=$DEV_BACKEND_PORT
DEV_FRONTEND_PORT=$DEV_FRONTEND_PORT

# Host IP for database access (rootless Docker cannot use localhost)
HOST_IP=$HOST_IP

# User identifier for container names
USER=$USER
EOF

    # Combine base .env with user-specific settings
    cat "$PROJECT_DIR/.env" "$PROJECT_DIR/.env.dev.user" > "$PROJECT_DIR/.env.combined"

    # Restore original umask
    umask "$old_umask"
}

cleanup_env() {
    rm -f "$PROJECT_DIR/.env.combined" "$PROJECT_DIR/.env.dev.user" 2>/dev/null || true
}

# Ensure cleanup on script exit
trap cleanup_env EXIT

# =============================================================================
# COMMANDS
# =============================================================================
cmd_up() {
    check_docker
    check_databases
    prepare_env

    echo -e "${CYAN}=== DEVELOPMENT ENVIRONMENT ===${NC}"
    echo -e "User: ${GREEN}$USER${NC} (UID: $USER_UID)"
    echo -e "Ports: ${GREEN}$PORT_BASE-$((PORT_BASE + 99))${NC}"
    echo -e "Database host: ${GREEN}$HOST_IP${NC}"
    echo ""

    cd "$PROJECT_DIR"
    docker compose --env-file .env.combined -f docker-compose.dev.yml -p "sujbot_$USER" up -d

    echo ""
    echo -e "${GREEN}Development environment started!${NC}"
    echo ""
    echo "Access points:"
    echo "  Web UI:     http://localhost:$DEV_HTTP_PORT"
    echo "  Backend:    http://localhost:$DEV_BACKEND_PORT"
    echo "  Frontend:   http://localhost:$DEV_FRONTEND_PORT (Vite dev server)"
    echo ""
    echo "Useful commands:"
    echo "  ./scripts/dev.sh logs           - View all logs"
    echo "  ./scripts/dev.sh logs backend   - View backend logs"
    echo "  ./scripts/dev.sh status         - Check container status"
}

cmd_down() {
    check_docker
    prepare_env

    echo -e "${YELLOW}Stopping development environment...${NC}"
    cd "$PROJECT_DIR"
    docker compose --env-file .env.combined -f docker-compose.dev.yml -p "sujbot_$USER" down
    echo -e "${GREEN}Development environment stopped.${NC}"
}

cmd_restart() {
    cmd_down
    cmd_up
}

cmd_logs() {
    check_docker
    prepare_env

    cd "$PROJECT_DIR"
    # Check if any containers are running
    local running
    running=$(docker compose --env-file .env.combined -f docker-compose.dev.yml -p "sujbot_$USER" ps -q 2>/dev/null)
    if [ -z "$running" ]; then
        echo -e "${YELLOW}No development containers running.${NC}"
        echo "Start with: ./scripts/dev.sh up"
        exit 0
    fi

    echo -e "${GREEN}Viewing development logs (Ctrl+C to exit)...${NC}"
    docker compose --env-file .env.combined -f docker-compose.dev.yml -p "sujbot_$USER" logs -f "${@:2}"
}

cmd_status() {
    check_docker
    prepare_env

    echo -e "${CYAN}=== DEVELOPMENT STATUS ($USER) ===${NC}"
    cd "$PROJECT_DIR"
    docker compose --env-file .env.combined -f docker-compose.dev.yml -p "sujbot_$USER" ps
}

cmd_build() {
    check_docker
    prepare_env

    echo -e "${YELLOW}Building development containers...${NC}"
    cd "$PROJECT_DIR"
    docker compose --env-file .env.combined -f docker-compose.dev.yml -p "sujbot_$USER" build "${@:2}"
    echo -e "${GREEN}Build complete.${NC}"
}

cmd_ports() {
    echo -e "${CYAN}=== PORT ALLOCATION FOR $USER ===${NC}"
    echo ""
    echo "User: $USER (UID: $USER_UID)"
    echo "Port range: $PORT_BASE-$((PORT_BASE + 99))"
    echo ""
    echo "Service ports:"
    echo "  Nginx HTTP:  $DEV_HTTP_PORT"
    echo "  Nginx HTTPS: $DEV_HTTPS_PORT"
    echo "  Backend:     $DEV_BACKEND_PORT"
    echo "  Frontend:    $DEV_FRONTEND_PORT"
    echo ""
    echo "Database host: $HOST_IP"
}

cmd_help() {
    echo "SUJBOT Development Environment (Rootless Docker)"
    echo ""
    echo "User: $USER (UID: $USER_UID)"
    echo "Port range: $PORT_BASE-$((PORT_BASE + 99))"
    echo ""
    echo "Usage: $0 <command>"
    echo ""
    echo "Commands:"
    echo "  up       Start development containers"
    echo "  down     Stop development containers"
    echo "  restart  Restart development containers"
    echo "  logs     View logs (optional: service name)"
    echo "  status   Show container status"
    echo "  build    Build containers (optional: service name)"
    echo "  ports    Show your allocated port numbers"
    echo "  help     Show this help message"
    echo ""
    echo "Prerequisites:"
    echo "  1. Rootless Docker: ./scripts/setup-rootless-docker.sh"
    echo "  2. Database services: ./scripts/db.sh up"
}

# =============================================================================
# MAIN
# =============================================================================
case "${1:-help}" in
    up)      cmd_up ;;
    down)    cmd_down ;;
    restart) cmd_restart ;;
    logs)    cmd_logs "$@" ;;
    status)  cmd_status ;;
    build)   cmd_build "$@" ;;
    ports)   cmd_ports ;;
    help)    cmd_help ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        cmd_help
        exit 1
        ;;
esac
