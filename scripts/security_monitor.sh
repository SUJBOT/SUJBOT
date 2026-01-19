#!/bin/bash
#
# SUJBOT2 Security Monitoring Script
# Runs via cron to perform security health checks using Claude Code CLI
#
# Usage:
#   ./scripts/security_monitor.sh                    # Regular 4h check (production)
#   ./scripts/security_monitor.sh --daily            # Daily 24h summary
#   ./scripts/security_monitor.sh --dry-run          # Test without notifications
#   ./scripts/security_monitor.sh --env dev          # Monitor dev environment
#   ./scripts/security_monitor.sh --env dev --dry-run # Dev environment dry-run
#

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="$PROJECT_DIR/config.json"
ENV_FILE="$PROJECT_DIR/.env"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Read log directory from config (with fallback)
if [ -f "$CONFIG_FILE" ]; then
    LOG_DIR=$(jq -r '.security_monitoring.logging.log_dir // "'"$PROJECT_DIR/logs/security"'"' "$CONFIG_FILE")
else
    LOG_DIR="$PROJECT_DIR/logs/security"
fi

# Convert relative paths to absolute (relative to PROJECT_DIR)
if [[ "$LOG_DIR" == ./* ]]; then
    LOG_DIR="$PROJECT_DIR/${LOG_DIR#./}"
fi

# Security: Validate LOG_DIR is within PROJECT_DIR to prevent path traversal
# Resolve to canonical path and check prefix
LOG_DIR_REAL=$(cd "$PROJECT_DIR" && mkdir -p "$LOG_DIR" 2>/dev/null && cd "$LOG_DIR" && pwd) || {
    echo "ERROR: Cannot create or access log directory: $LOG_DIR"
    exit 1
}
if [[ "$LOG_DIR_REAL" != "$PROJECT_DIR"* ]]; then
    echo "ERROR: Log directory must be within project directory (path traversal detected)"
    echo "  LOG_DIR: $LOG_DIR"
    echo "  Resolved: $LOG_DIR_REAL"
    echo "  PROJECT_DIR: $PROJECT_DIR"
    exit 1
fi
LOG_DIR="$LOG_DIR_REAL"
LOG_FILE="$LOG_DIR/monitor_$TIMESTAMP.log"

# ============================================================================
# Error Handler
# ============================================================================

error_handler() {
    local exit_code=$1
    local line_number=$2
    echo "[ERROR] Script failed on line $line_number with exit code $exit_code" | tee -a "$LOG_FILE"

    # Try to send error notification
    if [ -f "$SCRIPT_DIR/security_notify.py" ] && [ "$DRY_RUN" = false ]; then
        local notify_result
        notify_result=$(echo '{
            "timestamp": "'"$(date -u +"%Y-%m-%dT%H:%M:%SZ")"'",
            "overall_status": "critical",
            "summary": "Security monitor script selhal na radku '"$line_number"'",
            "findings": [{
                "category": "internal",
                "severity": "critical",
                "title": "Monitor Script Error",
                "description": "Security monitoring skript selhal s exit kodem '"$exit_code"' na radku '"$line_number"'",
                "recommendation": "Zkontrolujte logy v '"$LOG_FILE"'"
            }],
            "metrics": {}
        }' | uv run python "$SCRIPT_DIR/security_notify.py" --config "$CONFIG_FILE" 2>&1) || {
            echo "[ERROR] Failed to send error notification: $notify_result" >> "$LOG_FILE"
        }
    fi
    exit "$exit_code"
}

trap 'error_handler $? $LINENO' ERR

# ============================================================================
# Logging Function
# ============================================================================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

# ============================================================================
# Parse Arguments
# ============================================================================

DAILY_MODE=false
DRY_RUN=false
ENVIRONMENT=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --daily)
            DAILY_MODE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --env)
            if [[ -n "$2" && "$2" != --* ]]; then
                ENVIRONMENT="$2"
                shift 2
            else
                echo "Error: --env requires an argument (dev or prod)"
                exit 1
            fi
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--daily] [--dry-run] [--env dev|prod]"
            exit 1
            ;;
    esac
done

# Default environment to production if not specified
if [ -z "$ENVIRONMENT" ]; then
    ENVIRONMENT="prod"
fi

# Validate environment
if [[ "$ENVIRONMENT" != "dev" && "$ENVIRONMENT" != "prod" ]]; then
    echo "Error: Invalid environment '$ENVIRONMENT'. Use 'dev' or 'prod'"
    exit 1
fi

# ============================================================================
# Setup
# ============================================================================

# Ensure log directory exists
mkdir -p "$LOG_DIR"

log "=== SUJBOT2 Security Monitor Started ==="
log "Environment: $ENVIRONMENT"
log "Mode: $([ "$DAILY_MODE" = true ] && echo 'DAILY SUMMARY' || echo 'REGULAR CHECK')"
log "Dry run: $DRY_RUN"
log "Project: $PROJECT_DIR"

# Source environment variables (for cron context)
if [ -f "$ENV_FILE" ]; then
    set -a
    # shellcheck source=/dev/null
    source "$ENV_FILE"
    set +a
    log "Environment loaded from .env"
fi

# Ensure PATH includes common locations
export PATH="/usr/local/bin:/usr/bin:/bin:$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# ============================================================================
# Verify Prerequisites
# ============================================================================

# Check Claude Code CLI
if ! command -v claude &> /dev/null; then
    log "ERROR: Claude Code CLI not found in PATH"
    log "PATH: $PATH"
    exit 1
fi
log "Claude Code CLI: $(which claude)"

# Check config file
if [ ! -f "$CONFIG_FILE" ]; then
    log "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Check if monitoring is enabled
MONITORING_ENABLED=$(jq -r '.security_monitoring.enabled // false' "$CONFIG_FILE")
if [ "$MONITORING_ENABLED" != "true" ]; then
    log "Security monitoring is disabled in config. Exiting."
    exit 0
fi

# ============================================================================
# Load Configuration
# ============================================================================

# Get containers to monitor (environment-aware)
# Try environment-specific config first, fall back to default containers with prefix
# Using --arg for safe variable interpolation (prevents injection)
ENV_CONTAINERS=$(jq -r --arg env "$ENVIRONMENT" '.security_monitoring.environments[$env].containers // empty | join(" ")' "$CONFIG_FILE" 2>/dev/null)

if [ -n "$ENV_CONTAINERS" ]; then
    CONTAINERS="$ENV_CONTAINERS"
else
    # Fall back to default containers, applying environment prefix
    DEFAULT_CONTAINERS=$(jq -r '.security_monitoring.checks.docker_health.containers | join(" ")' "$CONFIG_FILE")
    if [ "$ENVIRONMENT" = "dev" ]; then
        # Replace sujbot_ prefix with sujbot_dev_
        CONTAINERS=$(echo "$DEFAULT_CONTAINERS" | sed 's/sujbot_/sujbot_dev_/g')
    else
        CONTAINERS="$DEFAULT_CONTAINERS"
    fi
fi
log "Containers to check: $CONTAINERS"

# Get log window based on mode
if [ "$DAILY_MODE" = true ]; then
    LOG_SINCE=$(jq -r '.security_monitoring.checks.log_analysis.log_since_daily // "24h"' "$CONFIG_FILE")
    PROMPT_FILE_KEY="daily_prompt_file"
else
    LOG_SINCE=$(jq -r '.security_monitoring.checks.log_analysis.log_since_regular // "4h"' "$CONFIG_FILE")
    PROMPT_FILE_KEY="prompt_file"
fi

MAX_LOG_LINES=$(jq -r '.security_monitoring.checks.log_analysis.max_log_lines // 500' "$CONFIG_FILE")
PROMPT_FILE=$(jq -r ".security_monitoring.claude_settings.$PROMPT_FILE_KEY // \"prompts/security_monitor.txt\"" "$CONFIG_FILE")
TIMEOUT=$(jq -r '.security_monitoring.claude_settings.timeout_seconds // 120' "$CONFIG_FILE")

log "Log window: $LOG_SINCE"
log "Prompt file: $PROMPT_FILE"

# ============================================================================
# Collect System Data
# ============================================================================

log "Collecting system data..."

# 1. Docker container status
DOCKER_PS="UNAVAILABLE"
if command -v docker &> /dev/null; then
    DOCKER_PS=$(docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" 2>&1) || DOCKER_PS="Docker command failed: $DOCKER_PS"
fi

# 2. Docker resource usage
DOCKER_STATS="UNAVAILABLE"
if command -v docker &> /dev/null; then
    DOCKER_STATS=$(docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}" 2>&1) || DOCKER_STATS="Docker stats failed: $DOCKER_STATS"
fi

# 3. Container logs (errors/warnings only)
CONTAINER_LOGS=""
# Use read with IFS to safely handle container names
while IFS= read -r container; do
    # Skip empty lines
    [ -z "$container" ] && continue
    # Validate container name (alphanumeric, underscore, dash only)
    if [[ ! "$container" =~ ^[a-zA-Z0-9_-]+$ ]]; then
        log "WARNING: Skipping invalid container name: $container"
        continue
    fi
    if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^${container}$"; then
        LOGS=$(docker logs --since "$LOG_SINCE" "$container" 2>&1 | grep -iE "(error|critical|fail|warning|exception)" | tail -n "$MAX_LOG_LINES") || true
        if [ -n "$LOGS" ]; then
            CONTAINER_LOGS+="=== $container ===
$LOGS

"
        fi
    else
        CONTAINER_LOGS+="=== $container ===
CONTAINER NOT RUNNING

"
    fi
done <<< "$(echo "$CONTAINERS" | tr ' ' '\n')"

# 4. System resources
DISK_USAGE=$(df -h "$PROJECT_DIR" 2>&1 | tail -1) || DISK_USAGE="UNAVAILABLE"
MEMORY_USAGE=$(free -h 2>&1 | grep -E "^Mem:" || echo "UNAVAILABLE")

# 5. Health endpoint check (environment-aware)
# Try environment-specific config first, fall back to default ports
# Using --arg for safe variable interpolation (prevents injection)
HEALTH_ENDPOINT=$(jq -r --arg env "$ENVIRONMENT" '.security_monitoring.environments[$env].health_endpoint // empty' "$CONFIG_FILE" 2>/dev/null)
if [ -z "$HEALTH_ENDPOINT" ]; then
    # Default: prod=8000 (internal), dev=8100 (exposed)
    if [ "$ENVIRONMENT" = "dev" ]; then
        HEALTH_ENDPOINT="http://localhost:8100/health"
    else
        HEALTH_ENDPOINT="http://localhost:8000/health"
    fi
fi
log "Health endpoint: $HEALTH_ENDPOINT"

HEALTH_STATUS="UNAVAILABLE"
HEALTH_RESPONSE=""
if command -v curl &> /dev/null; then
    HEALTH_STATUS=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 5 "$HEALTH_ENDPOINT" 2>&1) || HEALTH_STATUS="000"
    if [ "$HEALTH_STATUS" = "200" ]; then
        HEALTH_RESPONSE=$(curl -s --connect-timeout 5 "$HEALTH_ENDPOINT" 2>&1) || HEALTH_RESPONSE="{}"
    fi
fi

# 6. SSL certificate check (environment-aware)
# Check if SSL is enabled for this environment
# Using --arg for safe variable interpolation (prevents injection)
SSL_ENABLED=$(jq -r --arg env "$ENVIRONMENT" '.security_monitoring.environments[$env].ssl_enabled // true' "$CONFIG_FILE" 2>/dev/null)

if [ "$SSL_ENABLED" = "false" ] || [ "$ENVIRONMENT" = "dev" ]; then
    SSL_INFO="SSL check skipped (${ENVIRONMENT} environment)"
else
    SSL_INFO="NOT CONFIGURED"
    SSL_CERT_PATH=$(jq -r '.security_monitoring.checks.ssl_certificate.cert_path // ""' "$CONFIG_FILE")
    if [ -n "$SSL_CERT_PATH" ] && [ -f "$SSL_CERT_PATH" ]; then
        SSL_EXPIRY=$(openssl x509 -enddate -noout -in "$SSL_CERT_PATH" 2>&1) || SSL_EXPIRY="Unable to read certificate"
        if [[ "$SSL_EXPIRY" != "Unable"* ]]; then
            SSL_EXPIRY_DATE=$(echo "$SSL_EXPIRY" | cut -d= -f2)
            SSL_DAYS=$(( ($(date -d "$SSL_EXPIRY_DATE" +%s) - $(date +%s)) / 86400 )) || SSL_DAYS="?"
            SSL_INFO="Certificate: $SSL_CERT_PATH
Expiry: $SSL_EXPIRY_DATE
Days remaining: $SSL_DAYS"
        else
            SSL_INFO="$SSL_EXPIRY"
        fi
    fi
fi

# 7. Authentication events (from nginx logs)
AUTH_FAILURES=""
NGINX_LOG="$PROJECT_DIR/logs/nginx/access.log"
if [ -f "$NGINX_LOG" ]; then
    # Count 401/403 responses in the log window
    CUTOFF_TIME=$(date -d "-$LOG_SINCE" +%d/%b/%Y:%H:%M:%S 2>/dev/null || echo "")
    if [ -n "$CUTOFF_TIME" ]; then
        AUTH_FAILURES=$(grep -E "\" (401|403) " "$NGINX_LOG" 2>/dev/null | tail -n 100 || echo "No auth failures found")
    else
        AUTH_FAILURES=$(grep -E "\" (401|403) " "$NGINX_LOG" 2>/dev/null | tail -n 50 || echo "No auth failures found")
    fi
fi

# 8. Uptime
SYSTEM_UPTIME=$(uptime 2>&1 || echo "UNAVAILABLE")

# ============================================================================
# Compile System Data
# ============================================================================

SYSTEM_DATA="=== DOCKER CONTAINERS ===
$DOCKER_PS

=== DOCKER RESOURCE USAGE ===
$DOCKER_STATS

=== CONTAINER LOGS (errors/warnings from last $LOG_SINCE) ===
$CONTAINER_LOGS

=== SYSTEM RESOURCES ===
Disk: $DISK_USAGE
Memory: $MEMORY_USAGE
Uptime: $SYSTEM_UPTIME

=== HEALTH ENDPOINT ===
Backend /health: HTTP $HEALTH_STATUS
Response: $HEALTH_RESPONSE

=== SSL CERTIFICATE ===
$SSL_INFO

=== AUTHENTICATION EVENTS (last $LOG_SINCE) ===
$AUTH_FAILURES

=== REPORT METADATA ===
Timestamp: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
Environment: $ENVIRONMENT
Report Type: $([ "$DAILY_MODE" = true ] && echo 'daily_summary' || echo 'regular_check')
Period: $LOG_SINCE
Server: $(hostname)
"

log "System data collected successfully"

# ============================================================================
# Load and Prepare Prompt
# ============================================================================

FULL_PROMPT_FILE="$PROJECT_DIR/$PROMPT_FILE"
if [ ! -f "$FULL_PROMPT_FILE" ]; then
    log "ERROR: Prompt file not found: $FULL_PROMPT_FILE"
    exit 1
fi

PROMPT_TEMPLATE=$(cat "$FULL_PROMPT_FILE")

# Replace placeholder with actual data
# Using awk for safe substitution (handles special characters in data)
FULL_PROMPT=$(echo "$PROMPT_TEMPLATE" | awk -v data="$SYSTEM_DATA" '{gsub(/\{\{SYSTEM_DATA\}\}/, data); print}')

# ============================================================================
# Invoke Claude Code CLI
# ============================================================================

log "Invoking Claude Code CLI for analysis..."

ANALYSIS_FILE="$LOG_DIR/analysis_$TIMESTAMP.json"

# Run Claude Code with timeout
# Using --print flag to get raw output, no interactive mode
CLAUDE_OUTPUT=$(timeout "$TIMEOUT" claude --print --dangerously-skip-permissions -p "$FULL_PROMPT" 2>&1) || {
    CLAUDE_EXIT=$?
    log "WARNING: Claude Code exited with code $CLAUDE_EXIT"
    if [ $CLAUDE_EXIT -eq 124 ]; then
        log "ERROR: Claude Code timed out after ${TIMEOUT}s"
    fi
    CLAUDE_OUTPUT='{
        "timestamp": "'"$(date -u +"%Y-%m-%dT%H:%M:%SZ")"'",
        "overall_status": "warning",
        "summary": "Claude Code analyza selhala nebo vyprsela",
        "findings": [{
            "category": "internal",
            "severity": "warning",
            "title": "Analysis Incomplete",
            "description": "Claude Code CLI neodpovedel vcas nebo selhal",
            "recommendation": "Zkontrolujte Claude Code instalaci a pristup"
        }],
        "metrics": {}
    }'
}

# Try to extract JSON from Claude output (it might have extra text or markdown blocks)
# First, strip markdown code blocks if present (```json ... ```)
STRIPPED_OUTPUT=$(echo "$CLAUDE_OUTPUT" | sed 's/^```json//g; s/^```//g' | tr -d '\r')

# Look for JSON object in the output
if echo "$STRIPPED_OUTPUT" | jq . > /dev/null 2>&1; then
    CLAUDE_JSON="$STRIPPED_OUTPUT"
else
    # Try to extract JSON from mixed output using sed (more portable than grep -P)
    CLAUDE_JSON=$(echo "$STRIPPED_OUTPUT" | sed -n '/{/,/}/p' | head -1 || echo "$STRIPPED_OUTPUT")
    if ! echo "$CLAUDE_JSON" | jq . > /dev/null 2>&1; then
        log "WARNING: Could not parse JSON from Claude output"
        CLAUDE_JSON='{
            "timestamp": "'"$(date -u +"%Y-%m-%dT%H:%M:%SZ")"'",
            "overall_status": "warning",
            "summary": "Nelze parsovat vystup z Claude Code",
            "findings": [{
                "category": "internal",
                "severity": "warning",
                "title": "Parse Error",
                "description": "Vystup z Claude Code neni validni JSON",
                "recommendation": "Zkontrolujte prompt format a Claude Code vystup"
            }],
            "metrics": {},
            "raw_output": "'"$(echo "$CLAUDE_OUTPUT" | head -c 500 | tr '\n' ' ' | sed 's/"/\\"/g')"'"
        }'
    fi
fi

log "Claude Code analysis complete"

# Save analysis output
echo "$CLAUDE_JSON" > "$ANALYSIS_FILE"
log "Analysis saved to: $ANALYSIS_FILE"

# ============================================================================
# Send Notifications
# ============================================================================

if [ "$DRY_RUN" = true ]; then
    log "DRY RUN - Skipping notifications"
    log "Analysis output:"
    echo "$CLAUDE_JSON" | jq . 2>/dev/null || echo "$CLAUDE_JSON"
else
    log "Sending notifications..."
    if [ -f "$SCRIPT_DIR/security_notify.py" ]; then
        NOTIFY_OUTPUT=$(echo "$CLAUDE_JSON" | uv run python "$SCRIPT_DIR/security_notify.py" --config "$CONFIG_FILE" 2>&1) || {
            log "WARNING: Notification script failed with exit code $?"
            log "Notification output: $NOTIFY_OUTPUT"
        }
        echo "$NOTIFY_OUTPUT" | tee -a "$LOG_FILE"
    else
        log "WARNING: Notification script not found: $SCRIPT_DIR/security_notify.py"
    fi
fi

# ============================================================================
# Cleanup Old Logs
# ============================================================================

RETENTION_DAYS=$(jq -r '.security_monitoring.logging.retention_days // 30' "$CONFIG_FILE")
log "Cleaning up logs older than $RETENTION_DAYS days..."
find "$LOG_DIR" -name "*.log" -mtime +"$RETENTION_DAYS" -delete 2>/dev/null || true
find "$LOG_DIR" -name "*.json" -mtime +"$RETENTION_DAYS" -delete 2>/dev/null || true

# ============================================================================
# Done
# ============================================================================

log "=== Security Monitor Completed Successfully ==="
