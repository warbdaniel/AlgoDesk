#!/bin/bash
# AlgoDesk Deployment Script
# Pulls latest changes from GitHub and restarts affected services
# Usage: deploy.sh [--force] [--no-restart]

set -e

TRADING_DIR="/opt/trading-desk"
LOG_FILE="/opt/trading-desk/logs/system/deploy.log"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log() { echo -e "${GREEN}[DEPLOY]${NC} $(date '+%H:%M:%S') $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $(date '+%H:%M:%S') $1"; }
err() { echo -e "${RED}[ERROR]${NC} $(date '+%H:%M:%S') $1"; }

cd "$TRADING_DIR"

log "Starting deployment..."
log "Current commit: $(git rev-parse --short HEAD)"

# Stash any local changes
if [[ $(git status --porcelain) ]]; then
    warn "Local changes detected, stashing..."
    git stash
fi

# Pull latest
log "Pulling from origin/main..."
git pull origin main 2>&1

NEW_COMMIT=$(git rev-parse --short HEAD)
log "Now at commit: $NEW_COMMIT"

# Check what changed
CHANGED_FILES=$(git diff --name-only HEAD~1 HEAD 2>/dev/null || echo "first-deploy")

if [[ "$1" != "--no-restart" ]]; then
    # Restart affected services
    if echo "$CHANGED_FILES" | grep -q "scripts/regime-detector/"; then
        log "Regime detector changed - restarting..."
        pm2 restart regime-detector
    fi

    if echo "$CHANGED_FILES" | grep -q "dashboard/"; then
        log "Dashboard changed - restarting..."
        pm2 restart dashboard
    fi

    if echo "$CHANGED_FILES" | grep -q "guardian/"; then
        log "Guardian changed - restarting..."
        systemctl restart trading-guardian
    fi

    if echo "$CHANGED_FILES" | grep -q "scripts/fix-api/"; then
        log "FIX API changed - restarting..."
        pm2 restart fix-api 2>/dev/null || pm2 start scripts/fix-api/fix_api_server.py --name fix-api --interpreter python3
    fi

    if echo "$CHANGED_FILES" | grep -q "scripts/data-pipeline/"; then
        log "Data pipeline changed - restarting..."
        pm2 restart data-pipeline 2>/dev/null || pm2 start scripts/data-pipeline/data_api_server.py --name data-pipeline --interpreter python3
    fi

    if echo "$CHANGED_FILES" | grep -q "scripts/claude-api/"; then
        log "Claude API agent changed - restarting..."
        pm2 restart claude-api 2>/dev/null || pm2 start scripts/claude-api/claude_api_server.py --name claude-api --interpreter python3
    fi
fi

log "Deployment complete! Commit: $NEW_COMMIT"
pm2 list
