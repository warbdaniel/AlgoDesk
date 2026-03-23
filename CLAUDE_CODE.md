# Claude Code Integration Guide

## Architecture
Claude Code works on this repo via GitHub. The VPS at 46.225.98.97 pulls changes.

## Workflow
1. **Claude Code** edits files in the `AlgoDesk` GitHub repo
2. **Claude Code** commits and pushes to `main` branch
3. **VPS** pulls changes: run `deploy` command on VPS
4. **Deploy script** auto-restarts affected services (regime-detector, dashboard, guardian)

## Directory Structure (maps to VPS /opt/trading-desk/)
```
scripts/regime-detector/   → Market Regime Detector (FastAPI on port 5000)
dashboard/                 → Analytics Dashboard (FastAPI on port 5100)  
guardian/                  → Trading Desk Guardian (systemd daemon)
ctrader-mcp-server/       → cTrader MCP Server (14 tools)
scripts/utils/            → Utility scripts (deploy.sh)
configs/                  → Configuration files
```

## Services on VPS
- **regime-detector** (PM2) - FastAPI port 5000
- **dashboard** (PM2) - FastAPI port 5100
- **blackjack-advisor** (PM2) - separate project
- **trading-guardian** (systemd) - monitoring daemon
- **n8n** (Docker) - workflow automation port 5678

## Key Files
- `scripts/regime-detector/regime_detector.py` - Market regime classification
- `scripts/regime-detector/config.yaml` - Regime detector config
- `dashboard/dashboard.py` - Post-trade analytics backend
- `dashboard/templates/dashboard.html` - Dashboard frontend
- `guardian/guardian.py` - Server monitoring daemon
- `guardian/config.yaml` - Guardian config
- `guardian/guardian-ctl` - Guardian CLI tool

## VPS Environment
- Python 3.12.3 (venv at /opt/trading-desk/venv/)
- Key packages: fastapi, uvicorn, numpy, pandas, scikit-learn, hmmlearn, ctrader-sdk
- PM2 for process management
- Docker for n8n

## Important Notes
- `.env` files are in `.gitignore` - never commit secrets
- Database files (.db) are gitignored
- Logs directory is gitignored
- Data directories (historical, live, hmm-models) are gitignored
- The `venv/` directory is NOT in the repo - only on VPS
