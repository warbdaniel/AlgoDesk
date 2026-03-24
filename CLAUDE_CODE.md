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
scripts/fix-api/           → cTrader FIX 4.4 Connector (FastAPI on port 5200)
scripts/data-pipeline/     → Data Infrastructure (FastAPI on port 5300)
dashboard/                 → Analytics Dashboard (FastAPI on port 5100)
guardian/                  → Trading Desk Guardian (systemd daemon)
ctrader-mcp-server/       → cTrader MCP Server (14 tools)
scripts/alpha-engine/     → Project Alpha ML Scalping Engine (data foundation)
scripts/utils/            → Utility scripts (deploy.sh, deploy_webhook.py)
```

## Services on VPS
- **regime-detector** (PM2) - FastAPI port 5000
- **dashboard** (PM2) - FastAPI port 5100
- **fix-api** (PM2) - FastAPI port 5200
- **data-pipeline** (PM2) - FastAPI port 5300
- **blackjack-advisor** (PM2) - separate project
- **trading-guardian** (systemd) - monitoring daemon
- **deploy-webhook** (PM2) - GitHub webhook CI/CD port 9000
- **n8n** (Docker) - workflow automation port 5678

## Key Files
- `scripts/regime-detector/regime_detector.py` - Market regime classification
- `scripts/regime-detector/config.yaml` - Regime detector config
- `scripts/fix-api/fix_connector.py` - Core FIX 4.4 connector (SSL, heartbeat, reconnect)
- `scripts/fix-api/fix_price_client.py` - FIX price feed client
- `scripts/fix-api/fix_trade_client.py` - FIX trade client with risk limits
- `scripts/fix-api/fix_api_server.py` - FIX REST API server
- `scripts/data-pipeline/data_store.py` - SQLite market data store (ticks + candles)
- `scripts/data-pipeline/historical_data.py` - Historical data manager (backfill, CSV, gaps)
- `scripts/data-pipeline/feature_engine.py` - Feature engineering (indicators + ML vectors)
- `scripts/data-pipeline/data_bus.py` - Event bus (pub/sub + webhooks)
- `scripts/data-pipeline/data_api_server.py` - Data pipeline REST API
- `dashboard/dashboard.py` - Post-trade analytics backend
- `dashboard/templates/dashboard.html` - Dashboard frontend
- `guardian/guardian.py` - Server monitoring daemon
- `guardian/config.yaml` - Guardian config
- `guardian/guardian-ctl` - Guardian CLI tool
- `scripts/alpha-engine/config.py` - Alpha engine configuration (AlphaConfig)
- `scripts/alpha-engine/tick_buffer.py` - High-performance tick ring buffer (TickBufferManager)
- `scripts/alpha-engine/scalping_features.py` - Microstructure feature engine (ScalpingFeatureEngine)
- `scripts/alpha-engine/label_engine.py` - ML label generation: triple barrier, forward returns (LabelEngine)
- `scripts/alpha-engine/dataset_builder.py` - ML dataset assembly with temporal splits (DatasetBuilder)
- `scripts/alpha-engine/alpha_store.py` - SQLite persistence for features, labels, datasets (AlphaStore)

## VPS Environment
- Python 3.12.3 (venv at /opt/trading-desk/venv/)
- Key packages: fastapi, uvicorn, numpy, pandas, scikit-learn, hmmlearn, ctrader-sdk, simplefix, requests
- PM2 for process management
- Docker for n8n

## Data Pipeline (Phase 2)
The data pipeline at `scripts/data-pipeline/` provides:
- **MarketDataStore**: SQLite WAL-mode database for tick + multi-timeframe OHLCV candle persistence
- **HistoricalDataManager**: cTrader OHLC backfill, gap detection, CSV import/export
- **FeatureEngine**: 25+ technical indicators computed from candle data (SMA, EMA, MACD, RSI, ADX, ATR, Bollinger, Stochastic, price action) with ML-ready vector output
- **DataBus**: Pub/sub event bus with in-process callbacks and HTTP webhook delivery for cross-service events (tick, candle_close, regime_change, signal, order_fill, risk_alert)
- **REST API**: Port 5300 - ticks, candles, features, historical management, event bus, webhooks

## Alpha Engine (Project Alpha - Phase 3)
The alpha engine at `scripts/alpha-engine/` provides the data foundation for ML-based scalping:
- **TickBufferManager**: Per-symbol ring buffers (50k ticks default) with O(1) append, rolling spread EMA, tick-rate stats
- **ScalpingFeatureEngine**: 35+ microstructure features — price velocity/acceleration, spread z-score, order-flow imbalance, microprice, realized volatility, tick intensity, level proximity
- **LabelEngine**: Forward-looking label generation — multi-horizon returns (5s–120s), triple-barrier (TP/SL/time-expiry), ternary direction classification
- **DatasetBuilder**: ML dataset assembly — temporal train/val/test splits with purge gaps, outlier clipping, z-score/minmax normalisation (fitted on train only)
- **AlphaStore**: SQLite WAL-mode persistence for computed features, labels, dataset metadata, and normalisation statistics
- **Integration**: Consumes ticks from FIX API (port 5200) via DataBus, stores to data-pipeline (port 5300), features computed from TickRing buffers

## Important Notes
- `.env` files are in `.gitignore` - never commit secrets
- Database files (.db) are gitignored
- Logs directory is gitignored
- Data directories (historical, live, hmm-models) are gitignored
- The `venv/` directory is NOT in the repo - only on VPS
