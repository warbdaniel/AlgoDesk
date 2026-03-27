"""
Orchestrator Configuration
==========================
Central config pulling from CLS config + orchestrator-specific settings.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path("/opt/trading-desk")
sys.path.insert(0, str(PROJECT_ROOT / "cls"))

from config import (
    SYMBOL_MAP, SYMBOL_NAME_TO_ID, ALL_SYMBOLS, PIP_SIZES, DEFAULT_PIP_SIZE,
    SYMBOL_SPREAD_PIPS, DEFAULT_SPREAD_PIPS, PROMOTION_CRITERIA, REGIMES,
    OPTUNA_TRIALS, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, PURGE_GAP_BARS,
    FORWARD_HORIZON_BARS, TP_PIPS, SL_PIPS, SLIPPAGE_PIPS,
    REGIME_STRONG_TREND_ADX, REGIME_MILD_TREND_ADX, REGIME_RANGING_ADX,
    CLS_DB, MODELS_DIR, FIX_API_URL, DATA_PIPELINE_URL,
)

# ── Orchestrator paths ──────────────────────────────────────────────────────
ORCH_DIR = PROJECT_ROOT / "orchestrator"
ORCH_DB = PROJECT_ROOT / "data" / "orchestrator.db"
RESULTS_DIR = ORCH_DIR / "results"
LOGS_DIR = PROJECT_ROOT / "logs" / "orchestrator"
PARQUET_DIR = PROJECT_ROOT / "data" / "historical" / "candles_5m"

# ── Service URLs ─────────────────────────────────────────────────────────────
FIX_URL = FIX_API_URL          # http://localhost:5200
DATA_URL = DATA_PIPELINE_URL   # http://localhost:5300
CLS_URL = "http://localhost:5500"
REGIME_URL = "http://localhost:5000"
DASHBOARD_URL = "http://localhost:5100"
ORCH_API_PORT = 5600
ORCH_API_HOST = "0.0.0.0"

# ── Risk limits ──────────────────────────────────────────────────────────────
MAX_ACCOUNT_DRAWDOWN = 0.40       # 40% HARD KILL SWITCH
DAILY_LOSS_LIMIT = 0.05           # 5% daily
PER_TRADE_RISK = 0.02             # 2% per trade
MAX_CORRELATED_EXPOSURE = 0.06    # 6% total for correlated group
MAX_OPEN_POSITIONS = 10
MIN_LOT_SIZE = 0.01
MAX_LOT_SIZE = 1.0

# ── Correlation groups (reduce sizing when multiple from same group open) ────
CORRELATION_GROUPS = {
    "USD_MAJORS": ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF"],
    "JPY_CROSSES": ["USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "CADJPY", "CHFJPY", "NZDJPY"],
    "GBP_CROSSES": ["GBPUSD", "EURGBP", "GBPJPY", "GBPCHF", "GBPCAD", "GBPNZD", "GBPAUD"],
    "AUD_NZD": ["AUDUSD", "NZDUSD", "AUDNZD", "AUDJPY", "AUDCAD", "AUDCHF", "NZDCAD", "NZDCHF"],
    "METALS": ["XAUUSD", "XAGUSD", "XAUEUR", "XPTUSD", "XPDUSD"],
    "ENERGY": ["XTIUSD", "XBRUSD", "XNGUSD"],
}

# ── Schedule intervals (seconds) ────────────────────────────────────────────
HEARTBEAT_INTERVAL = 30
DATA_BACKFILL_INTERVAL = 300      # 5 min
SIGNAL_SCAN_INTERVAL = 60         # 1 min
POSITION_MANAGE_INTERVAL = 30     # 30 sec
RISK_CHECK_INTERVAL = 15          # 15 sec
PERFORMANCE_LOG_INTERVAL = 3600   # 1 hour
BACKTEST_INTERVAL = 86400         # Daily
WEEKLY_REOPTIMIZE_INTERVAL = 604800  # Weekly

# ── Backtest parameters ─────────────────────────────────────────────────────
WALK_FORWARD_WINDOWS = 5
MIN_BARS_FOR_BACKTEST = 2000

# ── Drift detection ─────────────────────────────────────────────────────────
DRIFT_SHARPE_THRESHOLD = 0.50     # Flag if live Sharpe < 50% of backtest
STRATEGY_RETIREMENT_MONTHS = 3    # Retire after 3 consecutive negative months

# ── PM2 service names ───────────────────────────────────────────────────────
PM2_SERVICES = [
    "fix-api", "data-pipeline", "regime-detector",
    "dashboard", "cls", "guardian", "adam-trader",
]

# Ensure dirs exist
for d in [RESULTS_DIR, LOGS_DIR, PARQUET_DIR, ORCH_DB.parent]:
    d.mkdir(parents=True, exist_ok=True)
