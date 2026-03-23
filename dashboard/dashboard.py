#!/usr/bin/env python3
"""
Trading Desk Analytics Dashboard v1.0.0
Post-trade analytics with Expected Value, Sharpe Ratio, Profit Factor.
Serves on port 5100. Receives trade logs via POST /api/trades.
"""

import os
import json
import sqlite3
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path
from contextlib import contextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from typing import Optional, List
import numpy as np

# =============================================================
#  Configuration
# =============================================================
DASHBOARD_DIR = Path(__file__).parent
DB_PATH = DASHBOARD_DIR / "trades.db"
TEMPLATES_DIR = DASHBOARD_DIR / "templates"
STATIC_DIR = DASHBOARD_DIR / "static"

# =============================================================
#  Database Setup
# =============================================================
def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn

def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS trades (
            id TEXT PRIMARY KEY,
            symbol TEXT NOT NULL,
            direction TEXT NOT NULL CHECK(direction IN ('BUY', 'SELL')),
            entry_price REAL NOT NULL,
            exit_price REAL NOT NULL,
            volume REAL NOT NULL DEFAULT 0.01,
            pnl REAL NOT NULL,
            pnl_pips REAL DEFAULT 0,
            risk_percent REAL DEFAULT 1.0,
            regime TEXT DEFAULT 'UNKNOWN',
            confidence REAL DEFAULT 0,
            entry_time TEXT NOT NULL,
            exit_time TEXT NOT NULL,
            duration_seconds INTEGER DEFAULT 0,
            signal_source TEXT DEFAULT 'manual',
            orderflow_context TEXT DEFAULT '{}',
            reasoning TEXT DEFAULT '',
            tags TEXT DEFAULT '[]',
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
        CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time);
        CREATE INDEX IF NOT EXISTS idx_trades_regime ON trades(regime);
        CREATE INDEX IF NOT EXISTS idx_trades_direction ON trades(direction);
    """)
    conn.commit()
    conn.close()

# =============================================================
#  Pydantic Models
# =============================================================
class TradeCreate(BaseModel):
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    volume: float = 0.01
    pnl: float
    pnl_pips: float = 0
    risk_percent: float = 1.0
    regime: str = "UNKNOWN"
    confidence: float = 0
    entry_time: str
    exit_time: str
    duration_seconds: int = 0
    signal_source: str = "manual"
    orderflow_context: Optional[dict] = {}
    reasoning: str = ""
    tags: Optional[List[str]] = []

class TradeBatch(BaseModel):
    trades: List[TradeCreate]

# =============================================================
#  Analytics Engine
# =============================================================
class AnalyticsEngine:
    """Calculates all trading performance metrics."""

    @staticmethod
    def calculate(trades: list) -> dict:
        if not trades:
            return AnalyticsEngine._empty_metrics()

        pnls = [t["pnl"] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        breakevens = [p for p in pnls if p == 0]

        total_trades = len(pnls)
        win_count = len(wins)
        loss_count = len(losses)

        # Win Rate
        win_rate = (win_count / total_trades) * 100 if total_trades > 0 else 0

        # Average Win / Average Loss
        avg_win = sum(wins) / win_count if win_count > 0 else 0
        avg_loss = sum(losses) / loss_count if loss_count > 0 else 0

        # Expected Value (EV) per trade
        # EV = (Win% * Avg Win) + (Loss% * Avg Loss)
        # Note: avg_loss is already negative
        win_prob = win_count / total_trades if total_trades > 0 else 0
        loss_prob = loss_count / total_trades if total_trades > 0 else 0
        expected_value = (win_prob * avg_win) + (loss_prob * avg_loss)

        # Profit Factor = Gross Profit / |Gross Loss|
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0

        # Sharpe Ratio (annualized, assuming ~252 trading days)
        if len(pnls) >= 2:
            mean_return = np.mean(pnls)
            std_return = np.std(pnls, ddof=1)
            sharpe_ratio = (mean_return / std_return) * math.sqrt(252) if std_return > 0 else 0
        else:
            sharpe_ratio = 0

        # Profit Ratio (Reward:Risk) = Avg Win / |Avg Loss|
        profit_ratio = avg_win / abs(avg_loss) if avg_loss != 0 else float('inf') if avg_win > 0 else 0

        # Max Drawdown
        cumulative = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        max_drawdown = float(np.max(drawdown)) if len(drawdown) > 0 else 0

        # Equity curve
        equity_curve = [0] + list(np.cumsum(pnls).tolist())

        # Consecutive wins/losses
        max_consec_wins = 0
        max_consec_losses = 0
        current_wins = 0
        current_losses = 0
        for p in pnls:
            if p > 0:
                current_wins += 1
                current_losses = 0
                max_consec_wins = max(max_consec_wins, current_wins)
            elif p < 0:
                current_losses += 1
                current_wins = 0
                max_consec_losses = max(max_consec_losses, current_losses)
            else:
                current_wins = 0
                current_losses = 0

        # Net P&L
        net_pnl = sum(pnls)

        # Best / Worst trade
        best_trade = max(pnls)
        worst_trade = min(pnls)

        # Rolling EV (last 20 trades)
        rolling_window = 20
        rolling_evs = []
        for i in range(len(pnls)):
            window = pnls[max(0, i - rolling_window + 1):i + 1]
            w = [p for p in window if p > 0]
            l = [p for p in window if p < 0]
            w_count = len(w)
            l_count = len(l)
            t_count = len(window)
            w_prob = w_count / t_count if t_count > 0 else 0
            l_prob = l_count / t_count if t_count > 0 else 0
            a_win = sum(w) / w_count if w_count > 0 else 0
            a_loss = sum(l) / l_count if l_count > 0 else 0
            rolling_evs.append((w_prob * a_win) + (l_prob * a_loss))

        return {
            "total_trades": total_trades,
            "win_count": win_count,
            "loss_count": loss_count,
            "breakeven_count": len(breakevens),
            "win_rate": round(win_rate, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "expected_value": round(expected_value, 2),
            "profit_factor": round(profit_factor, 3) if profit_factor != float('inf') else "∞",
            "sharpe_ratio": round(sharpe_ratio, 3),
            "profit_ratio": round(profit_ratio, 3) if profit_ratio != float('inf') else "∞",
            "max_drawdown": round(max_drawdown, 2),
            "net_pnl": round(net_pnl, 2),
            "gross_profit": round(gross_profit, 2),
            "gross_loss": round(gross_loss, 2),
            "best_trade": round(best_trade, 2),
            "worst_trade": round(worst_trade, 2),
            "max_consec_wins": max_consec_wins,
            "max_consec_losses": max_consec_losses,
            "equity_curve": [round(e, 2) for e in equity_curve],
            "rolling_ev": [round(e, 2) for e in rolling_evs],
        }

    @staticmethod
    def by_regime(trades: list) -> dict:
        regimes = {}
        for t in trades:
            r = t.get("regime", "UNKNOWN")
            if r not in regimes:
                regimes[r] = []
            regimes[r].append(t)
        return {r: AnalyticsEngine.calculate(tlist) for r, tlist in regimes.items()}

    @staticmethod
    def by_symbol(trades: list) -> dict:
        symbols = {}
        for t in trades:
            s = t["symbol"]
            if s not in symbols:
                symbols[s] = []
            symbols[s].append(t)
        return {s: AnalyticsEngine.calculate(tlist) for s, tlist in symbols.items()}

    @staticmethod
    def _empty_metrics():
        return {
            "total_trades": 0, "win_count": 0, "loss_count": 0,
            "breakeven_count": 0, "win_rate": 0, "avg_win": 0,
            "avg_loss": 0, "expected_value": 0, "profit_factor": 0,
            "sharpe_ratio": 0, "profit_ratio": 0, "max_drawdown": 0,
            "net_pnl": 0, "gross_profit": 0, "gross_loss": 0,
            "best_trade": 0, "worst_trade": 0,
            "max_consec_wins": 0, "max_consec_losses": 0,
            "equity_curve": [0], "rolling_ev": [],
        }

# =============================================================
#  FastAPI App
# =============================================================
app = FastAPI(
    title="Trading Desk Analytics Dashboard",
    version="1.0.0",
    description="Post-trade analytics: EV, Win Rate, Sharpe Ratio, Profit Factor"
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

@app.on_event("startup")
async def startup():
    init_db()

# --- Health ---
@app.get("/health")
async def health():
    conn = get_db()
    count = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
    conn.close()
    return {
        "status": "healthy",
        "service": "trading-desk-dashboard",
        "version": "1.0.0",
        "trade_count": count,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

# --- Dashboard HTML ---
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

# --- Log a single trade ---
@app.post("/api/trades")
async def create_trade(trade: TradeCreate):
    trade_id = str(uuid.uuid4())[:12]
    conn = get_db()
    try:
        conn.execute("""
            INSERT INTO trades (id, symbol, direction, entry_price, exit_price,
                volume, pnl, pnl_pips, risk_percent, regime, confidence,
                entry_time, exit_time, duration_seconds, signal_source,
                orderflow_context, reasoning, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade_id, trade.symbol, trade.direction.upper(),
            trade.entry_price, trade.exit_price, trade.volume,
            trade.pnl, trade.pnl_pips, trade.risk_percent,
            trade.regime, trade.confidence,
            trade.entry_time, trade.exit_time, trade.duration_seconds,
            trade.signal_source,
            json.dumps(trade.orderflow_context or {}),
            trade.reasoning,
            json.dumps(trade.tags or [])
        ))
        conn.commit()
        return {"status": "ok", "trade_id": trade_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        conn.close()

# --- Log batch of trades ---
@app.post("/api/trades/batch")
async def create_trades_batch(batch: TradeBatch):
    conn = get_db()
    ids = []
    try:
        for trade in batch.trades:
            trade_id = str(uuid.uuid4())[:12]
            conn.execute("""
                INSERT INTO trades (id, symbol, direction, entry_price, exit_price,
                    volume, pnl, pnl_pips, risk_percent, regime, confidence,
                    entry_time, exit_time, duration_seconds, signal_source,
                    orderflow_context, reasoning, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade_id, trade.symbol, trade.direction.upper(),
                trade.entry_price, trade.exit_price, trade.volume,
                trade.pnl, trade.pnl_pips, trade.risk_percent,
                trade.regime, trade.confidence,
                trade.entry_time, trade.exit_time, trade.duration_seconds,
                trade.signal_source,
                json.dumps(trade.orderflow_context or {}),
                trade.reasoning,
                json.dumps(trade.tags or [])
            ))
            ids.append(trade_id)
        conn.commit()
        return {"status": "ok", "count": len(ids), "trade_ids": ids}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        conn.close()

# --- Get all trades ---
@app.get("/api/trades")
async def get_trades(
    symbol: Optional[str] = None,
    regime: Optional[str] = None,
    limit: int = 500,
    offset: int = 0
):
    conn = get_db()
    query = "SELECT * FROM trades WHERE 1=1"
    params = []
    if symbol:
        query += " AND symbol = ?"
        params.append(symbol)
    if regime:
        query += " AND regime = ?"
        params.append(regime)
    query += " ORDER BY entry_time DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]

# --- Get analytics ---
@app.get("/api/analytics")
async def get_analytics(
    symbol: Optional[str] = None,
    regime: Optional[str] = None
):
    conn = get_db()
    query = "SELECT * FROM trades WHERE 1=1"
    params = []
    if symbol:
        query += " AND symbol = ?"
        params.append(symbol)
    if regime:
        query += " AND regime = ?"
        params.append(regime)
    query += " ORDER BY entry_time ASC"
    rows = conn.execute(query, params).fetchall()
    conn.close()
    trades = [dict(r) for r in rows]
    overall = AnalyticsEngine.calculate(trades)
    by_regime = AnalyticsEngine.by_regime(trades)
    by_symbol = AnalyticsEngine.by_symbol(trades)
    return {
        "overall": overall,
        "by_regime": by_regime,
        "by_symbol": by_symbol,
        "generated_at": datetime.now(timezone.utc).isoformat()
    }

# --- Get distinct symbols and regimes for filters ---
@app.get("/api/filters")
async def get_filters():
    conn = get_db()
    symbols = [r[0] for r in conn.execute("SELECT DISTINCT symbol FROM trades ORDER BY symbol").fetchall()]
    regimes = [r[0] for r in conn.execute("SELECT DISTINCT regime FROM trades ORDER BY regime").fetchall()]
    conn.close()
    return {"symbols": symbols, "regimes": regimes}

# --- Delete a trade ---
@app.delete("/api/trades/{trade_id}")
async def delete_trade(trade_id: str):
    conn = get_db()
    result = conn.execute("DELETE FROM trades WHERE id = ?", (trade_id,))
    conn.commit()
    if result.rowcount == 0:
        conn.close()
        raise HTTPException(status_code=404, detail="Trade not found")
    conn.close()
    return {"status": "deleted", "trade_id": trade_id}

# =============================================================
#  Run
# =============================================================
if __name__ == "__main__":
    import uvicorn
    init_db()
    uvicorn.run(app, host="0.0.0.0", port=5100, log_level="info")
