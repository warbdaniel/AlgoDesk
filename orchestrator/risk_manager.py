"""
Position Sizer & Risk Manager
==============================
Kelly criterion sizing, drawdown protection, correlation awareness.
40% max drawdown is a HARD KILL SWITCH.
"""

import logging
import math
import time
import json
import requests
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

import sys
sys.path.insert(0, str(Path(__file__).parent))
from orch_config import (
    FIX_URL, MAX_ACCOUNT_DRAWDOWN, DAILY_LOSS_LIMIT, PER_TRADE_RISK,
    MAX_CORRELATED_EXPOSURE, MAX_OPEN_POSITIONS, MIN_LOT_SIZE, MAX_LOT_SIZE,
    CORRELATION_GROUPS, PIP_SIZES, DEFAULT_PIP_SIZE,
)
from db import db

logger = logging.getLogger("orchestrator.risk_manager")

REQUEST_TIMEOUT = 10


class RiskManager:
    def __init__(self):
        self.peak_equity = 0
        self.daily_start_equity = 0
        self.daily_pnl = 0
        self.kill_switch_active = False
        self.last_equity_check = 0
        self._account_cache = {}
        self._cache_ts = 0

    def initialize(self):
        """Initialize from account state."""
        account = self._get_account()
        if account:
            equity = account.get("equity", 0)
            balance = account.get("balance", 0)
            self.peak_equity = max(equity, balance, self.peak_equity)
            self.daily_start_equity = equity

            # Restore peak from DB
            stored_peak = db.get_state("peak_equity", 0)
            if stored_peak and stored_peak > self.peak_equity:
                self.peak_equity = stored_peak

            logger.info(f"Risk manager initialized: equity={equity}, peak={self.peak_equity}")

    def check_risk(self) -> dict:
        """Full risk check. Returns status dict."""
        account = self._get_account()
        if not account:
            return {"status": "error", "reason": "cannot_reach_fix_api"}

        equity = account.get("equity", 0)
        balance = account.get("balance", 0)
        margin = account.get("margin_used", account.get("margin", 0))

        if equity <= 0:
            return {"status": "error", "reason": "zero_equity"}

        # Update peak
        if equity > self.peak_equity:
            self.peak_equity = equity
            db.set_state("peak_equity", self.peak_equity)

        # Drawdown calculation
        drawdown_pct = (self.peak_equity - equity) / self.peak_equity if self.peak_equity > 0 else 0

        # Daily P&L
        if self.daily_start_equity > 0:
            self.daily_pnl = (equity - self.daily_start_equity) / self.daily_start_equity
        else:
            self.daily_pnl = 0

        # Log equity
        open_positions = len(db.get_open_trades())
        db.log_equity(balance, equity, margin, open_positions, self.daily_pnl,
                       drawdown_pct, self.peak_equity)

        # ── HARD KILL SWITCH: 40% drawdown ──
        if drawdown_pct >= MAX_ACCOUNT_DRAWDOWN:
            if not self.kill_switch_active:
                logger.critical(
                    f"KILL SWITCH ACTIVATED: Drawdown {drawdown_pct:.1%} >= {MAX_ACCOUNT_DRAWDOWN:.0%} "
                    f"(equity={equity}, peak={self.peak_equity})"
                )
                self._activate_kill_switch()
            return {
                "status": "kill_switch",
                "drawdown_pct": drawdown_pct,
                "equity": equity,
                "peak": self.peak_equity,
                "daily_pnl": self.daily_pnl,
            }

        # Daily loss limit
        daily_exceeded = False
        if self.daily_pnl <= -DAILY_LOSS_LIMIT:
            daily_exceeded = True
            logger.warning(f"Daily loss limit hit: {self.daily_pnl:.2%}")

        return {
            "status": "ok" if not daily_exceeded else "daily_limit",
            "can_trade": not daily_exceeded and not self.kill_switch_active,
            "equity": equity,
            "balance": balance,
            "peak_equity": self.peak_equity,
            "drawdown_pct": round(drawdown_pct, 4),
            "daily_pnl_pct": round(self.daily_pnl, 4),
            "daily_limit_remaining": round(DAILY_LOSS_LIMIT + self.daily_pnl, 4),
            "kill_switch_active": self.kill_switch_active,
            "open_positions": open_positions,
        }

    def calculate_position_size(self, symbol: str, entry_price: float,
                                  sl_price: float, confidence: float = 0.5) -> float:
        """
        Kelly criterion-based position sizing.
        Returns lot size clamped to limits.
        """
        if self.kill_switch_active:
            return 0

        account = self._get_account()
        if not account:
            return 0

        equity = account.get("equity", 0)
        if equity <= 0:
            return 0

        pip_size = PIP_SIZES.get(symbol, DEFAULT_PIP_SIZE)
        sl_pips = abs(entry_price - sl_price) / pip_size
        if sl_pips < 1:
            sl_pips = 1

        # Kelly fraction: f = (bp - q) / b
        # where b = reward/risk, p = win prob (confidence), q = 1 - p
        # Use half-Kelly for safety
        p = min(max(confidence, 0.3), 0.8)
        b = 1.5  # Assume 1.5:1 R:R average
        q = 1 - p
        kelly = (b * p - q) / b
        half_kelly = max(kelly / 2, 0.01)

        # Risk amount: equity * per_trade_risk * kelly_adjustment
        risk_amount = equity * PER_TRADE_RISK * min(half_kelly / 0.1, 1.0)

        # Pip value (approximate - varies by pair)
        # For most pairs: 1 standard lot = $10/pip for 4-decimal, $1000/pip for 2-decimal
        if pip_size >= 0.01:  # JPY pairs
            pip_value_per_lot = 1000 * pip_size
        elif pip_size >= 0.001:  # XNGUSD
            pip_value_per_lot = 100 * pip_size
        elif pip_size >= 0.1:  # Gold/Platinum
            pip_value_per_lot = 100 * pip_size
        elif pip_size >= 1.0:  # BTC
            pip_value_per_lot = pip_size
        else:
            pip_value_per_lot = 10  # Standard forex

        # Lot size = risk_amount / (sl_pips * pip_value_per_lot)
        pip_val_safe = max(pip_value_per_lot, 0.01)
        lots = risk_amount / (sl_pips * pip_val_safe)

        # Correlation adjustment
        corr_factor = self._correlation_factor(symbol)
        lots *= corr_factor

        # Clamp
        lots = max(MIN_LOT_SIZE, min(lots, MAX_LOT_SIZE))
        lots = round(lots, 2)

        return lots

    def _correlation_factor(self, symbol: str) -> float:
        """Reduce size if correlated positions are open."""
        open_trades = db.get_open_trades()
        open_symbols = [t["symbol"] for t in open_trades]

        if not open_symbols:
            return 1.0

        # Check how many correlated positions
        correlated_count = 0
        for group_name, group_symbols in CORRELATION_GROUPS.items():
            if symbol in group_symbols:
                correlated_count += sum(1 for s in open_symbols if s in group_symbols)

        if correlated_count == 0:
            return 1.0
        elif correlated_count == 1:
            return 0.7
        elif correlated_count == 2:
            return 0.4
        else:
            return 0.2

    def can_open_trade(self, symbol: str) -> tuple:
        """Check if we can open a new trade. Returns (bool, reason)."""
        if self.kill_switch_active:
            return False, "kill_switch_active"

        risk = self.check_risk()
        if not risk.get("can_trade", False):
            return False, risk.get("status", "unknown")

        open_trades = db.get_open_trades()
        if len(open_trades) >= MAX_OPEN_POSITIONS:
            return False, f"max_positions ({MAX_OPEN_POSITIONS})"

        # Check for existing trade on same symbol
        for t in open_trades:
            if t["symbol"] == symbol:
                return False, "already_in_trade"

        return True, "ok"

    def _activate_kill_switch(self):
        """Emergency halt: close all positions, activate FIX kill switch."""
        self.kill_switch_active = True
        db.set_state("kill_switch_active", True)
        db.set_state("kill_switch_time", datetime.now(timezone.utc).isoformat())

        try:
            requests.post(f"{FIX_URL}/kill-switch/activate", timeout=REQUEST_TIMEOUT)
            logger.critical("FIX API kill switch activated")
        except Exception as e:
            logger.error(f"Failed to activate FIX kill switch: {e}")

    def deactivate_kill_switch(self):
        """Manual deactivation only."""
        self.kill_switch_active = False
        db.set_state("kill_switch_active", False)
        try:
            requests.post(f"{FIX_URL}/kill-switch/deactivate", timeout=REQUEST_TIMEOUT)
        except Exception:
            pass
        logger.info("Kill switch deactivated manually")

    def reset_daily(self):
        """Reset daily counters (call at market open)."""
        account = self._get_account()
        if account:
            self.daily_start_equity = account.get("equity", 0)
        self.daily_pnl = 0
        logger.info(f"Daily risk reset: start_equity={self.daily_start_equity}")

    def _get_account(self) -> dict:
        now = time.time()
        if now - self._cache_ts < 5 and self._account_cache:
            return self._account_cache
        try:
            resp = requests.get(f"{FIX_URL}/account", timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                self._account_cache = resp.json()
                self._cache_ts = now
                return self._account_cache
        except Exception:
            pass
        return self._account_cache or {}

    def get_status(self) -> dict:
        risk = self.check_risk()
        return {
            **risk,
            "max_account_drawdown": MAX_ACCOUNT_DRAWDOWN,
            "daily_loss_limit": DAILY_LOSS_LIMIT,
            "per_trade_risk": PER_TRADE_RISK,
            "max_open_positions": MAX_OPEN_POSITIONS,
        }


risk_manager = RiskManager()
