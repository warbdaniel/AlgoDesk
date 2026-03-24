"""
Backtest Evaluator
==================

Simulates trades from model predictions on the test set, computes
performance metrics, generates equity curve plots, and supports
walk-forward evaluation with periodic retraining.

Spread assumptions (per round-trip, in pips):
  - Majors (EURUSD, GBPUSD, USDJPY, USDCHF, AUDUSD, USDCAD, NZDUSD): 1.0
  - Crosses (EURGBP, EURJPY, GBPJPY, CHFJPY, CADJPY, NZDJPY): 2.0
  - Gold (XAUUSD): 30.0
  - BTC (BTCUSD): 50.0
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np

logger = logging.getLogger("evaluator")

# Spread in pips (round-trip)
_MAJORS = {"EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"}
_CROSSES = {"EURGBP", "EURJPY", "GBPJPY", "CHFJPY", "CADJPY", "NZDJPY"}

SPREAD_PIPS = {}
for _s in _MAJORS:
    SPREAD_PIPS[_s] = 1.0
for _s in _CROSSES:
    SPREAD_PIPS[_s] = 2.0
SPREAD_PIPS["XAUUSD"] = 30.0
SPREAD_PIPS["BTCUSD"] = 50.0

# pip sizes (matching candle_config.py)
PIP_SIZES = {
    "EURUSD": 0.0001, "GBPUSD": 0.0001, "USDJPY": 0.01,
    "USDCHF": 0.0001, "AUDUSD": 0.0001, "USDCAD": 0.0001,
    "NZDUSD": 0.0001, "EURGBP": 0.0001, "EURJPY": 0.01,
    "GBPJPY": 0.01, "CHFJPY": 0.01, "CADJPY": 0.01,
    "NZDJPY": 0.01, "XAUUSD": 0.1, "BTCUSD": 1.0,
}


# ---------------------------------------------------------------------------
# Trade result
# ---------------------------------------------------------------------------
@dataclass
class TradeResult:
    ts: float
    direction: int  # 1=long
    entry_price: float
    exit_price: float
    pnl_pips: float
    spread_pips: float
    net_pnl_pips: float
    barrier_candles: int


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
@dataclass
class BacktestMetrics:
    symbol: str = ""
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown_pips: float = 0.0
    expected_value_pips: float = 0.0
    total_pnl_pips: float = 0.0
    avg_win_pips: float = 0.0
    avg_loss_pips: float = 0.0

    def to_dict(self) -> dict:
        return {k: round(v, 4) if isinstance(v, float) else v
                for k, v in self.__dict__.items()}


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------
class BacktestEvaluator:
    """Simulate trades and compute performance metrics."""

    def __init__(self, prob_threshold: float = 0.5):
        self.prob_threshold = prob_threshold

    def evaluate(
        self,
        y_proba: np.ndarray,
        labels: list[dict],
        symbol: str,
    ) -> tuple[BacktestMetrics, list[float]]:
        """Evaluate model predictions against actual barrier outcomes.

        Args:
            y_proba: P(win) predictions from model
            labels: label dicts with barrier_label, barrier_return, barrier_candles
            symbol: trading symbol

        Returns:
            (metrics, equity_curve) where equity_curve is cumulative PnL in pips.
        """
        pip_size = PIP_SIZES.get(symbol, 0.0001)
        spread = SPREAD_PIPS.get(symbol, 1.5) * pip_size

        trades: list[TradeResult] = []
        equity = [0.0]

        for i, prob in enumerate(y_proba):
            if prob < self.prob_threshold:
                continue

            lbl = labels[i]
            barrier_ret = lbl.get("barrier_return", 0.0)
            barrier_candles = int(lbl.get("barrier_candles", 0))

            # PnL in pips (barrier_return is in price units)
            gross_pips = barrier_ret / pip_size
            spread_pips = SPREAD_PIPS.get(symbol, 1.5)
            net_pips = gross_pips - spread_pips

            trade = TradeResult(
                ts=lbl.get("ts", 0.0) if isinstance(lbl, dict) else 0.0,
                direction=1,
                entry_price=0.0,
                exit_price=0.0,
                pnl_pips=gross_pips,
                spread_pips=spread_pips,
                net_pnl_pips=net_pips,
                barrier_candles=barrier_candles,
            )
            trades.append(trade)
            equity.append(equity[-1] + net_pips)

        metrics = self._compute_metrics(trades, symbol)
        return metrics, equity

    def _compute_metrics(
        self, trades: list[TradeResult], symbol: str,
    ) -> BacktestMetrics:
        m = BacktestMetrics(symbol=symbol)
        if not trades:
            return m

        pnls = [t.net_pnl_pips for t in trades]
        m.total_trades = len(trades)
        m.wins = sum(1 for p in pnls if p > 0)
        m.losses = sum(1 for p in pnls if p <= 0)
        m.win_rate = m.wins / m.total_trades if m.total_trades > 0 else 0.0
        m.total_pnl_pips = sum(pnls)

        gross_profit = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p < 0))
        m.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        win_pnls = [p for p in pnls if p > 0]
        loss_pnls = [p for p in pnls if p <= 0]
        m.avg_win_pips = np.mean(win_pnls).item() if win_pnls else 0.0
        m.avg_loss_pips = np.mean(loss_pnls).item() if loss_pnls else 0.0

        m.expected_value_pips = m.total_pnl_pips / m.total_trades

        # Sharpe (annualised, assuming ~252 trading days, ~57 5M bars/day)
        if len(pnls) > 1:
            pnl_arr = np.array(pnls)
            mean_pnl = pnl_arr.mean()
            std_pnl = pnl_arr.std()
            if std_pnl > 0:
                m.sharpe_ratio = (mean_pnl / std_pnl) * math.sqrt(252 * 57)

        # Max drawdown
        cumulative = np.cumsum(pnls)
        peak = np.maximum.accumulate(cumulative)
        drawdown = peak - cumulative
        m.max_drawdown_pips = float(drawdown.max()) if len(drawdown) > 0 else 0.0

        return m


# ---------------------------------------------------------------------------
# Walk-forward evaluator
# ---------------------------------------------------------------------------
class WalkForwardEvaluator:
    """Walk-forward backtest: retrain every N months, test on next M months."""

    def __init__(
        self,
        retrain_months: int = 2,
        test_months: int = 1,
        prob_threshold: float = 0.5,
    ):
        self.retrain_months = retrain_months
        self.test_months = test_months
        self.evaluator = BacktestEvaluator(prob_threshold)

    def run(
        self,
        db_path: str,
        symbol: str,
        dataset_id: int | None = None,
    ) -> dict:
        """Run walk-forward evaluation on all available data.

        Loads all samples (train+val+test), splits into walk-forward
        windows, retrains every retrain_months, tests on the next
        test_months.
        """
        from lib.trainer import load_split_from_db

        # Load all splits and combine chronologically
        X_train, y_train, feat_names, ts_train = load_split_from_db(
            db_path, symbol, "train", dataset_id,
        )
        X_val, y_val, _, ts_val = load_split_from_db(
            db_path, symbol, "val", dataset_id,
        )
        X_test, y_test, _, ts_test = load_split_from_db(
            db_path, symbol, "test", dataset_id,
        )

        if len(X_train) == 0:
            return {"symbol": symbol, "status": "no_data"}

        X_all = np.vstack([X_train, X_val, X_test])
        y_all = np.concatenate([y_train, y_val, y_test])
        ts_all = np.concatenate([ts_train, ts_val, ts_test])

        # Replace NaN/Inf
        np.nan_to_num(X_all, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        # Load labels for barrier return info
        conn = sqlite3.connect(db_path, timeout=30)
        conn.row_factory = sqlite3.Row

        ds_row = conn.execute(
            "SELECT id FROM candle_datasets WHERE symbol=? "
            "ORDER BY created_at DESC LIMIT 1",
            (symbol,),
        ).fetchone()
        did = dataset_id or (ds_row["id"] if ds_row else None)

        all_labels = []
        for split in ["train", "val", "test"]:
            rows = conn.execute(
                "SELECT labels FROM candle_dataset_samples "
                "WHERE dataset_id=? AND split_name=? AND symbol=? ORDER BY ts",
                (did, split, symbol),
            ).fetchall()
            all_labels.extend([json.loads(r["labels"]) for r in rows])
        conn.close()

        # Walk-forward windows
        secs_per_month = 30.44 * 24 * 3600
        retrain_secs = self.retrain_months * secs_per_month
        test_secs = self.test_months * secs_per_month

        t_start = ts_all[0]
        t_end = ts_all[-1]

        all_equity = [0.0]
        all_metrics_list = []
        window = 0

        cursor = t_start + retrain_secs  # first test window starts after initial train

        while cursor < t_end:
            test_end = min(cursor + test_secs, t_end)

            # Train on everything before cursor
            train_mask = ts_all < cursor
            test_mask = (ts_all >= cursor) & (ts_all < test_end)

            n_train = train_mask.sum()
            n_test = test_mask.sum()

            if n_train < 100 or n_test < 10:
                cursor = test_end
                continue

            from lightgbm import LGBMClassifier
            model = LGBMClassifier(
                n_estimators=500, max_depth=6, num_leaves=63,
                learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                random_state=42, verbosity=-1,
            )
            model.fit(X_all[train_mask], y_all[train_mask])

            y_proba = model.predict_proba(X_all[test_mask])[:, 1]

            test_indices = np.where(test_mask)[0]
            test_labels = [all_labels[j] for j in test_indices]

            wf_metrics, wf_equity = self.evaluator.evaluate(
                y_proba, test_labels, symbol,
            )
            wf_metrics.symbol = f"{symbol}_wf{window}"

            # Extend equity curve
            if len(wf_equity) > 1:
                base = all_equity[-1]
                all_equity.extend([base + e for e in wf_equity[1:]])

            all_metrics_list.append(wf_metrics.to_dict())
            window += 1
            cursor = test_end

        # Aggregate
        total_trades = sum(m["total_trades"] for m in all_metrics_list)
        total_pnl = sum(m["total_pnl_pips"] for m in all_metrics_list)
        total_wins = sum(m["wins"] for m in all_metrics_list)

        return {
            "symbol": symbol,
            "status": "ok",
            "walk_forward_windows": window,
            "total_trades": total_trades,
            "total_pnl_pips": round(total_pnl, 2),
            "overall_win_rate": round(total_wins / total_trades, 4) if total_trades else 0,
            "equity_curve": all_equity,
            "window_metrics": all_metrics_list,
        }


# ---------------------------------------------------------------------------
# Plot helper
# ---------------------------------------------------------------------------
def plot_equity_curve(
    equity: list[float],
    symbol: str,
    output_path: str | Path,
    title: str = "",
) -> str:
    """Save equity curve as PNG. Returns path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(equity, linewidth=1.0, color="#2196F3")
    ax.fill_between(range(len(equity)), equity, alpha=0.15, color="#2196F3")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_title(title or f"{symbol} — Equity Curve (pips)")
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Cumulative PnL (pips)")
    ax.grid(True, alpha=0.3)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Equity curve saved: %s", out)
    return str(out)


def plot_calibration_curve(
    calibration: list[dict],
    symbol: str,
    output_path: str | Path,
) -> str:
    """Save calibration curve as PNG. Returns path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    predicted = [b["avg_predicted"] for b in calibration if b["count"] > 0]
    actual = [b["avg_actual"] for b in calibration if b["count"] > 0]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
    ax.plot(predicted, actual, "o-", color="#FF5722", label="Model")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(f"{symbol} — Calibration Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(out)
